
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Optional

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor

# ---------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------

META_TARGET_COLS = [
    "forward_returns",
    "risk_free_rate",
    "market_forward_excess_returns",
]

LAG_LABEL_COLS = [
    "forward_returns",
    "risk_free_rate",
    "market_forward_excess_returns",
]

def add_lagged_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged_* versions of target columns to mirror the test schema."""
    df = df.copy()
    for col in LAG_LABEL_COLS:
        lag_name = f"lagged_{col}"
        if col in df.columns and lag_name not in df.columns:
            df[lag_name] = df[col].shift(1)
    return df


def drop_sparse_and_constant(
    X: pd.DataFrame,
    min_frac: float = 0.20,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop columns with coverage < min_frac or zero variance.
    Returns filtered DataFrame and the list of kept columns.
    """
    nn_frac = X.notnull().mean()
    keep = nn_frac[nn_frac >= min_frac].index.tolist()
    X_sub = X[keep]
    var = X_sub.var(numeric_only=True)
    keep_final = [c for c in keep if var.get(c, 0.0) > 0.0]
    return X[keep_final].copy(), keep_final


def get_feature_columns(train: pd.DataFrame, test: Optional[pd.DataFrame]) -> List[str]:
    """
    Determine usable feature columns by intersecting train and test columns,
    and dropping obvious non feature columns.
    """
    drop_cols = set(META_TARGET_COLS + ["is_scored"])
    if test is not None:
        common_cols = sorted(list(set(train.columns).intersection(test.columns)))
    else:
        common_cols = list(train.columns)
    feat_cols = [c for c in common_cols if c not in drop_cols]
    return feat_cols


# ---------------------------------------------------------------------
# Position mapping and metric proxy
# ---------------------------------------------------------------------

def winsorize(x: np.ndarray, low_q: float = 0.01, high_q: float = 0.99) -> np.ndarray:
    lo, hi = np.nanpercentile(x, [low_q * 100.0, high_q * 100.0])
    return np.clip(x, lo, hi)


def map_to_position(y_pred: np.ndarray, k: float) -> np.ndarray:
    """
    Linear map from predicted excess return to position in [0, 2].
    """
    yp = winsorize(np.asarray(y_pred, dtype=float))
    pos = 1.0 + k * yp
    return np.clip(pos, 0.0, 2.0)


@dataclass
class CVFoldResult:
    k_star: float
    mean_return: float
    vol_ratio: float
    sharpe_like: float


def evaluate_k_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    vol_cap: float = 1.2,
    k_grid: np.ndarray = np.linspace(0.0, 60.0, 121),
) -> CVFoldResult:
    """
    For one validation fold, choose k that maximizes mean(position * return)
    under a volatility ratio cap.
    """
    y_true = np.asarray(y_true, dtype=float)
    yp = winsorize(np.asarray(y_pred, dtype=float))
    base_vol = float(np.nanstd(y_true) + 1e-12)

    best = CVFoldResult(k_star=0.0, mean_return=-1e18, vol_ratio=0.0, sharpe_like=0.0)

    for k in k_grid:
        pos = np.clip(1.0 + k * yp, 0.0, 2.0)
        r = pos * y_true
        m = float(np.nanmean(r))
        s = float(np.nanstd(r) + 1e-12)
        vol_ratio = s / base_vol if base_vol > 0 else np.inf
        if vol_ratio <= vol_cap * 1.01:
            sharpe_like = m / s if s > 0 else 0.0
            if (m > best.mean_return + 1e-12) or (
                abs(m - best.mean_return) <= 1e-12 and sharpe_like > best.sharpe_like
            ):
                best = CVFoldResult(
                    k_star=float(k),
                    mean_return=m,
                    vol_ratio=float(vol_ratio),
                    sharpe_like=float(sharpe_like),
                )

    if best.mean_return <= -1e17:
        best = CVFoldResult(k_star=0.0, mean_return=0.0, vol_ratio=0.0, sharpe_like=0.0)
    return best


# ---------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------

def make_elasticnet_model(alpha: float = 5e-4, l1_ratio: float = 0.2) -> Pipeline:
    return Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler(with_mean=True, with_std=True)),
            ("est", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42, max_iter=5000)),
        ]
    )


def make_ridge_model(alpha: float = 1.0) -> Pipeline:
    return Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler(with_mean=True, with_std=True)),
            ("est", Ridge(alpha=alpha, random_state=42)),
        ]
    )


def make_rf_model(
    n_estimators: int = 200,
    min_samples_leaf: int = 2,
    max_features: str = "sqrt",
    random_state: int = 42,
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=-1,
        random_state=random_state,
    )


def make_pca_ridge_model(
    n_components: int = 30,
    alpha: float = 1.0,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler(with_mean=True, with_std=True)),
            ("pca", PCA(n_components=n_components, random_state=42)),
            ("est", Ridge(alpha=alpha, random_state=42)),
        ]
    )


def make_pls_model(
    n_components: int = 8,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler(with_mean=True, with_std=True)),
            ("pls", PLSRegression(n_components=n_components)),
        ]
    )


def make_hgb_model(
    max_depth: int = 3,
    learning_rate: float = 0.05,
    max_iter: int = 300,
    min_samples_leaf: int = 20,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        min_samples_leaf=min_samples_leaf,
        validation_fraction=None,
        random_state=42,
    )


# ---------------------------------------------------------------------
# Time series CV helper
# ---------------------------------------------------------------------

def time_series_cv_indices(
    n_samples: int,
    n_splits: int = 3,
    gap: int = 3,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Wrapper around TimeSeriesSplit that returns explicit index arrays.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    return list(tscv.split(np.arange(n_samples)))


# ---------------------------------------------------------------------
# Model training with k selection
# ---------------------------------------------------------------------

@dataclass
class TrainedModel:
    model: object
    feature_cols: List[str]
    k_final: float
    calib: Optional[IsotonicRegression] = None
    meta: Optional[dict] = None


def train_with_cv_and_k(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 3,
    gap: int = 3,
    vol_cap: float = 1.2,
    k_grid: np.ndarray = np.linspace(0.0, 60.0, 121),
    use_isotonic: bool = False,
) -> TrainedModel:
    """
    Generic helper:
      - run time series CV,
      - select k per fold,
      - aggregate to a final k,
      - optionally fit isotonic calibration on OOF predictions.
    """
    n_samples = len(X)
    splits = time_series_cv_indices(n_samples, n_splits=n_splits, gap=gap)

    k_list = []
    oof_pred = np.full(n_samples, np.nan)
    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        model.fit(X_tr, y_tr)
        pred_va = model.predict(X_va)
        oof_pred[va_idx] = pred_va
        res = evaluate_k_grid(y_va.values, pred_va, vol_cap=vol_cap, k_grid=k_grid)
        k_list.append(res.k_star)

    k_final = float(np.median(k_list)) if k_list else 0.0

    # Fit on full data
    model.fit(X, y)

    calib = None
    if use_isotonic:
        # Use OOF preds and positions from global k_final to fit monotone map
        mask = ~np.isnan(oof_pred)
        if mask.any():
            p_target = map_to_position(oof_pred[mask], k_final)
            calib = IsotonicRegression(y_min=0.0, y_max=2.0, out_of_bounds="clip")
            calib.fit(oof_pred[mask], p_target)

    meta = {
        "k_list": k_list,
        "k_final": k_final,
    }
    return TrainedModel(model=model, feature_cols=list(X.columns), k_final=k_final, calib=calib, meta=meta)


# ---------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------

def predict_positions(
    tm: TrainedModel,
    X_new: pd.DataFrame,
) -> np.ndarray:
    """
    Predict positions in [0, 2] for new feature matrix using trained model and calibration if present.
    """
    X = X_new[tm.feature_cols]
    y_pred = tm.model.predict(X)

    if tm.calib is not None:
        p = tm.calib.predict(y_pred)
        return np.clip(p, 0.0, 2.0)
    else:
        return map_to_position(y_pred, tm.k_final)
