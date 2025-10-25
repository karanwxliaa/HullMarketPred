# Hull Market Prediction

## Overview
Predict next day S and P five hundred returns and decide the daily allocation. The target is to beat the index while keeping volatility within one hundred twenty percent of the market. The metric is a Sharpe like score that penalizes excess volatility and underperformance.

## Problem statement
We predict next day market return and next day excess return over the risk free rate. We choose an allocation from zero to two. We aim to outperform the index within the stated risk limit. We will use the daily data provided by the competition.

## Data
Train has eight thousand nine hundred ninety rows and ninety eight columns. Features come in families D E I M P S V. The targets are forward returns and market forward excess returns. Family D has no missing values. Families M S V E have notable missing values.

## EDA summary
The mean of next day excess return is near zero with a small positive tilt. Daily standard deviation is about one percent. Lag one autocorrelation is minus zero point zero four four seven five which suggests mild mean reversion.
By realized volatility terciles we see a U shape. Low volatility days have a small positive average. Mid volatility days have a small negative average. High volatility days have a small positive average.
Top absolute correlations with next day excess return include M four at minus zero point zero six six. V thirteen at plus zero point zero six two. M one at plus zero point zero four six. D two and D one at about plus zero point zero three four. These are in sample and small. Treat them with care.
Decile analysis shows shapes that are useful. M four is negative at the top deciles and positive at the low deciles. V thirteen and M one rise with the feature value. This hints at non linear effects.
PCA suggests many components are needed. The feature space is not very low rank.

## Files
The main notebook is hull_tactical_eda dot ipynb. It expects data under C colon backslash mnt backslash data backslash hull_tactical_data. Adjust paths if needed.

## Next steps
1. Use time safe cross validation with purge and embargo.
2. Impute missing values with forward safe methods.
3. Build small signals from the strongest families. Start with D and M. Include V features that show consistent shape.
4. Use simple models first. Try linear with regularization and tree based methods. Add monotonic bins if needed.
5. Track realized volatility of the allocation. Keep it within the rule.
6. Blend decorrelated signals and test in rolling windows. Validate out of sample.

## How to run the EDA
Create the conda env as shared earlier. Install the listed packages. Run the notebook. The Altair cells will render once Altair is installed.


