# MethodesQuantitatives

## Exercise 1 (5 points) – Time Series Decomposition
Decompose a real-world time series into trend, seasonal, and residual components.
Dataset: Monthly retail sales (file dataset1.txt)
1. Use regression techniques to separate the series into:
- Trend component
- Seasonal component
- Residual component
2. Visualize each component in separate plots.
3. Analyze the residuals: Is the decomposition satisfactory? Are there patterns left to be better
caught with finer analysis or decomposition?
4. Forecast values for 2025 and compare available ones with forecasts. How good is the model you
obtained? How can its forecasting capabilities be quantified?
## Exercise 2 (8 points) – SARIMA Modeling
Model and forecast a real-world seasonal time series using SARIMA.
Dataset: Electric Power Consumption
1. Data Exploration
• Load the dataset and inspect the first few rows to understand its structure.
• Handle missing values or anomalies if present.
2. Stationarity Analysis
• Use statistical tests (e.g., Augmented Dickey-Fuller) to check if the series is stationary.
• Apply differencing if necessary to achieve stationarity.
3. SARIMA Parameter Identification
• Use ACF and PACF plots to identify optimal parameters (p, d, q) and seasonal parameters (P,
D, Q)s.
• Consider the seasonal patterns of the data when selecting parameters.
4. Model Building
• Fit a SARIMA model using a library such as statsmodels in Python with the chosen
parameters.
• Train the model on the historical data.
5. Model Evaluation
• Evaluate the model performance.
6. Forecasting and Visualization
• Forecast future electricity consumption using the trained SARIMA model.
• Plot the forecast against actual data to assess the model’s accuracy.
## Exercise 3 (7 points) – Hidden Markov Model (HMM) for Regime Detection
Detect underlying regimes in a financial time series using HMM.
Dataset: S&P 500 index data (file dataset3.xlsx)
1. Assume there are hidden market regimes (e.g., “bull” and “bear” markets) that influence
returns.
2. Fit a Gaussian HMM to the returns to identify hidden states.
3. Assign each day to a regime and visualize the time series with regime coloring.
4. Analyze the characteristics of each regime (mean, variance) and interpret them in a financial
context.

Hint: Start with 2 or 3 hidden states and experiment with different numbers if needed.
