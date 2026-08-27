# Time Series Forecasting – Superstore Sales
This project performs time series forecasting on monthly sales data using two models:

- ARIMA

- Prophet

The workflow includes data loading, resampling, decomposition, autocorrelation analysis, model training, forecasting, evaluation and model comparison.

## Project Objective
Forecast monthly sales using classical statistical models and modern additive models.
The project demonstrates skills in:

- Time series preprocessing

- STL decomposition

- ACF/PACF analysis

- ARIMA modeling

- Prophet modeling

- Evaluation metrics (MAE, RMSE)

- Model comparison

## Dataset
The dataset is loaded from a CSV file containing Superstore sales data:

```python
df = pd.read_csv("Sample - Superstore.csv", parse_dates=["Order Date"], encoding="latin1") #insert the adress for the file .csv in the first part of the command

```
## Monthly resampling:

```python
ts = df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"].sum()
ts = ts.asfreq("ME")
```
## Time Series Visualization
```python
plt.figure(figsize=(12,5))
plt.plot(ts, label="Sales per month")
plt.title("Sales per month")
plt.xlabel("Year")
plt.ylabel("Sales")
plt.legend()
plt.show()
```
## STL Decomposition
```python
stl = STL(ts, period=12)
res = stl.fit()

fig = res.plot()
fig.set_size_inches(12,8)
plt.show()
```
The decomposition highlights:

- trend component

- seasonal component

- residual component

## ACF/PACF Analysis
```python
fig, ax = plt.subplots(2,1, figsize=(12,8))
plot_acf(ts.dropna(), ax=ax[0])
plot_pacf(ts.dropna(), ax=ax[1])
plt.show()
```
These plots help identify autoregressive and moving average components.

## ARIMA Model
### Train/test split:

```python
train = ts[:-12]
test = ts[-12:]
```
### Model fitting:

```python
model = ARIMA(train, order=(1,1,1))
arima_fit = model.fit()
print(arima_fit.summary())
```
### Forecast:

```python
arima_forecast = arima_fit.forecast(steps=12)
```
### Metrics:

```python
mae_arima = mean_absolute_error(test, arima_forecast)
rmse_arima = np.sqrt(mean_squared_error(test, arima_forecast))
```
### Plot:

```python
plt.figure(figsize=[12,5])
plt.plot(train, label="Train")
plt.plot(test, label="Test")
plt.plot(arima_forecast, label="ARIMA Forecast")
plt.legend()
plt.show()
```
## Prophet Model
### Data preparation:

```python
prophet_df = ts.reset_index()
prophet_df.columns = ["ds", "y"]
```
### Train/test split:

```python
train_prophet = prophet_df[:-12]
test_prophet = prophet_df[-12:]
```
### Model fitting:

```python
m = Prophet()
m.fit(train_prophet)
```
### Forecast:

```python
future = m.make_future_dataframe(periods=12, freq="ME")
forecast = m.predict(future)
```
### Metrics:

```python
prophet_pred = forecast["yhat"][-12:].values

mae_prophet = mean_absolute_error(test_prophet["y"], prophet_pred)
rmse_prophet = np.sqrt(mean_squared_error(test_prophet["y"], prophet_pred)
```
### Plot:

```python
m.plot(forecast)
plt.title("Forecast Prophet")
plt.show()
```
### Model Comparison
```python
print("ARIMA - MAE:", mae_arima, "RMSE:", rmse_arima)
print("Prophet - MAE:", mae_prophet, "RMSE:", rmse_prophet)

if rmse_prophet < rmse_arima:
    print("The best model is: PROPHET")
else:
    print("The best model is: ARIMA")
```
## Key Results
- STL decomposition reveals clear trend and seasonality.

- ACF/PACF suggest ARIMA(1,1,1) as a reasonable baseline model.

- Prophet achieves lower MAE and RMSE compared to ARIMA.

- Prophet is selected as the best model for this dataset.

## Technologies Used
- Python

- Pandas

- NumPy

- Matplotlib

- Seaborn

- Statsmodels

- Scikit‑learn

- Prophet

## Purpose of This Project
This project demonstrates:

- ability to handle real-world time series data

- understanding of decomposition and autocorrelation

- competence with ARIMA and Prophet models

- evaluation and comparison of forecasting models

- reproducible workflow for time series forecasting


## Possible Extensions
- SARIMA modeling

- Hyperparameter tuning

- Cross‑validation for time series

- Forecasting with LSTM or GRU

- Multivariate forecasting
