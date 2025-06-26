from statsmodels.tsa.holtwinters import ExponentialSmoothing

def train_ets_model(train_data, seasonal_periods):
    model = ExponentialSmoothing(
        train_data,
        seasonal_periods=seasonal_periods,
        trend='add',
        seasonal='add',
        initialization_method="estimated",
        use_boxcox=True,
    ).fit()
    return model
