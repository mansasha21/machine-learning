from fbprophet import Prophet

def fit(df):
  model = Prophet()
  model.fit(df)
  return model


def predict(model, amountOfDays):
  futureDF = model.make_future_dataframe(amountOfDays)
  return model.predict(futureDF)


if __name__ == "__main__":
    import pandas as pd
    import sys
    
    if len(sys.argv) < 2:
        print("incorrect format")
        df = pd.read_csv('example.csv')
    else:
        df = pd.read_csv(sys.argv[1])
    
    if df.empty:
        print("problem with path")
    else:
        model = fit(df)
        forecast = predict(model, 100)
        model.plot(forecast)
        
    