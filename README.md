# Ex.No: 6               HOLT WINTERS METHOD

### AIM:
To implement the Holt Winters Method Model using Python.
### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
Importing necessary modules
```
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
```
Load the dataset,perform data exploration
```
data = pd.read_csv('/content/AirPassengers.csv', parse_dates=['Month'],index_col='Month'
data.head()
```
Resample and plot data
```
data_monthly = data.resample('MS').sum() #Month start
data_monthly.head()
data_monthly.plot()
```
Scale the data and check for seasonality
```
scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten
scaled_data.plot() # The data seems to have additive trend and multiplicative seasonality
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(data_monthly, model="additive")
decomposition.plot()
plt.show()
```
Split test,train data,create a model using Holt-Winters method, train with train data and Evaluate
the model predictions against test data
```
scaled_data=scaled_data+1 # multiplicative seasonality cant handle non postive values, ye
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul').fit()
test_predictions_add = model_add.forecast(steps=len(test_data))
ax=train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add","test_data"])
ax.set_title('Visual evaluation')
np.sqrt(mean_squared_error(test_data, test_predictions_add))
np.sqrt(scaled_data.var()),scaled_data.mean()
```
Create teh final model and predict future data and plot it
```
final_model = ExponentialSmoothing(data_monthly, trend='add', seasonal='mul', seasonal_pe
final_predictions = final_model.forecast(steps=int(len(data_monthly)/4)) #for next year
ax=data_monthly.plot()
final_predictions.plot(ax=ax)
ax.legend(["data_monthly", "final_predictions"])
ax.set_xlabel('Number of monthly passengers')
ax.set_ylabel('Months')
ax.set_title('Prediction')
```

### OUTPUT:
Scaled_data plot:

![Screenshot 2025-04-26 160343](https://github.com/user-attachments/assets/94b06921-6d66-465b-96ea-4237a9b6047a)

Decomposed plot:

![Screenshot 2025-04-26 160427](https://github.com/user-attachments/assets/86dc81c4-70e3-40d4-be3f-30aac930d4fd)

TEST_PREDICTION


![Screenshot 2025-04-26 160502](https://github.com/user-attachments/assets/0a1733ed-7106-4353-9f96-02945394ae52)

Model performance metrics:
RMSE:
![Screenshot 2025-04-26 160516](https://github.com/user-attachments/assets/790326ef-89d8-43cf-a5af-8223eaf3c5a8)

Standard deviation and mean:

![Screenshot 2025-04-26 160535](https://github.com/user-attachments/assets/2f6c3598-aec7-4af1-8163-73227e6330e0)

FINAL_PREDICTION

![Screenshot 2025-04-26 160554](https://github.com/user-attachments/assets/d26adfb0-5d62-4ec2-9d76-4cc5d1f6259f)


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
