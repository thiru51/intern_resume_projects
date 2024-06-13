import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf 
import tensorflow as tf
from tensorflow.keras.models import load_model


start= '2009-01-01'
end = '2019-12-31'

st.title('Stock Trend Prediction ')
user_input=st.text_input('Enter Stock Ticker', 'AAPL')
df=yf.download(user_input , start= start, end= end)

#Describe Data
st.subheader('Data from 2009 - 2019')
st.write(df.describe)


#visualization
st.subheader('Closing price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price vs Time Chart with 100MA & 200MA')
ma100= df.Close.rolling(100).mean
ma200 = df.Close.rolling(200).mean
fig = plt.figure(figsize = (12,6))
plt.plot(ma200, 'r')
plt.plot(ma100, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

#Splitting the datada into training and testing data
data_training =  pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing  =  pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])

 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


#splitting the data into x_train and y_train
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i-100 : i])
        y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)



#load my model
model = load_model('keras_model.h5')

#testing part 
past_100_days = data_training.tail(100)
final_df= pd.concat([past_100_days,data_testing], ignore_index=True)
input_data =  scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])


x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predeict(x_test)
scaler = scaler.scale_

scaler_factor = 1/scaler[0]
y_predicted = y_predicted * scaler_factor
y_test = y_test * scaler_factor



#final graph
st.subheader('Predicted vs Original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r' , label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
