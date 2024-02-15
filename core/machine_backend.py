import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
import datetime as dt
#selenium imports
from selenium import webdriver   #used to interact with web browser
from selenium.webdriver.common.keys import Keys   #used to insert data using keystrokes
from selenium.webdriver.common.by import By   #used to select elements on the webpage
import time
#Sci-kit learning module imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
#matplotlibs imports
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
#Sequential imports
import numpy as np 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, LSTM

#used to get the list of all s&p500 companies
def get_sp500_members(url):
    #send get request
        response = requests.get(url)

        #checking if request is sent(status code)

        if response.status_code == 200:
                print(f"Request was sent {response.status_code}")
                #create beautiful soup object
                soup = BeautifulSoup(response.text, 'html.parser')

                table = soup.find('table', class_='wikitable sortable')


                data = {}
                symbols = []
                companies = []
                sectors = []
            
                if table:
                    rows = table.find_all('tr')[1:]
                    #collecting data for ticker symbols
                    for row in rows:
                            for a in row.find_all('td')[0]:
                                    if not a == '\n':
                                        symbols.append(a.text)
                    #remove unnecessary words
                    unnecessary_words = ['(Class A)', '(Class B)', '(Class C)', '(Previously PerkinElmer)'] 
                    for row in rows:
                        for a in row.find_all('td')[1]:
                            if a.text.strip() not in unnecessary_words:
                                companies.append(a.text)

                    #collecting data
                    for row in rows:
                            for a in row.find_all('td')[2]:
                                   sectors.append(a.text)

                    data['symbols'] = symbols
                    data['companies' ] = companies
                    data['sectors'] = sectors

                    #create a data frame
                    df = pd.DataFrame(data)


                    return df
                
                else:
                    return "Unable to get data object"
                  
        else:
                print(f"Request not sent: {response.status_code}")

#saves data as a txt file format


def get_stock_price_by_api(symbol, start_date, end_date):
    stock_data = yf.download (symbol, start_date, end_date)
    return stock_data

#Sequential
start_date = dt.datetime(2016, 1, 1)
end_date = dt.datetime(2021, 1, 1 )
def get_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, start_date, end_date)
    return data

def get_current_share_price(symbol, start_date, end_date=dt.datetime.now()):
    current_price = get_stock_data(symbol, start_date, end_date)
    return current_price


def generate_data(symbol, data, yesterday_date):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_days = 60
    X_train = []
    y_train = []
    
    for x in range(prediction_days, len(scaled_data)):
        X_train.append(scaled_data[x-prediction_days:x,0])
        y_train.append(scaled_data[x,0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train , (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout (0.2)) 
    model.add(LSTM(units=50, return_sequences=True)) 
    model.add(Dropout(0.2)) 
    model.add(LSTM(units=50)) 
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=25, batch_size=34)

    # Prepare the test data
    test_start_date = dt.datetime(2021, 1, 1)
    test_end_date = dt.datetime.now()
    test_data = get_stock_data(symbol, test_start_date, test_end_date)
    # Actual prices from the market
    actual_prices = test_data['Close'].values
    # Concatenate the training data with the test data
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days: ].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)
    X_test = []

    for x in range(prediction_days, len(model_inputs)):
        X_test.append(model_inputs[x-prediction_days:x, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_prices = model.predict(X_test)

    predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the test predictions
    fig, ax = plt.subplots()
    ax.plot(actual_prices, color='black', label=f'Actual {symbol} Prices' )
    ax.plot(predicted_prices, color='red', label=f'Predicted {symbol} Prices')
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{symbol} Share Price') 
    ax.set_title(f'{symbol} Share Price')

    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data. shape[0], real_data.shape[1], 1))
    
    current_share_price = get_current_share_price(symbol, yesterday_date).iloc[-1]['Close']


    
    future_share_price = model.predict(real_data)
    future_share_price = scaler.inverse_transform(future_share_price)

    percentage_difference = ((future_share_price - current_share_price) / future_share_price) * 100

    return{'graph': fig, 'prediction': future_share_price.tolist(), 'percentage':percentage_difference.tolist()}
    
def predicted_data(symbol):
    #url for page to scrape from

    start_date = dt.datetime(2019,1,1)
    end_date = dt.datetime(2024,1,1)


    stock_data = get_stock_price_by_api(symbol, start_date, end_date)

    #predictions = model.predict(X_test)

    #evalute model
    #mse = mean_squared_error(y_test, predictions)

    #print(f"mean squared error: {mse}")

    #predict future data(day, month, year)
    """last_day_features = features.tail(1)
    next_day_predictions = model.predict(last_day_features)
    print(next_day_predictions)"""

    #graph = plot_actual_vs_prediction(symbol, y_test, predictions)
    
    #return graph


"""
Xray current price 2/11 - 32.79

XRAY Tommorow Price 2/12 - predicted - 34.14, actual - 33.52

"""