'''
Step 1- Gather Data: Gather Historical data on s&P 500 dividend aristocrats.  Get data from financial APIS, databases, or webscraping(preffered)

Step 2- Data Preprocessing: clean and organize data. this includes missing values, removing columns, and convert data into machine 
learning algorithms

Step 3- Splitting data: dividing data into training, validation, and test sets
    - training set is used to train mode
    - validation set used to train hyperparameters
    - test set is used to evalute the performace of a model

Step 4- selecting a model: choose a machine learning algorithm suitable for regression task
    - linear regression
    - desicion trees
    - random forests
    - complex algorithms are gradient boosting or neural network

Step 5: Training the model: Once we choose the model, we will train the model on the training data. Adjust hyperparameters to improve 
performance, using validation set for tuning.

Step 6: Evalute the Model: Assess the Model's performance on the test set using appropriate evaluation metrics 
(e.g Mean Squared, Mean Absolute, R-squared.)

Step 7: Iterating and improving: If the model; performance is not satisfactory, consider refining the feature engineering, trying 
different algorithms, or adjusting hyperparameters


'''


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
"""
def save_data_to_txt(df, fileName):
                    fileName = 'sample.txt'
                    with open (fileName, 'w') as f:
                        f.write(df.to_string())
                    print(f"Data saved to: {fileName}")

#saves data as a csv format
def save_data_to_csv(df, fileName):
        csvFileName = 'sample.csv'
        df.to_csv(csvFileName, index=False)
        print(f"Data saved into: {fileName}")

#saves everything as an Excel/xlsx sheet
def save_data_to_excel(df, fileName):
        with pd.ExcelWriter(fileName, engine='openpyxl') as writer:
                        df.to_excel(writer,sheet_name = 'sheet_one')
"""



def get_stock_price_by_api(symbol, start_date, end_date):
    stock_data = yf.download (symbol, start_date, end_date)
    return stock_data

#get stock data using web scraping using selenium
"""
def get_stock_price_by_selenium(url, symbol = None):
    driver = webdriver.Chrome()

    driver.get(url)

    driver.maximize_window()

    time.sleep(1)

    historical_section = driver.find_element(By.XPATH, "//*[@id='quote-nav']/ul/li[4]/a")
    historical_section.click()


    time.sleep(3)


    time_period_dropdown = driver.find_element(By.XPATH, "//*[@id='Col1-1-HistoricalDataTable-Proxy']/section/div[1]/div[1]/div[1]/div/div/div")
    time_period_dropdown.click()

    time.sleep(3)

    five_year_duration = driver.find_element(By.XPATH, "//span[text()='5Y']")
    five_year_parent = five_year_duration.find_element(By.XPATH, "..")
    five_year_parent.click()

    time.sleep(2)

    #update table
    apply_button = driver.find_element(By.XPATH, "//span[text()='Apply']")
    parent_apply_button = apply_button.find_element(By.XPATH, "..")
    parent_apply_button.click()

    time.sleep(3)

    table = driver.find_element(By.XPATH, "//table[@data-test='historical-prices']")

    for i in range(13):
           driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
           time.sleep(2)

    rows = table.find_elements(By.TAG_NAME, "tr")
    print(len(rows))

    #extract data
    historcal_data      =[]

    date_list           =[]
    open_price_list     =[]
    high_price_list     =[]
    low_price_list      =[]
    close_price_list    =[]
    adj_close_list      =[]
    volume_list         =[] 


    for row in rows[1:]:
            columns = row.find_elements(By.TAG_NAME, 'td')

            try:
                date = columns[0].text
                open_price = columns[1].text
                high_price = columns[2].text
                low_price = columns[3].text
                close_price = columns[4].text
                adj_close = columns[5].text
                volume = columns[6].text
            except IndexError:
                date          = ""
                open_price    = ""
                high_price    = ""
                low_price     = ""
                close_price   = ""
                adj_close     = ""
                volume        = ""


            date_list.append(date)
            open_price_list.append(open_price)
            high_price_list.append(high_price)
            low_price_list.append(low_price)
            close_price_list.append(close_price)
            adj_close_list.append(adj_close)
            volume_list.append(volume) 


            historcal_data_dict = {
                'Date':date_list,
                'Open': open_price_list,
                'High': high_price_list,
                'Low': low_price_list,
                'Close': close_price_list,
                'Adj Close': adj_close_list,
                'Volume': volume_list
                       }
                
    time.sleep(5)

    driver.quit()

    return historcal_data_dict

def get_df_object(dict):
       df = pd.DataFrame(dict)
       return df
"""
"""
creates features and target variables from historical stock data
parameters:
    data: dataframe with historical date
    target_col: column representing target variable
    windowsize - number of past days used as feature
returns:
    X: dataframe with features
    y: series with target variables
"""
"""
def create_features_and_target(stock_data, target_col='Close', window_size=10 ):
    stock_data['Date'] = stock_data.index
    stock_data.reset_index(drop=True, inplace=True)

    for i in range(1, window_size + 1):
           col_name = f'{target_col}_lag_{i}'
           stock_data[col_name] = stock_data[target_col].shift(i)

    stock_data.dropna(inplace=True)

    X = stock_data.drop(['Date', target_col], axis=1)
    y=stock_data[target_col]

    return X,y
"""
"""
def train_linear_regression_model(X_train, y_train):
       model = LinearRegression()
       model.fit(X_train, y_train)
       return model
"""
"""
def plot_actual_vs_prediction(symbol, y_test, predictions):


       fig, ax = plt.subplots()
       ax.plot (y_test.index, y_test, label="Actual closing Prices", marker="o") 
       ax.plot(y_test.index, predictions, label="predicted closing Prices", linestyle="--", marker="x")
       ax.set_xlabel('Date')
       ax.set_ylabel('closing Prices')
       ax.set_title(f'Actual vs Predicted Prices for symbol: {symbol}')
       return fig
"""

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

    """features, target = create_features_and_target(stock_data)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.2,random_state=42)

    # Train the model
    model = train_linear_regression_model (X_train, y_train)

    #predict model"""
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