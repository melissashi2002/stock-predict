'''
Useage:
  Currently Workable:
    predict_single_firm_naive(CompanyID):
'''

# import
from datetime import date
from datetime import timedelta
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GET_METHOD='get'

import requests
from pytrends.request import TrendReq

'''
List of words:
Positive Adjectives:
These words often highlight the strengths and desirable qualities of a company, such as:

Innovative
Responsive
Empowered
Progressive
Collaborative
Trusting
Rewarding
Zealous
Engaging
Transparent
Negative Adjectives:
These are used to critique or bring attention to less favorable aspects of a company, such as:

Abusive
Bureaucratic
Deceitful
Toxic
Rigid
Unsupportive
Combative
Disrespectful
Restrictive
Neutral Adjectives:
These adjectives are more objective and can be seen in both positive and negative contexts depending on the perspective:

Established
Competitive
Formal
Hierarchical
Traditional
Structured
'''

# from pytrends.request import TrendReq as UTrendReq

'''
headers = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'en-US,en;q=0.9',
    'cache-control': 'max-age=0',
    # 'cookie': '__utma=10102256.911987879.1711738581.1711738598.1711738598.1; __utmc=10102256; __utmz=10102256.1711738598.1.1.utmcsr=trends.google.com|utmccn=(referral)|utmcmd=referral|utmcct=/; __utmb=10102256.2.10.1711738598; HSID=ADlQ9DWzIFc945Xhf; SSID=AcTl7Dy8S3-9mgPYu; APISID=hx4s73b9WLMfjP1W/AmWKFW5CWT_Oo2tML; SAPISID=MvqP5fUj-Or92LKj/AOMYL2BEG23g1XV5W; __Secure-1PAPISID=MvqP5fUj-Or92LKj/AOMYL2BEG23g1XV5W; __Secure-3PAPISID=MvqP5fUj-Or92LKj/AOMYL2BEG23g1XV5W; SID=g.a000iAjoJv0rqCM2r97tQIJvyVaaAh9e-B5iyLk5GUdrCDkMU95t3KgQuQswYz34Jfnb1DJfGgACgYKAYgSAQASFQHGX2MiuRLPPMY6v26fCHIPY9wd2BoVAUF8yKqgYNXOollyfeCoTOQ6Cf3t0076; __Secure-1PSID=g.a000iAjoJv0rqCM2r97tQIJvyVaaAh9e-B5iyLk5GUdrCDkMU95tKaQJWPidikN7xMskOcf0ngACgYKAecSAQASFQHGX2MidZwwGRfLiwmmKEGjJm1CZxoVAUF8yKobPMbcbN6aW-Pub4QOYpWT0076; __Secure-3PSID=g.a000iAjoJv0rqCM2r97tQIJvyVaaAh9e-B5iyLk5GUdrCDkMU95tabLecRlZIZOXbmuUCprDogACgYKAYkSAQASFQHGX2MijrfVUpOx-_a8yrU09qtNfxoVAUF8yKq0VXddkyLqV5i-HAlowo1g0076; 1P_JAR=2024-03-29-18; SEARCH_SAMESITE=CgQI45oB; AEC=Ae3NU9PMwsDGKaO16SqENTNWaBlw5PZ_mLqtBLvkYusmzpcNoOM-HhnwYQ; NID=512=GbddApdtB_QJ-dZuyp53fQFYUni-Q1pAIbXG5ULr6T32EjREM8kYDSfrkgNabtoqd2P47SSfe7C0sARlS5DR5YZYEYCpPYBDU1mqSPXsRiIz3oVXy1UNz4Xszm3XN8U4BBDv7qWPeeIbrmbmDpECvnUtKk8celS6AgU2VvN4qcfOJ4iXQ1iVm0wEL48MRywnLF98dn-ogqnA9BC92f9MNqvhVYyXzpoL0vTgsZ7yHcmjtOYnYTltSK9WsQlnX-yAx9SkOeJ1CCrFrCXa6fI78ay3_j26glCze0gAOjhWtaqzqFuWyd8SVPt4u1jBTmOaZC18; _gid=GA1.3.475312357.1711738581; OTZ=7490576_76_80_104160_76_446820; _ga=GA1.3.911987879.1711738581; _ga_VWZPXDNJJB=GS1.1.1711738581.1.1.1711738972.0.0.0; __Secure-1PSIDTS=sidts-CjEB7F1E_F-Wpw5jiBUNhU-Pqn3viF_V8adw5iEt69AlUkC3dyTeWcCLGPRv4SeRSWbXEAA; __Secure-3PSIDTS=sidts-CjEB7F1E_F-Wpw5jiBUNhU-Pqn3viF_V8adw5iEt69AlUkC3dyTeWcCLGPRv4SeRSWbXEAA; SIDCC=AKEyXzXGvIcvfXBV4hv2BMZeq3vmfLJt6Ner19Rx2r6u2qaMU-km-gTydfHCDPYC_Im9vhJaKAo; __Secure-1PSIDCC=AKEyXzXMMRgLELfnNJ3bOTqj-JPHvW8ipV-G_cOKi65XH-EAOz7c0iBrQCrYD_zyUir0MgYbpX8; __Secure-3PSIDCC=AKEyXzWi6AeUWswAiDg9RBZYhWL-n1mTQcwS3D_xZKsfCLVcwmAh1TRD-QkJ3ZTZ_WZ1gpIkyHsT',
    'sec-ch-ua': '"Google Chrome";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
    'sec-ch-ua-arch': '"arm"',
    'sec-ch-ua-bitness': '"64"',
    'sec-ch-ua-full-version': '"123.0.6312.87"',
    'sec-ch-ua-full-version-list': '"Google Chrome";v="123.0.6312.87", "Not:A-Brand";v="8.0.0.0", "Chromium";v="123.0.6312.87"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-model': '""',
    'sec-ch-ua-platform': '"macOS"',
    'sec-ch-ua-platform-version': '"13.5.2"',
    'sec-ch-ua-wow64': '?0',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'none',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'x-client-data': 'CIq2yQEIo7bJAQiJksoBCKmdygEIqpfLAQiTocsBCIagzQEI2/XNAQjP/80BGI/OzQEYyvjNARjrjaUX',
}
'''

# class TrendReq(UTrendReq):
#     def _get_data(self, url, method=GET_METHOD, trim_chars=0, **kwargs):
#         return super()._get_data(url, method=GET_METHOD, trim_chars=trim_chars, headers=headers, **kwargs)
# set up
word_list = ['innovative', 'competitive', 'abusive', 'progressive', 'threatening', 'engaging']
# word_list = ['valuable', 'successful', 'candid', 'organic', 'steadfast', 'talented', 'unique', 'interesting', 'functional', 'educated', 'ready', 'credible', 'substantial', 'workable', 'perfect', 'possible', 'supreme', 'focused', 'maintainable', 'threatening']
# word_list = ['valuable']
pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25), retries=3, backoff_factor=0.1, requests_args={'verify':False})

''' 
helper function that produce a computer name that is more likely to be searched
'''
def smart_comp_name(comp_name):
  print(comp_name)
  cn_list = comp_name.split(' ')
  smart = ''
  if len(cn_list) >= 2:
    if cn_list[1] == 'Inc' or cn_list[1] == 'Inc.':
      smart = cn_list[0]
    else:
      smart = cn_list[0] + " " + cn_list[1]
  else:
    smart = cn_list[0]
  return smart

'''
helper function that add a new row to data, fetch day in history (for example, if fetch day is from monday, then we fetch data from last friday)
  data: CompanyID Date LastPrice Word1 Word2 ... GrowthRate
  firm_struct: CompanyId CompanyName
  date: datetime instance
  word_list: list of words to fetch data from pytrends
  pytrends: TrendReq instance
  can fetch data from at most 1 month ago. If date is today, fetch data from yesterday
'''
def add_par_day_firm(data, firm_struct, fetch_date, word_list, pytrends):
  smart_name = smart_comp_name(firm_struct['CompanyName'])
  
  # get actual date (only if today), if time is before 9pm, return
  if fetch_date == date.today() and datetime.now().hour < 21:
    print("this function can only fetch data from yesterday or before")
    return

  # check if fetch date is already in data
  if fetch_date in data['Date']:
    print(firm_struct['CompanyName'] + " already has data for " + str(fetch_date))
    return
  
  # get stock price day before fetch date
  stock_info = yf.download(firm_struct['CompanyId'], str(fetch_date - timedelta(days = 7)), str(fetch_date + timedelta(days = 1)))
  if stock_info.empty:
    print(firm_struct['CompanyName'] + " has no stock price data")
    return
  
  #get stock price day after fetch date
  stock_reverse = stock_info.sort_index(ascending=False)
  if stock_reverse.iloc[0].name.date() != fetch_date:
    print(stock_reverse)
    print(firm_struct['CompanyName'] + " has no stock price data at fetch date " + str(fetch_date))
    return
  fetch_day_price = stock_reverse.iloc[0]['Close']
  last_day_price = stock_reverse.iloc[1]['Close']

  last_day = stock_reverse.iloc[1].name.date()
  
  # calculate growth rate
  growth_rate = (fetch_day_price - last_day_price) / last_day_price


  # fetch data for word list from pytrends
  search_list = ['"' + smart_name +  '" + "' + e + '"' for e in word_list]
  # if date is within 7 days, fetch data for 7 days
  if last_day > date.today() - timedelta(days = 7):
    fetch_timeframe = "now 7-d"
  elif last_day > date.today() - timedelta(days = 30):
    fetch_timeframe = "today 1-m"
  else:
    fetch_timeframe = "today 3-m"

  word_data_list = []
  
  for i in range(len(word_list)):
    sub_list = search_list[i:i+1]
    pytrends.build_payload(sub_list, cat=0, timeframe=fetch_timeframe, geo='US')
    trend = pytrends.interest_over_time()
    trend = trend.drop(['isPartial'], axis=1)
    trend = trend.resample('D').sum()
    trend_data = trend.loc[str(last_day):str(fetch_date - timedelta(days = 1))].mean()[0]
    word_data_list.append(trend_data)
  
  # insert new data into data
  to_insert = [firm_struct['CompanyId'], fetch_date, last_day_price] + word_data_list + [growth_rate]
  data.loc[len(data)] = to_insert

'''
helper function to get data for a month, using "today 1-m" as timeframe. Get last 29 days of data
'''
def add_par_month_firm(data, x_data, firm_struct, word_list, pytrends, fetch_date = date.today() - timedelta(days = 1), fetch_timeframe = "today 3-m"):
  smart_name = smart_comp_name(firm_struct['CompanyName'])
  today = fetch_date
  
  # get stock price day before fetch date
  stock_info = yf.download(firm_struct['CompanyId'], str(today - timedelta(days = 90)), str(today + timedelta(days = 1)))
  if stock_info.empty:
    print(firm_struct['CompanyName'] + " has no stock price data")
    return

  stock_reverse = stock_info.sort_index(ascending=False)

  X = pd.DataFrame

  pytrend_dict = {}

  search_list = ['"' + smart_name +  '" + "' + e + '"' for e in word_list]

  for i in range(len(word_list)):
    sub_list = search_list[i:i+1]
    pytrends.build_payload(sub_list, cat=0, timeframe=fetch_timeframe, geo='US')
    trend = pytrends.interest_over_time()
    trend = trend.drop(['isPartial'], axis=1)
    trend = trend.resample('D').sum()
    pytrend_dict[word_list[i]] = trend
  

  # loop through stock_info
  for i in range(len(stock_reverse) - 2):
    i_date = stock_reverse.iloc[i].name.date()
    last_date = stock_reverse.iloc[i + 1].name.date()
    if i_date == today:
      # update x_data
      current_day_price = stock_reverse.iloc[i]['Close']
      next_day_price = stock_reverse.iloc[i + 1]['Close']
      last_growth_rate = (next_day_price - current_day_price) / current_day_price
      word_data_list = []
      for j in range(len(word_list)):
        trend_for_word = pytrend_dict[word_list[j]]
        trend_data = trend_for_word.loc[str(last_date):str(i_date - timedelta(days = 1))].mean()[0]
        word_data_list.append(trend_data)
      to_insert = [firm_struct['CompanyId'], current_day_price, last_growth_rate] + word_data_list
      x_data.loc[len(x_data)] = to_insert
    else:
      next_next_day_price = stock_reverse.iloc[i + 2]['Close']
      next_day_price = stock_reverse.iloc[i + 1]['Close']
      current_day_price = stock_reverse.iloc[i]['Close']
      growth_rate = (next_day_price - current_day_price) / current_day_price
      last_growth_rate = (next_next_day_price - next_day_price) / next_day_price
      
      word_data_list = []
      for j in range(len(word_list)):
        trend_for_word = pytrend_dict[word_list[j]]
        trend_data = trend_for_word.loc[str(last_date):str(i_date - timedelta(days = 1))].mean()[0]
        word_data_list.append(trend_data)
      to_insert = [firm_struct['CompanyId'], stock_reverse.iloc[i + 1].name.date(), current_day_price, last_growth_rate] + word_data_list + [growth_rate]
      data.loc[len(data)] = to_insert



'''
helper function that get X for linear regression, it should be in the form of
'''
def get_X_linear_reg(firm_struct, word_list):
  # get today's date
  today = date.today() - timedelta(days = 1)
  # create a empty dataframe
  X = pd.DataFrame()
  add_par_day_firm(X, firm_struct, today, word_list, pytrends)
  # drop last column
  X = X.iloc[:, :-1]
  return X


'''
'''
def update_monthly_data():
  return

'''
helper function that update data from a range of dates, maximum from 85 days ago.
  data: CompanyID Date LastPrice Word1 Word2 ... GrowthRate
  firms: list of firm_struct
  word_list: list of words to fetch data from pytrends
  bg_date: datetime instance of beginning date
  ed_date: datetime instance of ending date
  pytrends: TrendReq instance
'''
def update_data_in_date_range(data, firms, word_list, bg_date, ed_date, pytrends):
  if ed_date > date.today():
    print("cannot fetch data from future")
    return
  if ed_date < bg_date:
    print("end date cannot be before beginning date")
    return
  if bg_date < date.today() - timedelta(days = 85):
    print("can only fetch data from 85 days ago")
    return
  
  last_day = bg_date

  while last_day <= ed_date:
    for index, firm in firms.iterrows():
      add_par_day_firm(data, firm, last_day, word_list, pytrends)
    last_day = last_day + timedelta(days = 1)

'''
helper function that update data several days till today, maximum from 85 days ago.
'''
def update_data_till_today(data, firms, word_list, num_days, pytrends):
  if num_days > 85:
    print("can only fetch data from 85 days ago")
    return
  update_data_in_date_range(data, firms, word_list, date.today() - timedelta(days = num_days), date.today() - timedelta(days = 1), pytrends)

'''
update data for past 30 days, saving the data_path
'''
def update_data(data_path, x_path, firms):
  # load data
  # word_list = ['valuable', 'successful', 'candid', 'organic', 'steadfast', 'talented', 'unique', 'interesting', 'functional', 'educated', 
  #              'ready', 'credible', 'substantial', 'workable', 'perfect', 'possible', 'supreme', 'focused', 'maintainable', 'threatening']
  # try:
  #   # Try to read the CSV file
  #   data = pd.read_csv(data_path)
  # except FileNotFoundError:
  #   print("File not found. Creating a new DataFrame.")
  #   data = pd.DataFrame(columns = ['CompanyID', 'Date', 'LastPrice'] + word_list + ['GrowthRate'])

  data = pd.DataFrame(columns = ['CompanyID', 'Date', 'LastPrice'] + word_list + ['GrowthRate'])
  x_data = pd.DataFrame(columns = ['CompanyID', 'LastPrice'] + word_list)
  for index, firm in firms.iterrows():
    add_par_month_firm(data, x_data, firm, word_list, pytrends)
  # save data
  data.to_csv(data_path, index=False)
  x_data.to_csv(x_path, index=False)

# def get_parx_day_firm(data, firm_struct, fetch_date, word_list, pytrends):

def train_firm_linear_reg(data, firm_struct):
  # train a linear regression model to predict growth rate
  # get data for firm
  firm_data = data[data['CompanyID'] == firm_struct['CompanyId']]
  firm_data = firm_data.drop(['CompanyID'], axis=1)
  
  # sum the data of the other firms to get the average data group by date, then concatinate rows to the desire date of firm_data, change the column names to not be the same as firm_data, then drop the date column
  other_firms = data[data['CompanyID'] != firm_struct['CompanyId']]
  # other_firms = other_firms.drop(['CompanyID'], axis=1)
  other_firms = other_firms.groupby('Date').mean()
  other_firms = other_firms.reset_index()
  other_firms.columns = ['Date'] + [e + '_avg' for e in other_firms.columns[1:]]
  firm_data = pd.merge(firm_data, other_firms, on='Date')
  firm_data = firm_data.drop(['Date'], axis=1)

  print(firm_data)

  # fit the model without test data
  X = firm_data.drop(['GrowthRate'], axis=1)
  y = firm_data['GrowthRate']
  model = LinearRegression()
  model.fit(X, y)

  return model

def train_all_firms_linear_reg(data):
  firms = pd.read_csv('companies_data.csv')
  firms = firms[['CompanyId', 'CompanyName']]
  models = {}
  for index, firm in firms.iterrows():
    models[firm['CompanyID']] = train_firm_linear_reg(data, firm)
  return models

def predict_firm_growth_rate_linear_reg(firm_struct, model):
  # get X data
  X = get_X_linear_reg(firm_struct, word_list)
  # predict growth rate
  growth_rate = model.predict(X)
  return growth_rate

def predict_all_firms_growth_rate_linear_reg():
  firms = pd.read_csv('companies_data.csv')
  firms = firms[['CompanyId', 'CompanyName']]
  data = pd.read_csv('test_data.csv')
  models = train_all_firms_linear_reg(data)
  predictions = {}
  for index, firm in firms.iterrows():
    predictions[firm['CompanyID']] = predict_firm_growth_rate_linear_reg(firm, models[firm['CompanyID']])
  return predictions


def predict_single_firm(CompanyID):
  firms = pd.read_csv('companies_data.csv')
  firm = firms[firms['CompanyId'] == CompanyID]

  data = pd.read_csv('train_data.csv')
  model = train_firm_linear_reg(data, firm)
  growth_rate = predict_firm_growth_rate_linear_reg(firm, model)
  return growth_rate

# model 0: linear regression, 1: Support Vector Regression, 2: Random Forest Regression
def predict_single_firm_naive(CompanyID, model = 1):
  firms = pd.read_csv('companies_data.csv')
  firm = firms[firms['CompanyId'] == CompanyID].iloc[0]

  firm_data = pd.DataFrame(columns = ['CompanyID', 'Date', 'LastPrice', 'LastGrowthRate'] + word_list + ['GrowthRate'])
  x_data = pd.DataFrame(columns = ['CompanyID', 'LastPrice', 'LastGrowthRate'] + word_list)


  add_par_month_firm(firm_data, x_data, firm, word_list, pytrends)

  firm_data = firm_data.drop(['CompanyID'], axis=1)
  firm_data = firm_data.drop(['Date'], axis=1)

  print(firm_data)

  # fit the model without test data
  X = firm_data.drop(['GrowthRate'], axis=1)
  y = firm_data['GrowthRate']
  x_data = x_data.drop(['CompanyID'], axis=1)

  X, x_data = data_scaler(X, x_data)

  if model == 0:
    model = LinearRegression()
    model.fit(X, y)
  elif model == 1:
    model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    model.fit(X, y)
  

  p_growth_rate = model.predict(x_data)[0]
  p_price = x_data.iloc[0]['LastPrice'] * (1 + p_growth_rate)

  return p_growth_rate, p_price

'''
Given X data pandas dataframe, each row is 'LastPrice', 'LastGrowthRate', and the word_list, return the scaled X data
'''
def data_scaler(X, x_data):
  # sclaer is a list of scalers for each column, the number x for each feature meaning that the feature value would be ranged from 0 to x.
  # for each value x in the row, we would update it to (x - min(row)) / (max(row) - min(row)) * scaler
  scaler = [5, 1] + [0.5 for i in range(len(word_list))]
  for i in range(len(scaler)):
    # get the min and max of the column
    min_val = min(X.iloc[:, i])
    max_val = max(X.iloc[:, i])
    # update the column
    X.iloc[:, i] = (X.iloc[:, i] - min_val) / (max_val - min_val) * scaler[i]
    x_data.iloc[0, i] = (x_data.iloc[0, i] - min_val) / (max_val - min_val) * scaler[i]
  return X, x_data 

  

firms = pd.read_csv('companies_data.csv')
firms = firms[['CompanyId', 'CompanyName']]
# firms = firms.head(2)
# update_data('train_data.csv', 'x_data.csv', firms)
test_firm = firms.sample(n=1, random_state=2)
p_growth_rate, p_price = predict_single_firm_naive(test_firm['CompanyId'].values[0])
print("Test Firm: ", test_firm['CompanyName'].values[0])
print('Predicted Growth Rate: ', p_growth_rate)
print('Predicted Price: ', p_price)

