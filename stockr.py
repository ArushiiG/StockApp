
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 13:52:39 2021
@author: aaron, arushi, kashish, cyan
"""

# =============================================================================
# Front-End framework : https://docs.streamlit.io/library/api-reference
# In the below we have used streamlit for front end of Application
# =============================================================================


from datetime import datetime, timedelta # For Datetime calculation
from dateutil.relativedelta import relativedelta #For dynamically calculating the month/ year difference
from fbprophet import Prophet #For Non linear model
import numpy as np # For faster Mathematical calulations
import pandas as pd # For creating and manipulating  dataframes
import plotly.graph_objects as go # For Visualizations
import plotly.express as px # For Visualizations
from sklearn.linear_model import LinearRegression # For Linear Model
from sklearn.metrics import mean_squared_error # To calculate root mean square error
from sklearn.metrics import r2_score # To calculate the r2 score
import streamlit as st #For the UI of the application
import yfinance as yf # For fetching stocks data


# =============================================================================
# To Retain the information between reruns and set Page layout
# The function is intialising the session state variables if they are not
# intialised before
# st.set_page_config -> To set the layout as wide
# st.spinner -> Loading section
# =============================================================================
def intialise_session_variables():
    st.set_page_config(layout="wide")
    with st.spinner("Intialising Session variables"):      
        if 'all_data' not in st.session_state:
            st.session_state.all_data = ''
        if 'symbol' not in st.session_state:
            st.session_state.symbol = ''
        if 'all_info' not in st.session_state:
            st.session_state.all_info = ''
        if 'training_period' not in st.session_state:
            st.session_state.training_period = ''
        if 'days_for_prediction' not in st.session_state:
            st.session_state.days_for_prediction = ''
        if 'companylist' not in st.session_state:
            st.session_state.companylist = ''
        if 'lag_date' not in st.session_state:
            st.session_state.lag_date = 0
           
           
# =============================================================================
# To check the date time format of the start date and the end date
# =============================================================================
def date_checker(start,end,df):
    if start>=df.iloc[0,0] and end <= df.iloc[-1,0]:
        return True
    else:
        return False
   
# =============================================================================
# The start of Visualization for STOCKr
# st.write -> Enters the text in the webpage in a Markdown fashion
# st.session_state.X -> Variable X is mentioned in "intialise_session_variable"
# st.sidebar.selectbox -> Dropdown option for the passed values
# yf.download -> To download all the stock data of the ticker selected by the user
# st.metric -> To provide 1-Day Close Price Change
# =============================================================================
def intializations():
    st.write(""" # STOCKr :bar_chart: """)
    st.session_state.companylist = pd.read_csv('companylist.csv')
    company_df = st.session_state.companylist
    company_name = st.sidebar.selectbox('Provide the Name of the Company',company_df['Name'])
    ticker = st.sidebar.selectbox("Stock Symbol",company_df[company_df['Name']==company_name])
    if st.session_state.symbol!=ticker:
        st.session_state.symbol=ticker
        st.session_state.all_data = yf.download(ticker)
    temp_df = st.session_state.all_data
    if len(temp_df) > 100 :
        if 'Date' not in temp_df.columns:
            historical_df = st.session_state.all_data
            # In order to get date column
            historical_df.reset_index(inplace=True)      
        else:
            historical_df = st.session_state.all_data          
        st.session_state.all_info = yf.Ticker(st.session_state.symbol).info
        # The try except block checks if there are any extra information provided for the company, if not then it will throw an error
        try:
            business_summary(ticker,company_name)
        except:
            historical_df = st.session_state.all_data
            st.metric(label = f"{company_name} (Current Close Price)",value=round(historical_df.iloc[-1].Close,2),delta =round(historical_df.iloc[-1].Close-historical_df.iloc[-2].Close,2) )
            st.write(f"#### _Stock Record for {ticker} found: {len(historical_df)}_")
        return historical_df
    else:
        return pd.DataFrame(columns=['a','b'])
   
# =============================================================================
# Provides a numerical summary about the company
# st.container -> Used to segregate the visualisation at one place
# st.columns -> Used to bifurcate the visualization in different columns
# pd.DataFrame -> To create dataframes to store and manipulate data
# st.sidebar.X -> Widget(X) will be shown in the sidebar of the application
# =============================================================================
def business_summary(ticker,company_name):
    ebitda = st.session_state.all_info['ebitda']
    prev_close = st.session_state.all_info['previousClose']
    payout_ratio = st.session_state.all_info['payoutRatio']
    open_price = st.session_state.all_info['open']
    st.write(f"## {ticker}'s Business Summary")
    with st.container():
        col1,col2 = st.columns(2)
        with col1:           
            business_info_data= [
                ('Ebitda',round(ebitda,2)),
                ('Previous Close',round(prev_close,2)),
                ('Payout Ratio',round(payout_ratio*100,2)),
                ('Open',round(open_price,2))              
                ]
            business_info_df = pd.DataFrame(business_info_data,columns=['Statistic','Values'])
            business_info_df.set_index('Statistic',inplace=True)
            st.table(business_info_df)
        with col2:
            historical_df = st.session_state.all_data
            st.metric(label = f"{company_name} (Current Close Price)",value=round(historical_df.iloc[-1].Close,2),delta =round(historical_df.iloc[-1].Close-historical_df.iloc[-2].Close,2))
            st.write(f"## _Stock Record for {ticker} found: {len(historical_df)}_")
    st.sidebar.write(f"# {st.session_state.all_info['longName']} Business Summary")
    st.sidebar.write(f"_{st.session_state.all_info['longBusinessSummary']}_")
  
# =============================================================================
# The start date and the end date is taken as an input from the user
# After validating the date via date_checker the code either take input from set frequency or date
# input depending on user input
# After providing the Summary statistics check for conditions for Visualisations
# If data is more than 2 then show Volume Trend
# st.date_input -> To take user input in the date format
# =============================================================================
def descriptive_statistics(historical_df):
    with st.container():
        # Date - input
        start_date_col,end_date_col = st.columns(2)      
        with start_date_col:
            # By default the start date will be set for 1 year earlier            
            start_date = st.date_input("Start Date",min_value=historical_df.iloc[0,0].to_pydatetime(),max_value=historical_df.iloc[-1,0].to_pydatetime(),value=historical_df.iloc[0,0].to_pydatetime())
        with end_date_col:    
            end_date = st.date_input("End Date",min_value=start_date,max_value=historical_df.iloc[-1,0].to_pydatetime(),value=historical_df.iloc[-1,0].to_pydatetime())
        # Select time range of the data  
        date_true = date_checker(start_date,end_date,historical_df)
        if date_true:
            data_range = all_time_range("descriptive_time_range",st.session_state.all_data)
            if len(data_range) == 0:  
                stockdf = historical_df[(historical_df['Date'] >= datetime.strftime(start_date,"%Y-%m-%d")) &( historical_df['Date']<=datetime.strftime(end_date,"%Y-%m-%d"))].copy()
                st.write(f'## Summary Statistics of Close Price [{start_date}] to [{end_date}]')
            else:
                stockdf = data_range.copy()
                start_date = datetime.strftime(stockdf.iloc[0,0],"%Y-%m-%d")
                end_date = datetime.strftime(stockdf.iloc[-1,0],"%Y-%m-%d")
                st.write(f'## Summary Statistics of Close Price [{start_date}] to [{end_date}]')      
            if len(stockdf) != 0:
                # Statistics for Column 1
                desp_col1_data= [
                    ('Average Closing Price',round(np.mean(stockdf.Close),2)),
                    ('Median',round(np.median(stockdf.Close),2))
                    ]
                desp_col1_df = pd.DataFrame(desp_col1_data,columns=['Statistic','Values'])             
                desp_col1_df.set_index('Statistic',inplace=True)
                # Statistics for Column 2
                desp_col2_data= [
                    ('Inter Quartile Range',round((np.percentile(stockdf.Close, 75))-(np.percentile(stockdf.Close, 25)),2)),
                    ('Standard Variation',round(np.std(stockdf.Close),2))
                    ]
                desp_col2_df = pd.DataFrame(desp_col2_data,columns=['Statistic','Values'])      
                desp_col2_df.set_index('Statistic',inplace=True)      
                desp_col1,desp_col2 = st.columns(2)
                with desp_col1:
                    st.table(desp_col1_df)
                with desp_col2:
                    st.table(desp_col2_df)            
                candlestick(stockdf) # Calling Candlestick Visualization
                if len(stockdf)>2:
                    volume(stockdf) # Calling Volume Trend Visualization
                else:
                    st.info('More information required for Volume Trend')
                with st.container():
                    if len(stockdf) > 30:
                        macd(stockdf) # Calling MACD Visualization
                    else:
                         st.info('Minimum 30 days worth of data required for MACD')                      
                    ma_and_bollinger_band(stockdf) # Calling Simple Moving Average and Bollinger Band Visualization
            else:
                st.info("No data available for this time range")
        else:
            st.error("Please enter the correct date format")
                   
# =============================================================================
# Candlestick Visualization
# https://plotly.com/python/candlestick-charts/
# go.Figure -> Used to insert the type of graphs required
# st.plotly_chart -> Used to print the plot on the streamlit interface
# =============================================================================
def candlestick(stockdf):
    st.write("## Stock Candle Stick Trend ")
    fig = go.Figure(data=go.Candlestick(x=stockdf['Date'],open=stockdf['Open'], high=stockdf['High'], low=stockdf['Low'], close=stockdf['Close']))
    st.plotly_chart(fig,use_container_width=True)
   
# =============================================================================
# Moving Average Converge Divergence (MACD) Visualization  
# px.line -> for line graph in Plotly  
# =============================================================================
def macd(stockdf):
    st.write("## MACD")        
    #Adding columns for the chart
    stockdf['EMA12'] = stockdf['Adj Close'].ewm(span=12).mean()
    stockdf['EMA26'] = stockdf['Adj Close'].ewm(span=26).mean()            
    stockdf['MACD'] = stockdf['EMA12']-stockdf['EMA26']
    stockdf['Signal_line'] = stockdf['MACD'].ewm(span=9).mean()      
    st.plotly_chart(px.line(stockdf,x='Date',y=['Signal_line','MACD']),use_container_width=True)    

# =============================================================================
# Moving Average and Bollinger Band Visualization
# https://medium.com/codex/how-to-calculate-bollinger-bands-of-a-stock-with-python-f9f7d1184fc3
# st.info -> Used as a pop-up message for error handling
# =============================================================================
def ma_and_bollinger_band(stockdf):
    try:
        st.write('## Simple Moving Average')
        minimum = 2
        maximum = int(len(stockdf)*0.25)
        ma_val = (minimum+maximum)/2
        if ma_val<2:
            ma_val=2
        else:
            ma_val = int(ma_val)
        ma_number = int(st.number_input('Enter the moving average number',step=1,value=ma_val,min_value=minimum,max_value=maximum,help=f"Enter the value between {minimum} to {maximum}"))      
        # Calculation for Simple Moving Average and Bollinger Band    
        # Rolling mean and standard deviation
        stockdf['MA'] = stockdf['Close'].rolling(window = ma_number).mean()
        stockdf['SD'] = stockdf['Close'].rolling(window = ma_number).std()
        stockdf['UBB'] = stockdf['MA'] + stockdf['SD'] * 2
        stockdf['LBB'] = stockdf['MA'] - stockdf['SD'] * 2
        stockdf.dropna(inplace=True) # Dropping rows with NA values
       
        # Simple Moving Average Graph
        if len(stockdf) != 0:
            fig = go.Figure()        
            fig.add_trace(go.Candlestick(x=stockdf['Date'],open=stockdf['Open'], high=stockdf['High'], low=stockdf['Low'], close=stockdf['Close'],name= "Stock Trend"))
            fig.add_trace(go.Scatter(x=stockdf['Date'], y=stockdf['MA'],
                                mode='lines',
                                name='Moving Average',marker=dict(
                  color='LightSkyBlue',
                  line=dict(width=3)
               )))
           
            st.plotly_chart(fig,use_container_width=True)
            # Bollinger Band graph
            st.write("## Bollinger Band")  
            fig2 = go.Figure()
            fig2.add_trace(go.Candlestick(x=stockdf['Date'],open=stockdf['Open'], high=stockdf['High'], low=stockdf['Low'], close=stockdf['Close'],name = "Stock Trend"))
            fig2.add_trace(go.Scatter(x=stockdf['Date'], y=stockdf['MA'],
                                mode='lines',
                                name='Moving Average'))
            fig2.add_trace(go.Scatter(x=stockdf['Date'], y=stockdf['UBB'],
                                mode='lines',
                                name='Upper Bollinger Band'))
            fig2.add_trace(go.Scatter(x=stockdf['Date'], y=stockdf['LBB'],
                                mode='lines',
                                name='Lower Bollinger Band'))
            st.plotly_chart(fig2,use_container_width=True)
        else:
            st.info('More data required for Moving Average and Bollinger Band')
    except:
        st.info('More data required for Moving Average and Bollinger Band')
           
# =============================================================================
# Volume Trend front end      
# =============================================================================
def volume(stockdf):
    st.write("## Volume Trend")
    st.plotly_chart(px.line(stockdf,x="Date",y=['Volume']),use_container_width=True)

# =============================================================================
# Stock Market Predictions
# From here the code will call the linear and the non-linear model    
# =============================================================================
def stock_market_prediction(historical_df):
    st.write("## Stock Market Prediction")  
   
    min_training_val = int(len(historical_df)*.6)
    max_training_val = int(len(historical_df)*.8)
    with st.spinner('Loading Linear Model Graph for you!'):
        with st.container():
            st.write('#### _Simple Linear Regression (Linear model)_')
            col1,col2 = st.columns(2)
            with col1:
                st.session_state.training_period = int(st.number_input("Please enter the training period in days",value=min_training_val,min_value=min_training_val,max_value=max_training_val,help=f"You can choose the value from {min_training_val} to {max_training_val}",step=1))
            with col2:
                st.session_state.days_for_prediction = int(st.number_input("Enter the number of days for which you want to predict",value=15,min_value=1,max_value = 30,help="You can predict for upto 30 days"))            
            if historical_df['Date'].iloc[-1] < datetime.now():              
                if st.session_state.lag_date ==0:
                    st.session_state.lag_date = (datetime.now()-historical_df['Date'].iloc[-1]).days        
                    st.session_state.days_for_prediction += st.session_state.lag_date
                else:
                    st.session_state.days_for_prediction += st.session_state.lag_date
            with st.spinner("Calculating Linear Model"):
                linear_predictor(historical_df)
            st.write('#### _FB Prophet (Non-Linear Model)_')    
            if st.button('Predict'):
               
                    fbprophet_predictor(historical_df)
     
# =============================================================================
# Code Reference: #https://www.kaggle.com/ahmetax/fbprophet-and-plotly-example       
# =============================================================================
def fbprophet_predictor(historical_df):
    with st.spinner('Calculating your Non-linear model'):
        time_frame = datetime.strftime(datetime.now()-timedelta(days=st.session_state.training_period), '%Y-%m-%d')
        df = historical_df[historical_df['Date']>=time_frame].copy()
        #st.write(df)
        model = Prophet(changepoint_prior_scale=0.5,daily_seasonality=True,yearly_seasonality=True)
        #model = Prophet()
        df[['ds','y']] = df[['Date','Close']]
        model.fit(df)      
        future= model.make_future_dataframe(periods =st.session_state.days_for_prediction,freq = 'D' )
        forecast = model.predict(future)        
        fig = go.Figure()
        with st.spinner('Almost Done'):
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'],
                            mode='lines',
                            name='Original Data',marker=dict(
              color='LightSkyBlue',
              line=dict(width=3)
           )))
            fig.add_trace(go.Scatter(x=list(forecast['ds']), y=list(forecast['yhat']),
                    mode='lines',
                    name='Prediction',marker=dict(
              color='#FFBAD2',
              line=dict(width=3)
           )))
            fig.add_trace(go.Scatter(x=list(forecast['ds']), y=list(forecast['yhat_upper']),
                    mode='lines',
                    name='Prediction_upperband',marker=dict(
              color='#57b88f',
              line=dict(width=3)
           )))
            fig.add_trace(go.Scatter(x=list(forecast['ds']), y=list(forecast['yhat_lower']),
                    mode='lines',
                    name='Prediction_lower_band',marker=dict(
              color='#1705ff',
              line=dict(width=3)
           )))        
            st.plotly_chart(fig,use_container_width=True)
            rms = round(mean_squared_error(df['Close'], forecast[forecast['ds']<=datetime.strftime(df['Date'].iloc[-1],'%Y-%m-%d')]['yhat'],squared=False),2)
            r2 = round(r2_score(df['Close'], forecast[forecast['ds']<=datetime.strftime(df['Date'].iloc[-1],'%Y-%m-%d')]['yhat']),2)
            st.markdown(f"<h6 style='text-align: center; color: red;'>Root Mean Square: {rms}</h5>", unsafe_allow_html=True)
            st.markdown(f"<h6 style='text-align: center; color: red;'>R2 Score: {r2}</h5>", unsafe_allow_html=True)
                                
# =============================================================================
# Front end of the Linear Regression
# https://sumit-khedkar.medium.com/stock-market-prediction-using-python-article-1-the-straight-line-c23f26579b4d  
# =============================================================================
def linear_predictor(historical_df):
            predictor_df = historical_df        
            time_frame = datetime.strftime(datetime.now()-timedelta(days=st.session_state.training_period), '%Y-%m-%d')
            cropped_data = predictor_df[predictor_df['Date']>=time_frame].copy()  
            cropped_data.index = cropped_data['Date']          
            cropped_data.index = (cropped_data.index - pd.to_datetime(time_frame)).days
            y_learned,y_predict,newindex = prediction(cropped_data,'Close')
            #rf_prediction(cropped_data,'Close')
            old_x = pd.to_datetime(cropped_data.index, origin=time_frame, unit='D')
            old_df = pd.DataFrame(old_x,columns=['Date'])            
            # Here we will have to check if we are taking the proper data or not -- Required for Linear
            old_df['Close'] = y_learned.tolist()
            old_df['Close']=[float(i[0]) for i in old_df['Close']]      
            old_df['Tagging'] = 'Old_Predict'
            future_x = pd.to_datetime(newindex, origin=time_frame, unit='D')
            future_df = pd.DataFrame(future_x,columns=['Date'])
            future_df['Close']=y_predict.tolist()
            future_df['Close'] = [float(i[0]) for i in future_df['Close']]
            #st.write(future_df['Close'])
            future_df['Tagging'] = 'New_Predict'    
            final_df = old_df.append(future_df[1:])    
            #Linear predictor
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cropped_data['Date'], y=cropped_data['Close'],
                    mode='lines',
                    name='Original Data',marker=dict(
      color='LightSkyBlue',
      line=dict(width=3)
   )))
            fig.add_trace(go.Scatter(x=final_df['Date'], y=final_df['Close'],
                    mode='lines',
                    name='Trend Line',marker=dict(
      color='red',
      line=dict(width=2)
   )))
            st.plotly_chart(fig,use_container_width=True)
            rms = round(mean_squared_error(cropped_data['Close'], old_df['Close'],squared=False),2)
            r2 = round(r2_score(cropped_data['Close'], old_df['Close']),2)
            st.markdown(f"<h6 style='text-align: center; color: red;'>Root Mean Square: {rms}</h5>", unsafe_allow_html=True)
            st.markdown(f"<h6 style='text-align: center; color: red;'>R2 Score: {r2}</h5>", unsafe_allow_html=True)
                   
# =============================================================================
# Backend of the Linear Regression Model
# =============================================================================
def prediction(cropped_data,column_name):      
            y = np.asarray(cropped_data[column_name])
            x = np.asarray(cropped_data.index.values)  
            regression_model = LinearRegression()
            regression_model.fit(x.reshape(-1, 1), y.reshape(-1, 1)) # transpose the data
            y_learned = regression_model.predict(x.reshape(-1, 1))
            # To add more timeframes for the future dates as well
            newindex = np.asarray(pd.RangeIndex(start=x[-1], stop=x[-1] + st.session_state.days_for_prediction+1))    
            y_predict = regression_model.predict(newindex.reshape(-1, 1))
            return y_learned,y_predict,newindex
        
# =============================================================================
# Function is used to provide the time range pressed by the button
# For example: 1D, 5D etc  
# relativedelta -> To calculate the time frames on month and year level    
# =============================================================================
def all_time_range(key,data):
    col1,col2,col3,col4,col5,col6,col7,col8,col9,col10 = st.columns(10)
    temp = pd.DataFrame(columns=['a','b'])
    with col1:
        if st.button('1D',key=key):
            # The for loop here is to ensure we do not miss the last traded price          
            for i in range(1,5):
                temp = data[data['Date'] >= datetime.strftime(datetime.now()-timedelta(i), '%Y-%m-%d')]
                if len(temp)!=0:
                    break
    with col2:
        if st.button('5D',key=key):
            temp = data[data['Date'] >= datetime.strftime(datetime.now()-timedelta(5), '%Y-%m-%d')]                    
    with col3:
        if st.button('1M',key=key):
            temp = data[data['Date'] >= datetime.strftime(datetime.now()-relativedelta(months=1), '%Y-%m-%d')]          
    with col4:    
        if st.button('3M',key=key):
            temp = data[data['Date'] >= datetime.strftime(datetime.now()-relativedelta(months=3), '%Y-%m-%d')]
    with col5:    
        if st.button('6M',key=key):
            temp = data[data['Date'] >= datetime.strftime(datetime.now()-relativedelta(months=6), '%Y-%m-%d')]        
    with col6:    
        if st.button('YTD',key=key):
            temp = data[data['Date'] >= datetime.strftime(datetime.now()-relativedelta(month=1,day=1), '%Y-%m-%d')]
    with col7:    
        if st.button('1Y',key=key):
           temp = data[data['Date'] >= datetime.strftime(datetime.now()-relativedelta(years=1), '%Y-%m-%d')]
    with col8:    
        if st.button('2Y',key=key):
            temp =data[data['Date'] >= datetime.strftime(datetime.now()-relativedelta(years=2), '%Y-%m-%d')]
    with col9:    
        if st.button('5Y',key=key):
            temp = data[data['Date'] >= datetime.strftime(datetime.now()-relativedelta(years=5), '%Y-%m-%d')]
    with col10:    
        if st.button('Max',key=key):
            temp = st.session_state.all_data
    return temp

# =============================================================================
# Main will call all the required functions in the order that it is mentioned
# =============================================================================
def main():
    intialise_session_variables()  
    historical_df = intializations()
    if len(historical_df) == 0:
        st.info('Data for the company is currently not available :( ')
    else:
        descriptive_statistics(historical_df)
        stock_market_prediction(historical_df)
# =============================================================================
# Code will start from here and call main function    
# =============================================================================
if __name__ == "__main__":
    main()