import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt

start = "2011-01-01"
today = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction")

stock_list = pd.read_csv("https://www1.nseindia.com/content/indices/ind_nifty500list.csv")
stock_list = list(stock_list["Symbol"])

stock = st.selectbox("Select Stock :",stock_list)

n_days = st.text_input("Days of Prediction : ")

@st.cache
def load_data(ticker) :
    data = yf.download(ticker+'.NS',start,today)
    data.reset_index(inplace=True)
    return data



if st.button("Click Me!") :
    period = int(n_days)

    status = st.text("Loading Data")
    data = load_data(stock)
    status.text("Done..")


    df_train = data[['Date','Close']]
    df_train = df_train.iloc[:(-1*period),]
    df_train = df_train.rename(columns={"Date":"ds", "Close":"y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    fig1 = plot_plotly(m,forecast)
    st.plotly_chart(fig1)

    data2 = data[(-1*period):]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'],
                        mode='lines',
                        name='Past'))
    fig.add_trace(go.Scatter(x=data2['Date'], y=data2['Close'],
                        mode='lines',
                        name='Real'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                        mode='lines', 
                        name='Predicted'))
    
    fig.update_layout(width=1000)
    fig.update_layout(height=650)
    st.plotly_chart(fig)