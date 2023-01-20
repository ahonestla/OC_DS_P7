import pickle
import pandas as pd
import sklearn
import streamlit as st
import requests

# Title
st.title('Credit scoring application')

# Select the customer
cust = st.selectbox("Select the customer", ("Mark", "HTTP", "HTTPS"))

if cust == "HTTP":
    # Get data
    response = requests.get("http://scoringapp-api.azurewebsites.net/test")
    print(response.json())
    st.write(response.json())
elif cust == "HTTPS":
    # Get data
    response = requests.get("https://scoringapp-api.azurewebsites.net/test")
    print(response.json())
    st.write(response.json())
else:
    st.write('Select someone')