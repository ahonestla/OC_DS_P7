import pickle
import pandas as pd
import sklearn
import streamlit as st
import requests

# Title
st.title('Credit scoring application')

# Select the customer
cust = st.selectbox("Select the customer", ("Mark", "Pierre"))

# Get data
response = requests.get("http://127.0.01:5000/test")
print(response.json())

# Write
st.write(response.json())