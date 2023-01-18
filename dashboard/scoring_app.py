import pickle
import pandas as pd
import sklearn
import streamlit as st
import requests

# Title
st.title('Credit scoring application')

# Select the customer
cust = st.selectbox("Select the customer", ("Mark", "api:8000", "backend:8000", "api:123"))

if cust == "api:8000":
    # Get data
    response = requests.get("http://api:8000/test")
    print(response.json())
    st.write(response.json())
elif cust == "backend:8000":
    # Get data
    response = requests.get("http://backend:8000/test")
    print(response.json())
    st.write(response.json())
elif cust == "api:123":
    # Get data
    response = requests.get("http://api:123/test")
    print(response.json())
    st.write(response.json())
else:
    st.write('Select someone')