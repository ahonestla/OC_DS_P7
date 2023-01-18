import pickle
import pandas as pd
import sklearn
import streamlit as st
import requests

# Title
st.title('Credit scoring application')

# Select the customer
cust = st.selectbox("Select the customer", ("Mark", "Pierre"))

if cust == "Pierre":
    # Get data
    response = requests.get("http://backend:8080/test")
    print(response.json())

    st.write(response.json())
else:
    st.write('Select someone')