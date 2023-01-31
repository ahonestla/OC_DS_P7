import pickle
import pandas as pd
import sklearn
import streamlit as st
import requests
import json
import names

# API url
API_URL = "http://0.0.0.0:8000/"
# API_URL = "https://scoringapp-api.azurewebsites.net/"

# Title
st.title('Credit scoring application')
st.subheader("Victor BARBIER - Data Scientist - Projet 7")


# Functions
# Get list of SK_IDS
@st.cache
def get_cust_ids():
    response = requests.get(API_URL + "ids/")
    content = json.loads(response.content)
    return content['ids']

# Create array of random names
@st.cache
def create_customer_names(n):
    return [names.get_full_name() for _ in range(n)]




# Get customer ids
cust_ids = get_cust_ids()
cust_names = create_customer_names(len(cust_ids))
print(cust_names)

# Select the customer
cust_select = st.sidebar.selectbox("Select the customer", cust_ids,
                                   format_func=lambda x: str(x) + " - " + cust_names[x])

st.sidebar.write("You selected index " + str(cust_select))

# if cust == "LOCAL":
#     # Get data
#     response = requests.get(API_URL + "test")
#     print(response.json())
#     st.write(response.json())
# elif cust == "HTTP":
#     # Get data
#     response = requests.get("http://scoringapp-api.azurewebsites.net/test")
#     print(response.json())
#     st.write(response.json())
# elif cust == "HTTPS":
#     # Get data
#     response = requests.get("https://scoringapp-api.azurewebsites.net/test")
#     print(response.json())
#     st.write(response.json())
# else:
#     st.write('Select someone')