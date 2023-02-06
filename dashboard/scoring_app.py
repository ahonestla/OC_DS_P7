import pickle
import json
import pandas as pd
import sklearn
import streamlit as st
import streamlit.components.v1 as components
import requests
import names

# API url
API_URL = "http://0.0.0.0:8000/"
# API_URL = "https://scoringapp-api.azurewebsites.net/"

# Timeout for requests
TIMEOUT = 5

# Title
st.title('Credit scoring application')
st.subheader("Victor BARBIER - Data Scientist - Projet 7")
st.write("--")

# Functions
@st.cache
def get_cust_ids():
    """ Get list of customers ids """
    response = requests.get(API_URL + "ids/", timeout=TIMEOUT)
    content = json.loads(response.content)
    return content['ids']


@st.cache
def create_customer_names(cust_numbers):
    """ Create array of random names """
    return [names.get_full_name() for _ in range(cust_numbers)]

@st.cache
def get_explanation(cust_id):
    """ Get explanation """
    response = requests.get(API_URL + "explain/id=" + str(cust_id), timeout=TIMEOUT)
    content = json.loads(response.content)
    return content['explanation']


# Get customer ids
cust_ids = get_cust_ids()
cust_names = create_customer_names(len(cust_ids))
print(cust_names)

# Select the customer
cust_select_id = st.sidebar.selectbox(
    "Select the customer",
    cust_ids,
    format_func=lambda x: str(x) + " - " + cust_names[x])

st.sidebar.write("You selected the customer with id=" + str(cust_select_id))


# Display explanation
html_string = get_explanation(cust_select_id)
components.html(html_string, height=600)

st.write("--")

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