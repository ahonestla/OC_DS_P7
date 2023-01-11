import pickle
import pandas
import streamlit as st

# Title
st.title('Credit scoring application')

# Select the customer
cust = st.selectbox("Select the customer", ("Mark", "Pierre"))

# Unpick classifier
clf = pickle.load(open('models/randomforest_v1.pckl', 'rb'))
# Get parameters
params = clf.get_params(deep=True)

# Output prediction
st.text(f"This model parameters are :\n {str(params)}")
