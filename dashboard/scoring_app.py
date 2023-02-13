import json
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import altair as alt
import shap
import requests
import names

# API url
API_URL = "http://0.0.0.0:8000/"
# API_URL = "https://scoringapp-api.azurewebsites.net/"

# Timeout for requests
TIMEOUT = 5

# Prediction classes
CLASSES_NAMES = ['REPAY SUCCESS', 'REPAY FAILURE']
CLASSES_COLORS = ['green', 'red']

# Functions
@st.cache
def create_customer_names(cust_numbers):
    """ Create array of random names """
    return [names.get_full_name() for _ in range(cust_numbers)]


@st.cache
def get_cust_ids():
    """ Get list of customers ids """
    response = requests.get(API_URL + "ids/", timeout=TIMEOUT)
    content = json.loads(response.content)
    return content['ids']


@st.cache
def get_cust_columns(cust_id):
    """ Get customer main columns """
    response = requests.get(API_URL + "columns/id=" + str(cust_id), timeout=TIMEOUT)
    content = json.loads(json.loads(response.content))
    return pd.Series(content)


@st.cache
def get_predictions(cust_id):
    """ Get customer prediction """
    response = requests.get(API_URL + "predict/id=" + str(cust_id), timeout=TIMEOUT)
    content = json.loads(response.content)
    return content


@st.cache
def get_shap_values():
    """ Get all customers SHAP values """
    response = requests.get(API_URL + "shap", timeout=TIMEOUT)
    content = json.loads(response.content)
    explanation = shap.Explanation(np.asarray(content['values']),
                                   np.asarray(content['base_values']),
                                   feature_names=content['features'])
    return explanation


@st.cache
def get_shap_explanation(cust_id):
    """ Get customer SHAP explanation """
    response = requests.get(API_URL + "shap/id=" + str(cust_id), timeout=TIMEOUT)
    content = json.loads(response.content)
    explanation = shap.Explanation(np.asarray(content['values']), 
                                   content['base_values'],
                                   feature_names=content['features'])
    return explanation


@st.cache
def get_feature_importances():
    """ Get feature importance """
    response = requests.get(API_URL + "importances", timeout=TIMEOUT)
    content = json.loads(json.loads(response.content))
    return pd.DataFrame(content)


def xgb_importances_chart():
    """ Return altair chart of xgboost feature importances """
    imp_df = get_feature_importances()
    imp_sorted = imp_df.sort_values(by='importances', ascending=False)
    imp_chart = alt.Chart(imp_sorted.reset_index(), title="Top 15 feature importances").mark_bar().encode(
        x='importances',
        y=alt.Y('index', sort=None, title='features'))
    return imp_chart


def st_shap(plot, height=None):
    """ Create a shap html component """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


# Title
st.title('Credit scoring application')
st.subheader("Victor BARBIER - Data Scientist - Projet 7")
st.write("")

# Settings on sidebar
st.sidebar.subheader("Settings")
# Select source of feature importance
feat_imp_source = st.sidebar.radio("Feature importance source :", ('XGBoost', 'SHAP'))
# Select the prediction threshold
pred_thresh = st.sidebar.radio("Prediction threshold : ", ('Default', 'Optimized'))
# Select type of explanation
shap_plot_type = st.sidebar.radio("Select the plot type :", ('Waterfall', 'Bar'))

# Create tabs
tab_all, tab_single = st.tabs(["All customers", "Single customer"])

with tab_all:

    st.subheader("Feature importances (" + feat_imp_source + ")")
    st.write("")
    
    if (feat_imp_source == 'XGBoost'):
        # Display XGBoost feature importance
        st.altair_chart(xgb_importances_chart(), use_container_width=True)
        expander = st.expander("See explanation")
        expander.write("Blablabla features importances nanani nanana")

    else:
        # Display SHAP feature importance
        shap_values = get_shap_values()
        fig, _ = plt.subplots()
        fig.suptitle('Top 15 feature importances (test set)', fontsize=18)
        shap.summary_plot(shap_values, max_display=15, plot_type='bar', plot_size=[12, 6], show=False)
        st.pyplot(fig)
        expander = st.expander("See explanation")
        expander.write("Blablabla features importances nanani nanana")


with tab_single:
    # Get customer ids
    cust_ids = get_cust_ids()
    cust_names = create_customer_names(len(cust_ids))
    print(cust_names)

    # Select the customer
    cust_select_id = st.selectbox(
        "Select the customer",
        cust_ids,
        format_func=lambda x: str(x) + " - " + cust_names[x])

    # Display the columns
    st.subheader("Customer information")
    cust_df = get_cust_columns(cust_select_id).rename("id="+str(cust_select_id))
    st.dataframe(cust_df)

    # Display prediction
    st.subheader("Customer prediction")
    predictions = get_predictions(cust_select_id)
    pred = predictions['default'] if pred_thresh == 'Default' else predictions['custom']
    pred_text = "**:" + CLASSES_COLORS[pred] + "[" + CLASSES_NAMES[pred] + "]**"
    st.markdown("The model prediction is " + pred_text)

    # Display shap force plot
    shap_explanation = get_shap_explanation(cust_select_id)
    st_shap(shap.force_plot(shap_explanation))

    # Display shap bar/waterfall plot
    fig, _ = plt.subplots()
    if (shap_plot_type == 'Waterfall'):
        shap.plots.waterfall(shap_explanation, show=False)
    else:
        shap.plots.bar(shap_explanation, show=False)
    plt.title("Shap explanation plot", fontsize=16)
    fig.set_figheight(6)
    fig.set_figwidth(9)
    st.pyplot(fig)

st.write("")