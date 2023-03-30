import os
import pandas as pd
import streamlit as st

from src.components.eda_functions import *
from src.utils import load_object
from numerize.numerize import numerize
from dataclasses import dataclass

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

@dataclass
class StreamlitFieldConfig:
    index_no: int = -1
    class_val: str = ""
    time_val: float = 0
    amount_val: float = 0
    v1_28_val: str = ""

curr_path = "/Users/keval_78/Keval/Data Science/Loyalist/Term 4/AIP/Credit-Card-Fraud-Detection"

model_folder_path: str = os.path.join(curr_path, 'artifacts', 'models')

image_folder_path: str = os.path.join(curr_path, 'images')

models = {
    "LogisiticRegression": [],
    "KNearest": [],
    "SupportVectorClassifier": [],
    "DecisionTreeClassifier": [],
    "RandomForestClassifier": [],
    "XGBoostClassifier": []
}

st.set_page_config(layout="wide")

@st.cache_data
def get_data(path):
    df = pd.read_csv(path)
    return df


test_data_path: str = os.path.join('artifacts', "test.csv")
df = get_data(test_data_path)


# * Define Global Fields for Use.
global_fields = StreamlitFieldConfig()

# !============================
# ! SIDEBAR LOGIC STARTS...
# !============================

if 'global_fields' not in st.session_state:
    st.session_state.global_fields = global_fields

def select_params(sess, global_fields):
    index_no = int(global_fields.index_no)
    global_fields.class_val = "Fraud" if df.iloc[index_no][-1] == 1 else "Non-Fraud"
    global_fields.time_val = df.iloc[index_no][1]
    global_fields.amount_val = df.iloc[index_no][-2]
    global_fields.v1_28_val = df.iloc[index_no]

st.sidebar.title("CreditCard Fraud Detection 🎨")
st.sidebar.caption("Test your models for different Test transaction.")
st.sidebar.markdown("---")
st.sidebar.markdown("---")

st.sidebar.write("Enter the values")
index_no = st.sidebar.text_input("`Enter the Number`", value=0, help="Row Number from the Test dataframe.")

if st.session_state.global_fields.class_val == "Fraud":
    st.sidebar.error(f"Selected Transaction Class is: {st.session_state.global_fields.class_val}")
if st.session_state.global_fields.class_val == "Non-Fraud":
    st.sidebar.success(f"Selected Transaction Class is: {st.session_state.global_fields.class_val}")

date_and_time = st.sidebar.text_input("`Enter Date & Time in seconds`", value = st.session_state.global_fields.time_val)

amount = st.sidebar.number_input("`Amount in $`", value=st.session_state.global_fields.amount_val)

st.sidebar.write("`Select V1-V28 PCA Components`")
# Dataframe Details
enh_expander = st.sidebar.expander("Selected PCA Values...", expanded=False)
with enh_expander:
    enh_expander.write(st.session_state.global_fields.v1_28_val)


if st.session_state.global_fields.index_no != index_no:
    if not isinstance(index_no, int):
        index_no = -1
    st.session_state.global_fields.index_no = index_no
    select_params(st.session_state, st.session_state.global_fields)



# Clustering Model 
model_name = st.sidebar.selectbox("`Machine Learning Model`", models.keys(), help="Machine Learning model to use predicting frauds.")
sklearn_info = st.sidebar.empty()

st.sidebar.markdown("---")

if st.sidebar.button("Detection Result"):
    # Load selected model file.
    model_path = os.path.join(model_folder_path, "{}.pkl".format(model_name))
    ml_model = load_object(file_path=model_path)
    
    index_no = int(index_no)
    data_input = [df.iloc[index_no][df.columns[:-1]].to_list()]
    preds = ml_model.predict(data_input)
    class_val = "Fraud" if preds[0] == 1 else "Non-Fraud"
    
    st.sidebar.markdown("---")
    message = f"Prediction for {model_name} is: {class_val}"
    if st.session_state.global_fields.class_val == class_val:
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)
    st.sidebar.markdown("---")


st.sidebar.write("---\n")
st.sidebar.caption("You can check out the source code [here](https://github.com/Keval78/Credit-Card-Fraud-Detection).")

# ! SIDEBAR LOGIC ENDS...


# !============================
# !  APP LOGICS STARTS...
# !============================
st.image(os.path.join(image_folder_path, "DeepSightAnalytics.png"))
st.title("Fraudulent Transaction Detection Web App")

# provide options to either select an image form the gallery, upload one, or fetch from URL
eda_tab, syndata_tab = st.tabs(["EDA", "Sythetic Data Generation"])

with eda_tab:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    train_df = get_data(train_data_path)

    total_transactions = len(train_df)
    total_clicks = float(0)
    total_spent = float(0)
    total_conversions= float(0) 
    total_approved_conversions = float(0)


    total1,total2,total3,total4,total5 = st.columns(5,gap='large')
    with total1:
        st.image('images/impression.png',use_column_width='Auto')
        st.metric(label = 'Total Impressions', value= numerize(total_transactions))
    with total2:
        st.image('images/tap.png',use_column_width='Auto')
        st.metric(label='Total Clicks', value=numerize(total_clicks))
    with total3:
        st.image('images/hand.png',use_column_width='Auto')
        st.metric(label= 'Total Spend',value=numerize(total_spent,2))
    with total4:
        st.image('images/conversion.png',use_column_width='Auto')
        st.metric(label='Total Conversions',value=numerize(total_conversions))
    with total5:
        st.image('images/app_conversion.png',use_column_width='Auto')
        st.metric(label='Approved Conversions',value=numerize(total_approved_conversions))

    Q1,Q2 = st.columns(2)
    labels = ['Non-Fraudulent', 'Fraudulent']
    values = df['Class'].value_counts()
    title_ = "Distribution fraudulent Transactions"
    with Q1:
        pieplot_fig = pieplot(labels, values, title_)
        st.plotly_chart(pieplot_fig, use_container_width=True)
    with Q2:
        barplot_fig = barplot(labels, values, title_)
        st.plotly_chart(barplot_fig, use_container_width=True)

    # Q1,Q2 = st.columns(2)
    # with Q1:
    #     with st.spinner('Creating TSNE plot...'):
    #         st.plotly_chart(tsne_plot(train_df), use_container_width=True)
    # with Q2:
    #     with st.spinner('Creating TSNE plot...'):
    #         st.plotly_chart(tsne_plot(train_df), use_container_width=True)

    Q1,Q2 = st.columns(2)
    with Q1:
        title_ = "Distribution of Time for each Transaction"
        st.plotly_chart(histplot(train_df["Time"], title_), use_container_width=True)
    with Q2:
        title_ = "Distribution of Amount for each Transaction"
        st.plotly_chart(histplot(train_df["Amount"], title_), use_container_width=True)
    Q1,Q2 = st.columns(2)
    with Q1:
        title_ = "Distribution of Time for Fraud Transaction"
        st.plotly_chart(histplot(train_df["Time"][train_df["Class"]==1], title_), use_container_width=True)
    with Q2:
        title_ = "Distribution of Amount for Fraud Transaction"
        st.plotly_chart(histplot(train_df["Amount"][train_df["Class"]==1], title_), use_container_width=True)
    
    st.plotly_chart(correlation_heatmap(train_df, "Correalation Matrix for Original Dataset"), use_container_width=True)

    st.markdown("""
                ### After Applying Sampling SMOTE + ENN
                """)

    # SMOTE Technique (OverSampling) After splitting and Cross Validating
    train_data_path: str = os.path.join('artifacts', "train1.csv")
    sm_train_df = get_data(train_data_path)
    st.plotly_chart(correlation_heatmap(sm_train_df, "Correalation Matrix for Applying SMOTE + ENN"), use_container_width=True)




with syndata_tab:
    file_name = st.selectbox("Select Art for mdksmk", options={})



st.markdown("""
## About
Credit card fraud can be defined as an illegal activity where an individual steals 
someone else's credit card information without their consent. 
This is considered a type of identity theft and the purpose of doing so is to make purchases or 
withdraw money from the account linked to the credit card without authorization..

The notebook, model and documentation(Dockerfiles, FastAPI script, Streamlit App script) are available on [GitHub.](https://github.com/Nneji123/Credit-Card-Fraud-Detection)        
*Project Developed by :* 
- *Kiranmayee Porla*
- *Keval Padsala*
- *Nishi Agrawal*
- *Sahishn Gaddam*
- *Siddharth Agarwal*
- *Vaibhav Kumar*
""")

