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

curr_path = os.getcwd()
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
st.sidebar.markdown("""
## About Us
Credit card fraud can be defined as an illegal activity where an individual steals 
someone else's credit card information without their consent. 
This is considered a type of identity theft and the purpose of doing so is to make purchases or 
withdraw money from the account linked to the credit card without authorization.

---\n

The notebook, model and documentation(Dockerfiles, FastAPI script, Streamlit App script) are available on [GitHub.](https://github.com/Keval78/Credit-Card-Fraud-Detection)

---\n

Project Developed by :
1. Kiranmayee Porla
2. Keval Padsala
3. Nishi Agrawal
4. Sahishn Gaddam
5. Siddharth Agarwal
6. Vaibhav Kumar
""")

st.sidebar.write("---\n")
st.sidebar.caption("You can check out the source code [here](https://github.com/Keval78/Credit-Card-Fraud-Detection).")

# ! SIDEBAR LOGIC ENDS...



# !============================
# !  APP LOGICS STARTS...
# !============================
st.image(os.path.join(image_folder_path, "DeepSightAnalytics.png"))
st.title("Fraudulent Transaction Detection Web App")

# provide options to either select an image form the gallery, upload one, or fetch from URL
modeling_tab, eda_tab, syndata_tab = st.tabs(["Model Demo", "EDA", "Sythetic Data Generation"])

with modeling_tab:
    if 'global_fields' not in st.session_state:
        st.session_state.global_fields = global_fields


    def select_params(sess, global_fields):
        index_no = int(global_fields.index_no)
        global_fields.class_val = "Fraud" if df.iloc[index_no][-1] == 1 else "Non-Fraud"
        global_fields.time_val = df.iloc[index_no][1]
        global_fields.amount_val = df.iloc[index_no][-2]
        global_fields.v1_28_val = df.iloc[index_no]

    st.subheader("CreditCard Fraud Detection 🎨")
    st.caption("Test your models for different Test transaction.")
    st.markdown("---")
    
    st.write("Enter the values")
    index_no = st.text_input("`Enter the Number`", value=0, help="Row Number from the Test dataframe.")

    if st.session_state.global_fields.index_no != index_no:
        st.session_state.global_fields.index_no = index_no
    select_params(st.session_state, st.session_state.global_fields)
    
    if st.session_state.global_fields.class_val == "Fraud":
        st.error(f"Selected Transaction Class is: {st.session_state.global_fields.class_val}")
    if st.session_state.global_fields.class_val == "Non-Fraud":
        st.success(f"Selected Transaction Class is: {st.session_state.global_fields.class_val}")
    
    date_and_time = st.text_input("`Enter Date & Time in seconds`", value = st.session_state.global_fields.time_val)

    amount = st.number_input("`Amount in $`", value=st.session_state.global_fields.amount_val)

    st.write("`Select V1-V28 PCA Components`")
    
    # Dataframe Details
    enh_expander = st.expander("Selected PCA Values...", expanded=False)
    with enh_expander:
        enh_expander.write(st.session_state.global_fields.v1_28_val)
    
    # Clustering Model 
    model_name = st.selectbox("`Machine Learning Model`", models.keys(), help="Machine Learning model to use predicting frauds.")
    sklearn_info = st.empty()

    st.markdown("---")

    # button_icon = "https://cdn-icons-png.flaticon.com/512/3438/3438003.png"
    # button_label = "Detect Result"

    if st.button("Detect Result"):
        # Load selected model file.
        model_path = os.path.join(model_folder_path, "{}".format(model_name))
        ml_model = load_object(file_path=model_path)

        index_no = int(index_no)
        data_input = [df.iloc[index_no][df.columns[:-1]].to_list()]
        preds = ml_model.predict(data_input)
        class_val = "Fraud" if preds[0] == 1 else "Non-Fraud"
    
        st.markdown("---")
        message = f"\n\nPrediction for {model_name} is: {class_val}\n\n"
        if st.session_state.global_fields.class_val == class_val:
            st.success(message)
        else:
            st.error(message)
        st.markdown("---")

    st.write("---\n")




with eda_tab:
    @dataclass
    class DataConfig:
        total_transactions = numerize(284807)
        total_columns = numerize(31)
        avg_fraud_val = f'${numerize(122.211321,2)}'
        total_frauds = numerize(492)
        total_genuine = numerize(284315)
        
    total1,total2,total3,total4,total5 = st.columns(5,gap='large')
    data_config = DataConfig()
    with total1:
        st.image('images/impression.png',use_column_width='Auto')
        st.metric(label = 'Total Impressions', value=data_config.total_transactions)
    with total2:
        st.image('images/tap.png',use_column_width='Auto')
        st.metric(label='Total Columns', value=data_config.total_columns)
    with total3:
        st.image('images/hand.png',use_column_width='Auto')
        st.metric(label= 'Average Fraud Value',value=data_config.avg_fraud_val)
    with total4:
        st.image('images/conversion.png',use_column_width='Auto')
        st.metric(label='Total Frauds',value=data_config.total_frauds)
    with total5:
        st.image('images/app_conversion.png',use_column_width='Auto')
        st.metric(label='Genuine Transactions',value=data_config.total_genuine)
    
    Q1,Q2 = st.columns(2)
    with Q1:
        st.plotly_chart(pieplot("Distribution fraudulent Transactions"), use_container_width=True)
    with Q2:
        st.plotly_chart(barplot("Distribution fraudulent Transactions"), use_container_width=True)
    
    Q1,Q2 = st.columns(2)
    fig_q1, fig_q2 = histplots("Distribution of Time for each Transaction", "Distribution of Amount for each Transaction")
    with Q1:
        st.plotly_chart(fig_q1, use_container_width=True)
    with Q2:
        st.plotly_chart(fig_q2, use_container_width=True)
    
    Q1,Q2 = st.columns(2)
    fig_q1, fig_q2 = histplots_fraud("Distribution of Time for each Fraud Transaction", "Distribution of Amount for each Fraud Transaction")
    with Q1:
        st.plotly_chart(fig_q1, use_container_width=True)
    with Q2:
        st.plotly_chart(fig_q2, use_container_width=True)
    
    Q1,Q2 = st.columns(2)
    with Q1:
        st.plotly_chart(tsne_plot_original("TSNE plot for the 5% Samples of the dataset."), use_container_width=True)
    with Q2:
        st.plotly_chart(tsne_plot_smoteenn("TSNE plot for SMOTE+ENN the 5% Sample of the dataset."), use_container_width=True)
    
    st.markdown("""### Original Dataset""")
    st.plotly_chart(correlation_heatmap_original("Correalation Matrix for Original Dataset"), use_container_width=True)
    st.markdown("""### After Applying Sampling SMOTE + ENN""")
    st.plotly_chart(correlation_heatmap_smoteenn("Correalation Matrix for SMOTE+ENN Dataset"), use_container_width=True)
    
    st.markdown("""### Confusion Matrix""")
    st.plotly_chart(confusion_matrix(), use_container_width=True)
    
    st.markdown("""### ROC Curve""")
    st.plotly_chart(roc_curve(), use_container_width=True)



with syndata_tab:
    file_name = st.selectbox("Select Art for mdksmk", options={})



