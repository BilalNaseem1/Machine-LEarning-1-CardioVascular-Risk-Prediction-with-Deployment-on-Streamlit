import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats.mstats import winsorize
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
import pandas as pd
from sklearn.preprocessing import PowerTransformer
import streamlit as st
from PIL import Image
import pickle
import shap
import numpy as np
import lime
from lime import lime_tabular
from sklearn.linear_model import LogisticRegression

def get_clean_data():
  data = pd.read_csv("cvd_risk_data.csv")
  data = data.dropna(subset=['BPMeds'])

  imputing_median1 = data[data['is_smoking'] == 'YES']['cigsPerDay'].median()
  data['cigsPerDay'] = data['cigsPerDay'].fillna(imputing_median1)
  data = data.drop(['id', 'education', 'is_smoking', 'prevalentStroke'], axis=1)

  # Imputation
  for var in ['totChol', 'BMI', 'heartRate']:
    imputer = IterativeImputer(random_state=0)
    data[[var]] = imputer.fit_transform(data[[var]])

  imputer = IterativeImputer(random_state = 0)
  imputer.fit(data[['glucose']])

  data[['glucose']] = imputer.transform(data[['glucose']])
  data['mean_BP'] = (data['sysBP'] + data['diaBP']) / 2
  data.drop(['sysBP', 'diaBP'], axis=1, inplace=True)



  data['age'] = winsorize(data['age'], limits=(0, 0))
  data['cigsPerDay'] = winsorize(data['cigsPerDay'], limits=(0.7, 0.01))
  data['totChol'] = winsorize(data['totChol'], limits=(0, 0))

  data = pd.get_dummies(data, drop_first=True)

  data['BMI'] = winsorize(data['BMI'], limits=(0, 0))
  data['heartRate'] = winsorize(data['heartRate'], limits=(0, 0))
  data['glucose'] = winsorize(data['glucose'], limits=(0.7, 0.2))
  data['sex_M'] = winsorize(data['sex_M'], limits=(0, 0))
  data['prevalentHyp'] = winsorize(data['prevalentHyp'], limits=(0, 0.4))

  return data


def add_sidebar():
  st.sidebar.header("Features")
  
  data = get_clean_data()
  
  slider_labels = [
        ("Age", "age"),
        ("Cigarettes per Day", "cigsPerDay"),
        ("Total Cholestrol", "totChol"),
        ('BMI', "BMI"),
        ("Heart Rate", "heartRate"),
        ("Glucose", "glucose"),
        ("Mean Blood Pressure", "mean_BP"),
        # ('Sex', 'sex_M'),
        ("Blood Pressure Medication", "BPMeds"),
        ("Prevelant Hypertension", "prevalentHyp"),
        ("Diabetes", "diabetes"),
       
    ]

  input_dict = {}

  for label, key in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      value=float(data[key].mean())
    )
    
  return input_dict


def get_scaled_values(input_dict):
    data = get_clean_data()

    X = data.drop(['TenYearCHD'], axis=1)  # Exclude the target variable

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()

        # Check if denominator is zero or if min and max values are equal
        if (max_val - min_val) == 0 or np.isnan(max_val) or np.isnan(min_val):
            scaled_value = 0.0  # Assign a default value for invalid cases
        else:
            scaled_value = (value - min_val) / (max_val - min_val)

        scaled_dict[key] = scaled_value

    return scaled_dict


  

def get_radar_chart(input_data):
  
  input_data = get_scaled_values(input_data)
  
  categories = ['Age', 'cigsPerDay', 'totChol', 
              'BMI', 'Heart_Rate',
              'Glucose', 'Mean_BP',
              'Blood_Pressure_Medication',
              'Prevelant_Hypertension', 'Diabetes']


  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
           input_data['age'], input_data['cigsPerDay'], input_data['totChol'],
           input_data['BMI'], input_data['heartRate'], input_data['glucose'],
           input_data['mean_BP'], np.logical_xor(st.sidebar.radio("BPMeds", [0, 1], index=0), 1),
           np.logical_xor(st.sidebar.radio("Diabetes", [0, 1], index=0), 1)
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
  ))


  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
  return fig


def add_predictions(input_data):
    
    model = pickle.load(open("model/LR_model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    data = get_clean_data()

    # Preprocess the input data to match the processed training data
    input_df = pd.DataFrame(input_data, index=[0])
    input_df = input_df.reindex(columns=data.drop('TenYearCHD', axis=1).columns, fill_value=0)

    # Scale the input data
    input_array_scaled = scaler.transform(input_df)

    prediction = model.predict(input_array_scaled)

    st.subheader("CardioVascular Risk Predictor")
    st.write("CVD Risk Present?")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>No</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'Yes</span>", unsafe_allow_html=True)

    st.write("Probability of having No Risk: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of having Risk: ", model.predict_proba(input_array_scaled)[0][1])

    st.write("Machine Learning I Final Project: CVD Risk Predictor.\n Submitted by: Bilal Naseem - ERP: 13216 \n Kanza Nasim ERP: 27259")

def run_lime_prediction():
    data = get_clean_data()
    model = LogisticRegression()
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    X = data.drop(['TenYearCHD'], axis=1)
    y = data['TenYearCHD']
    model.fit(X,y)
    feats = X.columns
    explainer = lime.lime_tabular.LimeTabularExplainer(X.values, feature_names=X.columns, class_names=['0', '1'])

    # Select an instance from the test data for explanation
    instance = X.iloc[len(X)-1]

    # Explain the prediction for the selected instance
    explanation = explainer.explain_instance(instance.values, model.predict_proba, num_features=len(X.columns))

    # Plot the Lime prediction graph
    fig = explanation.as_pyplot_figure()
    
    # Display the Lime plot in Streamlit using matplotlib's figure
    st.subheader('LIME Prediction')
    st.pyplot(fig)

def run_shap_prediction():
    import shap
    data = get_clean_data()
    model = pickle.load(open("model/LR_model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    X = data.drop(['TenYearCHD'], axis=1)
    feats = X.columns
    # Select an input instance for which you want to explain the predictions
    input_instance = X.iloc[len(X)-1]  # Replace with your desired input instance
    

    # Create a SHAP explainer
    explainer = shap.Explainer(model, feats)

    # Generate SHAP values for the input instancec
    shap_values = explainer(input_instance)
    shap_values_matrix = np.array([shap_values.values])

    # Plot the SHAP values
    figshap = shap.summary_plot(shap_values_matrix, feature_names=feats.columns)
    st.subheader('SHAP Prediction')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(figshap)


def main():
  st.set_page_config(
    page_title="Cardio Vascular Risk Predictor",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
  )
  
  with open("assets/style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
  
  input_data = add_sidebar()
  
  with st.container():
    st.title("CardioVascular Risk Predictor")
    st.write("This app predicts using a machine learning model whether a person is likely to Develop Risk of CVD.")
  
  col1, col2 = st.columns([4,1])
  
  with col1:
    radar_chart = get_radar_chart(input_data)
    st.plotly_chart(radar_chart)
    run_lime_prediction()
    # run_shap_prediction()

  
  with col2:
    add_predictions(input_data)
 
if __name__ == '__main__':
  main()