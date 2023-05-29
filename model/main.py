import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
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


def create_model(data): 
  X = data.drop(['TenYearCHD'], axis=1)
  y = data['TenYearCHD']
  

  # scale the data
  scaler = MinMaxScaler()
  X = scaler.fit_transform(X)
  
  # split the data
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
  )
  
  # train the model
  model = LogisticRegression(C= 100.0, penalty ='l2', solver= 'saga')
  model.fit(X_train, y_train)
  
  # test model
  threshold = 0.1
  y_pred_proba = model.predict_proba(X_test)[:, 1]
  y_pred = (y_pred_proba > threshold).astype(int)
  print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
  print("Classification report: \n", classification_report(y_test, y_pred))
  
  return model, scaler


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
  data['BPMeds'] = winsorize(data['BPMeds'], limits=(0, 0))
  data['prevalentHyp'] = winsorize(data['prevalentHyp'], limits=(0, 0.4))



  return data



def main():
  data = get_clean_data()

  model, scaler = create_model(data)

  with open('model/LR_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
  with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
  

if __name__ == '__main__':
  main()