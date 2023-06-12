# Machine-Learning-I-CardioVascular-Risk-Prediction-with-Deployment-on-Streamlit
This report presents details the steps and findings for the Machine Learning I Final Project, IBA Spring 2023.
The Project focused on predicting cardiovascular disease. The dataset consisted of 3,390 observations and 17 attributes, encompassing demographic, behavioral, and medical information. The dataset was chosen after extensive search and came with challenges such as imbalanced target variable, null values, and skewed distributions. Extensive exploratory
data analysis was conducted to gain insights into the dataset’s characteristics and uncover
patterns. Extensive Feature engineering was applied, by dealing with nulls, outliers, feature
extraction and reduction, resulting in the creation of over 25 derived datasets and 18
Python working notebooks. Multiple ensemble models, including neural networks, soft
voting, and stacking, were employed.

![image](https://github.com/BilalNaseem1/Machine-Learning-I-CardioVascular-Risk-Prediction-with-Deployment-on-Streamlit/assets/31243659/d3ae70ce-9cae-49ac-a0e3-0cd960bdb51c)


Hyperparameter optimization was performed using
various techniques, with the objective of maximizing the area under the precision-recall
curve (AUC-PR). Different probability thresholds were explored to achieve the best
recall while maintaining a good balance of precision. 

![image](https://github.com/BilalNaseem1/Machine-Learning-I-CardioVascular-Risk-Prediction-with-Deployment-on-Streamlit/assets/31243659/50466988-2264-4a18-9ea6-c538a28f1e45)

![image](https://github.com/BilalNaseem1/Machine-Learning-I-CardioVascular-Risk-Prediction-with-Deployment-on-Streamlit/assets/31243659/2795db3f-2d86-47cb-a9c6-b15f6486c8fb)


Additionally, Synthetic Minority
Over-sampling Technique (SMOTE) was attempted to address class imbalance but did not
yield a significant improvement in the model’s performance.
To ensure explainability, Lime, Shap, and counterfactual explanations were applied to the
top four models obtained from hyperparameter optimization. The interpretability analysis
focused on understanding the models’ results, identifying their strengths, weaknesses, and
insights derived from the explanations. Finally, based on the evaluation metrics and interpretability analysis, logistic regression emerged as the best-performing model, demonstrating
the highest AUC-PR. This model was selected for deployment on Streamlit, providing a
user-friendly interface to predict cardiovascular disease risk based on the input features.

The streamlit app link can be found below:

https://bilalnaseem1-machine-learning-i-cardiovascular-risk--app-jesx2q.streamlit.app/
