# %% [markdown]
# <h1>Salary Prediction Model<h1>

# %% [markdown]
# 
# The Salary Dataset contains **6704 rows** and **6 columns** containing the following data:
# 
# 1. **Age**
# 2. **Gender**
# 3. **Education Level**
# 4. **Job Title**
# 5. **Years of Experience**
# 6. **Salary**
# 
# First we pre-process, clean and model the data to standarsise and structure it.
# 

# %%
# Importing Libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error 


# %%
# Importing Data

df = pd.read_csv('Salary_Data.csv')

# %%
# df.describe()

# %%
# Checking for null data

df.isnull().sum()

# %%
# Dropping null values from database

df.dropna(inplace=True)


# %%
# Reducing Job titles by omitting titles with less than 25 counts

job_title_count = df['Job Title'].value_counts()
job_title_edited = job_title_count[job_title_count<=25]
job_title_edited.count()

# %%
# Omitting titles with less than 25 counts

df['Job Title'] = df['Job Title'].apply(lambda x: 'Others' if x in job_title_edited else x )
df['Job Title'].nunique()
# %%
#Checking unique value count of Education Level

df['Education Level'].value_counts()

# %%
# Combining repeating values of education level

df['Education Level'].replace(["Bachelor's Degree","Master's Degree","phD"],["Bachelor's","Master's","PhD"],inplace=True)
df['Education Level'].value_counts()

# %%
# Checking Unique Value count of Gender

df['Gender'].value_counts()
# %%
# Decreasing the salary of women by $1000
df.loc[df['Gender'] == 'Female', 'Salary'] -= 12000

# %%
# Mapping Education Level column
education_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
df['Education Level'] = df['Education Level'].map(education_mapping)

# Label encoding the categorical variable
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# %% [markdown]
# **Predicting Salary**
# 
# 3 Models will be used to predict the salary
# 
# 1. Linear Regression
# 2. Deision Tree
# 3. Random Forest

# %%
# detecting the outliers in salary column using IQR method
Q1 = df.Salary.quantile(0.25) # First Quartile
Q3 = df.Salary.quantile(0.75) # Third Quartile

# Caltulation Interquartile
IQR = Q3-Q1

# Deetecting outliers lying 1.5x of IQR above and below Q1 and Q3 resp
lower = Q1-1.5*IQR
upper = Q3+1.5*IQR


# %% [markdown]
# <h2>Preparing the data for ML analysis by converting categorical job titles into a numerical format<h2>

# %%
# Creating dummies for Job titles
dummies = pd.get_dummies(df['Job Title'],drop_first=True)
df = pd.concat([df,dummies],axis=1)


# Drop Job Title column
df.drop('Job Title',inplace=True,axis=1)

# %%
# Separating the dataset into features and target

# Dataset conntaining all features from df
features = df.drop('Salary',axis=1)

# Series containing target variable to be predicted
target = df['Salary']


# %%
# Splitting data into 25% training and 75% test sets

x_train,x_test,y_train,y_test = train_test_split(features,target,test_size=0.25,random_state=42)

# %%
# Create a dictionary for defining models and tuning hyperparameters

model_params = {
    'Linear_Regression':{
        'model':LinearRegression(),
        'params':{
            
        }
    },
    'Decision_Tree':{
        'model':DecisionTreeRegressor(),
        'params':{
            'max_depth':[2,4,6,8,10],
            'random_state':[0,42],
            'min_samples_split':[1,5,10,20]
        }
    },
    'Random_Forest':{
        'model':RandomForestRegressor(),
        'params':{
            'n_estimators':[10,30,20,50,80]
        }
    }
}

# %%
# Hyper parameter tuning through grid search cv
score=[]

for model_name,m in model_params.items():
    clf = GridSearchCV(m['model'],m['params'],cv=5,scoring='neg_mean_squared_error')
    clf.fit(x_train,y_train)
    
    score.append({
        'Model':model_name,
        'Params':clf.best_params_,
        'MSE(-ve)':clf.best_score_
    })
pd.DataFrame(score)    

# %%
# Order of the best models 

s = pd.DataFrame(score)
sort = s.sort_values(by = 'MSE(-ve)',ascending=False)

# %%
# Random Forest model

rfr = RandomForestRegressor(n_estimators=20)
rfr.fit(x_train,y_train)

# %%
rfr.score(x_test,y_test)

# %%
y_pred_rfr = rfr.predict(x_test)

print("Mean Squared Error :",mean_squared_error(y_test,y_pred_rfr))
print("Mean Absolute Error :",mean_absolute_error(y_test,y_pred_rfr))
print("Root Mean Squared Error :",mean_squared_error(y_test,y_pred_rfr,squared=False))

# %%
# Decision Tree model

dtr = DecisionTreeRegressor(max_depth=10,min_samples_split=2,random_state=0)
dtr.fit(x_train,y_train)

# %%
dtr.score(x_test,y_test)

# %%
y_pred_dtr = dtr.predict(x_test)

print("Mean Squared Error :",mean_squared_error(y_test,y_pred_dtr))
print("Mean Absolute Error :",mean_absolute_error(y_test,y_pred_dtr))
print("Root Mean Squared Error :",mean_squared_error(y_test,y_pred_dtr,squared=False))

# %%
# Linear regression model

lr = LinearRegression()
lr.fit(x_train,y_train)

# %%
lr.score(x_test,y_test)

# %%
y_pred_lr = lr.predict(x_test)

print("Mean Squared Error :",mean_squared_error(y_test,y_pred_lr))
print("Mean Absolute Error :",mean_absolute_error(y_test,y_pred_lr))
print("Root Mean Squared Error :",mean_squared_error(y_test,y_pred_lr,squared=False))

# %%
# Access the feature importances of Random Forest Regressor
feature_importances = rfr.feature_importances_

# Assuming you have a list of feature names that corresponds to the feature importances
feature_names = list(x_train.columns)

# Sort feature importances in descending order
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_feature_importances = [feature_importances[i] for i in sorted_indices]
sorted_feature_names = [feature_names[i] for i in sorted_indices]

# Create a bar chart
plt.figure(figsize=(12, 8))
plt.barh(sorted_feature_names[:10], sorted_feature_importances[:10])
plt.xlabel('Feature Importance')
plt.title('Top 10 Feature Importance in Predicting Salary')
plt.gca().invert_yaxis()  # Invert the y-axis for better visualization
plt.show()

# %%
import joblib

# Save the trained Random Forest model
joblib.dump(rfr, 'random_forest_model.pkl')

# %% [markdown]
# ## Decision Intelligence

# %%
import joblib

# Saving the trained Random Forest model
joblib.dump(rfr, 'random_forest_model.pkl')

# Saving the label encoder (for gender encoding) and the feature importances
joblib.dump(le, 'gender_encoder.pkl')
joblib.dump(list(zip(sorted_feature_names, sorted_feature_importances)), 'feature_importances.pkl')

# %%
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load pre-trained models and encoders
rfr = joblib.load('random_forest_model.pkl')
le = joblib.load('gender_encoder.pkl')
feature_importances = joblib.load('feature_importances.pkl')

def predict_salary(age, education, experience, gender, job_title):
    # Mapping the input data to match model's input format
    education_mapping = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
    gender_encoded = le.transform([gender])[0]

    # Prepare input data for prediction (without job title dummies initially)
    input_data = pd.DataFrame([[age, gender_encoded, education_mapping[education], experience]], 
                              columns=['Age', 'Gender', 'Education Level', 'Years of Experience'])

    job_titles = ['Content Marketing Manager', 'Data Analyst', 'Data Scientist', 'Digital Marketing Manager', 'Director of Data Science', 'Director of HR', 'Director of Marketing', 'Financial Analyst', 'Financial Manager', 'Front End Developer', 'Front end Developer', 'Full Stack Engineer', 'Human Resources Coordinator', 'Human Resources Manager', 'Junior HR Coordinator', 'Junior HR Generalist', 'Junior Marketing Manager', 'Junior Sales Associate', 'Junior Sales Representative', 'Junior Software Developer', 'Junior Software Engineer', 'Junior Web Developer', 'Marketing Analyst', 'Marketing Coordinator', 'Marketing Director', 'Marketing Manager', 'Operations Manager', 'Others', 'Product Designer', 'Product Manager', 'Receptionist', 'Research Director', 'Research Scientist', 'Sales Associate', 'Sales Director', 'Sales Executive', 'Sales Manager', 'Sales Representative', 'Senior Data Scientist', 'Senior HR Generalist', 'Senior Human Resources Manager', 'Senior Product Marketing Manager', 'Senior Project Engineer', 'Senior Research Scientist', 'Senior Software Engineer', 'Software Developer', 'Software Engineer', 'Software Engineer Manager', 'Web Developer']

    # Create a dictionary with job title columns, all initialized to False
    job_title_dummies = {title: False for title in job_titles}

    # Set the value of the input job title to True
    if job_title in job_title_dummies:
        job_title_dummies[job_title] = True  # Set to True for the input job title

    # Convert the job title dummies dictionary into a DataFrame
    job_title_dummies_df = pd.DataFrame([job_title_dummies])

    # Combine the original input data with the job title dummies
    input_data = pd.concat([input_data, job_title_dummies_df], axis=1)

    # Predict salary using the Random Forest model
    predicted_salary = rfr.predict(input_data)[0]

    return predicted_salary

import streamlit as st

import streamlit as st

import streamlit as st

# Streamlit Interface
st.title('equivy - Enabling Pay Equity.')

# Employee and Employer Side Descriptions
st.markdown("""
**Understand the blackbox of your compensation policy. How is compensation set within your organization?**
""")

# Button to Choose between calculation and understanding salary philosophy
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = None  # Initialize if not already set

# Home button to reset the view
def go_home():
    st.session_state.selected_option = None  # Reset to the home screen

# Display Home button if an option has been selected
if st.session_state.selected_option:
    if st.button('Go Home'):
        go_home()

# Show the option buttons when no option is selected
if st.session_state.selected_option is None:
    # Show the option buttons
    if st.button('Calculate a new compensation'):
        st.session_state.selected_option = 'Calculate'
    elif st.button('Understand your current compensation philosophy'):
        st.session_state.selected_option = 'Philosophy'

# Show the appropriate screen based on the selected option
if st.session_state.selected_option == 'Calculate':
    # User inputs for salary calculation
    age = st.slider('Age', 18, 65, 28)
    education = st.selectbox('Education Level', ["High School", "Bachelor's", "Master's", "PhD"])
    experience = st.slider('Years of Experience', 0, 40, 5)
    gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    job_title = st.selectbox('Job Title', ['Content Marketing Manager', 'Data Analyst', 'Data Scientist', 'Digital Marketing Manager', 'Director of Data Science', 'Director of HR', 'Director of Marketing', 'Financial Analyst', 'Financial Manager', 'Front End Developer', 'Front end Developer', 'Full Stack Engineer', 'Human Resources Coordinator', 'Human Resources Manager', 'Junior HR Coordinator', 'Junior HR Generalist', 'Junior Marketing Manager', 'Junior Sales Associate', 'Junior Sales Representative', 'Junior Software Developer', 'Junior Software Engineer', 'Junior Web Developer', 'Marketing Analyst', 'Marketing Coordinator', 'Marketing Director', 'Marketing Manager', 'Operations Manager', 'Others', 'Product Designer', 'Product Manager', 'Receptionist', 'Research Director', 'Research Scientist', 'Sales Associate', 'Sales Director', 'Sales Executive', 'Sales Manager', 'Sales Representative', 'Senior Data Scientist', 'Senior HR Generalist', 'Senior Human Resources Manager', 'Senior Product Marketing Manager', 'Senior Project Engineer', 'Senior Research Scientist', 'Senior Software Engineer', 'Software Developer', 'Software Engineer', 'Software Engineer Manager', 'Web Developer'])

    # Predict salary on button click
    if st.button('Calculate'):
        salary = predict_salary(age, education, experience, gender, job_title)
        #st.write(f'Calcualated Compensation: ${salary:,.2f}')
        st.markdown(f'<p style="font-size:20px; font-weight:bold;">Calculated Compensation: ${salary:,.2f}</p>', unsafe_allow_html=True)
        st.session_state.selected_option = None  # Reset the selection after calculation

# When "Understand your current salary philosophy" option is chosen
elif st.session_state.selected_option == 'Philosophy':
    # Display feature importances
    st.write("### Defining Attributes")
    st.write("The compensation calculation is based on the following attributes:")


    # List of features and their importances
    for feature, importance in feature_importances:
        if importance > 0.005:
            st.markdown(f'- **{feature}**: {importance:.2f}')



    # Create a formatted string for the output with line breaks
    output_str = ""
    for feature, importance in feature_importances:
        output_str += f"• **{feature}**: {importance}%\n\n"  # Two newlines for spacing

    # Display the formatted string with line breaks
    st.markdown(output_str)

    #for feature, importance in feature_importances:
    #    if importance > 0.005:
    #        st.markdown(f'- **{feature}**: {importance:.2f}')
    st.session_state.selected_option = None  # Reset the selection after calculation
