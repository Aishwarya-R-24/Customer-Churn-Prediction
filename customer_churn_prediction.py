# Customer Churn Prediction using Logistic Regression in Telecommunication Sector

# import the required libraries

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
# %matplotlib inline

# uploading dataset

data=pd.read_excel('/content/update project datas.xlsx')

# data specification

data.head()

data.tail()

data.shape

data.size

# lower case the variable name

data.columns

data.columns = data.columns.str.lower().str.replace('','')
string_columns = list(data.dtypes[data.dtypes == 'object'].index)
for col in string_columns:
  data[col].str.lower().str.replace('','')
data.columns

# data characteristics

data.dtypes

data.info()

data.isnull().sum()

data.duplicated().sum()

# basic data cleaning

data['totalcharges'].dtype

data['totalcharges'] = pd.to_numeric(data['totalcharges'],errors = 'coerce')
data['totalcharges'].dtype

data['customerid'].dtype

data['customerid'] = data['customerid'].astype('object')
data['customerid'].dtype

# exploratory data analysis

data.skew(numeric_only = True)

data.corr(numeric_only = True)

# numerical feature distribution

numerical_features=['age','tenure','monthlycharges','totalcharges']
data[numerical_features].describe()

# numerical feature by churn
fig, ax =plt.subplots(2,2,figsize=(10,10))
data[data.churn =="No"][numerical_features].hist(bins=30, color="blue", alpha=0.5, ax=ax)
data[data.churn =="Yes"][numerical_features].hist(bins=30, color="red", alpha=0.5, ax=ax)

# monthly charges and total charges
sns.lmplot(data, x='monthlycharges', y='totalcharges', fit_reg=False)

# monthly charges by churn
Mth = sns.kdeplot(data.monthlycharges[(data["churn"] == "No") ],
                color="Red", fill=True)
Mth = sns.kdeplot(data.monthlycharges[(data["churn"] == "Yes") ],
                ax =Mth, color="blue", fill=True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly Charges by Churn')
plt.show()

# total charges by churn
Tot = sns.kdeplot(data.totalcharges[(data["churn"] == "No") ],
                color="Red", fill =True)
Tot = sns.kdeplot(data.totalcharges[(data["churn"] == "Yes") ],
                ax =Tot, color="blue", fill= True)
Tot.legend(["No Churn","Churn"],loc='upper right')
Tot.set_ylabel('Density')
Tot.set_xlabel('Total Charges')
Tot.set_title('Total Charges by Churn')
plt.show()

# heatmap of numerical feature
plt.figure(figsize=(6,4))
sns.heatmap(data[numerical_features].corr(), cmap="Blues")

# categorical feature distribution

categorical_features=['gender', 'phoneservice',
       'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup',
       'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies',
       'contract', 'paperlessbilling', 'paymentmethod']

fig, axes = plt.subplots(7, 2, figsize=(16, 28))
for i, predictor in enumerate(categorical_features):
    row = i // 2
    col = i % 2
    sns.countplot(data=data, x=predictor, hue='churn', ax=axes[row, col])
    axes[row, col].set_title(predictor)
    plt.tight_layout()
plt.show()

# target variable distribution
target='churn'
data[target].value_counts().plot(kind='bar').set_title('churned')

# outliers analysis with IQR method
x = ['tenure','monthlycharges']
def count_outliers(data,col):
        q1 = data[col].quantile(0.25,interpolation='nearest')
        q2 = data[col].quantile(0.5,interpolation='nearest')
        q3 = data[col].quantile(0.75,interpolation='nearest')
        q4 = data[col].quantile(1,interpolation='nearest')
        IQR = q3 -q1
        global LLP
        global ULP
        LLP = q1 - 1.5*IQR
        ULP = q3 + 1.5*IQR
        if data[col].min() > LLP and data[col].max() < ULP:
            print("No outliers in",i)
        else:
            print("There are outliers in",i)
            x = data[data[col]<LLP][col].size
            y = data[data[col]>ULP][col].size
            a.append(i)
            print('Count of outliers are:',x+y)
global a
a = []
for i in x:
    count_outliers(data,i)

# cleaning and transforming data

data.drop(['customerid'],axis=1,inplace=True)

data.head()

#one hot encoding
data1=pd.get_dummies(data=data,columns=['gender', 'phoneservice',
       'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup',
       'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies',
       'contract', 'paperlessbilling', 'paymentmethod', 'churn'],drop_first=True)

data1.head()

data1.columns

data1.shape

# missing value treatment
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
data1.totalcharges=imputer.fit_transform(data1["totalcharges"].values.reshape(-1,1))
data1['totalcharges'].describe()

# feature scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(data1.drop(['churn_Yes'],axis=1))
scaled_features=scaler.transform(data1.drop('churn_Yes',axis=1))

# feature selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc

# prediction using logistic regression

X=scaled_features
Y=data1['churn_Yes']

# split the data into training and testing sets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=44)

# fit the model on the training data
logmodel=LogisticRegression()
logmodel.fit(X_train,Y_train)

# predict the churn labels for test data
Y_prob = logmodel.predict_proba(X_test)[:, 1]
predLR=logmodel.predict(X_test)
predLR

Y_test

# coefficients of the features

# Convert X to a pandas DataFrame
X = pd.DataFrame(X)
# Get coefficients and intercept
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': logmodel.coef_[0]
})
intercept = logmodel.intercept_

# Create a dictionary with the coefficients and feature names
coefficients_dict = dict(zip(coefficients['Feature'], coefficients['Coefficient']))

# Create a dictionary with the column names and indices
column_dict = dict(enumerate(data1.columns))

# Create a new DataFrame to store the results
results_df = pd.DataFrame({
    'Feature': coefficients['Feature'],
    'Coefficient': coefficients['Coefficient'],
    'Index': [column_dict[col] for col in coefficients['Feature']]
})

# Print the results
print(results_df)

# Sort coefficients by absolute value
coefficients['Absolute_Coefficient'] = coefficients['Coefficient'].abs()
coefficients_sorted = coefficients.sort_values(by='Absolute_Coefficient', ascending=False)

# Print coefficients influencing churn prediction
print("Coefficients influencing churn prediction:")
print(coefficients_sorted[['Feature' , 'Coefficient']])

intercept = logmodel.intercept_
print("Intercept:",intercept)

# calculate the classification report

print(classification_report(Y_test,predLR))

report = classification_report(Y_test, predLR, target_names=['Churn_No', 'Churn_Yes'])
# split the report into lines
lines = report.split('\n')
# split each line into parts
parts = [line.split() for line in lines[2:-5]]
# extract the metrics for each class
class_metrics = dict()
for part in parts:
    class_metrics[part[0]] = {'precision': float(part[1]), 'recall': float(part[2]), 'f1-score': float(part[3]), 'support': int(part[4])}

# create a bar chart for each metric
fig, ax = plt.subplots(1, 4, figsize=(12, 4))
metrics = ['precision', 'recall', 'f1-score', 'support']
for i, metric in enumerate(metrics):
    ax[i].bar(class_metrics.keys(), [class_metrics[key][metric] for key in class_metrics.keys()])
    ax[i].set_title(metric)
# display the plot
plt.show()

# create a confusion matrix using matshow()
confusion_matrix_LR = confusion_matrix(Y_test, predLR)
plt.matshow(confusion_matrix(Y_test, predLR))
# add labels for the x and y axes
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
for i in range(2):
    for j in range(2):
        plt.text(j, i, confusion_matrix_LR[i, j], ha='center', va='center')
# add custom labels for x and y ticks
plt.xticks([0, 1], ["Not Churned", "Churned"])
plt.yticks([0, 1], ["Not Churned", "Churned"])
plt.show()

# performance of the model

logmodel.score(X_train, Y_train)

roc_auc_score(Y_test, predLR)

#roc curve
fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()