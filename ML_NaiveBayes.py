##     NAIVE BAYES      ##

# Importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Load and print the dataset
data_set= pd.read_csv("Iris.csv")
df=pd.DataFrame(data_set)
print(df.to_string())
df.info()
df.describe()
# Checking for null values
df.isnull().sum()
# Checking for duplicate values
df.duplicated().sum()
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
# Assuming df is your DataFrame
X = df.drop('Species', axis=1)
y = df['Species']
# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()
# Train the classifier
nb_classifier.fit(X_train, y_train)
# Make predictions on the test set
y_pred = nb_classifier.predict(X_test)
# predicting the accuracy score
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print("r2 score is ",score*100,"%")
report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)
