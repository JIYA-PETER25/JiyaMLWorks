##     RANDOM FOREST      ##

# Importing modules
import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
# Load and print the dataset
data_set= pd.read_csv('hypertension_data.csv')
df=pd.DataFrame(data_set)
print("Actual Dataset")
print(df.to_string())
# Checking for null values
df.isnull().sum()
# Removing the null values
df=df.dropna()
df.isnull().sum()
# Assigning the values to independent and dependent variables
x= data_set.iloc[:,2:13].values
y= data_set.iloc[:, 13].values
# Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)
# Fitting Random Forest classifier to the training set
from sklearn.ensemble import RandomForestClassifier
classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy")
classifier.fit(x_train, y_train)
# Predicting the test set result
y_pred= classifier.predict(x_test)
print("------------PREDICTION----------")
df2=pd.DataFrame({"Actual Result-Y":y_test,"PredictionResult":y_pred})
print(df2.to_string())
from sklearn.metrics import accuracy_score
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))