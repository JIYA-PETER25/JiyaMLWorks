##     SUPPORT VECTOR MACHINE      ##

# Importing modules
import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
# Load and print dataset
data_set= pd.read_csv('riceClassification.csv')
df=pd.DataFrame(data_set)
print(df.to_string())
df.info()
df.describe()
# Checking for null values
df.isnull().sum()
# Checking for duplicate values
df.duplicated().sum()
# Removing duplicates
df=df.drop_duplicates()
# Assigning the independent and dependent variables
x = df.drop(columns=['id', 'Class'])
y = df['Class']
# Splitting the dataset into training and testing set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)
# Fitting the SVM classifier for the training data
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(x_train, y_train)
# Predicting the test set result
y_pred= classifier.predict(x_test)
# Calculating the accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred,y_test)*100
print(accuracy)
# Dataframe showing actual and predicted values
df2=pd.DataFrame({"Actual Y_Test":y_test,"PredictionData":y_pred})
print("prediction status")
print(df2.to_string())