##     SIMPLE LINEAR REGRESSION      ##

# Importing modules
import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
# Load data
data_set= pd.read_csv('Student_Marks.csv')
print(data_set.describe())
# Print data
print("Dataset")
df=pd.DataFrame(data_set)
print(df.to_string())
# Checking for null values
df.isnull().sum()
# Assigning the independent variable to X
X= data_set.iloc[:,1].values
print(X)
# Assigning the dependent variable to y
y = df.iloc[:, 2].values
print(y)
# Reshaping the array
x=X.reshape(-1,1)
print(x)
# Load dataset slicing module
from sklearn.model_selection import train_test_split
# Splitting the dataset into training and testing set
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size= .2, random_state=0)
# Load liniear regression class
from sklearn.linear_model import LinearRegression
# Create an instance of linear regression
regressor= LinearRegression()
# Fitting the Simple Linear Regression model to the training dataset
regressor.fit(x_train,y_train)
# Prediction result on train data
x_pred= regressor.predict(x_train)
print("Prediction result on Test Data")
y_pred = regressor.predict(x_test)
# Create a dataframe to show the actual and predicted value
df2 = pd.DataFrame({'Actual Y-Data': y_test,'Predicted Y-Data': y_pred})
print(df2.to_string())
# Visualizing the Test set results
mtp.scatter(x_test, y_test, color="blue")
mtp.plot(x_test, y_pred, color="black")
mtp.title("Time study vs marks (Training Dataset)")
mtp.xlabel("Time study")
mtp.ylabel("marks")
mtp.show()
# Calculating the r2_score
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print("r2 score is ",score*100,"%")
# Plotting a scatter plot for actual and predicted values
mtp.scatter(y_test,y_pred,c="magenta")
mtp.xlabel('y test')
mtp.ylabel('predicted y')
mtp.grid()
mtp.show()