##     K-MEANS CLUSTERING      ##

# Importing modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Importing the dataset
data_set= pd.read_csv("mobilepricedata.csv")
# Form a dataframe
df=pd.DataFrame(data_set)
print(df.to_string())
df.info()
df.describe()
# Checking for null values
df.isnull().sum()
# Checking for duplicates
df.duplicated().sum()
# Removing duplicates
df=df.drop_duplicates()
#Extracting the matrix of features
x=df.iloc[:,[3,5]].values
x
# Finding optimal number of clusters using the elbow method
from sklearn.cluster import KMeans
wcss_list= [] #Initializing the list for the values of WCSS
#Using for loop for iterations from 1 to 10.
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)
print(wcss_list)
plt.plot(range(1, 11), wcss_list)
plt.title('The Elobw Method Graph')
plt.xlabel('Number of clusters(k)')
plt.ylabel('wcss_list')
plt.show()
# Training the K-means model on a dataset
kmeans=KMeans(n_clusters=3)
y_kmeans=kmeans.fit_predict(x)
print(y_kmeans)
# Visualizing the clusters
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=50,c='red',label='cluster 1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=50,c='blue',label='cluster 2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=50,c='green',label='cluster 3')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,c='yellow',label='centroids')
plt.title('Clustering')
plt.xlabel('BrandCategory ')
plt.ylabel('Rating')
plt.legend()
plt.show()