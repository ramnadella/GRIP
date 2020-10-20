#!/usr/bin/env python
# coding: utf-8

# ### TASK 2: K- Means Clustering
# 
# This notebook will walk through some of the basics of K-Means Clustering.

# In[1]:


# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# In[3]:


# Load the iris dataset
iris = datasets.load_iris()
iris_df=pd.DataFrame(iris.data, columns = iris.feature_names)

#iris_df=pd.DataFrame(iris.data)
#we can above command if we use it we can extract dataset without col names i.e sepal length etc.
iris_df.head()


# #### How do you find the optimum number of clusters for K Means? How does one determine the value of K?

# In[4]:


# Finding the optimum number of clusters for k-means classification

x = iris_df.iloc[:, [0, 1, 2, 3]].values
print(x)


# In[6]:


#Using the elbow method to find out the optimal number of #clusters. 
#KMeans class from the sklearn library.

from sklearn.cluster import KMeans
wcss = []

#this loop will fit the k-means algorithm to our data and 
#second we will compute the within cluster sum of squares and 
#appended to our wcss list.



for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
        
        
#i above is between 1-10 numbers. init parameter is the random 
#initialization method  
#we select kmeans++ method. max_iter parameter 
#the maximum number of iterations there can be to 
#find the final clusters when the K-meands algorithm is running. we 
#enter the default value of 300
#the next parameter is n_init which is the number of times the
#K_means algorithm will be run with
#different initial centroid.

#kmeans.fit(x)

#kmeans algorithm fits to the X dataset

#wcss.append(kmeans.inertia_)

#kmeans inertia_ attribute is:  Sum of squared distances of samples 
#to their closest cluster center.

#.Plot the elbow graph
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# You can clearly see why it is called 'The elbow method' from the above graph, the optimum clusters is where the elbow occurs. This is when the within cluster sum of squares (WCSS) doesn't decrease significantly with every iteration.
# 
# From this we choose the number of clusters as ** '3**'.

# In[9]:


# According to the Elbow graph we deterrmine the clusters number as 
# Applying k-means algorithm to the X dataset.
kmeans = KMeans(n_clusters=3 )


#                           OR
# WE CAN USE
#kmeans = KMeans(n_clusters = 3, init = 'k-means++',
#                max_iter = 300, n_init = 10, random_state = 0)


# We are going to use the fit predict method that returns for each 
#observation which cluster it belongs to. The cluster to which 
#client belongs and it will return this cluster numbers into a 
#single vector that is  called y K-means

y_kmeans = kmeans.fit_predict(x)
print(y_kmeans)


# In[10]:


iris_df['clusters']=y_kmeans
iris_df.head()


# In[11]:


kmeans.cluster_centers_


# In[12]:


# Visualising the clusters - On the first two columns
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# In[ ]:




