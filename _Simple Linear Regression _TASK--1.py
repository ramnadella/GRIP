#!/usr/bin/env python
# coding: utf-8

# #                               Simple Linear Regression 
# # In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.
# 

# In[1]:


import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


dataset = pd.read_csv('https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv')
#print(dataset)
dataset.head(25)


# In[4]:


# Plotting the distribution of scores
dataset.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs scores')  
plt.xlabel('Hours Studied')  
plt.ylabel('scores get')  
plt.show()


# In[5]:


X = dataset.iloc[:, :-1].values 

y = dataset.iloc[:, 1].values
# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[6]:


# Plotting the regression line
Y = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X,Y);
plt.show()


# In[7]:


print(X_train) # Testing data - In Hours
y_pred = regressor.predict(X_train) # Predicting the scores


# In[8]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_train, 'Predicted': y_pred})  
df 


# In[9]:


# You can also test with your own data
Hours = [[9.25]]
own_pred =regressor.predict(Hours)
print("No of Hours = {}".format(Hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[ ]:





# In[ ]:




