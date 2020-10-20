#!/usr/bin/env python
# coding: utf-8

# # **TASK-3 Decision Tree on Iris Dataset**
# 

# In[42]:



# Importing libraries in Python
import sklearn.datasets as datasets
import pandas as pd


# In[43]:


# Loading the iris dataset
iris=datasets.load_iris()

# Forming the iris dataframe
df=pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head(5))


# In[44]:


df.info()


# In[45]:


df.describe()


# In[46]:


import seaborn as sns
sns.pairplot(df)


# In[47]:


y=iris.target
print(y)


# **Let split the data into train and test sets**

# In[48]:


from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(df,y,random_state=0,test_size=0.2)


# In[49]:


print(train_y)
print(train_y.shape)


# ### **We need to define Decision tree Algorithm**

# In[50]:


# Defining the decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
# Decision Trees are a type of Supervised Learning Algorithms(meaning that they were given labeled data to train on). 
# The training data is continuously split into two more sub-nodes according to a certain parameter.
# The tree can be explained by two things, leaves and decision nodes.
# The decision nodes are where the data is split.
# The leaves are the decisions or the final outcomes.
# You can think of a decision tree in programming terms as a tree that has a bunch of “if statements” 
#for each node until you get to a leaf node (the final outcome).
dtree=DecisionTreeClassifier()
dtree.fit(train_x,train_y)

print('Decision Tree Classifer Created')


# In[51]:



print("Accuracy on test set: ",(dtree.score(test_x,test_y)*100),"%")


# In[52]:


get_ipython().system('pip install pydotplus')
get_ipython().system('apt-get install graphviz -y')


# In[53]:


# Import necessary libraries for graph viz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


# In[54]:


# Visualize the graph
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, feature_names=iris.feature_names,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# In[54]:




