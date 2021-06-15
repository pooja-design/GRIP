#!/usr/bin/env python
# coding: utf-8

# ### Name: Pooja D S
# 
# ### Graduate Rotational Internaship Program
# 
# ## Data Science and Business Analytics Tasks

# # Task 1 :Prediction Using Supervised ML

# ## Problem statement: Predict the percentage of an student based on the no. of study hours.

# #### Importing requird Libraries:

# In[200]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn import metrics


# #### Importing dataset provided: 

# In[201]:


dataset = pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")


# In[202]:


dataset


# ####  Number of rows and coloumns present in the dataset :

# In[203]:


dataset.shape


# #### Visualisation of the dataset :

# In[204]:


plt.scatter(dataset['Hours'], dataset['Scores'])
plt.title('Hours vs Score')
plt.xlabel('Studied Hours')
plt.ylabel('Score')
plt.show()


# #### X contains explanatory variable's values and y contains target feature's values :

# In[205]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[207]:


X_train.shape


# In[208]:


X_test.shape


# Training set contains 17 values and test set contains 8 values.
# 

# #### Training the Simple Linear Regression model on the Training set :

# In[238]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# #### Plotting the regression line :

# In[239]:


line = regressor.coef_*X+regressor.intercept_


# #### Plotting for the rest of the data

# In[240]:


plt.scatter(X, y)
plt.plot(X, line,color="green");
plt.show()


# #### Visualising Trainset reults :

# In[241]:


plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('Hours vs. Score (Training set)')
plt.xlabel('Hours studied')
plt.ylabel('Score')
plt.show()


# #### Visualising Test set results : 

# In[242]:


plt.scatter(X_test, y_test, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('Hours vs. Score (Test set)')
plt.xlabel('Hours studied')
plt.ylabel('Score')
plt.show()


# #### Predicted Test set results:

# In[243]:


y_pred = regressor.predict(X_test)
print(y_pred)


# #### Comparing Actual values with Predicted values :

# In[244]:


dataset = pd.DataFrame({'Actual Score': y_test, 'Predicted Score': y_pred})  
dataset


# In[248]:


dataset = np.array(9.25)
dataset


# In[249]:


dataset = dataset.reshape(-1, 1)
dataset


# In[250]:


pred = regressor.predict(dataset)
print("If the student studies for 9.25 hours/day, the score he may get is",pred)


# #### Evaluating the model :
# 

# In[251]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


# In[252]:


print("The R-Square of the model is: ",metrics.r2_score(y_test,y_pred))


# As model has r2_score (coefficient of determination) is high this model fits the data well.

# ## If student studies for 9.25 hours/day he may score 92.9
