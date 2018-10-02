
# coding: utf-8

# In[20]:


from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


# In[21]:


iris = load_iris()


# In[22]:


x = iris.data
y = iris.target


# In[23]:


model = DecisionTreeClassifier()
model.fit(x, y)


# In[42]:


print(model.predict([[5.0, 10.0, 4.0, 10.0]]))


# In[43]:


model.score


# In[47]:


print(model.score(x, y))

