#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score


# In[2]:


data = pd.read_csv("cleaned_data.csv")


# In[3]:


data.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')


# In[4]:


data.head()


# In[5]:


# Data cleaning and preprocessing
# Handle missing values
data = data.dropna()


# In[6]:


# Normalize numerical data
scaler = StandardScaler()
data[['price', 'time_taken', 'Days_Left']] = scaler.fit_transform(
    data[['price', 'time_taken', 'Days_Left']]
)


# In[7]:


# Histogram/Bar Chart
plt.figure(figsize=(8, 5))
sns.histplot(data['price'], kde=True)
plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# In[8]:


# K-means clustering with elbow and silhouette method
range_clusters = range(1, 11)
distortions = []

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data[['price', 'time_taken', 'Days_Left']])
    distortions.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range_clusters, distortions, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')
plt.show()

optimal_clusters = 3  # Determined from the elbow plot
kmeans_model = KMeans(n_clusters=optimal_clusters, random_state=42)
clusters = kmeans_model.fit_predict(data[['price', 'time_taken', 'Days_Left']])
data['Cluster'] = clusters

silhouette_avg = silhouette_score(data[['price', 'time_taken', 'Days_Left']], clusters)
print(f"Silhouette Score: {silhouette_avg}")

# Scatter plot for clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(x=data['price'], y=data['time_taken'], hue=data['Cluster'], palette='Set1')
plt.title('K-Means Clustering Results')
plt.xlabel('Price')
plt.ylabel('Time Taken')
plt.legend(title='Cluster')
plt.show()


# In[9]:


# Line fitting (Linear Regression)
model = LinearRegression()
X = data[['Days_Left']].values
y = data['price'].values
model.fit(X, y)

# Predictions and visualization
predictions = model.predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(data['Days_Left'], data['price'], label='Actual Data', alpha=0.6)
plt.plot(data['Days_Left'], predictions, color='red', label='Fitted Line')
plt.title('Price vs Days Left (Linear Regression)')
plt.xlabel('Days Left')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[10]:


from sklearn.preprocessing import LabelEncoder

# Apply label encoding to all categorical columns
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le  # Store the encoder for reference if needed

# Compute correlation matrix and plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[ ]:




