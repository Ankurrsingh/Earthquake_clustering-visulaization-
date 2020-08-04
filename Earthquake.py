#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import pandas as pd
import numpy as np

from tqdm import tqdm

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

from ipywidgets import interactive

from collections import defaultdict

import hdbscan
import folium
import re


cols = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
        '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', 
        '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', 
        '#000075', '#808080']*25


# In[2]:


df=pd.read_csv("Earthquake.csv")


# In[3]:


df.head(10)


# In[4]:


df.isna().count()


# In[5]:


df.drop(["time", "depth","mag","magType","nst","gap","dmin","rms","net","id","updated","type",
         "horizontalError","depthError","magError","magNst","status","locationSource","magSource"], axis = 1, inplace = True) 


# In[6]:


dff=df


# In[7]:


df.head(5)


# In[8]:


df.duplicated(subset=['longitude', 'latitude']).values.any()


# In[9]:


df.isna().values.any()


# In[10]:


print(f'Before dropping NaNs and dupes\t:\tdf.shape = {df.shape}')
df.dropna(inplace=True)
df.drop_duplicates(subset=['longitude', 'latitude'], keep='first', inplace=True)
print(f'After dropping NaNs and dupes\t:\tdf.shape = {df.shape}')


# In[11]:


df.head(5)


# In[12]:


X = np.array(df[[ 'latitude','longitude']], dtype='float64')


# In[13]:


print(X[:10])


# In[14]:


plt.scatter(X[:,0], X[:,1], alpha=0.2, s=10)


# In[15]:


m = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=9, 
               tiles='Stamen Toner')

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row.latitude, row.longitude],
        radius=5,
        popup=re.sub(r'[^a-zA-Z ]+', '', row.place),
        color='#1787FE',
        fill=True,
        fill_colour='#1787FE'
    ).add_to(m)


# In[16]:


m


# In[17]:


X = np.array(df[['latitude', 'longitude']], dtype='float64')
k = 5
model = KMeans(n_clusters=k, random_state=17).fit(X)
class_predictions = model.predict(X)
df[f'CLUSTER_kmeans{k}'] = class_predictions


# In[18]:


df.head()


# In[19]:


def create_map(df, cluster_column):
    m = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=9, tiles='Stamen Toner')

    for _, row in df.iterrows():

        if row[cluster_column] == -1:
            cluster_colour = '#000000'
        else:
            cluster_colour = cols[row[cluster_column]]

        folium.CircleMarker(
            location= [row['latitude'], row['longitude']],
            radius=5,
            popup= row[cluster_column],
            color=cluster_colour,
            fill=True,
            fill_color=cluster_colour
        ).add_to(m)
        
    return m

m = create_map(df, 'CLUSTER_kmeans5')
print(f'K={k}')
print(f'Silhouette Score: {silhouette_score(X, class_predictions)}')


# In[20]:


m


# In[ ]:





# In[21]:


best_silhouette, best_k = -1, 0

for k in tqdm(range(2, 100)):
    model = KMeans(n_clusters=k, random_state=1).fit(X)
    class_predictions = model.predict(X)
    
    curr_silhouette = silhouette_score(X, class_predictions)
    if curr_silhouette > best_silhouette:
        best_k = k
        best_silhouette = curr_silhouette
        
print(f'K={best_k}')
print(f'Silhouette Score: {best_silhouette}') 


# In[22]:


model = DBSCAN(eps=0.01, min_samples=5).fit(X)
class_predictions = model.labels_

df['CLUSTERS_DBSCAN'] = class_predictions
class_predictions


# In[23]:


# code for indexing out certain values
dummy = np.array([-1, -1, -1, 2, 3, 4, 5, -1])

new = np.array([(counter+2)*x if x==-1 else x for counter, x in enumerate(dummy)])


# In[24]:


m = create_map(df, 'CLUSTERS_DBSCAN')

    
print(f'Number of clusters found: {len(np.unique(class_predictions))}')
print(f'Number of outliers found: {len(class_predictions[class_predictions==-1])}')

print(f'Silhouette ignoring outliers: {silhouette_score(X[class_predictions!=-1], class_predictions[class_predictions!=-1])}')

no_outliers = 0
no_outliers = np.array([(counter+2)*x if x==-1 else x for counter, x in enumerate(class_predictions)])
print(f'Silhouette outliers as singletons: {silhouette_score(X, no_outliers)}')


# In[25]:


m


# In[26]:


model = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=500, 
                        cluster_selection_epsilon=0.01)
#min_cluster_size
#min_samples
#cluster_slection_epsilon

class_predictions = model.fit_predict(X)
df['CLUSTER_HDBSCAN'] = class_predictions


# In[27]:


m = create_map(df, 'CLUSTER_HDBSCAN')

print(f'Number of clusters found: {len(np.unique(class_predictions))-1}')
print(f'Number of outliers found: {len(class_predictions[class_predictions==-1])}')

print(f'Silhouette ignoring outliers: {silhouette_score(X[class_predictions!=-1], class_predictions[class_predictions!=-1])}')

no_outliers = np.array([(counter+2)*x if x==-1 else x for counter, x in enumerate(class_predictions)])
print(f'Silhouette outliers as singletons: {silhouette_score(X, no_outliers)}')


# In[29]:


best_silhouette, best_min_cluster_size,best_min_samples = -1, 0,100

for k in tqdm(range(2, 103,20)):
    for m in tqdm(range(100,500,50)):
        model = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=500, 
                        cluster_selection_epsilon=0.01)
        class_predictions = model.fit_predict(X)
    
        curr_silhouette = silhouette_score(X, class_predictions)
        if curr_silhouette > best_silhouette:
            best_min_cluster_size = k
            best_silhouette = curr_silhouette
            best_min_samples=m
        
print(f'min_cluster_size={best_min_cluster_size}')
print(f'Silhouette Score: {best_silhouette}') 
print(f'min samples: {m}')


# In[30]:


model = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=450, 
                        cluster_selection_epsilon=0.2,algorithm='best')
#min_cluster_size
#min_samples
#cluster_slection_epsilon

class_predictions = model.fit_predict(X)
df['CLUSTER_HDBSCAN'] = class_predictions


# In[31]:


m = create_map(df, 'CLUSTER_HDBSCAN')

print(f'Number of clusters found: {len(np.unique(class_predictions))-1}')
print(f'Number of outliers found: {len(class_predictions[class_predictions==-1])}')

print(f'Silhouette ignoring outliers: {silhouette_score(X[class_predictions!=-1], class_predictions[class_predictions!=-1])}')

no_outliers = np.array([(counter+2)*x if x==-1 else x for counter, x in enumerate(class_predictions)])
print(f'Silhouette outliers as singletons: {silhouette_score(X, no_outliers)}')


# In[32]:


'''Trying on some more values(changing Hyperparameters)'''
model = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=450, 
                        cluster_selection_epsilon=0.2,algorithm='prims_kdtree',allow_single_cluster=False)
#min_cluster_size
#min_samples
#cluster_slection_epsilon

class_predictions = model.fit_predict(X)
df['CLUSTER_HDBSCAN'] = class_predictions
m = create_map(df, 'CLUSTER_HDBSCAN')

print(f'Number of clusters found: {len(np.unique(class_predictions))-1}')
print(f'Number of outliers found: {len(class_predictions[class_predictions==-1])}')

print(f'Silhouette ignoring outliers: {silhouette_score(X[class_predictions!=-1], class_predictions[class_predictions!=-1])}')

no_outliers = np.array([(counter+2)*x if x==-1 else x for counter, x in enumerate(class_predictions)])
print(f'Silhouette outliers as singletons: {silhouette_score(X, no_outliers)}')


# In[38]:


model = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=450, 
                        cluster_selection_epsilon=0.4,algorithm='boruvka_balltree',cluster_selection_method='leaf',allow_single_cluster=False)
#min_cluster_size
#min_samples
#cluster_slection_epsilon

class_predictions = model.fit_predict(X)
df['CLUSTER_HDBSCAN'] = class_predictions
m = create_map(df, 'CLUSTER_HDBSCAN')
print(f'Number of clusters found: {len(np.unique(class_predictions))-1}')
print(f'Number of outliers found: {len(class_predictions[class_predictions==-1])}')

print(f'Silhouette ignoring outliers: {silhouette_score(X[class_predictions!=-1], class_predictions[class_predictions!=-1])}')

no_outliers = np.array([(counter+2)*x if x==-1 else x for counter, x in enumerate(class_predictions)])
print(f'Silhouette outliers as singletons: {silhouette_score(X, no_outliers)}')


# In[39]:


m


# In[35]:


get_ipython().run_line_magic('pinfo', 'hdbscan.HDBSCAN')


# In[40]:


model = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=50, 
                        cluster_selection_epsilon=0.4,algorithm='boruvka_balltree',cluster_selection_method='leaf',allow_single_cluster=False)
#min_cluster_size
#min_samples
#cluster_slection_epsilon

class_predictions = model.fit_predict(X)
df['CLUSTER_HDBSCAN'] = class_predictions
m = create_map(df, 'CLUSTER_HDBSCAN')
print(f'Number of clusters found: {len(np.unique(class_predictions))-1}')
print(f'Number of outliers found: {len(class_predictions[class_predictions==-1])}')

print(f'Silhouette ignoring outliers: {silhouette_score(X[class_predictions!=-1], class_predictions[class_predictions!=-1])}')

no_outliers = np.array([(counter+2)*x if x==-1 else x for counter, x in enumerate(class_predictions)])
print(f'Silhouette outliers as singletons: {silhouette_score(X, no_outliers)}')


# In[41]:


m


# In[ ]:


'''
Hierarchical Density-Based Spatial Clustering of Applications with Noise
min_cluster_size: The minimum size of clusters (default=5)
min_samples: The number of samples in a neighbourhood for a point to be considered a core point
cluster_selection_epsilon(float): A distance threshold. Clusters below this value will be merged.(default=0.0)
algorithm: Exactly which algorithm to use(default=best)
    other algorithms are:best
                         generic
                         prims_kdtree
                         prims_balltree
                         boruvka_kdtree
                         boruvka_balltree
allow_single_cluster: By default HDBSCAN* will not produce a single cluster, 
                    setting this to True will override this and allow single cluster results 


fit(X, y=None): Perform HDBSCAN clustering from features or distance matrix.
fit_predict(X, y=None): Performs clustering on X and returns cluster labels.
'''


# In[42]:


classifier = KNeighborsClassifier(n_neighbors=1)


# In[46]:


df_train = df[df.CLUSTER_HDBSCAN!=-1]
df_predict = df[df.CLUSTER_HDBSCAN==-1]


# In[49]:


X_train = np.array(df_train[['latitude', 'longitude']], dtype='float64')
y_train = np.array(df_train['CLUSTER_HDBSCAN'])
X_predict = np.array(df_predict[['latitude','longitude']], dtype='float64')


# In[50]:


classifier.fit(X_train, y_train)


# In[51]:


predictions = classifier.predict(X_predict)


# In[52]:


df['CLUSTER_hybrid'] = df['CLUSTER_HDBSCAN']


# In[53]:


df.loc[df.CLUSTER_HDBSCAN==-1, 'CLUSTER_hybrid'] = predictions


# In[54]:


df


# In[55]:


m = create_map(df, 'CLUSTER_hybrid')


# In[56]:


m


# In[57]:


df.CLUSTER_hybrid.nunique()


# In[ ]:




