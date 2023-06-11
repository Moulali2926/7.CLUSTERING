#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import pandas as pd
import matplotlib.pylab as plt 
evil = pd.read_csv("crime_data.csv")
evil    


# In[45]:


evil.describe()


# In[ ]:





# In[46]:


evil.info()


# In[47]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)  


# In[48]:


df_norm = norm_func(evil.iloc[:,1:])
df_norm


# In[49]:


from sklearn.preprocessing import MinMaxScaler
trans = MinMaxScaler()
data = pd.DataFrame(trans.fit_transform(evil.iloc[:,1:]))
data   


# In[50]:


from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 
#p = np.array(df_norm) # converting into numpy array format 
z = linkage(df_norm, method="average",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram( z,)
plt.show()
  


# In[51]:


from sklearn.cluster import AgglomerativeClustering 
import warnings 
warnings.filterwarnings('ignore')
h_complete = AgglomerativeClustering(n_clusters=5, linkage='average',affinity = "euclidean").fit(df_norm) 

cluster_labels=pd.Series(h_complete.labels_)
cluster_labels
evil['clust']=cluster_labels # creating a  new column and assigning it to new column 
evil  


# In[52]:


evil.iloc[:,1:].groupby(evil.clust).mean()   


# In[53]:


data = evil[(evil.clust==1)]
data   


# In[54]:


data = evil[(evil.clust==0)]
data    


# In[55]:


data = evil[(evil.clust==3)]
data     


# In[56]:


data = evil[(evil.clust==2)]
data    


# In[57]:


# create clusters
hc=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='complete')
hc 


# In[58]:


y_hc=hc.fit_predict(df_norm)


# In[59]:


y_hc


# In[60]:


evil['h_clusterid']=hc.labels_


# In[61]:


evil


# In[ ]:





# In[ ]:





# # # kmeans

# In[62]:


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# In[63]:


evil=pd.read_csv('crime_data.csv')


# In[64]:


def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)


# In[66]:


df_norm=norm_func(evil.iloc[:,1:])


# In[67]:


wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow curv')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[68]:


model=KMeans(n_clusters=4)
model.fit(df_norm)
model.labels_


# In[70]:


x=pd.Series(model.labels_)
evil['Clust']=x


# In[71]:


evil


# In[72]:


evil.iloc[:,1:5].groupby(evil.Clust).mean()


# In[ ]:





# ## DBSCAN

# In[73]:


from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# In[75]:


evil


# In[76]:


evil.info()


# In[77]:


evil.values


# In[79]:


ev=evil.iloc[:,1:5]


# In[81]:


ev.values


# In[83]:


stscaler=StandardScaler().fit(ev.values)
x=stscaler.transform(ev.values)


# In[84]:


x


# In[ ]:





# In[85]:


dbscan=DBSCAN(eps=2,min_samples=5)
dbscan.fit(x)
DBSCAN(eps=2)
dbscan.labels_


# In[86]:


cl=pd.DataFrame(dbscan.labels_,columns=['cluster'])
cl


# In[87]:


pd.concat([evil,cl],axis=1)


# In[ ]:




