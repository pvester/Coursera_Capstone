
# coding: utf-8

# ### Loading Packages and Reading Table Data
# Import packages and request wikipedia url. Find all values in the wikipedia table in-between th/td and assume these are all the table values we are interested in. This assumption has been double checked.

# In[10]:


from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd

wiki_url = requests.get('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M')
soup = BeautifulSoup(wiki_url.text, 'lxml')
for items in soup.find_all("table", class_="wikitable sortable"):
    data = [' '.join(item.text.split()) for item in items.find_all(['th','td'])]
table_data = np.reshape(data, (-1,3)) 


# ### Read data into Pandas Dataframe
# read the table_data into a panda dataframe. Replace "not assigned" neighborhood with corresponding value from Boroughs. Remove rows with "not assigned" boroughs.
# Group and sum strings by PostalCode/Borough and make sure they are separated by comma. At last find shape of new cleaned datafram.

# In[11]:


df = pd.DataFrame()
df['PostalCode'] = table_data[1:,0]
df['Borough'] = table_data[1:,1]
df['Neighborhood'] = table_data[1:,2]
df['Neighborhood'].replace("Not assigned", df['Borough'], inplace=True)
df['Borough'].replace("Not assigned", pd.np.nan, inplace=True)
df.dropna(axis = 0, subset=["Borough"], inplace=True)
df.reset_index()

for i in range(0,len(df['Neighborhood'])):
    df.Neighborhood.iloc[i] = df.Neighborhood.iloc[i] + ", "
    
df = df.groupby(["PostalCode", "Borough"], as_index = False, sort = False).sum()
for i in range(0,len(df['Neighborhood'])):
    if df.Neighborhood.iloc[i].endswith(", "):
        df.Neighborhood.iloc[i] = df.Neighborhood.iloc[i][:-2]
        
df.head()


# In[12]:


df.shape


# ### Assign Longitudes/Lattitude to Postal Codes
# Import data from .csv file, and merge the two data sets on the postal code.

# In[13]:


data = pd.read_csv('http://cocl.us/Geospatial_data')


# In[14]:


df['Lattitude'] = "NAN"
df['Longitude'] = "NAN"
for i in range(0, len(df['PostalCode'])):
    for ii in range(0, len(data)):
        if df['PostalCode'].iloc[i] == data.iloc[ii][0]:
            df['Lattitude'].iloc[i] = float(data.iloc[ii][1])
            df['Longitude'].iloc[i] = float(data.iloc[ii][2])
            break
df.head()


# ### Visualization

# Visualize only boroughs with Toronto in their names on a map.

# In[15]:


get_ipython().system(u'conda install -c conda-forge folium=0.5.0 --yes ')
import folium

for i in range(0,len(df)):
    if "Toronto" not in df['Borough'].iloc[i]:
        df['Borough'].iloc[i] = pd.np.nan
df.dropna(axis = 0, subset=["Borough"], inplace=True)

map_clusters = folium.Map(location=[df["Lattitude"].mean(), df["Longitude"].mean()], zoom_start=12)
for lat, lon in zip(df['Lattitude'], df['Longitude']):
   folium.CircleMarker(
        [lat, lon],
        radius=5,
       fill=True,
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# ### Construct Clustering Model and Compare with Real Districts

# We know there are four district with Toronto in its name, lets just for fun and as an illustration show how accurate a k mean clustering algorithm with k = 4 can predict the shape of these districts

# In[16]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=0).fit(df.iloc[:,3:5].values.reshape(-1,2))
kmeans


# plot the results on a map. The clustering algorihm decides to cluster the data into four "districts" in south, west, east and center.

# In[17]:


import matplotlib.cm as cm
import matplotlib.colors as colors


map_clusters = folium.Map(location=[df["Lattitude"].mean(), df["Longitude"].mean()], zoom_start=12)

# set color scheme for the clusters
x = np.arange(19)
ys = [i+x+(i*x)**2 for i in range(19)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

for lat, lon, cluster in zip(df['Lattitude'], df['Longitude'], kmeans.labels_):
   folium.CircleMarker(
        [lat, lon],
       color='black',
        radius=7,
       fill=True,
       fill_color=rainbow[4*cluster+1],
        fill_opacity=0.9).add_to(map_clusters)
       
map_clusters


# Now figure out how well the four clusters corresponds to the four real districts

# In[18]:


yhat = []
for i in range(0,len(df)):
    if "Downtown" in df["Borough"].iloc[i]:
        yhat.append(0)
    elif "West" in df["Borough"].iloc[i]:
        yhat.append(2)
    elif "East" in df["Borough"].iloc[i]:
        yhat.append(3)
    else:
        yhat.append(1)


# In[19]:


accuracy = sum(yhat == kmeans.labels_)/len(yhat)
accuracy


# almost 95 percent accurate in predicting the shape of the OFFICIAL district using a k mean clustering method!
