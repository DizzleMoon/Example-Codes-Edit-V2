#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gmaps
import gmaps.datasets
#configure api
# gmaps.configure(api_key='AIzaSyB5lwTRqQqVSr4uxEAHeeF1cyVKq-G_rD8')
# gmaps.configure(api_key='AIzaSyDMSV5BvQcWkwku88X_F4EIsfjiPlYnVVE')
# gmaps.configure(api_key='AIzaSyA86cB5j_1VqxdLI4Pz5FiSyNt7VOxdYN8')
# gmaps.configure(api_key='AIzaSyCZO2_TlszLSJ0l8EWDJnO9IyHZnpNUbmg')
gmaps.configure(api_key='AIzaSyA86cB5j_1VqxdLI4Pz5FiSyNt7VOxdYN8')


# In[2]:


new_york_coordinates = (40.75, -74.00)
gmaps.figure(center=new_york_coordinates, zoom_level=12)


# In[3]:


# import gmaps
# import gmaps.datasets
# # Use google maps api
# gmaps.configure(api_key='AIzaSyB5lwTRqQqVSr4uxEAHeeF1cyVKq-G_rD8') # Fill in with your API key
# # Get the dataset
# earthquake_df = gmaps.datasets.load_dataset_as_df('earthquakes')
# #Get the locations from the data set
# locations = earthquake_df[['latitude', 'longitude']]
# #Get the magnitude from the data
# weights = earthquake_df['magnitude']
# #Set up your map
# fig = gmaps.figure()
# fig.add_layer(gmaps.heatmap_layer(locations, weights=weights))
# fig


# In[4]:


import gmaps
import gmaps.datasets
# gmaps.configure(api_key='AIzaSyCo8CDTfE1QHVlG8mSkgtbZs2vOU3cwlhM')
#Define location 1 and 2
Durango = (37.2753,-107.880067)
SF = (37.7749,-122.419416)
# Markers
markers_loc = [Durango,SF]
markers = gmaps.marker_layer(markers_loc)
#Create the map
fig = gmaps.figure()
#create the layer
layer = gmaps.directions.Directions(Durango, SF,mode='car')
print(layer)
#Add the layer
fig.add_layer(layer)
fig.add_layer(markers)
fig


# In[5]:


# import googlemaps
# from datetime import datetime

# gmaps = googlemaps.Client(key = 'AIzaSyB5lwTRqQqVSr4uxEAHeeF1cyVKq-G_rD8' )

# now = datetime.now()
# directions_result = gmaps.directions(Durango,SF, mode='driving')

# fig = gmaps.figure(map_type = 'HYBRID')
# fig.add_layer(directions_result)
# fig


# In[ ]:





# In[ ]:




