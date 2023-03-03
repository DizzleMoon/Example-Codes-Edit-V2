#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gmaps
from datetime import datetime
now = datetime.now()

#configure api
# gmaps.configure(api_key='AIzaSyD9PKg58F9-aBpbRiGegcjJzlJrWR6Cu40')
gmaps.configure(api_key='AIzaSyA86cB5j_1VqxdLI4Pz5FiSyNt7VOxdYN8')

start =(0.3302,32.5737) #this is our start address
end = (0.3302,32.5737) #my start address is same as end address

# Markers
marker_locations = [
(0.3302,32.5737),
(0.3302,32.5737),
(0.332186,32.570529),(0.333551,32.570625),(0.331071,32.574760),(0.338699,32.561229)
]
#Create the map
# fig = gmaps.Map()
# Markers
markers = gmaps.marker_layer(marker_locations)

#the addresses my route has to move through
waypoints = [(0.332186,32.570529),(0.333551,32.570625),(0.331071,32.574760),(0.338699,32.561229)]


#Create the map
fig = gmaps.figure()
#create the layer
layer = gmaps.directions.Directions(start, end,waypoints = waypoints,optimize_waypoints=True,
                                    mode='car')
# layer = gmaps.directions.Directions(start, end,waypoints = waypoints,optimize_waypoints=True,
#                                     mode='car',api_key='AIzaSyD9PKg58F9-aBpbRiGegcjJzlJrWR6Cu40',departure_time = now)
#Add the layer
fig.add_layer(layer)
# fig.add_layer(gmaps.traffic_layer())
# fig.add_layer(markers)
fig


# In[ ]:





# In[ ]:




