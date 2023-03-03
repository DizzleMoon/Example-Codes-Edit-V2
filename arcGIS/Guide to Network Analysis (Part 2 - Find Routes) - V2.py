#!/usr/bin/env python
# coding: utf-8

# In[1]:



from copy import deepcopy
from datetime import datetime
from IPython.display import HTML
import json
import pandas as pd
from arcgis.gis import GIS
import arcgis.network as network
import arcgis.geocoding as geocoding
import arcgis


# In[2]:


# Connect to the routing service and call it
api_key = "AAPK584c0e1d93c94b7184b16cab622fc22akbl58IK1U9Xlnhgaqa3L0U4RG0U_Fkb21i2nHHRKngcdkmbcUBML_sGizmYFZpPN"
my_gis = arcgis.GIS("https://www.arcgis.com", api_key=api_key)
# my_gis = GIS('https://www.arcgis.com', user_name, password)


# In[3]:


route_service_url = my_gis.properties.helperServices.route.url
route_service_url


# In[4]:


route_service = network.RouteLayer(route_service_url, gis=my_gis)
route_service


# In[5]:


# inputs
stops = "-122.68782,45.51238;-122.690176,45.522054;-122.614995,45.526201"


# In[6]:


route_layer = network.RouteLayer(route_service_url, gis=my_gis)
result = route_layer.solve(stops=stops,
                           return_directions=False, return_routes=True, 
                           output_lines='esriNAOutputLineNone',
                           return_barriers=False, return_polygon_barriers=False, 
                           return_polyline_barriers=False)

travel_time = result['routes']['features'][0]['attributes']['Total_TravelTime']
print("Total travel time is {0:.2f} min".format(travel_time))


# In[7]:


stop1_address = 'Miami'
stop2_address = 'San Francisco'

stop1_geocoded = geocoding.geocode(stop1_address)
stop2_geocoded = geocoding.geocode(stop2_address)

stops = '{0},{1}; {2},{3}'.format(stop1_geocoded[0]['attributes']['X'],
                                  stop1_geocoded[0]['attributes']['Y'],
                                  stop2_geocoded[0]['attributes']['X'],
                                  stop2_geocoded[0]['attributes']['Y'])

route_layer = network.RouteLayer(route_service_url, gis=my_gis)
result = route_layer.solve(stops=stops, return_directions=False, return_routes=True, 
                           output_lines='esriNAOutputLineNone', return_barriers=False, 
                           return_polygon_barriers=False, return_polyline_barriers=False)

travel_time = result['routes']['features'][0]['attributes']['Total_TravelTime']
print("Total travel time is {0:.2f} min".format(travel_time))


# In[8]:


import datetime

start_time = int(datetime.datetime.utcnow().timestamp() * 1000)

route_layer = network.RouteLayer(route_service_url, gis=my_gis)
result = route_layer.solve(stops=stops, 
                           directions_language='en-US', return_routes=False,
                           return_stops=False, return_directions=True,
                           directions_length_units='esriNAUKilometers',
                           return_barriers=False, return_polygon_barriers=False,
                           return_polyline_barriers=False, start_time=start_time,
                           start_time_is_utc=True)


# In[9]:


records = []
travel_time, time_counter = 0, 0
distance, distance_counter = 0, 0

for i in result['directions'][0]['features']:
    time_of_day = datetime.datetime.fromtimestamp(i['attributes']['arriveTimeUTC'] / 1000).strftime('%H:%M:%S')
    time_counter = i['attributes']['time']
    distance_counter = i['attributes']['length']
    travel_time += time_counter
    distance += distance_counter
    records.append( (time_of_day, i['attributes']['text'], 
                     round(travel_time, 2), round(distance, 2))  ) 


# In[10]:


pd.set_option('display.max_colwidth', 100)
df = pd.DataFrame.from_records(records, index=[i for i in range(1, len(records) + 1)], 
                               columns=['Time of day', 'Direction text', 
                                        'Duration (min)', 'Distance (km)'])
HTML(df.to_html(index=False))


# In[11]:


td_data = ['<td align="left">{:s}</td>            <td align="left">{:s}</td>            <td align="left">{:.2f}</td>            <td align="left">{:.2f}</td>'.format(*values) 
           for values in records]
tr_data = ['<tr>{}</tr>'.format(i) for i in td_data]

display(HTML('<table>              <tr> <th> Time of day </th>              <th>Instruction</th>              <th>Time</th>              <th>Distance</th> </tr> {0} </table>'.format(''.join(tr_data))))


# In[12]:


stops = '''-3.203062,55.906437; -3.190080,55.935570'''

route_layer = network.RouteLayer(route_service_url, gis=my_gis)
result = route_layer.solve(stops=stops, 
                           directions_language='en-US', return_routes=True,
                           return_stops=True, return_directions=False,
                           return_barriers=False, return_polygon_barriers=False,
                           return_polyline_barriers=False)


# In[13]:


my_map = my_gis.map('San Francisco', zoomlevel=13)
my_map


# In[14]:


# my_map.clear_graphics()
stop_count = result['routes']['features'][0]['attributes']['StopCount']
travel_time = result['routes']['features'][0]['attributes']['Total_TravelTime']
distance = result['routes']['features'][0]['attributes']['Total_Kilometers']

data = [('Number of stops', stop_count),
        ('Total travel time', '{0:.2f} min'.format(travel_time)),
        ('Total travel distance', '{0:.2f} km'.format(distance))]

df = pd.DataFrame.from_records(data)
styles = [    
    dict(selector="td", props=[("padding", "2px")]),
    dict(selector='.row_heading, .blank', props=[('display', 'none;')]),
    dict(selector='.col_heading, .blank', props=[('display', 'none;')])]

symbol = {
    "type": "esriSLS",
    "style": "esriSLSSolid",
    "color": [128,0,128,90],
    "width": 4
}

popup_route = {"title": "Route", 
               "content": df.style.set_table_styles(styles).render()}
popup_stop = {"title": "Stop {}", 
              "content": df.style.set_table_styles(styles).render()}

my_map.draw(result['routes']['features'][0]['geometry'], popup_route, symbol)

for stop in result['stops']['features']:
    address = geocoding.reverse_geocode(stop['geometry'])['address']['Match_addr']
    my_map.draw(stop['geometry'], 
                {"title": "Stop {}".format(stop['attributes']['Sequence']), 
                 "content": address})
my_map.zoom = 12
my_map


# In[ ]:




