#!/usr/bin/env python
# coding: utf-8

# In[1]:


import arcgis
from arcgis.gis import GIS
import pandas as pd
import datetime
import getpass
from IPython.display import HTML

from arcgis import geocoding
from arcgis.features import Feature, FeatureSet
from arcgis.features import GeoAccessor, GeoSeriesAccessor


# In[2]:


api_key = "AAPKb924aa22a2c04a43948522166a50233aV7220htxh7Kf9oyEw8YQhKMjEE5pfZ5caL_2ApPUClsueGMtRw4eqirTDtPK0ofP"
#     api_key = "YOUR_API_KEY"
my_gis = arcgis.GIS("https://www.arcgis.com", api_key=api_key)

# api_key = "AAPK6815d252f60c4322aea51e58fd28be5ayfl7xP7lV_4qTMtEDBjaz_eSSWNtiIJJixWgXZBHSSROjs6Q883scxme54jS2T5Y"
# #     api_key = "YOUR_API_KEY"
# my_gis = arcgis.GIS("https://www.arcgis.com", api_key=api_key)


# In[3]:


# origin_coords = ['-117.187807, 33.939479', '-117.117401, 34.029346']
# origin_features = []

# for origin in origin_coords:
#     print(origin)
# #     reverse_geocode = geocoding.reverse_geocode({"x": origin.split(',')[0], 
# #                                               "y": origin.split(',')[1]}) 

#     reverse_geocode = geocoding.reverse_geocode({"x": coord[0]['x'], 
#                                              "y": coord[0]['y']})
#     print(reverse_geocode)
#     print('\n')

#     origin_feature = Feature(geometry=reverse_geocode['location'], 
#                            attributes=reverse_geocode['address'])
#     origin_features.append(origin_feature)
    
# print(origin_features)
# print('\n')

# origin_fset = FeatureSet(origin_features, geometry_type='esriGeometryPoint',
#                           spatial_reference={'latestWkid': 4326})
# print(origin_fset)


# In[4]:


addresses_item = my_gis.content.search('destinations_address', 'feature layer')[0]
addresses_item


# In[5]:



destinations_sdf = addresses_item.layers[0].query(as_df=True)
destinations_sdf


# In[6]:


destinations_sdf.columns


# In[7]:


len(destinations_sdf)


# In[8]:


dest = pd.DataFrame(destinations_sdf,columns=['OBJECTID','SHAPE'])
dest


# In[9]:


origin_coords = ['-117.187807, 33.939479', '-117.117401, 34.029346']
origin_features = []

Lat = destinations_sdf['Latitude'][0:2]
Long = destinations_sdf['Longitude'][0:2]

for i in range(len(Lat)):
    reverse_geocode = geocoding.reverse_geocode({"x": Long[i], 
                                             "y": Lat[i]})
    origin_feature = Feature(geometry=reverse_geocode['location'], 
                           attributes=reverse_geocode['address'])
    origin_features.append(origin_feature)
    
origin_fset = FeatureSet(origin_features, geometry_type='esriGeometryPoint',
                          spatial_reference={'latestWkid': 4326})
print(origin_fset)   

# for origin in origin_coords:
#     print(origin)
# #     reverse_geocode = geocoding.reverse_geocode({"x": origin.split(',')[0], 
# #                                               "y": origin.split(',')[1]}) 

#     reverse_geocode = geocoding.reverse_geocode({"x": coord[0]['x'], 
#                                              "y": coord[0]['y']})
#     print(reverse_geocode)
#     print('\n')

#     origin_feature = Feature(geometry=reverse_geocode['location'], 
#                            attributes=reverse_geocode['address'])
#     origin_features.append(origin_feature)
    
# print(origin_features)
# print('\n')

# origin_fset = FeatureSet(origin_features, geometry_type='esriGeometryPoint',
#                           spatial_reference={'latestWkid': 4326})
# print(origin_fset)


# In[10]:


# destinations_sdf['Latitude'][0]
# destinations_sdf['Longitude'][0]


# In[11]:


rev_geocode = []
origin_features = []

# destinations_sdf['Latitude'][0]
# destinations_sdf['Longitude'][0]

for i in range(len(destinations_sdf)):
    reverse_geocode = geocoding.reverse_geocode({"x": destinations_sdf['Longitude'][0], 
                                             "y": destinations_sdf['Latitude'][0]})
    rev_geocode.append(reverse_geocode['address']['Match_addr'])
    
    origin_feature = Feature(geometry=reverse_geocode['location'], 
                           attributes=reverse_geocode['address'])
    origin_features.append(origin_feature)

print(origin_features)
print('\n')

origin_fset = FeatureSet(origin_features, geometry_type='esriGeometryPoint',
                          spatial_reference={'latestWkid': 4326})
    
origin_fset


# In[12]:


dest.insert(0,'ADDRESS',rev_geocode)
dest


# In[13]:


destinations_sdf = dest.iloc[0:3,:]
destinations_sdf


# In[14]:


# destinations_sdf_2 = dest.iloc[3:6,:]
# destinations_sdf_2 

# origin_fset = FeatureSet(destinations_sdf_2 , geometry_type='esriGeometryPoint',
#                           spatial_reference={'latestWkid': 4326})

# origin_fset


# In[15]:


destinations_fset = destinations_sdf.spatial.to_featureset()
print(destinations_fset)


# In[16]:


get_ipython().run_cell_magic('time', '', "# solve OD cost matrix tool for the origns and destinations\nfrom arcgis.network.analysis import generate_origin_destination_cost_matrix\nresults = generate_origin_destination_cost_matrix(origins= origin_fset, #origins_fc_latlong, \n                                                destinations= destinations_fset, #destinations_fs_address,\n                                                origin_destination_line_shape='Straight Line')\nprint('Analysis succeeded? {}'.format(results.solve_succeeded))")


# In[17]:


od_df = results.output_origin_destination_lines.sdf
od_df


# In[18]:



# filter only the required columns
od_df2 = od_df[['DestinationOID','OriginOID','Total_Distance','Total_Time']]

# user pivot_table
od_pivot = od_df2.pivot_table(index='OriginOID', columns='DestinationOID')
od_pivot


# In[19]:


od_pivot.to_csv('OD_Matrix.csv')


# In[20]:


od_map = my_gis.map('New Hampshire')
od_map


# In[21]:


od_map.draw(results.output_origin_destination_lines)
od_map.draw(destinations_fset, symbol={"type": "esriSMS","style": "esriSMSSquare","color": [255,115,0,255], "size": 10})
od_map.draw(origin_fset, symbol={"type": "esriSMS","style": "esriSMSCircle","color": [76,115,0,255],"size": 8})


# In[ ]:




