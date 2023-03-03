#!/usr/bin/env python
# coding: utf-8

# In[1]:


from arcgis.gis import GIS
import arcgis.network as network
from arcgis.features import FeatureLayer, Feature, FeatureSet, use_proximity
import pandas as pd
import datetime as dt
import time


# In[2]:


"""
Find the 5, 10, and 15 minute drive-time polygons around all locations of a grocery store chain in a city.
"""

import arcgis
import pandas as pd


def print_result(result):
    """Print useful information from the result."""
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)

    output_polygons = result.service_areas.sdf
    print("\n-- Output Polygons -- \n")
    print(output_polygons[["StoreId", "StoreName", "Address",
                           "FromBreak", "ToBreak"]].to_string(index=False))


def main():
    """Program execution logic."""
    # inputs
    facilities = {
        "displayFieldName": "",
        "fieldAliases": {
            "StoreName": "Store Name",
            "Address": "Address",
            "StoreId": "Store ID"
        },
        "geometryType": "esriGeometryPoint",
        "spatialReference": {
            "wkid": 4326,
            "latestWkid": 4326
        },
        "fields": [
            {
                "name": "StoreName",
                "type": "esriFieldTypeString",
                "alias": "Name",
                "length": 50
            },
            {
                "name": "Address",
                "type": "esriFieldTypeString",
                "alias": "Name",
                "length": 256
            },
            {
                "name": "StoreId",
                "type": "esriFieldTypeString",
                "alias": "Store ID",
                "length": 16
            }
        ],
        "features": [
            {
                "attributes": {
                    "StoreName": "Store 1",
                    "Address": "1775 E Lugonia Ave, Redlands, CA 92374",
                    "StoreId": "120"
                },
                "geometry": {
                    "x": -117.14002999994386,
                    "y": 34.071219999994128
                }
            },
            {
                "attributes": {
                    "StoreName": "Store 2",
                    "Address": "1536 Barton Rd, Redlands, CA 92373",
                    "StoreId": "130"
                },
                "geometry": {
                    "x": -117.207329999671,
                    "y": 34.047980000203609
                }
            },
            {
                "attributes": {
                    "StoreName": "Store 3",
                    "Address": "11 E Colton Ave, Redlands, CA 92374",
                    "StoreId": "121"
                },
                "geometry": {
                    "x": -117.18194000041973,
                    "y": 34.06351999976232
                }
            }
        ]
    }


    # Connect to the Service area service
    api_key = "AAPK584c0e1d93c94b7184b16cab622fc22akbl58IK1U9Xlnhgaqa3L0U4RG0U_Fkb21i2nHHRKngcdkmbcUBML_sGizmYFZpPN"
    arcgis.GIS("https://www.arcgis.com", api_key=api_key)

    # Call the Service Area service
    result = arcgis.network.analysis.generate_service_areas(facilities=facilities,
                                                            break_values="5 10 15",
                                                            travel_direction="Towards Facility")
    print_result(result)
    return result
    
results = main()


# In[3]:


print(results)


# In[4]:


# poly_fset = FeatureSet(main())


# In[5]:


api_key = "AAPK584c0e1d93c94b7184b16cab622fc22akbl58IK1U9Xlnhgaqa3L0U4RG0U_Fkb21i2nHHRKngcdkmbcUBML_sGizmYFZpPN"
#     api_key = "YOUR_API_KEY"
my_gis = arcgis.GIS("https://www.arcgis.com", api_key=api_key)
map1 = my_gis.map('San Bernardino, CA', zoomlevel=12)
map1


# In[6]:


hospital_symbol = {"type":"esriPMS",
                   "url":"http://static.arcgis.com/images/Symbols/SafetyHealth/Hospital.png",
                   "contentType": "image/png", "width":20, "height":20}

map1.draw(results[0],symbol=hospital_symbol)


# In[7]:



# colors1 = {0: [255, 255, 128, 90], 
#           1: [128, 0, 128, 90],
#           2: [0, 128, 0, 90], 
#           3: [255, 255, 0, 90], 
#           4: [255, 0, 0, 90]}

# fill_symbol1 = {"type": "esriSFS","style": "esriSFSSolid",
#                "color": [115,76,0,255],
#                "outline":{"color":[0,0,128,255],"width":0.25,"type":"esriSLS","style":"esriSLSSolid"}}


# In[8]:


colors = {5: [0, 128, 0, 90], 
          10: [255, 255, 0, 90], 
          15: [255, 0, 0, 90]}

fill_symbol = {"type": "esriSFS","style": "esriSFSSolid",
               "color": [115,76,0,255],
               "outline":{"color":[0,0,128,255],"width":0.25,"type":"esriSLS","style":"esriSLSSolid"}}


# In[9]:


map1.clear_graphics()

for service_area in results.service_areas.features:
    
    #set color based on drive time
    fill_symbol['color'] = colors[service_area.attributes['ToBreak']]
    
    #set popup
    popup={"title": "Service area", 
           "content": "{} minutes".format(service_area.attributes['ToBreak'])}
    
    #draw service area
    map1.draw(service_area.geometry, symbol=fill_symbol, popup=popup, 
              attributes={"title": service_area.attributes['Name']})

map1.draw(results[0], symbol=hospital_symbol)


# In[10]:


# !curl https://route-api.arcgis.com/arcgis/rest/services/World/ServiceAreas/NAServer/ServiceArea_World/solveServiceArea? \
# -d "f=json" \
# -d "token=<AAPK584c0e1d93c94b7184b16cab622fc22akbl58IK1U9Xlnhgaqa3L0U4RG0U_Fkb21i2nHHRKngcdkmbcUBML_sGizmYFZpPN>" \
# -d "facilities=-117.133163,34.022445" \
# -d "defaultBreaks=2.5" \
# -d "travelDirection=esriNATravelDirectionToFacility" \
# -d "travelMode={'attributeParameterValues':[{'attributeName':'Avoid Private Roads','parameterName':'Restriction Usage','value':'AVOID_MEDIUM'},{'attributeName':'Walking','parameterName':'Restriction Usage','value':'PROHIBITED'},{'attributeName':'Preferred for Pedestrians','parameterName':'Restriction Usage','value':'PREFER_LOW'},{'attributeName':'WalkTime','parameterName':'Walking Speed (km/h)','value':5},{'attributeName':'Avoid Roads Unsuitable for Pedestrians','parameterName':'Restriction Usage','value':'AVOID_HIGH'}],'description':'Follows paths and roads that allow pedestrian traffic and finds solutions that optimize travel distance.','distanceAttributeName':'Kilometers','id':'yFuMFwIYblqKEefX','impedanceAttributeName':'Kilometers','name':'Walking Distance','restrictionAttributeNames':['Avoid Private Roads','Avoid Roads Unsuitable for Pedestrians','Preferred for Pedestrians','Walking'],'simplificationTolerance':2,'simplificationToleranceUnits':'esriMeters','timeAttributeName':'WalkTime','type':'WALK','useHierarchy':false,'uturnAtJunctions':'esriNFSBAllowBacktrack'}"


# In[11]:


list_of_breaks = [5,10,15]
# list_of_breaks = [5,10,15, 20,25,30,35,40,45]
if isinstance(list_of_breaks, list):
    string_of_breaks = ' '.join(map(str, list_of_breaks))
    print(string_of_breaks)


# In[12]:


results[0]


# In[13]:


# %%time

# import datetime

# current_time = dt.datetime.now() 

result1 = network.analysis.generate_service_areas(facilities=results[3], break_values=string_of_breaks, 
                                                  break_units="Minutes")


# In[14]:


result1


# In[15]:


pd.DataFrame(result1[3])


# In[16]:



# cols = ['FromBreak', 'ToBreak', 'COUNTY_CODE', 'COUNTY_NAME', 'DBA_ADDRESS1', 'DBA_CITY',  \
#         'DBA_ZIP_CODE', 'FACILITY_LEVEL_DESC', 'FACILITY_NAME', 'FACILITY_STATUS_DATE', 'FACILITY_STATUS_DESC', \
#         'LICENSE_CATEGORY_DESC', 'LICENSE_NUM', 'LICENSE_TYPE_DESC', 'Name', \
#         'OSHPD_ID', 'TOTAL_NUMBER_BEDS']


# In[17]:


# df = result1.service_areas.sdf[cols]
# df.sort_values('FromBreak', inplace=True, ascending=True)
# df.head()


# In[18]:


map3 = my_gis.map('SAN BERNARDINO, CA', zoomlevel=12)
map3


# In[19]:


colors = {5: [0, 128, 0, 90], 
          10: [255, 255, 0, 90], 
          15: [255, 0, 0, 90]}

fill_symbol = {"type": "esriSFS","style": "esriSFSSolid",
               "color": [115,76,0,255],
               "outline":{"color":[0,0,128,255],"width":0.25,"type":"esriSLS","style":"esriSLSSolid"}}


# In[20]:


map3.clear_graphics()

for service_area in result1.service_areas.features:
    
    #set color based on drive time
    fill_symbol['color'] = colors[service_area.attributes['ToBreak']]
    
    #set popup
    popup={"title": "Service area", 
           "content": "{} minutes".format(service_area.attributes['ToBreak'])}
    
    #draw service area
    map3.draw(service_area.geometry, symbol=fill_symbol, popup=popup, 
              attributes={"title": service_area.attributes['Name']})

map3.draw(result1[0], symbol=hospital_symbol)
map3


# In[21]:


map4 = my_gis.map('SAN BERNARDINO, CA', zoomlevel=12)
map4


# In[22]:


colors = {5: [0, 128, 0, 90], 
          10: [255, 255, 0, 90], 
          15: [255, 0, 0, 90]}

fill_symbol = {"type": "esriSFS","style": "esriSFSSolid",
               "color": [115,76,0,255],
               "outline":{"color":[0,0,128,255],"width":0.25,"type":"esriSLS","style":"esriSLSSolid"}}

hospital_symbol = {"type":"esriPMS",
                   "url":"http://static.arcgis.com/images/Symbols/SafetyHealth/Hospital.png",
                   "contentType": "image/png", "width":20, "height":20}


# In[23]:


map4.clear_graphics()

for service_area in result1.service_areas.features:
    
    #set color based on drive time
    fill_symbol['color'] = colors[service_area.attributes['ToBreak']]
    
    #set popup
    popup={"title": "Service area", 
           "content": "{} minutes".format(service_area.attributes['ToBreak'])}
    
    #draw service area
    map4.draw(service_area.geometry, symbol=fill_symbol, popup=popup, 
              attributes={"title": service_area.attributes['Name']})

map4.draw(results[0], symbol=hospital_symbol)
map4


# In[24]:


list_of_breaks = [5,10,15]
# list_of_breaks = [5,10,15, 20,25,30,35,40,45]
if isinstance(list_of_breaks, list):
    string_of_breaks = ' '.join(map(str, list_of_breaks))
    print(string_of_breaks)


# In[25]:


map5 = my_gis.map('SAN BERNARDINO, CA', zoomlevel=12)
map5


# In[26]:


colors = {5: [0, 128, 0, 90], 
          10: [255, 255, 0, 90], 
          15: [255, 0, 0, 90]}

fill_symbol = {"type": "esriSFS","style": "esriSFSSolid",
               "color": [115,76,0,255],
               "outline":{"color":[0,0,128,255],"width":0.25,"type":"esriSLS","style":"esriSLSSolid"}}


# In[27]:


map5.clear_graphics()

for service_area in result1.service_areas.features:
    
    #set color based on drive time
    fill_symbol['color'] = colors[service_area.attributes['ToBreak']]
    
    #set popup
    popup={"title": "Service area", 
           "content": "{} minutes".format(service_area.attributes['ToBreak'])}
    
    #draw service area
    map5.draw(service_area.geometry, symbol=fill_symbol, popup=popup, 
              attributes={"title": service_area.attributes['Name']})

map5.draw(result1[0], symbol=hospital_symbol)


# In[ ]:




