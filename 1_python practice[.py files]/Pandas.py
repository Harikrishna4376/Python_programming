#!/usr/bin/env python
# coding: utf-8

# In[6]:


"""All codes written below are regarding Pandas library. """
"""All codes written below are regarding GETTING STARTED with PANDAS. """


import pandas as pd
mydata = {
    'cars':["BMW","VOLVO","FORD"],
    'passings':[3,7,2]
}
myvar = pd.DataFrame(mydata)  #Converts data into DATAFRAME. 
print(myvar)

print(pd.__version__)  #Prints current downloaded version. 

"""All codes regarding GETTING STARTED with pandas are completed here. """


# In[28]:


"""All codes written below are regarding SERIES. """


import pandas as pd
a = [1,7,2]
myvar = pd.Series(a)
print(myvar)  #Prints series of the data. 

print(myvar[0]) #Displays elements according to index numbers. 

a = [1,7,2]
myvar = pd.Series(a,index=["x","y","z"])  #Changes the indexes from 0,1,2 to x,y,z.
print(myvar)

print(myvar["y"])

calories = {"day1":420,"day2":380,"day3":390}
myvar = pd.Series(calories)
print(myvar)

myvar = pd.Series(calories,index=["day1","day2"])
print(myvar)

data = {
    "calories":[420,380,390],
    "duration":[50,40,45]
}
myvar = pd.DataFrame(data)
print(myvar)

"""All codes regarding SERIES are completed here. """


# In[64]:


"""All codes written below are regarding DATAFRAMES. """


import pandas as pd
data = {
    "calories":[420,380,390],
    "duration":[50,40,45]
}
df = pd.DataFrame(data)  #Converts data into DATAFRAME. 
print(df)

print(df.loc[0])  #Returns ROW 0. 

print(df.loc[[0,1]])  #Returns ROW 0 and 1. 

import pandas as pd
data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}
df = pd.DataFrame(data, index = ["day1", "day2", "day3"])
print(df)

print(df.loc["day2"])

df = pd.read_csv('demofile.txt')
print(df)
""".csv or .txt both type of file can pandas read. """

"""All codes regarding DATAFRAMES are completed here. """


# In[80]:


"""All codes written below are regarding READ CSV files. """


import pandas as pd
#df = pd.read_csv('data.csv')
#print(df.to_string())

df = pd.read_csv('data.csv')
print(df)

print(pd.options.display.max_rows) #Displays max rows. 

pd.options.display.max_rows = 9999
df = pd.read_csv('data.csv')
print(df)

"""All codes regarding READ CSV files are completed here. """


# In[130]:


"""All codes written below are regarding READ JSON. """


import pandas as pd
df = pd.read_json('data1.json')
print(df)
print(df.to_string())

import pandas as pd
data = {
  "Duration":{
    "0":60,
    "1":60,
    "2":60,
    "3":45,
    "4":45,
    "5":60
  },
  "Pulse":{
    "0":110,
    "1":117,
    "2":103,
    "3":109,
    "4":117,
    "5":102
  },
  "Maxpulse":{
    "0":130,
    "1":145,
    "2":135,
    "3":175,
    "4":148,
    "5":127
  },
  "Calories":{
    "0":409,
    "1":479,
    "2":340,
    "3":282,
    "4":406,
    "5":300
  }
}
df1 = pd.DataFrame(data)
print(df1)

"""All codes regarding JSON files are completed here. """


# In[132]:


"""All codes written below are regarding ANALYSING DATAFRAMES. """


import pandas as pd

df = pd.read_csv('data.csv')

print(df.head(10))


print(df.tail()) 


print(df.info()) 


"""All codes regarding are completed here. """


# In[113]:


import json
import requests

# Step 1: Fetch the JavaScript data
url = "https://www.w3schools.com/python/pandas/data.js"
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Step 2: Extract the JavaScript content
    js_content = response.text

    # Step 3: Parse the JavaScript content to extract the JSON object
    start_index = js_content.find("{")
    end_index = js_content.rfind("}") + 1
    json_str = js_content[start_index:end_index]

    # Step 4: Convert the JSON string into a Python dictionary
    data = json.loads(json_str)

    # Step 5: Write the dictionary to a JSON file
    with open("data1.json", "w") as json_file:
        json.dump(data, json_file, indent=4)

    print("Data has been successfully written to data1.json")
else:
    print(f"Failed to fetch data. Status code: {response.status_code}")


# In[2]:


import pandas as pd

# Data provided
data = {
    'Duration': [60, 60, 60, 45, 45, 60, 60, 450, 30, 60, 60, 60, 60, 60, 60, 60, 60, 60, 45, 60, 45, 60, 45, 60, 45, 60, 60, 60, 60, 60, 60, 60],
    'Date': ['2020/12/01', '2020/12/02', '2020/12/03', '2020/12/04', '2020/12/05', '2020/12/06', '2020/12/07', '2020/12/08', '2020/12/09', '2020/12/10', 
             '2020/12/11', '2020/12/12', '2020/12/12', '2020/12/13', '2020/12/14', '2020/12/15', '2020/12/16', '2020/12/17', '2020/12/18', '2020/12/19',
             '2020/12/20', '2020/12/21', None, '2020/12/23', '2020/12/24', '2020/12/25', '2020/12/26', '2020/12/27', '2020/12/28', '2020/12/29', 
             '2020/12/30', '2020/12/31'],
    'Pulse': [110, 117, 103, 109, 117, 102, 110, 104, 109, 98, 103, 100, 100, 106, 104, 98, 98, 100, 90, 103, 97, 108, 100, 130, 105, 102, 100, 92, 103, 100, 102, 92],
    'Maxpulse': [130, 145, 135, 175, 148, 127, 136, 134, 133, 124, 147, 120, 120, 128, 132, 123, 120, 120, 112, 123, 125, 131, 119, 101, 132, 126, 120, 118, 132, 132, 129, 115],
    'Calories': [409.1, 479.0, 340.0, 282.4, 406.0, 300.0, 374.0, 253.3, 195.1, 269.0, 329.3, 250.7, 250.7, 345.3, 379.3, 275.0, 215.2, 300.0, None, 323.0, 243.0, 364.2, 282.0, 300.0, 246.0, 334.5, 250.0, 241.0, None, 280.0, 380.3, 243.0]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Convert DataFrame to CSV
df.to_csv('data2.csv', index=False)

# Verification message
print("CSV file 'data2.csv' has been created successfully.")


# In[22]:


"""All codes written below are regarding CLEANING DATA. """



import pandas as pd
df = pd.read_csv('data2.csv')
new_df = df.dropna()
print(new_df.to_string())

import pandas as pd
df = pd.read_csv('data2.csv')
df.dropna(inplace=True)
print(df.to_string())

df = pd.read_csv('data2.csv')
df.fillna(130,inplace=True)  #Fills the null value with 130.
print(df)

df["Calories"].fillna(130,inplace=True)

x = df["Calories"].mean()
df["Calories"].fillna(x,inplace=True)
print(df)

x = df["Calories"].median()
df["Calories"].fillna(x,inplace=True)
print(df)

x = df["Calories"].mode()[0]
df["Calories"].fillna(x,inplace=True)
print(df)

"""All codes regarding CLEANING EMPTY CELLS are completed here. """


# In[28]:


"""All codes written below are regarding CLEANING DATA OF WRONG FORMAT. """


import pandas as pd
df = pd.read_csv('data2.csv')
df["Date"] = pd.to_datetime(df['Date'])  #It converts to DATE. 
print(df.to_string())

df.dropna(subset=['Date'],inplace=True)
print(df)


"""All codes regarding CLEANING DATA OF WRONG FORMAT are completed here. """


# In[44]:


"""All codes written below are regarding FIXING WRONG DATA. """


df = pd.read_csv('data2.csv')
df.loc[7,'Duration']=45  #set 'Duration'=45 in row 7.
print(df)

for x in df.index:
    if df.loc[x,"Duration"]>120:
        df.loc[x,"Duration"]=120

for x in df.index:
    if df.loc[x,"Duration"]>120:
        df.drop(x,inplace=True)

print(df.to_string())


"""All codes regarding FIXING WRONG DATA are completed here. """


# In[48]:


"""All codes written below are regarding REMOVING DUPLICATES. """


print(df.duplicated())

df.drop_duplicates(inplace=True)

print(df.to_string())

"""All codes regarding REMOVING DUPLICATES are completed here. """


# In[58]:


"""All codes written below are regarding DATA CORRELATIONS. """


df = pd.read_csv('data3.csv')
df.corr()
print(df)

"""All codes regarding DATA CORRELATIONS are completed here. """


# In[70]:


"""All codes written below are regarding PLOTTING. """


import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('data3.csv')
df.plot()
plt.show()

df.plot(kind='scatter',x='Duration',y='Calories')
plt.show()

df.plot(kind='scatter',x = 'Duration',y = 'Maxpulse')
plt.show()

df["Duration"].plot(kind='hist')
plt.show()

"""All codes regarding PLOTTING are completed here. """
"""All codes regarding PANDAS LIBRARY are completed here. """


# In[ ]:




