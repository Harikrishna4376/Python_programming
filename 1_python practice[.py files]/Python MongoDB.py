#!/usr/bin/env python
# coding: utf-8

# In[5]:


"""All codes written below are regarding PYTHON MONGODB. """
"""All codes written below are regarding GETTING STARTED WITH PYMONGO. """


import pymongo

"""All codes regarding GETTING STARTED WITH PYMONGO are completed here. """


# In[21]:


"""All codes written below are regarding CREATING DATABASE. """


import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["mydatabase"]  #database will not be created untill it gets
                                                    #content.
print(myclient.list_database_names())

import pymongo
myclient = pymongo.MongoClient('mongodb://localhost:27017/')
dblist = myclient.list_database_names()
if "mydatabase" in dblist:
  print("The database exists.")

"""All codes regarding CREATING DATABASE are completed here. """


# In[27]:


import pymongo
myclient = pymongo.MongoClient('mongodb://localhost:27017/')
mydb = myclient["db1"]
mydb.collection.insert_one({"name":"test"})
dblist = myclient.list_database_names()
if "db1" in dblist:
    print("ok, db1 is created. no worries. ")


# In[39]:


"""All codes written below are regarding CREATING COLLECTION. """


import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["db2"]
mycol = mydb["customers"]  #collection will not be created untill content.

print(mydb.list_collection_names())

"""All codes regarding CREATING COLLECTION are completed here. """


# In[9]:


"""All codes written below are regarding INSERTING DOCUMENT. """
#document in mongo is same as record in SQL. 


import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["db2"]
mycol = mydb["customers"]
mydict = {
    "name":"John",
    "address":"Highway 37"
}
x = mycol.insert_one(mydict)
print(x)
print(mydb.list_collection_names())

mydict = {
    "name":"Peter",
    "address":"Lowstreet 27"
}
x = mycol.insert_one(mydict)
print(x.inserted_id)

doc = mydb.mycollection.find_one({"Peter":"Lowstreet 27"})
print(doc)

for x in mycol.find():
    print(x)

import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["db2"]
mycol = mydb["customers"]
mylist = [
    {"name":"Amy","address":"Apple st 652"},
    {"name":"Hannah","address":"Mountain 21"},
    {"name":"Michael","address":"Valley 345"},
    {"name":"Sandy","address":"Ocean blvd 2"},
    {"name":"Betty","address":"Green Grass 1"},
    {"name":"Richard","address":"Sky st 331"},
    {"name":"Susan","address":"One way 98"},
    {"name":"Vicky","address":"Yellow Garden 2"},
    {"name":"Ben","address":"Park Lane 38"},
    {"name":"William","address":"Central st 954"},
    {"name":"Chuck","address":"Main Road 989"},
    {"name":"Viola","address":"Sideway 1633"}
]
mycol.delete_many({})
x = mycol.insert_many(mylist)
print(x.inserted_ids)

import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["db2"]
mycol = mydb["customers"]
mylist = [
  { "_id": 1, "name": "John", "address": "Highway 37"},
  { "_id": 2, "name": "Peter", "address": "Lowstreet 27"},
  { "_id": 3, "name": "Amy", "address": "Apple st 652"},
  { "_id": 4, "name": "Hannah", "address": "Mountain 21"},
  { "_id": 5, "name": "Michael", "address": "Valley 345"},
  { "_id": 6, "name": "Sandy", "address": "Ocean blvd 2"},
  { "_id": 7, "name": "Betty", "address": "Green Grass 1"},
  { "_id": 8, "name": "Richard", "address": "Sky st 331"},
  { "_id": 9, "name": "Susan", "address": "One way 98"},
  { "_id": 10, "name": "Vicky", "address": "Yellow Garden 2"},
  { "_id": 11, "name": "Ben", "address": "Park Lane 38"},
  { "_id": 12, "name": "William", "address": "Central st 954"},
  { "_id": 13, "name": "Chuck", "address": "Main Road 989"},
  { "_id": 14, "name": "Viola", "address": "Sideway 1633"}
]
x = mycol.insert_many(mylist)
print(x.inserted_ids)

"""All codes regarding INSERT are completed here. """


# In[ ]:


import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient("db2")
mycol.create_index("name",unique=True)
try:
    mycol.insert_one({"name":"John","address":"Highway 37"})
    print("Inserted John.")
except pymongo.errors.DuplicateKeyError:
    print("Duplicate entry for John.")


# In[31]:


"""All codes written below are regarding FIND. """


import pymongo
pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["db2"]
mycol = mydb["customers"]
x = mycol.find_one()  #will find only one data. 
print(x)

import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["db2"]
mycol = mydb["customers"]
for x in mycol.find():  #will find all data.
    print(x)

import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["db2"]
mycol = mydb["customers"]
for x in mycol.find({},{"_id": 0,"name": 1,"address": 1}):
    print(x) #It will return only name,addresses not the ids.

import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["db2"]
mycol = mydb["customers"]
for x in mycol.find({},{"address": 0}):  #It will exclude address.
    print(x)

"""All codes regarding FIND are completed here. """


# In[41]:


"""All codes written below are regarding QUERY. """


import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["db2"]
mycol = mydb["customers"]
myquery = {"address": "Park Lane 38"}  #finding particular address data.
mydoc = mycol.find(myquery)
for x in mydoc:
    print(x)

import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["db2"]
mycol = mydb["customers"]
myquery = {"address": {"$gt": "S"}} #Finds data with address starting
mydoc = mycol.find(myquery)         #with "S" or higher.
for x in mydoc:
    print(x)

import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["db2"]
mycol = mydb["customers"]
myquery = {"address":{"$regex":"^S"}} #Finding data having address
mydoc = mycol.find(myquery)           #starting with "S".
for x in mydoc:
    print(x)

"""All codes regarding QUERY are completed here. """


# In[47]:


"""All codes written below are regarding SORT. """


import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["db2"]
mycol = mydb["customers"]
mydoc = mycol.find().sort("name") #will sort the data ascending. 
for x in mydoc:
    print(x)

import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["db2"]
mycol = mydb["customers"]
mydoc = mycol.find().sort("name",-1) #"-1" means inverse sort.
for x in mydoc:
    print(x)

"""All codes regardiing SORT are completed here. """


# In[59]:


"""All codes written below are regarding DELETE DOCUMENT. """


import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["db2"]
mycol = mydb["customers"]
myquery = {"address":"Mountain 21"}
mycol.delete_one(myquery) #It will delete one data. 

import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["db2"]
mycol = mydb["customers"]
myquery = {"address":{"$regex":"^S"}}
x = mycol.delete_many(myquery)
print(x.deleted_count, "documents deleted.")

import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["db2"]
mycol = mydb["customers"]
x = mycol.delete_many({})
print(x.deleted_count, "documents deleted. ")

"""All codes regarding DELETE are completed here. """


# In[75]:


import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["db2"]
mycol = mydb["customers"]
mylist = [
  { "_id": 1, "name": "John", "address": "Highway 37"},
  { "_id": 2, "name": "Peter", "address": "Lowstreet 27"},
  { "_id": 3, "name": "Amy", "address": "Apple st 652"},
  { "_id": 4, "name": "Hannah", "address": "Mountain 21"},
  { "_id": 5, "name": "Michael", "address": "Valley 345"},
  { "_id": 6, "name": "Sandy", "address": "Ocean blvd 2"},
  { "_id": 7, "name": "Betty", "address": "Green Grass 1"},
  { "_id": 8, "name": "Richard", "address": "Sky st 331"},
  { "_id": 9, "name": "Susan", "address": "One way 98"},
  { "_id": 10, "name": "Vicky", "address": "Yellow Garden 2"},
  { "_id": 11, "name": "Ben", "address": "Park Lane 38"},
  { "_id": 12, "name": "William", "address": "Central st 954"},
  { "_id": 13, "name": "Chuck", "address": "Main Road 989"},
  { "_id": 14, "name": "Viola", "address": "Sideway 1633"}
]
mycol.delete_many({})
x = mycol.insert_many(mylist)
print(x.inserted_ids)  #Adding back the data.


# In[79]:


"""All codes written below are regarding DROP COLLECTION. """


import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["db2"]
mycol = mydb["customers"]
#mycol.drop()

"""All codes regarding DROP COLLECTION are completed here. """


# In[89]:


"""All codes written below are regarding UPDATE. """

import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["db2"]
mycol = mydb["customers"]
myquery = {"address":"Valley 345"}
newvalues = {"$set":{"address":"Canyon 123"}}
mycol.update_one(myquery,newvalues)
for x in mycol.find():
    print(x)

import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["db2"]
mycol = mydb["customers"]
myquery = {"address":{"$regex":"^S"}}
newvalues = {"$set":{"name":"Minnie"}}
x = mycol.update_many(myquery,newvalues)
print(x.modified_count,"documents updated. ")

"""All codes regarding UPDATE are completed here. """


# In[99]:


"""All codes written below are regarding LIMIT. """


import pymongo
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["db2"]
mycol = mydb["customers"]
myresult = mycol.find().limit(5)  #Find only 5 data.
for x in myresult:
    print(x)

"""All codes regarding LIMIT are completed here. """
"""All codes regarding PYMONGO are completed here. """


# In[ ]:




