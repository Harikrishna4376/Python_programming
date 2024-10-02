#!/usr/bin/env python
# coding: utf-8

# In[31]:


"""All codes written below are regarding PYTHON MYSQL. """
"""All codes written below are regarding GETTING STARTED WITH MYSQL. """


import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "HariANDMahantji31@1022992398"
)
print(mydb)

"""All codes regarding GETTING STARTED with python mysql are completed. """


# In[67]:


"""All codes written below are regarding MYSQL CREATE DATABASE. """


import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    username = "root",
    password = "HariANDMahantji31@1022992398"
)
mycursor = mydb.cursor()
#mycursor.execute("CREATE DATABASE mydatabase")  #CREATING a database. 

import mysql.connector
"""mydb = mysql.connector.connect(
    host = "localhost",
    user = "root", 
    password = "HariANDMahantji31@1022992398"
)"""
#no ned to provide the details unless the new connection is created.  
mycursor = mydb.cursor()
mycursor.execute("SHOW DATABASES")
for x in mycursor:
    print(x)

import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "HariANDMahantji31@1022992398",
    database = "mydatabase"
)

"""All codes regarding CREATING DATABASE are completed here. """


# In[109]:


"""All codes written below are regarding CREATING TABLE. """


import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "HariANDMahantji31@1022992398",
    database = "mydatabase"
)
mycursor = mydb.cursor()  #below line created a table. 
#mycursor.execute("CREATE TABLE customers (name VARCHAR(255),address VARCHAR(255))")

mycursor.execute("SHOW TABLES") #Checking whether the table exists or not. 
for x in mycursor:
    print(x)

import mysql.connector
mycursor = mydb.cursor()
#mycursor.execute("CREATE TABLE customers (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255),address VARCHAR(255))")
#above line is commented because table customers already exists and creates error. 

#mycursor.execute("ALTER TABLE customers ADD COLUMN id INT AUTO_INCREMENT PRIMARY KEY")
mycursor.execute("SHOW TABLES")
for x in mycursor:
    print(x)

tablename = "customers"
mycursor.execute(f"DESCRIBE {tablename}")
print(f"\nStructure of table '{tablename}':")
for column in mycursor:
    print(column)


"""All codes regarding CREATE TABLE are completed here. """


# In[163]:


"""All codes written below are regarding INSERT INTO TABLE. """


import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "HariANDMahantji31@1022992398",
    database = "mydatabase"
)
mycursor = mydb.cursor()
sql = "INSERT INTO customers (name,address) VALUES (%s,%s)"
val = ("John","Highway 21")
mycursor.execute(sql,val)
mydb.commit()
print(mycursor.rowcount,"record inserted. ")

import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "HariANDMahantji31@1022992398",
    database = "mydatabase"
)
mycursor = mydb.cursor()
sql = "INSERT IGNORE INTO customers (name,address) VALUES( %s,%s)"
val = [
    ('Peter','Lowstreet 4'),
    ('Amy','Apple st 652'),
    ('Hannah','Mountain 21'),
    ('Michael','Valley 345'),
    ('Sandy','Ocean blvd 2'),
    ('Betty','Green Grass 1'),
    ('Richard','Sky st 331'),
    ('Susan','One way 98'),
    ('Vicky','Yellow Garden 2'),
    ('Ben','Park Lane 38'),
    ('William','Central st 954'),
    ('Chuck','Main Road 989'),
    ('Viola','Sideway 1633')
]
mycursor.executemany(sql,val)
mydb.commit()
print(mycursor.rowcount, " was inserted. ")  #Entering values in database. 

import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "HariANDMahantji31@1022992398",
    database = "mydatabase"
)
mycursor = mydb.cursor()
sql = "INSERT INTO customers (name,address) VALUES(%s,%s)"
val = ("Michael","Blue Village")
mycursor.execute(sql,val)
mydb.commit()
print("1 record instead, ID: ",mycursor.lastrowid)

import mysql.connector
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="HariANDMahantji31@1022992398",
    database="mydatabase"
)
mycursor = mydb.cursor()
mycursor.execute("""
DELETE t1
FROM customers t1
JOIN customers t2 
ON t1.name = t2.name 
AND t1.address = t2.address
AND t1.id > t2.id;
""")
mydb.commit()
print(mycursor.rowcount, "duplicate rows were deleted.")

"""All codes regarding INSERTING INTO TABLE are completed here. """


# In[187]:


"""All codes written below are regarding SELECT FROM. """


import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "HariANDMahantji31@1022992398",
    database = "mydatabase"
)
mycursor = mydb.cursor()
mycursor.execute("SELECT * FROM customers")
myresult = mycursor.fetchall()
for x in myresult:
    print(x)

import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "HariANDMahantji31@1022992398",
    database = "mydatabase"
)
mycursor = mydb.cursor()
mycursor.execute("SELECT name, address FROM customers")
myresult = mycursor.fetchall()
for x in myresult:
    print(x)  #It will only print name and address columns. 

import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "HariANDMahantji31@1022992398",
    database = "mydatabase"
)
mycursor = mydb.cursor()
mycursor.execute("SELECT * FROM customers")
myresult = mycursor.fetchone()
print(myresult)

"""All codes regarding SELECT FROM are completed here. """


# In[199]:


"""All codes written below are regarding MYSQL WHERE. """


import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "HariANDMahantji31@1022992398",
    database = "mydatabase"
)
mycursor = mydb.cursor()
sql = "SELECT * FROM customers WHERE address = 'Park Lane 38'"
mycursor.execute(sql)
myresult = mycursor.fetchall()
for x in myresult:
    print(x)

import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "HariANDMahantji31@1022992398",
    database = "mydatabase"
)
mycursor = mydb.cursor()
sql = "SELECT * FROM customers WHERE address LIKE '%way%'"
mycursor.execute(sql)
myresult = mycursor.fetchall()
for x in myresult:
    print(x)

import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "HariANDMahantji31@1022992398",
    database = "mydatabase"
)
mycursor = mydb.cursor()
sql = "SELECT * FROM customers WHERE address = %s"
adr = ("Yellow Garden 2",)
mycursor.execute(sql,adr)
myresult = mycursor.fetchall()
for x in myresult:
    print(x)

"""All codes regarding MYSQL WHERE are completed here. """


# In[207]:


"""All codes written below are regarding MYSQL ORDER BY. """


import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "HariANDMahantji31@1022992398",
    database = "mydatabase"
)
mycursor = mydb.cursor()
sql = "SELECT * FROM customers ORDER BY name"
mycursor.execute(sql)
myresult = mycursor.fetchall()
for x in myresult:
    print(x)

import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "HariANDMahantji31@1022992398",
    database = "mydatabase"
)
mycursor = mydb.cursor()
sql = "SELECT * FROM customers ORDER BY name DESC"
mycursor.execute(sql)
myresult = mycursor.fetchall()
for x in myresult:
    print(x)

"""All codes regarding MYSQL ORDER BY are completed here. """


# In[225]:


"""All codes written below are regarding MYSQL DELETE FROM BY. """


import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "HariANDMahantji31@1022992398",
    database = "mydatabase"
)
mycursor = mydb.cursor()
sql = "DELETE FROM customers WHERE address = 'Mountain 21'"
mycursor.execute(sql)
mydb.commit()
print(mycursor.rowcount,"record(s) deleted")

import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "HariANDMahantji31@1022992398",
    database = "mydatabase"
)
mycursor = mydb.cursor()
sql = "DELETE FROM customers WHERE address = %s"
adr = ("Yellow Garden 2",)
mycursor.execute(sql,adr)
mydb.commit()
print(mycursor.rowcount,"record(s) deleted")

"""All codes regarding DELETE FROM BY are completed here. """


# In[235]:


"""All codes written below are rgarding DROP TABLE. """


import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "HariANDMahantji31@1022992398",
    database = "mydatabase"
)
mycursor = mydb.cursor()
sql = "DROP TABLE customers" #it will delete table. 
#mycursor.execute(sql)

sql = "DROP TABLE IF EXISTS customers"
mycursor.execute(sql) #The command is executed if no error occurs. 

"""All codes regarding DROP TABLE are completed here. """


# In[379]:


"""All codes written below are regarding UPDATE TABLE. """

import mysql.connector
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="HariANDMahantji31@1022992398",
    database="mydatabase"
)
mycursor = mydb.cursor()
try:
    mycursor.execute("CREATE TABLE IF NOT EXISTS t1 (name VARCHAR(255), address VARCHAR(255), UNIQUE (name, address))")
    print("Table 't1' is ready.")
except mysql.connector.Error as err:
    print(f"Error: {err}")
sql = "INSERT IGNORE INTO t1 (name, address) VALUES (%s, %s)"
val = [
    ('Peter', 'Lowstreet 4'),
    ('Amy', 'Apple st 652'),
    ('Hannah', 'Mountain 21'),
    ('Michael', 'Valley 345'),
    ('Sandy', 'Ocean blvd 2'),
    ('Betty', 'Green Grass 1'),
    ('Richard', 'Sky st 331'),
    ('Susan', 'One way 98'),
    ('Vicky', 'Yellow Garden 2'),
    ('Ben', 'Park Lane 38'),
    ('William', 'Central st 954'),
    ('Chuck', 'Main Road 989'),
    ('Viola', 'Sideway 1633')
]
try:
    mycursor.executemany(sql, val)
    mydb.commit()
    print(mycursor.rowcount, "records inserted (or ignored if duplicate).")
except mysql.connector.Error as err:
    print(f"Error: {err}")
mycursor.close()
mydb.close()

import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "HariANDMahantji31@1022992398",
    database = "mydatabase"
)
mycursor = mydb.cursor()
sql = "INSERT INTO t1 (name,address) VALUES (%s,%s) ON DUPLICATE KEY UPDATE address = VALUES(address)"
val = [
    ('Michael','Canyon 123')
]
mycursor.executemany(sql,val)
mydb.commit()
print(mycursor.rowcount,"record(s) affected")
#mycursor.close()
#mydb.close()

sql = "SELECT * FROM t1 ORDER BY name"
mycursor.execute(sql)
result = mycursor.fetchall()
for x in result:
    print(x)

sql = """
DELETE t1
FROM t1
JOIN (
    SELECT MIN(id) as min_id, name, address
    FROM t1
    GROUP BY name, address
    HAVING COUNT(*) > 1
) dup ON t1.name = dup.name AND t1.address = dup.address AND t1.id != dup.min_id;
"""
mycursor.execute(sql)
mydb.commit()
print(mycursor.rowcount,"duplicate rows deleted. ")
sql = "SELECT * FROM t1 ORDER BY name"
mycursor.execute(sql)
result1 = mycursor.fetchall()
for x in result1:
    print(x)
mycursor.close()
mydb.close()

import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "HariANDMahantji31@1022992398",
    database = "mydatabase"
)
mycursor = mydb.cursor()
sql = "UPDATE t1 SET name = %s,address = %s WHERE id = %s AND (name != %s OR address != %s);"
val = ('Canyon 123','Valley 345',1,'Valley 345','Canyon 123')
mycursor.execute(sql,val)
mydb.commit()
print(mycursor.rowcount,"record(s) updated (if different),")
#mycursor.close()
#mydb.close()

sql = "SELECT * FROM t1 ORDER BY name"
mycursor.execute(sql)
result = mycursor.fetchall()
for x in result:
    print(x)
mycursor.close()
mydb.close()

"""All regarding UPDATE TABLE are completed here. """


# In[397]:


"""All codes written below are regarding LIMIT. """


import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "HariANDMahantji31@1022992398",
    database = "mydatabase"
)
mycursor = mydb.cursor()
sql = "SELECT * FROM t1 LIMIT 5 "
mycursor.execute(sql)
result = mycursor.fetchall()
for s in result:
    print(x)

mycursor.execute("SELECT * FROM t1 LIMIT 5 OFFSET 2")
result = mycursor.fetchall()
for x in result:
    print(x)

"""All codes regarding LIMIT in MYSQL are completed here. """


# In[445]:


"""All codes written below are regarding MYSQL JOIN. """


import mysql.connector
mydb = mysql.connector.connect(
    host = "localhost",
    user = "root",
    password = "HariANDMahantji31@1022992398",
    database = "mydatabase"
)
mycursor = mydb.cursor()
sql1 = "CREATE TABLE IF NOT EXISTS n1 (id INT PRIMARY KEY,name VARCHAR(255),fav INT)"
sql2 = "CREATE TABLE IF NOT EXISTS n2 (id INT PRIMARY KEY,name VARCHAR(255))"
mycursor.execute(sql1) #Creating TABLE n1.
mycursor.execute(sql2) #Creating TABLE n2.
mydb.commit()

sql_n1 = "INSERT IGNORE INTO n1 (id,name,fav) VALUES (%s,%s,%s)"
val_n1 = [
    (1, 'John', 154),
    (2, 'Peter', 154),
    (3, 'Amy', 155),
    (4, 'Hannah', None),
    (5, 'Michael', None)
]
sql_n2 = "INSERT IGNORE INTO n2 (id,name) VALUES (%s,%s)"
val_n2 = [
     (154, 'Chocolate Heaven'),
    (155, 'Tasty Lemons'),
    (156, 'Vanilla Dreams')
]
mycursor.executemany(sql_n1,val_n1)
mycursor.executemany(sql_n2,val_n2)
mydb.commit()

sql = "SELECT * FROM n1"
mycursor.execute(sql)
result = mycursor.fetchall()
for x in result:
    print(x)
sql = "SELECT * FROM n2"
mycursor.execute(sql)
result = mycursor.fetchall()
for x in result:
    print(x)

sql = "SELECT \
  n1.name AS user, \
  n2.name AS favorite \
  FROM n1 \
  INNER JOIN n2 ON n1.fav = n2.id"
mycursor.execute(sql)  #IT USES INNER JOIN.
result = mycursor.fetchall()
for x in result:
    print(x)

sql = "SELECT \
  n1.name AS user, \
  n2.name AS favorite \
  FROM n1 \
  LEFT JOIN n2 ON n1.fav = n2.id"
mycursor.execute(sql)  #THIS IS LEFT JOIN.
result = mycursor.fetchall()
for x in result:
    print(x)

sql = "SELECT \
  n1.name AS user, \
  n2.name AS favorite \
  FROM n1 \
  RIGHT JOIN n2 ON n1.fav = n2.id"
mycursor.execute(sql)
result = mycursor.fetchall()
for x in result:
    print(x)

"""All codes regarding MYSQL JOIN are completed here. """

"""All codes regarding PYTHON MYSQL are completed here. """


# In[ ]:




