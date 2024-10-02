#!/usr/bin/env python
# coding: utf-8

# In[12]:


"""All codes written below are regarding FILE OPEN. """


f = open("demofile.txt")
f = open("demofile.txt","rt")

"""All codes regarding FILE OPEN are completed here. """


# In[46]:


"""All codes written below are regarding FILE OPEN. """

import os
f = open("demofile.txt","r")
print(f.read())

f = open("C:\\Users\\harikrishna\\anaconda3\\1_python practice\\demofile.txt", "r")
print(f.read())

f = open("demofile.txt","r")
print(f.read())

f = open("demofile.txt","r")
print(f.readline()) #It will only display the 1st line. 

f = open("demofile.txt","r")
print(f.readline())
print(f.readline())

f = open("demofile.txt","r")
for x in f:
    print(x)

f = open("demofile.txt","r")
print(f.readline())
f.close()

"""All codes regarding FILE OPEN are completed here. """


# In[52]:


"""All codes written below are regarding FILE WRITE. """


f = open("demofile2.txt","a")  #This will automatically create the .txt file. 
f.write("Now the file has more content!")
f.close()
f = open("demofile2.txt","r")
print(f.read())

f = open("demofile3.txt","a") #This will automatically create the .txt file. 
f.write("Woops!, I have deleted the content! ")
f.close()
f = open("demofile3.txt","r")
print(f.read())

#f = open("myfile.txt","x") It will automatically create the .txt file. 
#f = open("myfile2.txt","w") It will automatically create the .txt file. 

"""All codes regarding READ/WRITE FILE are completed here. """


# In[56]:


"""All codes written below are regarding DELETE FILE. """


import os
#os.remove(demofile.txt) it will automatically remove 'demofile.txt'. it will give error once the file is executed more than once. 

import os
"""if os.path.exists("demofile.txt"):
    os.remove("demofile.txt")
else:
    print("The file does not exists. ") """

import os
#os.rmdir("myfolder") it will automatically delete the folder. 


# In[ ]:




