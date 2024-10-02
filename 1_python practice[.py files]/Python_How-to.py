#!/usr/bin/env python
# coding: utf-8

# In[13]:


"""All codes written below are regarding HOW-TO. """


mylist = ["a","b","a","c","c"]
mylist = list(dict.fromkeys(mylist))
print(mylist) #removing duplicates.

def myfunc(x):
    return list(dict.fromkeys(x))
mylist = myfunc(["a","b","a","c","c"])
print(mylist) #creating a function.

txt = "Hello World!"[::-1]
print(txt) #reverses the string.

def myfunc(x):
    return x[::-1]
mytxt = myfunc("I wonder how tbis text looks like backwards.")
print(mytxt) #reversing using a function. 

x = 5
y = 10
print(x + y)

x = input("Type your number: ")
y = input("type your number: ")
sum = int(x) + int(y)
print("The sum is: ",sum) #adding numbers using userinput.



# In[ ]:




