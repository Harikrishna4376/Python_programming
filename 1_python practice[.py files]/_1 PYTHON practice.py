#!/usr/bin/env python
# coding: utf-8

# In[5]:


print("hello world!")


# In[1]:


if 5>2:     # this is an example to check whether the described condition is true or not...
    print("Five is greater then two!")


# In[5]:


#this is comment.
print("hello world!")


# In[6]:


"""This is comment
and only this is comment..
"""
print("hello world!")


# In[7]:


x = 5
y = "john"
print(x)
print(y)


# In[13]:


x = 4          # this code is an example of displaying datatypes of written input
y = "john"
print(type(x))
print(type(y))


# In[21]:


def myfunc():       # this code is written for topic local and global variables
    global x
    x = "fantastic"

myfunc()

print("python is " + x)


# In[5]:


import random      # this code is an example of printing random number..
print(random.randrange(1,10))


# In[62]:


my_str = "hello world!"      # this is an example of calculatng the length of the string character..
print(len(my_str))


# In[74]:


mystr = 'Hello world!'    # this is to print the string from backwards..
print(mystr[-5:-2])


# In[2]:


mystr = "hello world!"   #this is an example of converting test into uppercase...
print(mystr.upper())


# In[17]:


a = " hello world! "     # this is to split the string into more than single string..
print(a.split())


# In[19]:


mystr = "hello world!"    # this is to replace characters(one or more than one character)
print(mystr.replace('hello' , 'jay swaminarayan'))


# In[24]:


a = "Hello"    # this is to give space between "hello" and "world".
b = "World!"
c = a + " " + b 
print(c)


# In[31]:


books = 10         # this code is an example to add external data into added text.
pencils = 20     
erasers = 15
sharpners = 8
txt = " Hello , my name is harry and i need {0}books , {1}pencils , {2}erasers ,and {3}sharpners."
print(txt.format(books,pencils,erasers,sharpners))


# In[52]:


my_str = "Hello my name is harry.\nI m 19 years old. "      #this is an example to display text in below line.
print(my_str)


# In[78]:


#this are the basic methods that python offers to play with.

print("Basic methods that python offers to play with...   :-  ")

txt = "hello world!"          #method to capitalize first letter.
print(txt.capitalize())

txt = "HELLO WORLD!"          #method to convert entire sentence into lower case.
print(txt.casefold())

txt = "Hello world!"          #method to bring sentence into center.
centered_txt = txt.center(20)
print(centered_txt)

letters = ['a','b','c','a','e','f','a']   #method to count number of entities of data.
count_of_as = letters.count('a')
print("Count of a's :", count_of_as)

txt = "Hello world!"          #method to encode entered data into byte size.
encoded_bytes = txt.encode('utf-8')
print("Encoded Bytes: ", encoded_bytes)

txt = "Hello world!"      #This method is a example of displaying ends with command
if txt.endswith("world!"):
    print("The text ends with 'world!'")
else:
    print("The test does not ends with 'world!'")
if txt.endswith("world"):
    print("The text ends with 'world'")
else:
    print("The text does not ends with 'world'")

txt = "Hello\tworld!"    #this method is an example of expanding text.
print(txt.expandtabs())

txt = "Hello world!"      #this method is an example to find the location of any text if it exixts.
index = txt.find("world!")
print(index)
index = txt.find("paradise!")
print(index)

name = "Harry"    # this is method to display anpther text into different sentence.
age = 21
my_str = "Hello , my name is {0} and i m {1} years old.".format(name,age)
print(my_str)

person = {"Name": "Harry" , "age" : 21}     # Another one.  
my_str = "Hello , my name is {Name} and I am {age} years old.".format_map(person)
print(my_str)

txt1 = "Hello1234"     # method to check whether the text is alpha numeric or not.
txt2 = "Hello 1234"
txt3 = " "
print(txt1.isalnum())
print(txt2.isalnum())
print(txt3.isalnum())

txt = "Hello world!"    # Another one.(for checking alphabets)
print(txt.isalpha())

txt = "Hello world!"    #Another one.(for checking ascii)
print(txt.isascii())

txt = "Hello world!"    #Another one.(for checking decimal)
print(txt.isdecimal())

txt = "Hello world1!"    #Another one.(for checking digit)
print(txt.isdigit())

print("Hello".isidentifier()) #check for identifier (true).
print("!ellO".isidentifier())  #check for identifier (false).

print("hello world!".islower())  #checks if each letter is in lower case.(true)
print("HELLO WORLD!".islower())  #same as above..(false)

print("1234llo".isnumeric())    #checks if each letter is numeric or not..
print("12300000".isnumeric())    #same as above(true).

print("harry".isprintable())    # checks whether the text is printable or not..

print("  ".isspace())   #checks if entire text is whitespace or not..

print("Harry".istitle())   # checks if the text is matching the rules of title (true)
print("hello".istitle())   # checks whether the text matches the rules of title or not..(false)

print("HELLO WORLD!".isupper())   # checking all letters are in upper case..

my_str = ["Lemon" , "Apple" , "Dragonfruit"]   #example to join (,) into the my_str variable..
result = " , ".join(my_str)
print(result)

text = "    HELLO WORLD!    "    # Method to remove excess space before text...
stripped_text = text.lstrip()
print(stripped_text)

text = "Hello world!"   # used to replace word with another word..
print(text.replace("Hello" , "world"))

text = "HHHello world!"   # used to find the number of string characters that exists in the text.
print(text.rfind("H"))

text = "Hello everyone!"  # used to locate the index at which specific letter is placed.
print(text.rindex('e'))

text = "Harry is good boy!"   # used to split each word in sentance.
print(text.split())


print("Hello harry " , " how r u?".splitlines())  # used to splitlines..

print("Yello is sun.".startswith('Y'))   # checks whether the statement starts with the defined character or not..(true)

print(" tata interprise.pvt.ltd".title())  # converts first letter of every word into upper case..


# In[144]:


print(10>9)   # this is to check whether the condition is true or not...
print(10==9)
print(10<9)

a = 10        
b = 20
if(a>b):
    print("The entered condition is true.")
else:
    print("The entered condition is incorrect.")

x = "" 
y = "world!"
print(bool(x))
print(bool(y))

x = 12   # example of every arithmetic operator in python..
y = 9
print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(x%y)
print(x**y)
print(x//y)

x = 4           # checks whether both condition stated in single argument is true or not.
if(x < 5 and x > 22):
    print("The condition is completely true.")
else:
    print("Condition is incorrect.")

x = 10         #checks whether both data set are same or not.
y = 10 
print(x is y)

x = ["apple" , "banana"]     # stated whether the word is stated in x dataset.
print("banana" in x)
print("apple" not in x)


# In[43]:


Mylist = ['Apple' , 'Banana' , 'kiwi']     #this is to show that list is createdin [big braces]..
print(Mylist)

Mylist = ['Apple' , 'Mango' , 'Kiwi' , 'Dragon fruit' , 'Orange' , 'Lemon']   # calculates number or items in the list..
print(len(Mylist))

Mylist = ['Apple' , 'Banana' , 'kiwi']            # method to display the datatype of the list...
print(type(Mylist))

Mylist = ['Apple' , 'Banana' , 'kiwi']    #example to display last item in the list..
print(Mylist[-3])

Mylist = ['Apple' , 'Banana' , 'kiwi' , 'Dragon fruit' , 'Papaya' , 'Orange']   #example to display specific items in the list.
print(Mylist[2:4])

Mylist = ['Apple' , 'Banana' , 'kiwi' , 'Dragon fruit' , 'Papaya' , 'Orange']   
print(Mylist[-6:-4])                        #example to print particular item in the list ignoring specific items..

Mylist = ['Apple' , 'Banana' , 'kiwi']
Mylist[2] = "Blueberry"     #example to swap item with another item..
print(Mylist)

Mylist = ['Apple' , 'Banana' , 'kiwi' , 'Dragon fruit' , 'Papaya' , 'Orange']   
Mylist[1:4] = ["Blueberry" , "Mango" , "Grapes"]   #example to change range of items in list..
print(Mylist)

Mylist = ['Apple' , 'Banana' , 'kiwi' , 'Dragon fruit' , 'Papaya' , 'Orange']   
Mylist[1:2] = ["Blueberry" , "Grapes"]      #example to change one item with multiple items.
print(Mylist)

Mylist = ['Apple' , 'Banana' , 'kiwi' , 'Dragon fruit' , 'Papaya' , 'Orange']   
Mylist[1:4] = ["Blueberry"]   #example to change multiple items with one..
print(Mylist)

Mylist = ['Apple' , 'Banana' , 'kiwi' , 'Dragon fruit' , 'Papaya' , 'Orange']   
Mylist.insert(2 , "Watermelon")      #example to insert an item into list..
print(Mylist)

Mylist = ["Item1" , "Item2" , "Item3"]
Mylist.append("Orange")    #append() method to add item at the end of the list..
print(Mylist)

Mylist = ["Item1" , "Item2" , "Item3"]
Mylist.insert(0 , "Orange")        #insert() to add an item into the list..
print(Mylist)

Mylist = ["Books" , "Pencil" , "Erasers"]
Extra = ["Item1" , "Item2" , "Item3"]   
Mylist.extend(Extra)             #extend() to extend list my another items..
print(Mylist)

Mylist = ["Books" , "Pencil" , "Erasers"]
Mylist.remove("Pencil")         #remove() command to remove particular item from list..
print(Mylist)

Mylist = ["Books" , "Pencil" , "Erasers"]
Mylist.pop(0)                    #pop(0) command to remove specified index.
print(Mylist)

Mylist = ["Books" , "Pencil" , "Erasers"]
Mylist.pop()                      #pop() command to remove last item from list..
print(Mylist)

Mylist = ["Books" , "Pencil" , "Erasers"]
del Mylist[0]                    #del() command to remove specified item from list..
print(Mylist)

Mylist = ["Books" , "Pencil" , "Erasers"]
Mylist.clear()                       # clear() command to clear entire list...
print(Mylist)



# In[11]:


Mylist = ["Books" , "Pencil" , "Erasers"]
for x in Mylist:              #  loops through the list using for loop...
    print(x)

Mylist = ["Super glue" , "Fevigum" , "Pencils"]
for i in range(len(Mylist)):                # Another one.
    print(Mylist[i])

Mylist = ["Item1" , "Item2" , "Item3"]
i = 0                                     # display all the items in list using while loop..
while i< len(Mylist):
    print(Mylist[i])
    i = i + 1

Mylist = ["Books" , "Pencil" , "Erasers"]
print(*Mylist, sep = "\n")              # for loop directly used in print function....



# In[12]:


# this is the code to display the list of items containing letter "a".

fruits = ["Dragon friut" , "Blueberry" , "Orange" , "Raspberry" , "Apple" , "Kiwi"]
newlist = []
for i in fruits:
    if "a" in i:
        newlist.append(i)
print(newlist)

# Same concept as above..
fruits = ["Dragon friut" , "Blueberry" , "Orange" , "Raspberry" , "Apple" , "Kiwi"]
new_list = [x for x in fruits if "e" in x]
print(new_list)

        #   display item only if it is not apple..
newlist = [x for x in fruits if x != "Apple"]
print(newlist)

#this is the code to sort the tiems in the list..
thislist = ["Orange" , "Kiwi" , "Banana" , "Blueberry"]
thislist.sort()
print(thislist)

#another one..
Thislist = [100 , 40 , 45 , 23 , 33 , 21 , 90 , 12 , 0 , 10]
Thislist.sort()
print(Thislist)

#this is the code to sort items in reverse manner(descending)
Mylist = [100 , 32 , 43 , 45 , 32 , 11 , 89 , 76 , 45]
Mylist.sort(reverse = True)
print(Mylist)

#Another one..
Mylist = ["Dragon fruit" , "Blueberry" , "Blackberry" , "Kiwi" , "Cocao fruit"]
Mylist.sort(reverse = True)
print(Mylist)

#this is code to customize the sort function.
#this code is to sort the items that are closer to number 50..
def myfunc(n):
    return abs(n - 50)
mylist = [100 , 32 , 54 , 40 , 51 , 56 , 55]
mylist.sort(key = myfunc)
print(mylist)

#this code is to reverse the items in the list.
mylist = ["Banana" , "Kiwi" , "Dragon fruit" , "Blackberry" , "Orange"]
mylist.reverse()
print(mylist)

#make a copy of list using copy() method.
mylist = ["apple" , "Banana" , "cherry" , "dragon fruit"]
thislist = mylist.copy()
print(mylist)

#make a copy of list using list() method.
mylist = ["Banana" , "Kiwi" , "Dragon fruit" , "Blackberry" , "Orange"]
newlist = list(mylist)
print(newlist)

#this code is to join two declared lists.
list1 = ["Cherry" , "Cocao"]
list2 = ["dragon fruit" , "Blueberry fruit"]
list3 = list1 + list2
print(list3)

#this is to append list into another list..
list1 = ["a" , "c" , "b"]
list2 = ["f" , "g" , "h"]
for x in list2:
    list1.append(x)
print(list1)

#this is to extend list2 at the end of list1.
list1 = ["a" , "c" , "b"]
list2 = ["f" , "g" , "h"]
list1.extend(list2)
print(list1)

#the below are the built in methods for the list..
"""append()
clear()
copy()
count()
extend()
index()
insert()
pop()
remove()
reverse()
sort()
"""


# In[11]:


# All the below codes are defined for tuples only.

thistuple = ("Apple" , "Banana" , "cherry")   # Tuples allow duplicates as well..
print(thistuple)

Mytuple = ("Apple",)    # Inorder to create tuple with only one item , one must have to write comma.
print(type(Mytuple))

Mytuple = ("Apple")        #if comma is not added into the data structure will accept it as a string..
print(type(Mytuple))

tuple1 = ("abc" , 123 , True , 3456 , "Male")        #Tuple can contain multiple different data types..
print(tuple1)

thistuple = tuple(("Aprilia" , "Bike" , "Ducati"))    #Tuple constructor can also be created..
print(thistuple)

thistuple = ("Apple" , "Banana" , "cherry")
print(thistuple[1])                                   #Tuple banana is printed because python uses 0 indexing..

thistuple = ("Apple" , "Banana" , "cherry")
print(thistuple[-1])                                     # tuple also accepts -ve indexing.

thistuple = ("Apple" , "Banana" , "cherry")
print(thistuple[2:3])                               # indexing defining starting and ending index.

x = ("Apple" , "Banana" , "Cherry" , "Rubber" , "Pencil" , "Ducati")
y = list(x)
y[0] = "Kiwi"                              # changing an item name in tuple..
x = tuple(y)
print(x)

x = ("Apple" , "Banana" , "Cherry" , "Rubber" , "Pencil" , "Ducati")
y = list(x)                                # changing tuple into list.
print(y)                                
y.append("Orange")                         # Appending(adding) an item in list.
y= tuple(y)                               #converting the list back into tuple.
print(y)

x = ("Apple" , "Banana" , "Cherry" , "Pencil" , "Ducati")
y = ("Books",)
x += y                                     #create another tuple and then adding that tuple into the main tuple.
print(x)

x = ("Apple" , "Banana" , "Cherry" , "Ducati")
y = list(x)                      #convert tuple into list.
y.remove("Banana")                # remove item named banana.
x = tuple(y)                           #converting list back into tuple.
print(x)

x = ("Apple" , "Banana" , "Cherry" , "Rubber" , "Pencil" , "Ducati")
del x                          # del command will delete entire tuple completely.
#print(x)  this will create error in the output because the tuple is deleted and there is nothing to print.

thistuple = ("Apple" , "Banana" , "Cherry" , "Rubber" , "Pencil" , "Ducati")
for x in thistuple:
    print(x)                    #for loop can also be used to print tuple..

x = ("Apple" , "Banana" , "Cherry" , "Rubber" , "Pencil" , "Books")
for i in range(len(x)):
    print(len(x[i]))                 #this code displays the length of the items in the tuple..

Mytuple = ("Armada" , "Banana" , "Raspberry")
i = 0
while i < len(Mytuple):
    print(len(Mytuple[i]))
    i = i + 1                    #using while loop for tuple..

tuple1 = ("1" , "2" , "3")
tuple2 = ("4" , "5")
tuple3 = tuple1 + tuple2
print(tuple3)                     #joining two tuples together..

fruits = ("Apple" , "Cherry" , "Kiwi")
newtuple = fruits * 2
print(newtuple)                       #multiply the tuple by 2.

 #there are two main tuple methods:-
"""count()
index()"""


# In[23]:


#all the codes below are of set data type:-

Myset = {"Maths" , "Physics" , "Chemistry"}
print(Myset)          #bacics of creating and printing set.

thisset = {"Maths" , "Physics" , "Chemistry" , "Maths"}
print(thisset)                 #set does not allow duplicates.

thisset = {"Maths" , "Physics" , "Chemistry" , True , 1 , 2 , False , 0 , -1}
print(thisset)                   #[True and 1] and [False and 0] are considered tobe the same value in set. 

thisset = {"Maths" , "Physics" , "Chemistry" , "Biology"}
print(len(thisset))                     #length of the set.

setA = {"Maths" , "Physics" , "Chemistry" , "Biology"}
setB = {1,3,5,7,9}
setC = {True , False , True}
print(setA , setB , setC)

Myset = {"apple" , "banana" , "cherry"}
for x in Myset:
 print(x)                       #print using for loop.

Thisset = {"Orange" , "Kiwi" , "Melon"}
Thisset.add("Dragon fruit")
print(Thisset)                    #add() to add any item into the set.

thisset = {"honda" , "bmw" , "aprilia"}
tropical = {"harley davidson" , "ducati" , "suzuki"}
thisset.update(tropical)         #update() command used to merge or add tropical into thisset.
print(thisset)               

tropical = {"harley davidson" , "ducati" , "suzuki"}
tropical.remove("harley davidson")           #remove() commd used to remove any item from set..
print(tropical)                               #if the item written in remove does not exist in the set then it will create error

tropical = {"harley davidson" , "ducati" , "suzuki"}
tropical.discard("harley davidson")      # one can aso use discard instead of remove().
print(tropical)         #how ever discard will not create any error if the item written in the discard doesn't exist in set.

thisset = {"Kiwi" , "dragon fruit" , "Watermelon"}
thisset.pop()       #pop() command will delete random item in the set.
print(thisset)

Myset = {"Abc" , "DEF" , "XYZ"}
Myset.clear()
print()            #clear() cmd will clear entire set . and thats why it won't print anything in output

set1 = {"harley davidson" , "ducati" , "suzuki"}
set2 = {"Bmw" , "honda" , "aprilia"}
set3 = set1.union(set2)            #union will join both sets. [join() cmd will not work]
print(set3)

set1 = {"abc" , "def" , "xyz"}
set2 = {"xyz" , "abc"}
set3 = set1.intersection(set2)             #print the items that exists both in set1 ,2
print(set3)

set1 = {"def" , "abc" , "xyz"}
set2 = {"def" , "apple" , "kiwi"}
set3 = set1.symmetric_difference(set2)          #prints the items that are not common.
print(set3)
"""the codes for set are completed."""


# In[14]:


"""all the below examples are for dictionary."""

thisdict = {                       #basic to create a dictionary.
    "brand" : "Ford" ,
    "model" : "mustang" ,
    "year" : 1964
} 
print(thisdict)

thisdict = {
    "brand" : "ford" ,
    "Model" : "mustang" ,     #particularly created to print(brand)
    "year" : 1964
}
print(thisdict["brand"])

thisdic = {
    "brand" : "ford" ,
    "model" : "Mustang" ,           #used to check the length of the dictionary.
    "year"  : 1964 , 
}
print(len(thisdict))

thisdict = {
    "brand" : "peter england" , 
    "year" : 2014 ,                   #dictionary can be of any data type.
    "colours" : ["blue" , "red" , "Purple"]    
}
print(thisdict)

Mydict = {
    "name" : "Harry" , 
    "age" : 24 ,                #checking the datatype entered into the string..
    "hobby" : "gaming"
}
print(type(Mydict["age"]))


thisdict = dict(name = "Harry" , age = 24 , country = "india" )
print(thisdict)             #creating constructor.
x = thisdict.get("name")     #get() can also be used to get any item in the dictionary
x = thisdict.keys
print(x)

car = {
    "brand" : "Mercedes-Benz" ,
    "model" : "Maybach gls 600" ,
    "year" : 2022
}
x = car.keys()                   #used to get the list of keys.
print(x)
car["color"] = "Ice white"
print(x)

car = {
    "Brand" : "Bmw" ,
    "Model" : "Alpina B8" ,
    "year" : 2012
}
x = car.values()           #used to change the values of any item in the dictionary
print(x)
car["year"] = 2004
print(x)

car = {
    "Brand" : "Porsche" ,
    "Model" : "Porsche 911 GT3 RS" ,
    "year"  : 2023
}
if "model" in car:            #used to check whether Model exixts in the dictionary
    print("Yes , model is present as an entity in the dictionary")
else:
    print("Invalid text")

car = {
    "Brand" : "Audi" , 
    "Model" : "RS etron GT" , 
    "year"  : 2010
}
car.update({"year" : 2020})             #changing year of the model.
print(car)

car = {
    "Brand" : "Mercedes-Benz" , 
    "Model" : "Maybach gls 600" , 
    "Year"  : 2022
}
car.update({"colour" : "Two tone Red"}) #used to add any data into the dictionary
print(car)

Mydict ={
    "Name" : "Harikrishna"  ,
    "age" : 24 , 
    "favourite colour" : "Blue"
}
Mydict.pop("age")            #pop() used to remove any item in the dictionary.
print(Mydict)

thisdict ={
    "Item1" : ["apple" , "Mango" , "Kiwi"] , 
    "item2" : ["Pencil" , "Eraser" , "Books"] ,
    "item3" : ["Banana" , "Water melon"]
}

thisdict.clear()         #this will clear entire dict and hence will print "{}"
print(thisdict)

car = {
    "Brand" : "Mercedes-Benz" , 
    "Model" : "Maybach gls 600" , 
    "Year"  : 2022
}
car = dict(car)          #making a copy of dict..
print(car)

Myfamily = {
    "Child1" : {
        "name" : "Emily" , 
        "age"  : 12
    } , 
    "child2" : {
        "name" : "Williams" ,      #3 dictionaries inside one dictionary..
        "age"  : 14
    } ,
    "child3" : {
        "name" : "Henry" , 
        "age"  : 13
    }
}
print(Myfamily)

child1 = {
    "name" : "Henry" , 
    "age"  : 11
} ,
child2 = {
    "name" : "Sanheiser" , 
    "age"  : 10
} ,
child3 = {
    "name" : "Williams" , 
    "age" : 13
}
Myfamily = {                  #first created 3 dictionaries and then defined all dictionaries into one.
    "child1" : child1 ,
    "child2" : child2 , 
    "child3" : child3
}
print(Myfamily)

"""All of the exercises of dictionaries are over.."""


# In[36]:


"""Now the loops exercises start from the below"""

a = 10
b = 22
if b > a:                         #basic looping.
    print("B is greater than a")
else:
    print("B is smaller than a")

a = 1001
b = 120
if b > a:
    print("Yes, b is greater than a")
elif a == b:                             #multiple looping..
    print("Yes , a and b are equal")
else:
    print("a is greater than b")

a = 120
b = 222      
print("a") if a > b else print("b")       #one line loop.

a = 12
b = 20
if a > b:
    pass            #this will pass the statement .

i = 1
while i <= 5:
    print(i)
    i += 1            #increment command/

i = 1
while i < 6:
    print(i)
    if i == 4:
        break            #this will break the loop as soon as i=3.
    i += 1

i = 0
while i < 8:
    i += 1
    if i <6:
        print(i)
    elif i > 5:        
        print(i)

fruits = ["Apple" , "Banana" , "Cherry"]
for x in fruits:
    print(x)
    if x == "Banana":
        break

fruits = ["Apple" , "Banana" , "Cherry"]
for x in fruits:
    if x == "Banana":
        break
    print(x)

for x in range(6):
    print(x)

for x in range(2,30,3):       #3 at last is used here to display next 3rd number.
    print(x)

for x in range(6):
    if x == 3:
        break
        print(x)
    else:
        print("invalid text!")

"""All examples related to loops are completed here."""


# In[2]:


"""All the below codes are related to functions:-"""
def myfunc():               #myfunc(): is created.
    print("Hello world")
myfunc()                    #without calling the function output will print 0

def myfunc(fname):
    print(fname + " Stanford")   #this is used to attach a surname.
myfunc("Emily")
myfunc("William")
myfunc("Stella")

def my_function(fname , lname):
    print(fname + " " + lname)  #double columns are used to create space..
my_function("Emily" , "reeves")

def myfunc(*kids):
        print("The youngest child is " + kids[0])     #[0] will print 1st name in the function.
myfunc("aidin" , "bob" , "henry")

def myfunc(*, x):
    print(x)
myfunc(x = 3)

def myfunc(x):
    print(x)
myfunc(3)

def myfunc(a,b,/,*,c,d):     #/ is for positional datatypes only and after * are keywords
    print(a+b+c+d)
myfunc(1,1,c=1,d=1)
"""basically anything that is before / , value can be entered without describing
the position.
while if * is written then we have to describe position of that datatype"""

def myfunc(k):
    if(k > 0):
        result = k + myfunc(k-1)
        print(result)
    else:
        result = 0
    return result
print("\n The recursion example result is :- \n")
myfunc(6)
"""All the codes for function and resursion are completed."""


# In[24]:


"""All the codes starting below are of lambda."""

x = lambda a : a + 10    #lambda is a function .
print(x(5))

x = lambda a,b,c : (a+b)/c
print(x(5,6,7))

def func(n):
    return lambda a : a * n
mydoubler = func(2)          #mydoubler is just a func name not syntax.
print(mydoubler(22))

def function(k):
    return lambda a : a * k
mydoubler = function(2)         #a = 11 and k = 2,3 .
mytripler = function(3)
print(mydoubler(11))
print(mytripler(11))
"""Codes for all the lambda function are completed.."""


# In[49]:


"""All codes for arrays are shown below."""

car1 = "Ford"
car2 = "Toyota"
car3 = "Mercedes"
print(car1,car2,car3)

cars = ["Ford" , "Toyota" , "Nissan"]
print(cars[0])      #ford is printed because python uses 0 indexing.
print(len(cars))         #len() to check the langth of the array..

car = ["Volvo" , "Mercedes" , "Bmw"]
for c in car:
    print(c)
car.append(["Audi" , "Toyota"])     #append() is used to join an item in array.
print(car)
car.pop(3)             #pop() used here to delete 4thh element in array.
print(car)
car.insert(4 , "Audi")
print(car)
car.remove("Volvo")
print(car)
car.clear()
print(car)
car.extend(["Nissan" , "Mitsubishi"])   #extend() adds items at the end.
print(car)
car.index("Nissan")  #shows the position of the item..
print(car)
car.reverse()
print(car)
car.sort()
print(car)
"""All the code for arrays are completed above."""


# In[75]:


"""All codes describes below are of classes/objects ."""

class person:
    def __init__(self,name,age):
        self.name = name
        self.age = age
p1 = person("John",33)    #p1.person() is used to add name and age in class.
print(p1.name)
print(p1.age)

class person:
    def __init__(self,name,age):
        self.name = name
        self.age = age
p1 = person("John" , 33)
print(p1)      #without entering obj with str in class will not print data.

class person:
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def myfunc(self):
        print("Hello , my name is " + self.name + " , i m " + str(self.age) + " Years old.")
p1 = person("John wick" , 35)
p1.myfunc()       #function is used to make things easy..

class person:
    def __init__(obj,name,age):
        obj.name = name
        obj.age = age
    def myfunc(obj):
        print("Hello,my name is " + obj.name + " and i m " + str(obj.age) + " yrs old.")
p1 = person("john wick" , 36)
p1.myfunc()
p1.age = 46         #update the age.
p1.myfunc()        #if i use "del p1.age" it will create error as age is deleted.

"""All code example for python classesare completed here."""


# In[7]:


# """All codes starting from below are of inheritance regarding classes nd objects."""

class person:
    def __init__(self,fname,lname):
        self.firstname = fname
        self.lastname = lname
    def printname(self):
        print(self.firstname,self.lastname)
x = person("John" , "Wick")         #data entered using inheritance..
x.printname()

class pname:
    def __init__(self,fname,lname,pERNO):
        self.firstname = fname
        self.lastname = lname
        self.personERNO = pERNO
    def printdetails(self):
        print(self.firstname,self.lastname,self.personERNO)
x = pname("Tobey" , "Willson" , 210340131006)
x.printdetails()
x = pname("Mike" , "Olsen" , 210340131026)
x.printdetails()

class person:
    def __init__(self,fname,lname):
        self.firstname = fname
        self.lastname = lname
    def printname(self):
        print(self.firstname,self.lastname)
class student(person):
    def __init__(self,fname,lname):
        person.__init__(self,fname,lname)        #inheritance continued.
x = person("Nick" , "Collins")
x.printname()

class person:
    def __init__(self,fname,lname):
        self.firstname = fname
        self.lastname = lname
    def printname(self):
        print(self.firstname,self.lastname)
class student(person):
    def __init__(self,fname,lname):
        super().__init__(fname,lname)
        self.graduationyear = 2019
x = student("Denzel" , "Washington")
x.graduationyear

class person:
    def __init__(self,fname,lname):
        self.firstname = fname
        self.lastname = lname
    def printname(self):
        print(self.firstname,self.lastname)
class student(person):
    def __init__(self,fname,lname):
        super().__init__(fname,lname)
        self.graduationyear = 2022
x = student("Morgan" , "Freeman")
print(x.graduationyear)
x.printname()
"""All codes for inheritance are completed above."""


# In[2]:


"""All codes written below are regarding Iterator. """

mytuple = ("apple","banana","cherry")
myit = iter(mytuple)
print(next(myit))
print(next(myit))
print(next(myit))

mystr = "Banana"
myit = iter(mystr)
print(next(myit))
print(next(myit))
print(next(myit))
print(next(myit))
print(next(myit))
print(next(myit))

mytuple = ("apple","banana","cherry")
for x in mytuple:
    print(x)

mystr = "Banana"
for x in mystr:
    print(x)

class Mynumbers:
    def __iter__(self):
        self.a = 1
        return self
    def __next__(self):
        x = self.a
        self.a += 1
        return x
myclass = Mynumbers()
myiter = iter(myclass)
print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))
print(next(myiter))

class mynumbers:
    def __iter__(self):
        self.a = 1
        return self
    def __next__(self):
        if self.a <= 20:
            x = self.a
            self.a += 1
            return x
        else:
            raise StopIteration
myclass = mynumbers()
myiter = iter(myclass)
for x in myiter:
    print(x)

"""All codes regarding ITERATORS are completed here. """


# In[14]:


"""All codes written below are regarding POLYMORPHISM. """

x = "Hello World!"
print(len(x))

mytuple = ("apple","banana","cherry")
print(len(mytuple))

thisdict = {
    "brand":"Ford",
    "model":"Mustang",
    "year":1964
}
print(len(thisdict))

class car:
    def __init__(self,brand,model):
        self.brand = brand
        self.model = model
    def move(self):
        print("Drive!")
class Boat:
    def __init__(self,brand,model):
        self.brand = brand
        self.model = model
    def move(self):
        print("Sail!")
class Plane:
    def __init__(self,brand,model):
        self.brand = brand
        self.model = model
    def move(self):
        print("Fly!")
car1 = car("Ford","Mustang")
boat1 = Boat("Ibiza","Touring 20")
plane1 = Plane("Boeing","747")
for x in (car1,boat1,plane1):
    x.move()

class vehicle:
    def __init__(self,brand,model):
        self.brand = brand
        self.model = model
    def move(self):
        print("Move!")
class car(vehicle):
    pass
class boat(vehicle):
    def move(self):
        print("Sail!")
class plane(vehicle):
    def move(self):
        print("Fly!")
car1 = car("Ford","Mustang")
boat1 = boat("Ibise","Touring 20")
plane1 = plane("Boeing","747")
for x in (car1,boat1,plane1):
    print(x.brand)
    print(x.model)
    x.move()

"""All codes regarding POLYMORPHISM are completed here. """


# In[66]:


"""All codes written below are regarding SCOPE. """


def myfunc():
    x = 300
    print(x)
myfunc()

def myfunc():
    x = 300
    def myinnerfunc():
        print(x)
    myinnerfunc()
myfunc()

def myfunc():
    x = 100
    def myinnerfunc():
        print(x)
    myinnerfunc()
myfunc()

x = 780
def func():
    print(x)
myfunc()
print(x)

x = 10
def func():
    print(x)
myfunc()
print(x)

x = 1080
def func():
    x = 20
    print(x)
myfunc()
print(x)

def myfunc():
    global x
    x = 300
myfunc()
print(x)

x = 300
def myfunc():
    global x
    x = 200
myfunc()
print(x)

def myfunc1():
    x = "Jane"
    def myfunc2():
        nonlocal x
        x = "Hello!"
    myfunc2()
    return x
print(myfunc1())

"""All codes regarding SCOPE are completed here. """


# In[100]:


"""All codes written below are regarding MODULES. """

def greeting(name):
    print("Hello: " + name)  #Save this code in a file with ".py" extension. 

"""import mymodule
mymodule.greeting("John")""" #This code will give error because the file is not saved. 

"""person1 = {
    "name":"John",
    "Age":"36",
    "Country":"Norway"
}"""  #This will give error as well as the module is not saved. 

import platform
x = platform.system()
print(x)

import platform as p
x = dir(p)
print(x)

def greeting(name):
    print("Hello: " + name)
person1 = {
    "name":"John",
    "age":"36",
    "country":"Norway"
}

"""from mymodule import person1
print(person1['age'])"""  #It will cause error because the module is not saved. 

"""All codes regarding MODULES are completed here. """


# In[180]:


"""All codes written below are regarding PYTHON DATETIME. """

import datetime
x = datetime.datetime.now() #It will force the code to print the current "datetime".
print(x)

import datetime
x = datetime.datetime.now()
print(x.year)
print(x.strftime("%A"))  #"%A" will print YEAR and WEEKTIME of current. 

import datetime
x = datetime.datetime(2020,5,17) #It will print the listed datetime. 
print(x)

import datetime
x = datetime.datetime(2018,6,1)
print(x.strftime("%B"))

import datetime
x = datetime.datetime.now()
print(x)
print(x.strftime("%a")) #Prints WEEKDAYS in SHORT.
print(x.strftime("%A")) #Prints WEEKDAYS in FULL. 
print(x.strftime("%W")) #Prints WEEKNUMBER. 
print(x.strftime("%d")) #Prints DAY of the month. 
print(x.strftime("%B")) #Prints MONTH of the YEAR. 
print(x.strftime("%m")) #Prints MONTH of the year in NUMBER. 
print(x.strftime("%y")) #Prints the current YEAR in SHORT. 
print(x.strftime("%Y")) #Prints the current YEAR in FULL. 
print(x.strftime("%H")) #Prints HOUR in 24 HOUR format. 
print(x.strftime("%I")) #Prints HOUR in 12 HOUR format. 
print(x.strftime("%p")) #Prints AM or PM (before midday or after midday ).
print(x.strftime("%M")) #Prints current MINUTE. 
print(x.strftime("%S")) #Prints current SECOND. 
print(x.strftime('%f')) #Prints MICROSECOND. 
print(x.strftime("%z")) #Prints time OFFSET. 
print(x.strftime("%Z")) #Prints TIMEZONE. 
print(x.strftime("%j")) #Prints DAY NUMBER OF YEAR. 
print(x.strftime("%U")) # WEEKNUMBER as sunday as firstday. 
print(x.strftime("%W")) # WEEKNUMBER as monday as firstday. 
print(x.strftime("%c")) #LOCAL VERSION OF DATETIME. 
print(x.strftime("%C")) #CENTURY. 
print(x.strftime("%x")) #LOCAL VERSION OF DATE. 
print(x.strftime("%X")) #LOCAL VERSION OF TIME. 
print(x.strftime("%%")) # % character. 
print(x.strftime("%G")) # %YEAR. 
print(x.strftime("%u")) # %YEAR. 
print(x.strftime("%V")) # %WEEKNUMBER. 

"""All codes regarding DATETIME are completed here. """


# In[194]:


"""All codes written below are regarding MATH. """


x = min(5,10,15)
y = max(5,10,15)
print(x)
print(y)

x = abs(-7.25)
print(x)

x = pow(4,3)
print(x)

import math
x = math.sqrt(64)
print(x)

import math
x = math.ceil(1.4)
y = math.floor(1.4)
print(x)
print(y)

import math
x = math.pi
print(x)

"""All codes regarding MATH are completed here. """


# In[224]:


"""All codes written below are regarding JSON in python. """


import json
x = '{"name":"John","age":30,"city":"Newyork"}'
y = json.loads(x)
print(y["age"])

import json  #Converts data into JSON. 
x = {
    "name":"John",
    "age":30,
    "city":"New York"
}
y = json.dumps(x)
print(x)

import json
print(json.dumps({"name":"json","age":30}))
print(json.dumps(["apple","bananas"]))
print(json.dumps(("apple","bananas")))
print(json.dumps("Hello"))
print(json.dumps(42))
print(json.dumps(31.76))
print(json.dumps(True))
print(json.dumps(False))
print(json.dumps(None))

import json
x = {
    "name":"John",
    "age":30,
    "married":True,
    "divorced":False,
    "children":("Ann","Billy"),
    "pets":None,
    "cars":[
        {"model":"BMW 230","mpg":27.5},
        {"model":"Ford Edge","mpg":24.1}
    ]
}
print(json.dumps(x))

json.dumps(x,indent=4)

json.dumps(x,indent=4,separators=(".""="))

json.dumps(x,indent=4,sort_keys=True)

"""All codes regarding JSON are completed here. """


# In[274]:


"""All codes written below are regarding REGEX. """


import re
txt = "The rain in spain. "
x = re.search("^The.*Spain$",txt)

import re
txt = "The rain in Spain. "
x = re.findall("ai",txt)
print(x)

import re
txt = "The rain in Spain. "
x = re.findall("Portugal",txt)
print(x)

import re
txt = "The rain in Spain. "
x = re.search("\s",txt)
print("Tje first white-space character is located in position:",x.start())

import re
txt = "The rain in Spain. "
x = re.search("Portugal",txt)
print(x)

import re
txt = "The rain in Spain. "
x = re.split("\s",txt)
print(x)

import re
txt = "The rain in Spain. "
x = re.split("\s",txt,1)
print(x)

import re
txt = "The rain in Spain. "
x = re.sub("\s",txt,"g")
print(x)

import re
txt = "The rain in spain. "
x = re.sub("\s","9",txt,2)
print(x)

import re
txt = "The rain in Spain. "
x = re.search("ai",txt)
print(x)

import re
txt = "The rain in Spain. "
x = re.search(r"\bS\w+",txt)
print(x.span())

import re
txt = "The rain in Spain"
x = re.search(r"\bS\w+", txt)
print(x.string)

import re
txt = "The rain in Spain. "
x = re.search(r"\bS\w+",txt)
print(x.group())

"""All codes regarding REGEX. """


# In[284]:


"""All codes written below are regarding PIP. """


import camelcase
c = camelcase.CamelCase()
txt = "Hello world!"
print(c.hump(txt))

"""All codes regarding CAMELCASE are completed here. """


# In[312]:


"""All codes written below are regarding TRY EXCEPT. """


try:
  print(x)
except:
  print("An exception occurred")

try:
    print(x)
except NameError:
    print("Variable x is not defined. ")
except:
    print("Something else went wrong. ")

try:
    print("Hello! ")
except:
    print("Something else went wrong. ")
else:
    print("Nothing went wrong. ")

try:
    print(x)
except:
    print("Something else went wrong. ")
finally:
    print("The 'try except' is finished. ")

try:
    f = open("demofile.txt")
    try:
        f.write("Lorum Ipsum")
    except:
        print("Somethign else went wrong. ")
    finally:
        f.close()
except:
    print("Something went wrong when operating the file. ")

"""x = -1
if x < 0:
    raise Exception("Sorry, no numbers below zero. ")  #This will raise error exception. """

"""X = 'Hello!'
if not type(x) is int:
raise TypeError("Only integers are allowed. ")"""

"""All codes regarding TRY EXCEPT are completed here. """

"""All codes written below are regarding USER INPUT. """


"""username = input("Enter username: ")
print("Username is : " + username)"""

"""All codes regarding USERINPUT are completed here. """
# In[12]:


"""All codes written below are regarding STRING FORMATTING. """


txt = f"The price is 49 dollars. " # 'f' is used for string formatting. 
print(txt)

price = 59
txt = f"The price is {price} dollars. "
print(txt)

txt = f"The price is {price:.2f} dollars. "
print(txt)

txt = f"The price is {20 * 59} dollars. "
print(txt)

"""All codes regarding STRING FORMATTING are completed here. """


# In[ ]:





# In[1]:


type(int(12.3))


# In[ ]:




