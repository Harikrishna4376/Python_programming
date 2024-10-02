#!/usr/bin/env python
# coding: utf-8

# In[10]:


"""All codes written below are regarding NUMPY. """
"""All codes written below are regarding NUMPY INTRODUCTION. """


import numpy as np
arr = np.array([1,2,3,4,5])
print(arr)

import numpy as np
print(np.__version__) #IT will display the current version of numpy. 

"""All codes regarding NUMPY INTRODUCTION are completed here. """


# In[32]:


"""All codes written below are regarding CREATING ARRAYS. """


import numpy as np
arr = np.array([1,2,3,4,5])
print(arr)
print(type(arr))

import numpy as np
aarr = np.array((1,2,3,4,5)) #Inspite of using tuple to create array, it will display the array only in big braces. 
print(arr)

import numpy as np
arr = np.array(42)
print(arr)

import numpy as np
arr = np.array([1,2,3,4,5])
print(arr)

import numpy as np
arr = np.array([[1,2,3],[4,5,6]])
print(arr)

import numpy as np
arr = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(arr)

import numpy as np
a = np.array(42)
b = np.array([1,2,3,4,5])
c = np.array([[1,2,3],[4,5,6]])
d = np.array([[[1,2,3],[4,5,6]],[[1,2,3],[4,5,6]]])
print(a.ndim) #it will display the dimensions. 
print(b.ndim)
print(c.ndim)
print(d.ndim)

import numpy as np
arr = np.array([1,2,3,4],ndmin=5)
print(arr)
print('number of dimensions: ', arr.ndim)

"""All codes regarding CREATING ARRAYS are completed here. """


# In[56]:


"""All codes written below are regarding ARRAY INDEXING. """


import numpy as np
arr = np.array([1,2,3,4,5])
print(arr[0]) #It will print the array of written index. 

import numpy as np
arr = np.array([1,2,3,4,5])
print(arr[1]) #It will print the array of written index. 

print(arr[2] + arr[3] + arr[4]) #It will print the array of written index. 

arr = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print('2nd element on 1st row: ',arr[0,1])

print('5th element on 2nd row: ',arr[1,4]) #prints 1ST ROW, 4TH MEANS 5TH ELEMENT. 

arr = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(arr[0,1,2])  #prints 3rd element of 2nd row of 1st dimension. 

arr = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print('Last element from 2nd dim: ',arr[1,-1]) #Prints last element and row of 2nd dimension. 

"""All codes regarding ARRAY INDEXING are completed here. """


# In[76]:


"""All codes written below are regarding ARRAY SLICING. """


import numpy as np
arr = np.array([1,2,3,4,5,6,7])
print(arr[1:5])

import numpy as np
arr = np.array([1,2,3,4,5,6,7])
print(arr[4:])

print(arr[:4])

print(arr[-3:-1])

print(arr[1:5:2])  #Prints every other element from array from index to 5 with a gap of 2. 

print(arr[::2]) #prints every other element with a gap of 2

"""STEP means if the step is 2 , then it will not print every other element that is not 2nd. 
meaning if the step is 2 and the starting and ending is 1 and 5, means that the code 
will display the 2nd element and will consider as a step and after displaying the step,
it will count as first. """

arr = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(arr[1,1:4])

import numpy as np
arr = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(arr[0:2,2])  #It will print the both elements of index 2. 

print(arr[0:2,1:4])  #It will return 1 to 4 from both dimensions. 

"""All codes regarding SLICING ARRAYS. """


# In[94]:


"""All codes written below are regarding DATA TYPES. """


import numpy as np
arr = np.array([1,2,3,4,5])
print(arr.dtype)

arr = np.array(['apple','banana','cherry'])
print(arr.dtype)

arr = np.array([1,2,3,4,5],dtype = 'S')
print(arr)
print(arr.dtype)

arr = np.array([1,2,3,4],dtype = 'i4')
print(arr)
print(arr.dtype)

arr = np.array([1.2,2.3,3.4])
newarr = arr.astype('i')
print(newarr)
print(newarr.dtype)

arr = np.array([1.1,2.2,3.3])
newarr = arr.astype(int)
print(newarr)
print(newarr.dtype)

newarr = arr.astype(bool)
print(newarr)
print(newarr.dtype)

"""All codes regarding DATA TYPES are completed here. """


# In[104]:


"""All codes written below are regarding COPY VS VIEW. """


import numpy as np
arr = np.array([1,2,3,4,5])
x = arr.copy()
arr[0] = 42
print(arr)
print(x)

arr = np.array([1,2,3,4,5])
x = arr.view()
arr[0] = 42
print(arr)
print(x)

arr = np.array([1,2,3,4,5])
x = arr.view()
x[0] = 31
print(arr)
print(x)

arr = np.array([1,2,3,4,5])
x = arr.copy()
y = arr.view()
print(x.base)
print(y.base)

"""All codes regarding COPY VS VIEW are completed here. """


# In[118]:


"""All codes written below are regarding ARRAY SHAPE. """


import numpy as np
arr = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(arr.shape)

arr = np.array([1,2,3,4,5],ndmin = 5)
print(arr)
print('Shape of array :',arr.shape)

"""All codes regarding ARRAY SHAPE are completed here. """


# In[154]:


"""All codes written below are regarding RESHAPING ARRAYS. """


import numpy as np
arr = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
newarr = arr.reshape(2,6) #It will reshape the array into particular dimension or into 2*6,3*4,4*3, and soo on. 
print(newarr)

newarr = arr.reshape(2,3,2) #It will reshape the array into more dimensions. 
print(newarr)

arr = np.array([1,2,3,4,5,6,7,8])
print(arr.reshape(2,4).base)

newarr = arr.reshape(2,2,-1)
print(newarr)

arr = np.array([[1,2,3],[4,5,6]])
newarr = arr.reshape(-1) #It converts any dimensional array into one dimensional array. 
print(newarr)

"""All codes regarding ARRAY RESHAPING are completed here. """


# In[198]:


"""All codes written below are regarding ARRAY ITERATING. """


import numpy as np
arr = np.array([1,2,3])
for x in arr:
    print(x)

arr = np.array([[1,2,3],[4,5,6]])
for x in arr:
    print(x)  #Using for loop to display 'arr'.

arr = np.array([[1,2,3],[4,5,6]])
for x in arr:
    for y in x:
        print(y)  #It will display array in non array format. 

arr = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
for x in arr:
    print(x)
for x in arr:
    for y in x:
        for z in y:
            print(z)

arr = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
for x in np.nditer(arr):
    print(x)  #Another way to display array. 

arr = np.array([1,2,3])
for x in np.nditer(arr,flags = ['buffered'],op_dtypes = ['S']):
    print(x)

arr = np.array([[1,2,3,4],[5,6,7,8]])
for x in np.nditer(arr[:,::2]):  #It will print the step2 array from both dimensions. 
    print(x)

arr = np.array([1,2,3])
for idx,x in np.ndenumerate(arr):  #It ENUMERATES ARRAY. 
    print(idx,x)

arr = np.array([[1,2,3,4],[5,6,7,8]])
for idx,x in np.ndenumerate(arr):
    print(idx,x)

"""All codes regarding ITERATING ARRAYS are completed here. """


# In[230]:


"""All codes written below are regarding JOINING ARRAY. """


import numpy as np
arr1 = np.array([[1,2,3],[1,2,3]])
arr2 = np.array([[4,5,6],[4,5,6]])
arr = np.concatenate((arr1,arr2))
print(arr)  #Joining two arrays. 

arr = np.concatenate((arr1,arr2),axis=1)
print(arr)

arr1 = np.array([1,2,3])
arr2 = np.array([4,5,6])
arr = np.stack((arr1,arr2),axis=1)
print(arr)  #Using STACK to merge the arrays. 

arr = np.hstack((arr1,arr2))
print(arr)  #Using HORIZONTAL STACK. 

arr = np.vstack((arr1,arr2))
print(arr)

arr = np.dstack((arr1,arr2))
print(arr)  #It will display the stack array into height. with more dimensions. 

"""All codes regarding JOINING ARRAY are completed here. """


# In[252]:


"""All codes written below are regarding SPLITTING ARRAYS. """


import numpy as np
arr = np.array([1,2,3,4,5,6])
newarr = np.array_split(arr,3) #SPLITS array into 3 parts. 
print(newarr)

newarr = np.array_split(arr,4) #SPLITS ARRAY INTO 4. 
print(newarr)

newarr = np.array_split(arr,3)
print(newarr[0])
print(newarr[1])
print(newarr[2])

arr = np.array([[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]])
newarr = np.array_split(arr,3)
print(newarr)

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.array_split(arr,3)
print(newarr)

newarr = np.array_split(arr,3,axis = 1)
print(newarr)

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.hsplit(arr,3)
print(newarr)

"""All codes regarding SPLITTING ARRAY are completed here. """


# In[268]:


"""All codes written below are regarding SEARCHING ARRAYS. """


import numpy as np
arr = np.array([1,2,3,4,5,4,4])
x = np.where(arr == 4)
print(x)

arr = np.array([1,2,3,4,5,6,7,8])
x = np.where(arr%2 == 0)  #It will not print the values that are EVEN. 
print(x)

x = np.where(arr%2 == 1)
print(x)  #It iwll not display the ODD values. 

arr = np.array([6,7,8,9])
x = np.searchsorted(arr,7)  #It will print the searched array values. 
print(x)

arr = np.array([6,7,8,9])
x = np.searchsorted(arr,7,side = 'right') #Finds the indexes where the value 7 should be inserted starting form the right. 
print(x)

arr = np.array([1,3,5,7])
x = np.searchsorted(arr,[2,4,6])  #It will show the position where the value must be positioned. 
print(x)

"""All codes regarding SEARCHING ARRAYS are completed here. """


# In[278]:


"""All codes written below are regarding SORTING ARRAYS. """


import numpy as np
arr = np.array([3,2,0,1]) #It will automatically arrange(sort) the value elements. 
print(np.sort(arr))

arr = np.array(['banana','cherry','apple'])
print(np.sort(arr))  #It will arrange the elements in alphabetical vise manner. 

arr = np.array([True,False,True])
print(np.sort(arr))  #It will also sort the boolean values or Strings. 

arr = np.array([[3,2,4],[5,0,1]])
print(np.sort(arr))  #It will arrange the arrays according to dimension manner. 

"""All codes regarding SORTED ARRAYS are completed here. """


# In[290]:


"""All codes written below are regarding FILTER ARRAY. """


import numpy as np
arr = np.array([41,42,43,44])
x = [True,False,True,False]
newarr = arr[x]
print(newarr)  #Prints the array value of boolean equivalence. 

import numpy as np
arr = np.array([41,42,43,44])
filter_arr = []
for element in arr:
    if element > 42:
        filter_arr.append(True)
    else:
        filter_arr.append(False)
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)

filter_arr = arr > 42
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)

arr = np.array([1,2,3,4,5,6,7])
filter_arr = arr % 2 == 0
newarr = arr[filter_arr]
print(filter_arr)
print(newarr)

"""All codes regarding FILTER ARRAYS are completed here. """


# In[320]:


"""All codes written below are regarding RANDOM NUMBERS IN NUMPY. """


from numpy import random
x = random.randint(100)
print(x)

x = random.rand()
print(x)

x = random.randint(100,size=(5))
print(x)

x = random.randint(100,size=(3,5))
print(x)

x = random.rand(5)
print(x)

x = random.rand(3,5)
print(x)

x = random.choice([3,5,7,9])
print(x)

x = random.choice([3,5,7,9],size=(3,5))
print(x)

"""All codes regarding RANDOM NUMBERS are completed here. """


# In[332]:


"""All codes written below are regarding DATA DISTRIBUTION. """


from numpy import random
x = random.choice([3,5,7,9],p = [0.1,0.3,0.6,0.0],size=(100))
print(x)

x = random.choice([3,5,7,9],p=[0.1,0.3,0.6,0.0],size=(3,5))
print(x)

"""All codes regarding DATA DISTRIBUTION are completed here. """


# In[347]:


"""All codes written below are regarding RANDOM PERMUTATIONS. """


from numpy import random
import numpy as np
arr = np.array([1,2,3,4,5])
random.shuffle(arr)
print(arr)

print(random.permutation(arr))

"""All codes regarding RANDOM PERMUTATIONS are completed here. """


# In[357]:


"""All codes written below are regarding SEABORN. """


import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot([0,1,2,3,4,5])
plt.show()

sns.distplot([0,1,2,3,4,5],hist=False)
plt.show()

"""All codes regarding SEABORN are completed here. """


# In[371]:


"""All codes written below are regarding NORMAL GAUSSIAN DISTRIBUTION. """


from numpy import random
x = random.normal(size=(2,3))
print(x)

x = random.normal(loc=1,scale=2,size=(2,3))
print(x)

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.normal(size=1000),hist=False)
plt.show()

"""All codes regarding NORMAL GAUSSIAN DISTRIBUTION are completed here. """


# In[403]:


"""All codes written below are regarding BINOMIAL DISTRIBUTION. """


from numpy import random
x = random.binomial(n=10,p=0.5,size=10)
print(x)

sns.distplot(random.binomial(n=10,p=0.5,size=1000),hist=True,kde=False)
plt.show()

sns.distplot(random.normal(loc=50,scale=5,size=1000),hist=False,label='normal')
sns.distplot(random.binomial(n=100,p=0.5,size=1000),hist=False,label='binomial')
plt.show()

"""All codes regarding BINOMIAL DISTRIBUTION are completed here. """


# In[417]:


"""All codes written below are regarding POISSON DISTRIBUTION. """


from numpy import random
x = random.poisson(lam=2,size=10)
print(x)

from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.distplot(random.poisson(lam=2,size=1000),kde=True)
plt.show()

sns.distplot(random.normal(loc=50,scale=7,size=1000),hist=False,label='normal')
sns.distplot(random.poisson(lam=50,size=1000),hist=False,label='poisson')
plt.show()

sns.distplot(random.binomial(n=1000,p=0.01,size=1000),hist=False,label='binomial')
sns.distplot(random.poisson(lam=10,size=1000),hist=False,label='poisson')
plt.show()

"""All codes regarding POISSON DISTRIBUTION are completed here. """


# In[423]:


"""All codes written below are regarding UNIFORM DISTRIBUTION. """


from numpy import random
x = random.uniform(size=(2,3))
print(x)

sns.distplot(random.uniform(size=1000),hist=False)
plt.show()

"""All codes regarding UNIFORM DISTRIBUTION are completed here. """


# In[435]:


"""All codes written below are regarding LOGISTIC DISTRIBUTION. """


from numpy import random
x = random.logistic(loc=1,scale=2,size=(2,3))
print(x)

sns.distplot(random.logistic(size=1000),hist=False)
plt.show()

sns.distplot(random.normal(scale=2,size=1000),hist=False,label='normal')
sns.distplot(random.logistic(size=1000),hist=False,label='logistic')
plt.show()

"""All codes regarding LOGISTIC DISTRIBUTION are completed here. """


# In[439]:


"""All codes written below are regarding MULTINOMIAL DISTRIBUTION. """


from numpy import random
x = random.multinomial(n=6,pvals=[1/6,1/6,1/6,1/6,1/6,1/6])
print(x)

"""All codes regarding MULTINOMIAL DISTRIBUTION are completed here. """


# In[453]:


"""All codes written below are regarding EXPONENTIAL DISTRIBUTION. """


from numpy import random
x = random.exponential(scale=2,size=(2,3))
print(x)

sns.distplot(random.exponential(size=1000),hist=False)
plt.show()

sns.distplot(random.poisson(size=1000),hist=False,label='Poisson')
sns.distplot(random.exponential(size=1000),hist=False,label='Exponential')
plt.show()

"""All codes regarding EXPONENTIAL DISTRIBUTION are completed here. """


# In[459]:


"""All codes written below are regarding CHI SQUARE DISTRIBUTION. """


from numpy import random
x = random.chisquare(df=2,size=(2,3))
print(x)

sns.distplot(random.chisquare(df=1,size=1000),hist=False)
plt.show()

"""All codes regarding CHI SQUARE are completed here. """


# In[467]:


"""All codes written below are regarding RAYLEIGH DISTRIBUTION. """


from numpy import random
x = random.rayleigh(scale=2,size=(2,3))
print(x)

sns.distplot(random.rayleigh(size=1000),hist=False)
plt.show()

"""All codes regarding RAYLEIGH DISTRIBUTION are completed here. """


# In[473]:


"""All codes written below are regarding PARETO DISTRIBUTION. """


from numpy import random
x = random.pareto(a=2,size=(2,3))
print(x)

sns.distplot(random.pareto(a=2,size=1000),kde=True)
plt.show()

"""All codes regarding PARETO DISTRIBUTION are completed here. """


# In[479]:


"""All codes written below are regarding ZIPF DISTRIBUTION. """


from numpy import random
x = random.zipf(a=2,size=(2,3))
print(x)

sns.distplot(random.zipf(a=2,size=1000),hist=False,kde=True)
plt.show()

"""All codes regarding ZIPF DISTRIBUTION are completed here. """


# In[491]:


"""All codes written below are regarding NUMPY UFUNCS. """
"""NUMPY UFUNC stands for universal functions. it is used
for vectorization in NumPy which is way faster than 
iterating over elements. """
"""Converting iterative statements into vector
based operation is called vectorization. """


x = [1,2,3,4]
y = [5,6,7,8]
z = []
for i ,j in zip(x,y):
    z.append(i + j)  #It will perform summation. 
print(z)

z = np.add(x,y)
print(z)  #It also performs same operation. 

"""All codes regarding NUMPY UFUNC are completed here. """


# In[505]:


"""All codes written below are regarding CREATE YOUR OWN UFUNC. """


import numpy as np
def add(x,y):   #It also performs addition operation but it creates their own universal function. 
    return x+y
add = np.frompyfunc(add,2,1)
print(add([1,2,3,4],[5,6,7,8]))

print(type(np.add))  #It is displaying that the function used is indeed numpy function. 

print(type(np.concatenate))  #It combines two arrays. 

import numpy as np
if type(np.add) == np.ufunc:  #It will display whether the function is ufunc or not. 
    print('add is ufunc. ')
else:
    print('add is not ufunc. ')

"""All codes regarding NUMPY UFUNC are completed here. """


# In[533]:


"""All codes written below are regarding SIMPLE ARITHMETIC. """


import numpy as np
arr1 = np.array([10,11,12,13,14,15])
arr2 = np.array([20,21,22,23,24,25])
newarr = np.add(arr1,arr2)  #Addition of two arrays. 
print(newarr)

newarr = np.subtract(arr1,arr2)  #Performing substraction operation. 
print(newarr)

newarr = np.multiply(arr1,arr2)  #Performing multiplication operation. 
print(newarr)

newarr = np.divide(arr1,arr2)  #Performing division operation. 
print(newarr)

newarr = np.power(arr1,arr2)  #Raise the values in arr1 to the power of values in arr2. 
print(newarr)

newarr = np.mod(arr1,arr2)  #Modulo operation. 
print(newarr)

newarr = np.remainder(arr1,arr2)  #calculating remainder. 
print(newarr)

newarr = np.divmod(arr1,arr2)  #Return the quotient and mod. 
print(newarr)

arr3 = np.array([-1,-2,-3,1,3,2,-9,8])
newarr = np.absolute(arr3)  #Converts the negative intergers in to positive ones. 
print(newarr)

"""All codes regarding SIMPLE ARITHMETIC are completed here. """


# In[18]:


"""All codes written below are regarding ROUNDING DECIMALS. """


import numpy as np
arr = np.trunc([-3.1666,3.6667]) #TRUNCATENING DECIMALS.
print(arr)

arr = np.fix([-3.1666,3.6667])
print(arr)

arr = np.around([3.1666,2])
print(arr)

arr = np.floor([-3.1666,3.6667])
print(arr)

arr = np.ceil([-3.1666,3.6667])
print(arr)

"""All codes regarding ROUNDING DECIMALS are completed here. """


# In[28]:


"""All codes written below are regarding NUMPY LOGS. """


import numpy as np
arr = np.arange(1,10)
print(np.log2(arr))  #It will display the logs of the corresponding numbers. 

arr = np.arange(1,10)
print(np.log10(arr))

print(np.log(arr))

from math import log
nplog = np.frompyfunc(log,2,1)
print(nplog(100,15))

"""All codes regarding NUMPY LOGS are completed here. """


# In[40]:


"""All codes written below are regarding NUMPY SUMMATIONS. """


arr1 = np.array([1,2,3])
arr2 = np.array([1,2,3])
newarr = np.add(arr1,arr2)
print(newarr)

newarr = np.sum([arr1,arr2])
print(newarr)

newarr = np.sum([arr1,arr2],axis=1)
print(newarr)

newarr = np.cumsum(arr1)
print(newarr)

"""All codes regarding NUMPY SUMMATION are completed here. """


# In[50]:


"""All codes written below are regarding NUMPY PRODUCTS. """


import numpy as np
arr = np.array([1,2,3,4])
x = np.prod(arr)
print(x)

arr1 = np.array([1,2,3,4])
arr2 = np.array([5,6,7,8])
x = np.prod([arr1,arr2])
print(x)

newarr = np.prod([arr1,arr2],axis=1)
print(newarr)

newarr = np.cumprod(arr2)
print(newarr)


"""All codes regarding NUMPY PRODUCTS are completed here. """


# In[58]:


"""All codes written below are regarding NUMPY DIFFERENCES. """


import numpy as np
arr = np.array([10,15,20,5])
newarr = np.diff(arr)
print(newarr)

newarr = np.diff(arr,n=2)
print(newarr)

"""All codes regarding NUMPY DIFFERENCES are completed here. """


# In[68]:


"""All codes written below are regarding LCM [LOWEST COMMON FACTOR]. """


import numpy as np
num1 = 4
num2 = 6
x = np.lcm(num1,num2)   #Calculates LCM. 
print(x)

arr = np.array([3,6,9])
x = np.lcm.reduce(arr)
print(x)

arr = np.arange(1,11)
x = np.lcm.reduce(arr)
print(x)

"""All codes regarding LCM are completed here. """


# In[74]:


"""All codes written below are regarding GCD. """


import numpy as np
num1 = 6
num2 = 9
x = np.gcd(num1,num2)
print(x)

arr = np.array([20,8,32,36,16])
x = np.gcd.reduce(arr)
print(x)

"""All codes regarding GCD are completed here. """


# In[98]:


"""All codes written below are regarding TRIGONOMETRIC FUNCTIONS. """


import numpy as np
x = np.sin(np.pi/2)
print(x)

arr = np.array([np.pi/2,np.pi/3,np.pi/4,np.pi/5])
x = np.sin(arr)
print(x)

arr = np.array([90,180,270,360])
x = np.deg2rad(arr)
print(x)

arr = np.array([np.pi/2,np.pi,1.5*np.pi,2*np.pi])
x = np.rad2deg(arr)
print(x)

x = np.arcsin(1.0)
print(x)

arr = np.array([1,-1,0.1])
x = np.arcsin(arr)
print(x)

base = 3
perp = 4
x = np.hypot(base,perp)
print(x)

"""All codes regarding TRIGONOMETRIC FUNCTIONS are comleted here. """


# In[108]:


"""All codes written below are regarding HYPERBOLIC FUNCTIONS. """


import numpy as np
x = np.sinh(np.pi/2)
print(x)

arr = np.array([np.pi/2,np.pi/3,np.pi/4,np.pi/5])
x = np.cosh(arr)
print(x)

x = np.arcsin(1.0)
print(x)

arr = np.array([0.1,0.2,0.5])
x = np.arctanh(arr)
print(x)


"""All codes regarding HYPERBOLIC FUNCTIONS are completed here. """


# In[128]:


"""All codes written below are regarding SET OPERATIONS. """


import numpy as np
arr = np.array([1,1,1,2,3,4,5,5,6,7])
x = np.unique(arr)
print(x)

arr1 = np.array([1,2,3,4])
arr2 = np.array([3,4,5,6])
newrr = np.union1d(arr1,arr2)
print(newarr)

newarr = np.intersect1d(arr1,arr2,assume_unique=True)
print(newarr)

newarr = np.setdiff1d(arr1, arr2, assume_unique=True)
print(newarr)

newrr = np.setxor1d(arr1,arr2,assume_unique=True)
print(newrr)

"""All codes regarding SET OPERATIONS are completed here. """


# In[ ]:




