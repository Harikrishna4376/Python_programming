#!/usr/bin/env python
# coding: utf-8

# In[14]:


"""All codes written below are regarding SCIPY TUTORIALS. """
"""All codes written below are regarding GETTING STARTED WITH SCIPY. """


from scipy import constants
print(constants.liter)

import scipy
print(scipy.__version__)

"""All codes regarding GETTING STARTED WITH SCIPY are completed here. """


# In[42]:


"""All codes written below are regarding SCIPY CONSTANTS. """


from scipy import constants
print(constants.pi)

print(dir(constants))

print(constants.yotta)  #Prints the values that these constants holds. 
print(constants.zetta)
print(constants.exa)
print(constants.peta)   #Prints the Metric(SI) prefixes. 
print(constants.tera)
print(constants.giga)
print(constants.mega)
print(constants.kilo)
print(constants.hecto)
print(constants.deka)
print(constants.deci)
print(constants.centi)
print(constants.milli)
print(constants.micro)
print(constants.nano)
print(constants.pico)
print(constants.femto)
print(constants.atto)
print(constants.zepto)

print(constants.kibi)
print(constants.mebi)   #Prints the value of BINARY PREFIXES. 
print(constants.gibi)
print(constants.tebi)
print(constants.pebi)
print(constants.exbi)
print(constants.zebi)
print(constants.yobi)

print(constants.gram)
print(constants.metric_ton)
print(constants.grain)
print(constants.lb)
print(constants.pound)
print(constants.oz)
print(constants.ounce)
print(constants.stone)         #Prints MASS. 
print(constants.long_ton)
print(constants.short_ton)
print(constants.troy_ounce)
print(constants.troy_pound)
print(constants.carat)
print(constants.atomic_mass)
print(constants.m_u)
print(constants.u)

print(constants.degree)
print(constants.arcmin)
print(constants.arcminute)
print(constants.arcsecond)

print(constants.minute)
print(constants.hour)
print(constants.day)
print(constants.week)
print(constants.year)
print(constants.Julian_year)

print(constants.inch)              #0.0254
print(constants.foot)              #0.30479999999999996
print(constants.yard)              #0.9143999999999999
print(constants.mile)              #1609.3439999999998
print(constants.mil)               #2.5399999999999997e-05
print(constants.pt)                #0.00035277777777777776
print(constants.point)             #0.00035277777777777776
print(constants.survey_foot)       #0.3048006096012192
print(constants.survey_mile)       #1609.3472186944373
print(constants.nautical_mile)     #1852.0
print(constants.fermi)             #1e-15
print(constants.angstrom)          #1e-10
print(constants.micron)            #1e-06
print(constants.au)                #149597870691.0
print(constants.astronomical_unit) #149597870691.0
print(constants.light_year)        #9460730472580800.0
print(constants.parsec)            

print(constants.atm)         #101325.0
print(constants.atmosphere)  #101325.0
print(constants.bar)         #100000.0
print(constants.torr)        #133.32236842105263
print(constants.mmHg)        #133.32236842105263
print(constants.psi)

print(constants.hectare)
print(constants.acre)

print(constants.liter)            #0.001
print(constants.litre)            #0.001
print(constants.gallon)           #0.0037854117839999997
print(constants.gallon_US)        #0.0037854117839999997
print(constants.gallon_imp)       #0.00454609
print(constants.fluid_ounce)      #2.9573529562499998e-05
print(constants.fluid_ounce_US)   #2.9573529562499998e-05
print(constants.fluid_ounce_imp)  #2.84130625e-05
print(constants.barrel)           #0.15898729492799998
print(constants.bbl)   

print(constants.kmh)            #0.2777777777777778
print(constants.mph)            #0.44703999999999994
print(constants.mach)           #340.5
print(constants.speed_of_sound) #340.5
print(constants.knot)           #0.5144444444444445

print(constants.zero_Celsius)      #273.15
print(constants.degree_Fahrenheit) #0.5555555555555556

print(constants.eV)            #1.6021766208e-19
print(constants.electron_volt) #1.6021766208e-19
print(constants.calorie)       #4.184
print(constants.calorie_th)    #4.184
print(constants.calorie_IT)    #4.1868
print(constants.erg)           #1e-07
print(constants.Btu)           #1055.05585262
print(constants.Btu_IT)        #1055.05585262
print(constants.Btu_th)        #1054.3502644888888
print(constants.ton_TNT)       #4184000000.0

print(constants.hp)
print(constants.horsepower)

print(constants.dyn)
print(constants.dyne)
print(constants.lbf)
print(constants.pound_force)
print(constants.kgf)
print(constants.kilogram_force)


"""All codes regarding SCIPY CONSTANTS are completed here. """


# In[54]:


"""All codes written below are regarding SCIPY OPTIMIZERS. """


from scipy.optimize import root
from math import cos
def myfunc(x):
    return x + cos(x)
myroot = root(myfunc,0)
print(myroot.x)

print(myroot)

from scipy.optimize import minimize
def eqn(x):
    return x**2 + x + 2
mymin = minimize(eqn,0,method='BFGS')  #Minimizing equation with BFGS. 
print(mymin)

"""all codes regarding SCIPY OPTIMIZERS are completed here. """


# In[72]:


"""All codes written below are regarding SPARSE DATA. """


import numpy as np
from scipy.sparse import csr_matrix
arr = np.array([0,0,0,0,0,1,1,0,2])  #Creating CSR_MATRIX from an array. 
print(csr_matrix(arr))

arr = np.array([[0,0,0],[0,0,1],[1,0,2]])
print(csr_matrix(arr).data)

arr = np.array([[0,0,0],[0,0,1],[1,0,2]])
print(csr_matrix(arr).count_nonzero())

mat = csr_matrix(arr)
mat.eliminate_zeros()
print(mat)

mat = csr_matrix(arr)
mat.sum_duplicates()
print(mat)

newarr = csr_matrix(arr).tocsc()
print(newarr)

"""All codes regarding SPARSE DATA are completed here. """


# In[96]:


"""All codes written below are regarding SCIPY GRAPHS. """


import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
arr = np.array([
    [0,1,2],
    [1,0,0],
    [2,0,0]
])
newarr = csr_matrix(arr)
print(connected_components(newarr))

import numpy as np
from scipy.sparse.csgraph import dijkstra  #Calculating shortest dist 
from scipy.sparse import csr_matrix        #using dijkstras algorithm.
arr = np.array([
    [0,1,2],
    [1,0,0],
    [2,0,0]
])
newarr = csr_matrix(arr)
print(dijkstra(newarr,return_predecessors=True,indices=0))

import numpy as np
from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse import csr_matrix
arr = np.array([
    [0,1,2],
    [1,0,0],
    [2,0,0]
])
newarr = csr_matrix(arr)
print(floyd_warshall(newarr,return_predecessors=True))

import numpy as np
from scipy.sparse.csgraph import bellman_ford
from scipy.sparse import csr_matrix
arr = np.array([
    [0,-1,2],
    [1,0,0],
    [2,0,0]
])
newarr = csr_matrix(arr)
print(bellman_ford(newarr,return_predecessors=True,indices=0))

import numpy as np
from scipy.sparse.csgraph import depth_first_order
from scipy.sparse import csr_matrix
arr = np.array([
     [0, 1, 0, 1],
  [1, 1, 1, 1],
  [2, 1, 1, 0],
  [0, 1, 0, 1]
])
newarr = csr_matrix(arr)
print(depth_first_order(newarr,1))

import numpy as np
from scipy.sparse.csgraph import breadth_first_order
from scipy.sparse import csr_matrix
arr = np.array([
  [0, 1, 0, 1],
  [1, 1, 1, 1],
  [2, 1, 1, 0],
  [0, 1, 0, 1]
])
newarr = csr_matrix(arr)
print(breadth_first_order(newarr, 1))

"""All codes regarding SCIPY GRAPHS are completed here. """


# In[36]:


"""All codes written below are regarding SPATIAL DATA. """


import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
points = np.array([   #Creating triangulation from below points. 
    [2,4],
    [3,4],
    [3,0],
    [2,2],
    [4,1]
])
simplices = Delaunay(points).simplices
plt.triplot(points[:,0],points[:,1],simplices)
plt.scatter(points[:,0],points[:,1],color='r')
plt.show()

import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
points = np.array([
    [2, 4],
  [3, 4],
  [3, 0],
  [2, 2],
  [4, 1],
  [1, 2],
  [5, 0],
  [3, 1],
  [1, 2],
  [0, 2]
])
hull = ConvexHull(points)
hull_points = hull.simplices
plt.scatter(points[:,0],points[:,1])
for simplex in hull_points:
    plt.plot(points[simplex,0],
points[simplex,1],'k-')
plt.show()

from scipy.spatial import KDTree
points = [(1,-1),(2,3),(-2,3),(2,-3)]
kdtree = KDTree(points)
res = kdtree.query((1,1))  #Finding the nearest neighbor to (1,1).
print(res)

from scipy.spatial.distance import euclidean
p1 = (1,0)
p2 = (10,2)
res = euclidean(p1,p2)
print(res)

from scipy.spatial.distance import cityblock
p1 = (1,0)
p2 = (10,2)
res = cityblock(p1,p2)
print(res)

from scipy.spatial.distance import cosine
p1 = (1,0)
p2 = (10,2)
res = cosine(p1,p2)
print(res)

from scipy.spatial.distance import hamming
p1 = (True,False,True)
p2 = (False,True,True)
res = hamming(p1,p2)
print(res)

"""All codes regarding SPATIAL DATA are completed here. """


# In[46]:


"""All codes written below are regarding MATLAB ARRAYS. """


from scipy import io
import numpy as np
arr = np.arange(10)
io.savemat('arr.mat',{"vec":arr})  #Created an array mat file automatically.

from scipy import io
import numpy as np
arr = np.array([0,1,2,3,4,5,6,7,8,9])
io.savemat('arr.mat',{"vec":arr})
mydata = io.loadmat('arr.mat')
print(mydata)

print(mydata['vec'])  #Prints the ARRAY from the file. 

mydata = io.loadmat('arr.mat',squeeze_me = True)  #Makes the array compact.
print(mydata['vec'])

"""All codes regarding MATLAB ARRAYS are completed here. """


# In[60]:


"""All codes written below are regarding INTERPOLATION. """
"""Interpolation is a method for generating points between given points. 
For example: for points1 and 2, we may interpolate and find points 
1.33 and 1.66 .
Interpolation has many usage, in Machine Learning we often deal with 
missing data in dataset, interpolation is often used to 
substitute those values. """



from scipy.interpolate import interp1d
import numpy as np
xs = np.arange(10)
ys = 2*xs + 1
interp_func = interp1d(xs,ys)
newarr = interp_func(np.arange(2.1,3,0.1))
print(newarr)

from scipy.interpolate import UnivariateSpline
import numpy as np
xs = np.arange(10)
ys = xs**2 + np.sin(xs) + 1
interp_func = UnivariateSpline(xs,ys)
newarr = interp_func(np.arange(2.1,3,0.1))
print(newarr)

from scipy.interpolate import Rbf
import numpy as np
xs = np.arange(10)
ys = xs**2 + np.sin(xs) + 1
interp_func = Rbf(xs,ys)
newarr = interp_func(np.arange(2.1,3,0.1))
print(newarr)

"""All codes regarding INTERPOLATION are completed here. """


# In[80]:


"""All codes written below are regarding STATISTICAL SIGNIFICANCE. """
"""So, in statistics, statistical significance means that the result
that was produced has a reason behind it, it was not produced randomly,
or by chance. 
Scipy provides us with a module called scipy.stats which has functions for
performing statistical significance tests. """


import numpy as np
from scipy.stats import ttest_ind
v1 = np.random.normal(size=100)
v2 = np.random.normal(size=100)
res = ttest_ind(v1, v2)
print(res)

res = ttest_ind(v1,v2).pvalue
print(res)

import numpy as np
from scipy.stats import kstest
v = np.random.normal(size=100)
res = kstest(v,'norm')
print(res)

import numpy as np
from scipy.stats import describe
v = np.random.normal(size=100)
res = describe(v)
print(res)

import numpy as np
from scipy.stats import skew,kurtosis
v = np.random.normal(size=100)
print(skew(v))
print(kurtosis(v))

import numpy as np
from scipy.stats import normaltest
v = np.random.normal(size=100)
print(normaltest(v))

"""All codes regarding STATISTICAL SIGNIFICANCE are completed here. """
"""All codes regarding SCIPY are completed here. """


# In[ ]:




