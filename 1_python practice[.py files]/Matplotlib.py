#!/usr/bin/env python
# coding: utf-8

# In[12]:


"""All codes written below are regarding MATPLOTLIB. """
"""All codes written below are regarding MATPLOTLIB GETTING STARTED. """


import matplotlib
print(matplotlib.__version__)

"""All codes regarding GETTING STARTED with matplotlib is completed here. """


# In[16]:


"""All codes written below are regarding MATPLOTLIB.PYPLOT. """


import matplotlib.pyplot as plt
import numpy as np
xpoints = np.array([0,6])
ypoints = np.array([0,250])
plt.plot(xpoints,ypoints)
plt.show()

"""All codes regarding PYPLOT are completed here. """


# In[32]:


"""All codes written below are regarding PLOTTING. """


import matplotlib.pyplot as plt
import numpy as np
xpoints = np.array([1,8])
ypoints = np.array([3,10])
plt.plot(xpoints,ypoints)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
xpoints = ([1,8])
ypoints = ([3,10])
plt.plot(xpoints,ypoints,'*')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
xpoints = np.array([1,2,6,8])
ypoints = np.array([3,8,1,10])
plt.plot(xpoints,ypoints)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3,8,1,10,5,7])
plt.plot(ypoints)
plt.show()

"""All codes regarding PLOTTING are completed here. """


# In[74]:


"""All codes written below are regarding MARKERS. """


import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3,8,1,10])
plt.plot(ypoints,marker='o')
plt.show()

plt.plot(ypoints,marker='_')

#plt.plot(ypoints,marker='*')
#plt.plot(ypoints,marker='+')
#plt.plot(ypoints,marker='.')
#plt.plot(ypoints,marker=',')
#plt.plot(ypoints,marker='X')
#plt.plot(ypoints,marker='P')
#plt.plot(ypoints,marker='s')
#plt.plot(ypoints,marker='D')
#plt.plot(ypoints,marker='d')
#plt.plot(ypoints,marker='p')
#plt.plot(ypoints,marker='H')
#plt.plot(ypoints,marker='h')
#plt.plot(ypoints,marker='v')
#plt.plot(ypoints,marker='^')
#plt.plot(ypoints,marker='<')
#plt.plot(ypoints,marker='>')
#plt.plot(ypoints,marker='1')
#plt.plot(ypoints,marker='2')
#plt.plot(ypoints,marker='3')
#plt.plot(ypoints,marker='4')
#plt.plot(ypoints,marker='|')

import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3,8,1,10])
plt.plot(ypoints,'o:r')  #'o' is marker and 'r' is the color of the marker.
plt.show()

import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3,8,1,10])
plt.plot(ypoints,marker='o',ms = 15) #ms is 'marker size'.
plt.show()

plt.plot(ypoints,marker='*',ms = 15,mec = 'r') #mec is 'marker edge color'.
plt.show()

plt.plot(ypoints,marker='o',ms = 15,mfc='r') #mfc is 'marker face color'.
plt.show()

plt.plot(ypoints,marker='o',ms = 15,mec='r',mfc='b')
plt.show()

plt.plot(ypoints,marker='o',ms = 15,mec='#4CAF50',mfc = 'b')
plt.show()

plt.plot(ypoints,marker='o',ms=15,mec='hotpink',mfc='b')
plt.show()

"""All codes regarding MARKERS are completed here. """


# In[23]:


"""All codes written below are regarding MATPLOTLIB LINE. """


import matplotlib.pyplot as plt
import numpy as np
ypoints = np.array([3,8,1,10])
plt.plot(ypoints,linestyle='dotted')
plt.show()

plt.plot(ypoints,linestyle='dashed')
plt.show()

plt.plot(ypoints,ls = ':')
plt.show()

plt.plot(ypoints,color='r')
plt.show()

plt.plot(ypoints, color='#4CAF50')
plt.show()

plt.plot(ypoints,c='hotpink')
plt.show()

plt.plot(ypoints,linewidth='3.6')
plt.show()

y1 = np.array([3,8,1,10])
y2 = np.array([6,2,7,11])
plt.plot(y1)
plt.plot(y2)
plt.show()

x1 = np.array([0,1,4,6])
x2 = np.array([3,8,1,10])
y1 = np.array([0,1,2,3])
y2 = np.array([6,2,7,11])
#plt.plot(x1,x2,y1,y2)
plt.plot(x1,color='r')
plt.plot(x2,color='b')
plt.plot(y1,color='m')
plt.plot(y2,color='hotpink')
plt.show()

"""All codes regarding MATPLOTLIB LINE are completed here. """


# In[41]:


"""All codes written below are regarding LABELS AND TITLES. """


import numpy as np
import matplotlib.pyplot as plt
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x,y)
plt.xlabel('Avg Pulse')
plt.ylabel('Calorie Burnage')
plt.show()

plt.plot(x,y)
plt.title("Sports Watch Data")
plt.xlabel("Avg Pulse")
plt.ylabel("Calorie Burnage")
plt.show()

font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}
plt.title("Sports Watch Data",fontdict=font1)
plt.xlabel("Avg Pulse",fontdict = font2)
plt.ylabel("Calorie Burnage",fontdict=font2)
plt.plot(x,y)
plt.show()

plt.title("Sports Watch Data",loc='right')
plt.xlabel("Avg Pulse")
plt.ylabel("Calorie Burnage")
plt.plot(x,y)
plt.show()

"""All codes regarding LABELS AND TITLES are completed here. """


# In[53]:


"""All codes written below are regarding ADDING GRID LINES. """


import numpy as np
import matplotlib.pyplot as plt
x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.title("Sports Watch Data")
plt.xlabel("Avg Pulse")
plt.ylabel("Calorie Burnage")
plt.plot(x,y)
plt.grid()   #Creates into the graph. 
plt.show()

plt.plot(x,y)
plt.grid(axis='x')
plt.show()

plt.plot(x,y)
plt.grid(axis='y')
plt.show()

plt.plot(x,y)
plt.grid(color='green',linestyle='--',linewidth=0.5)  #Custom GRIDS. 
plt.show()

"""All codes regarding ADDING GRID LINES are completed here. """


# In[71]:


"""All codes written below are regarding DISPLAY MULTIPLE PLOTS. """


import numpy as np
import matplotlib.pyplot as plt
x = np.array([0,1,2,3])
y = np.array([3,8,1,10])
plt.subplot(1,2,1)
plt.plot(x,y)
x1 = np.array([0,1,2,3])
y1 = np.array([10,20,30,40])
plt.subplot(1,2,2)
plt.plot(x1,y1)
plt.show()

plt.subplot(2,1,2)
plt.plot(x,y)
plt.subplot(2,1,1)
plt.plot(x1,y1)
plt.show()

x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])
plt.subplot(2, 3, 1)
plt.plot(x,y)
x1 = np.array([0, 1, 2, 3])
y1 = np.array([10, 20, 30, 40])
plt.subplot(2, 3, 2)
plt.plot(x1,y1)
x2 = np.array([0, 1, 2, 3])
y2 = np.array([3, 8, 1, 10])
plt.subplot(2, 3, 3)
plt.plot(x2,y2)
x3 = np.array([0, 1, 2, 3])
y3 = np.array([10, 20, 30, 40])
plt.subplot(2, 3, 4)
plt.plot(x3,y3)
x4 = np.array([0, 1, 2, 3])
y4 = np.array([3, 8, 1, 10])
plt.subplot(2, 3, 5)
plt.plot(x4,y4)
x5 = np.array([0, 1, 2, 3])
y5 = np.array([10, 20, 30, 40])
plt.subplot(2, 3, 6)
plt.plot(x5,y5)
plt.show()

"""All codes regarding SUBPLOT are completed here. """


# In[151]:


"""All codes written below are regarding SCATTER. """


import matplotlib.pyplot as plt
import numpy as np
x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
plt.scatter(x,y)
plt.show()

plt.scatter(x,y)
x1 = np.array([2,2,8,1,15,8,12,9,7,3,11,4,7,14,12])
y1 = np.array([100,105,84,105,90,99,90,95,94,100,79,112,91,80,85])
plt.scatter(x1,y1)
plt.show()

plt.scatter(x,y,color='hotpink')
plt.scatter(x1,y1,color='#88c999')
plt.show

colors = np.array(["red","green","blue","yellow","pink","black","orange","purple","beige","brown","gray","cyan","magenta"])
plt.scatter(x,y,c=colors)
plt.show()

colors = np.array([0,10,20,30,40,45,50,55,60,70,80,90,100])
plt.scatter(x,y,c = colors,cmap='viridis')
plt.show()

colors = np.array([0,10,20,30,40,45,50,55,60,70,80,90,100])
plt.scatter(x,y,c = colors,cmap='Blues_r')  #inorder for cmap to work, the colors must be in numeric values, rather than string values. 
plt.colorbar()
plt.show()

colors = np.array([0,10,20,30,40,45,50,55,60,70,80,90,100])
sizes = np.array([20,50,100,200,500,1000,60,90,10,300,600,800,75])
plt.scatter(x,y,c=colors,s=sizes,cmap='twilight',alpha=0.7)  #ALPHA means he opacity of the markers. 
plt.show()

x = np.random.randint(100, size=(100)) #Random arrays. 
y = np.random.randint(100, size=(100))
colors = np.random.randint(100, size=(100))  #Random colors. 
sizes = 10 * np.random.randint(100, size=(100))  #Random sizes. 
plt.scatter(x, y, c=colors, s=sizes, alpha=0.5, cmap='nipy_spectral')
plt.colorbar()
plt.show()

"""All codes regarding SCATTER are completed here. """


# In[181]:


"""All codes written below are regarding BARS. """


import matplotlib.pyplot as plt
import numpy as np
x = np.array(["A","B","C","D"])
y = np.array([3,8,1,10])
plt.bar(x,y,color='m')
plt.show()

x1 = ["APPLES","BANANAS"]
y1 = [400,350]
plt.bar(x1,y1,color='green')
plt.show()

plt.barh(x,y,color='hotpink')
plt.show()

plt.bar(x,y,width=0.1) #Demonstrates thickness of the bar. 
plt.show()

plt.barh(x,y,height=0.2) #In horizontal bar, height is counted. 
plt.show()

"""All codes regarding BARS are completed here. """


# In[189]:


"""All codes written below are regarding HISTOGRAMS. """


import numpy as np
x = np.random.normal(170,10,250)
print(x)

plt.hist(x,width=0.1)
plt.show()

"""All codes regarding HISTOGRAMS are completed here. """


# In[213]:


"""All codes written below are regarding PIE CHARTS. """


import matplotlib.pyplot as plt
import numpy as np
y = np.array([35,25,25,15])
plt.pie(y)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
y = np.array([35,25,25,15])
mylabels = ["Apples","Bananas","Cherries","Dates"]
plt.pie(y,labels = mylabels)
plt.show()

plt.pie(y,labels = mylabels,startangle = 90)
plt.show()

myexplode=[0.2,0,0,0]  #Explode means to cut a piece of graph into display piece. 
plt.pie(y,labels = mylabels,explode = myexplode)
plt.show()

plt.pie(y,labels = mylabels,explode=myexplode,shadow=True)
plt.show()

mycolors = ["black","hotpink","b","#4CAF50"]
plt.pie(y,labels = mylabels,colors=mycolors)
plt.show()

plt.pie(y,labels=mylabels)
plt.legend()
plt.show()

"""All codes regarding PIE CHARTS are completed here. """


# In[ ]:




