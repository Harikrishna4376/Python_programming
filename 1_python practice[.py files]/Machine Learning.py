#!/usr/bin/env python
# coding: utf-8

# In[9]:


"""All codes written below are regarding MACHINE LEARNING. """
"""All codes written below are regarding MEAN MEDIAN MODE. """


import numpy as np
speed = [99,86,87,88,111,86,103,87,94,78,77,85,86]
x = np.mean(speed)
print(x)

x = np.median(speed)
print(x)

from scipy import stats
x = stats.mode(speed)
print(x)

"""All codes regarding MEAN MEDIAN MODE are completed here. """


# In[19]:


"""All codes written below are regarding STANDARD DEVIATION. """


import numpy as np
speed = [86,87,88,86,87,85,86]
x = np.std(speed)
print(x)

speed = [32,111,138,28,59,77,97]
x = np.std(speed)
print(x)

x = np.var(speed)
print(x)

"""All codes regarding STANDARD DEVIATION are completed here. """


# In[27]:


"""All codes written below are regarding PERCENTILES. """


ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
x = np.percentile(ages,75)
print(x)

x = np.percentile(ages,90)
print(x)

"""All codes regarding PERCENTILE are completed here. """


# In[9]:


"""All codes written below are regarding DATA DISTRIBUTION. """


import numpy as np
x = np.random.uniform(0.0,5.0,25)
print(x)

import numpy as np
import matplotlib.pyplot as plt
x = np.random.uniform(0.0,5.0,250)
plt.hist(x,25)
plt.show()

x = np.random.uniform(0.0,5.0,100000)
plt.hist(x,100)
plt.show()

"""All codes regarding DATA DISTRIBUTION are completed here. """


# In[15]:


"""All codes written below are regarding NORMAL DATA DISTRIBUTION. """


import numpy as np
import matplotlib.pyplot as plt
x = np.random.normal(5.0,1.0,100000)
plt.hist(x,100)
plt.show()

"""All codes regrding NORMAL DATA DISTRIBUTION are completed here. """


# In[21]:


"""All ocdes written below are regarding SCATTER PLOT. """


import matplotlib.pyplot as plt
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
plt.scatter(x,y)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
x = np.random.normal(5.0,1.0,1000)
y = np.random.normal(10.0,2.0,1000)
plt.scatter(x,y)
plt.show()

"""All codes regarding SCATTER PLOT are completed here. """


# In[47]:


"""All codes written below are regarding LINEAR REGRESSION. """


import matplotlib.pyplot as plt
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]
plt.scatter(x, y)
plt.show()

import matplotlib.pyplot as plt
from scipy import stats
slope,intercept,r,p,std_err = stats.linregress(x,y)
def func(x):
    return slope * x + intercept
mymodel=list(map(func,x))
plt.scatter(x,y)
plt.plot(x,mymodel)
plt.show()

slope,intercept,r,p, std_err = stats.linregress(x,y)
print(r)

slope,intercept,r,p,std_err = stats.linregress(x,y)
def func(x):
    return slope * x + intercept
speed = func(10)
print(speed)  #Predicts the speed of 10 year old car. 

x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]
slope,intercept,r,p,std_err = stats.linregress(x,y)
def func(x):
    return slope * x + intercept
mymodel = list(map(func,x))
plt.scatter(x,y)
plt.plot(x,mymodel)
plt.show()

import numpy as np
from scipy import stats
x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]
slope,intercept,r,p,str_err = stats.linregress(x,y)
print(r)

"""All codes regarding LINEAR REGRESSION are completed here. """


# In[73]:


"""All codes written below are regarding POLYNOMIAL REGRESSION. """


import matplotlib.pyplot as plt
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]
plt.scatter(x,y)
plt.show()

mymodel = np.poly1d(np.polyfit(x,y,3))
myline = np.linspace(1,22,100)
plt.scatter(x,y)
plt.plot(myline,mymodel(myline))
plt.show()

from sklearn.metrics import r2_score
print(r2_score(y,mymodel(x)))

speed = mymodel(17)
print(speed)

x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]
mymodel = np.poly1d(np.polyfit(x, y, 3))
myline = np.linspace(2, 95, 100)
plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()

print(r2_score(y,mymodel(x)))

"""All codes regarding POLYNOMIAL REGRESSION are completed here. """


# In[107]:


"""All codes written below are regarding MULTIPLE REGRESSION. """

import pandas as pd
df = pd.read_csv('data4.csv')
print(df)

import pandas as pd
from sklearn import linear_model
df = pd.read_csv("data4.csv")
x = df[['Weight','Volume']]
y = df['CO2']
model = linear_model.LinearRegression()
model.fit(x,y)
prediction = model.predict([[2300,1300]])
print(prediction)

df = pd.read_csv('data4.csv')
x = df[['Weight','Volume']]
y = df['CO2']
model = linear_model.LinearRegression()
model.fit(x,y)
print(model.coef_)

df = pd.read_csv('data4.csv')
x = df[['Weight','Volume']]
y = df['CO2']
model = linear_model.LinearRegression()
model.fit(x,y)
prediction = model.predict([[3300,1300]])
print(prediction)

"""all codes regarding MULTIPLE REGRESSION are completed here. """


# In[119]:


"""All codes written below are regarding SCALE. """


import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
df = pd.read_csv('data4.csv')
x = df[['Weight','Volume']]
scaledx = scale.fit_transform(x)
print(scaledx)

df = pd.read_csv('data4.csv')
x = df[['Weight','Volume']]
y = df['CO2']
scaledx = scale.fit_transform(x)
model = linear_model.LinearRegression()
model.fit(x,y)
scaled = scale.transform([[2300,1.3]])
prediction = model.predict([scaled[0]])
print(prediction)

"""All codes regarding SCALE are completed here. """


# In[151]:


"""All codes written below are regarding TRAIN/TEST. """


import numpy
import matplotlib.pyplot as plt
x = np.random.normal(3,1,100)
y = np.random.normal(150,40,100) / x
plt.scatter(x,y)
plt.show()

train_x = x[:80]
train_y = y[:80]
test_x = x[80:]
test_y = y[80:]
plt.scatter(train_x,train_y)
plt.show()

plt.scatter(test_x,test_y)
plt.show()

np.random.seed(2)
x = np.random.normal(3,1,100)
y = np.random.normal(150,40,100) / x
train_x = x[:80]
train_y = y[:80]
test_x = x[80:]
test_y = y[80:]
model = np.poly1d(numpy.polyfit(train_x,train_y,4))
myline = np.linspace(0,6,100)
plt.scatter(train_x,train_y)
plt.plot(myline,mymodel(myline))
plt.show()

np.random.seed(2)
x = np.random.normal(3,1,100)
y = np.random.normal(150,40,100) / x
train_x = x[:80]
train_y = y[:80]
test_x = x[80:]
test_y = y[80:]
model = np.poly1d(np.polyfit(train_x,train_y,4))
r2 = r2_score(train_y,model(train_x))
print(r2)

"""All codes regarding TRAIN/TEST are completed here. """


# In[93]:


import pandas as pd

# Create a DataFrame with the provided data
data = {
    "Car": ["Toyota", "Mitsubishi", "Skoda", "Fiat", "Mini", "VW", "Skoda", "Mercedes", "Ford", "Audi", "Hyundai", "Suzuki", 
            "Ford", "Honda", "Hundai", "Opel", "BMW", "Mazda", "Skoda", "Ford", "Ford", "Opel", "Mercedes", "Skoda", 
            "Volvo", "Mercedes", "Audi", "Audi", "Volvo", "BMW", "Mercedes", "Volvo", "Ford", "BMW", "Opel", "Mercedes"],
    "Model": ["Aygo", "Space Star", "Citigo", "500", "Cooper", "Up!", "Fabia", "A-Class", "Fiesta", "A1", "I20", "Swift", 
              "Fiesta", "Civic", "I30", "Astra", "1", "3", "Rapid", "Focus", "Mondeo", "Insignia", "C-Class", "Octavia", 
              "S60", "CLA", "A4", "A6", "V70", "5", "E-Class", "XC70", "B-Max", "2", "Zafira", "SLK"],
    "Volume": [1000, 1200, 1000, 900, 1500, 1000, 1400, 1500, 1500, 1600, 1100, 1300, 1000, 1600, 1600, 1600, 1600, 2200, 
               1600, 2000, 1600, 2000, 2100, 1600, 2000, 1500, 2000, 2000, 1600, 2000, 2100, 2000, 1600, 1600, 1600, 2500],
    "Weight": [790, 1160, 929, 865, 1140, 929, 1109, 1365, 1112, 1150, 980, 990, 1112, 1252, 1326, 1330, 1365, 1280, 
               1119, 1328, 1584, 1428, 1365, 1415, 1415, 1465, 1490, 1725, 1523, 1705, 1605, 1746, 1235, 1390, 1405, 1395],
    "CO2": [99, 95, 95, 90, 105, 105, 90, 92, 98, 99, 99, 101, 99, 94, 97, 97, 99, 104, 104, 105, 94, 99, 99, 99, 99, 
            102, 104, 114, 109, 114, 115, 117, 104, 108, 109, 120]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save DataFrame to a CSV file, overwriting if it already exists
df.to_csv('data4.csv', index=False)

print("Data written to data4.csv")



# In[153]:


import pandas as pd

# Define the data in a dictionary
data = {
    "Age": [36, 42, 23, 52, 43, 44, 66, 35, 52, 35, 24, 18, 45],
    "Experience": [10, 12, 4, 4, 21, 14, 3, 14, 13, 5, 3, 3, 9],
    "Rank": [9, 4, 6, 4, 8, 5, 7, 9, 7, 9, 5, 7, 9],
    "Nationality": ["UK", "USA", "N", "USA", "USA", "UK", "N", "UK", "N", "N", "USA", "UK", "UK"],
    "Go": ["NO", "NO", "NO", "NO", "YES", "NO", "YES", "YES", "YES", "YES", "NO", "YES", "YES"]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save DataFrame to a CSV file, overwriting if it already exists
df.to_csv('data5.csv', index=False)

print("Data written to data5.csv")


# In[167]:


"""All codes written below are regarding DECISION TREES. """


import pandas as pd
df = pd.read_csv('data5.csv')
print(df)

d = {'UK':0,"USA":1,'N':2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES':1,'NO':0}
df['Go'] = df['Go'].map(d)
print(df)

features = ['Age','Experience','Rank','Nationality']
x = df[features]
y = df['Go']
print(x)
print(y)

import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
df = pd.read_csv('data5.csv')
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)
features = ['Age','Experience','Rank','Nationality']
x = df[features]
y = df['Go']
dtree = DecisionTreeClassifier()
dtree = dtree.fit(x,y)
tree.plot_tree(dtree,feature_names = features)

print(dtree.predict([[40,10,7,1]]))

print(dtree.predict([[40,10,6,1]]))

"""All codes regarding DECISION TREE are completed here. """


# In[179]:


"""All codes written below are regarding CONFUSION MATRIX. """


import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
actual = np.random.binomial(1,.9,size=1000)
predicted = np.random.binomial(1,.9,size=1000)
confusion_matrix = metrics.confusion_matrix(actual,predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels = [0,1])
cm_display.plot()
plt.show()

Accuracy = metrics.accuracy_score(actual,predicted)

Precision = metrics.precision_score(actual,predicted)

Sensitivity_recall = metrics.recall_score(actual,predicted)

Specificity = metrics.recall_score(actual,predicted,pos_label=0)

F1_score = metrics.f1_score(actual,predicted)

print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"Specificity":Specificity,"F1_score":F1_score})


"""All codes regarding CONFUSION MATRIX are completed here. """


# In[201]:


"""All codes written below are regarding HIERARCHICAL CLUSTERING. """


import numpy as np
import matplotlib.pyplot as plt
x = [4,5,10,4,3,11,14,6,10,12]
y =[21,19,24,17,16,25,24,22,21,21]
plt.scatter(x,y)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram,linkage
data = list(zip(x,y))
linkage_data = linkage(data,method='ward',metric = 'euclidean')
dendrogram(linkage_data)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
data = list(zip(x, y))
hierarchical_cluster = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels = hierarchical_cluster.fit_predict(data)
plt.scatter(x, y, c=labels)
plt.show()

dendrogram(linkage_data)
plt.show()

"""All codes regarding HIERARCHICAL CLUSTERING are completed here. """


# In[211]:


"""All codes written below are regarding LOGISTIC REGRESSION. """


import numpy as np
from sklearn import linear_model
x = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
model = linear_model.LinearRegression()
model.fit(x,y)
prediction = model.predict(np.array([3.46]).reshape(-1,1))
print(prediction)

model = linear_model.LogisticRegression()
model.fit(x,y)
model_odds = model.coef_
odds = np.exp(model_odds)
print(odds)

model = linear_model.LogisticRegression()
model.fit(x,y)
def mod(model,x):
    model_odds = model.coef_ * x + model.intercept_
    odds = np.exp(model_odds)
    probability = odds / (1 + odds)
    return(probability)
print(mod(model,x))

"""All codes regarding LOGISTIC REGRESSION are completed here. """


# In[219]:


"""All codes written below are regarding GRID SEARCH. """


from sklearn import datasets
from sklearn.linear_model import LogisticRegression
iris = datasets.load_iris()
x = iris['data']
y = iris['target']
logit = LogisticRegression(max_iter = 10000)
print(logit.fit(x,y))
print(logit.score(x,y))

iris = datasets.load_iris()
c = [0.25,0.5,0.75,1,1.25,1.5,1.75,2]
scores = []
for choice in c:
    logit.set_params(C=choice)
    logit.fit(x,y)
    scores.append(logit.score(x,y))
print(scores)

"""All codes regarding GRID SEARCH are completed here. """


# In[235]:


"""All codes written below are regarding CATEGORIAL DATA. """


import pandas as pd
cars = pd.read_csv('data4.csv')
print(cars.to_string())

new_cars = pd.get_dummies(cars[['Car']])
print(new_cars)

x = pd.concat([cars[['Volume','Weight']],new_cars],axis=1)
y = cars['CO2']
model = linear_model.LinearRegression()
model.fit(x,y)
prediction = model.predict([[2300,1300,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]])
print(prediction)

import pandas as pd
colors = pd.DataFrame({'color':['blue','red']})
print(colors)

import pandas as pd
colors = pd.DataFrame({'color':['blue','red']})
dummies = pd.get_dummies(colors,drop_first = True)
print(dummies)

dummies['color'] = colors['color']
print(dummies)

"""All codes regarding CATEGORIAL DATA are completed here. """


# In[253]:


"""All codes written below are regarding K_MEANS. """


import matplotlib.pyplot as plt
x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
plt.scatter(x,y)
plt.show()

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]
data = list(zip(x, y))
inertias = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)
plt.plot(range(1, 11), inertias, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)
plt.scatter(x,y,c=kmeans.labels_)
plt.show()

"""All codes regarding K-MEANS are completed here. """


# In[13]:


"""All codes written below are regarding BOOTSTRAP AGGREGATION. """


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
data = datasets.load_wine(as_frame=True)
x = data.data
y = data.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 22)
dtree = DecisionTreeClassifier(random_state = 22)
dtree.fit(x_train,y_train)
y_pred = dtree.predict(x_test)
print("Train data accuracy:",accuracy_score(y_true=y_train,y_pred = dtree.predict(x_train)))
print("Test data accuracy:",accuracy_score(y_true = y_test,y_pred = y_pred))

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
data = datasets.load_wine(as_frame=True)
x = data.data
y = data.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 22)
estimator_range = [2,4,6,8,10,12,14,16]
models = []
scores = []
for n_estimators in estimator_range:
    clf = BaggingClassifier(n_estimators = n_estimators,random_state = 22)
    clf.fit(x_train,y_train)
    models.append(clf)
    scores.append(accuracy_score(y_true = y_test,y_pred = clf.predict(x_test)))
plt.figure(figsize = (9,6))
plt.plot(estimator_range,scores)
plt.xlabel("n_estimators",fontsize = 18)
plt.ylabel('score',fontsize = 18)
plt.tick_params(labelsize = 16)
plt.show()

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
data = datasets.load_wine(as_frame = True)
x = data.data
y = data.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 22)
oob_model = BaggingClassifier(n_estimators = 12,oob_score = True,random_state = 22)
oob_model.fit(x_train,y_train)
print(oob_model.oob_score_)

from sklearn.tree import plot_tree
clf = BaggingClassifier(n_estimators = 12,oob_score = True,random_state = 22)
clf.fit(x_train,y_train)
plt.figure(figsize = (30,20))
plot_tree(clf.estimators_[0],feature_names = x.columns)

"""All codes regarding BOOTSTRAP AGGREGATION are completed here. """


# In[55]:


"""All codes written below are regarding CROSS VALIDATION. """


from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold,cross_val_score
x,y = datasets.load_iris(return_X_y = True)
clf = DecisionTreeClassifier(random_state = 42)
k_folds = KFold(n_splits = 5)
scores = cross_val_score(clf,x,y,cv = k_folds)
print("Cross Validation Scores: ",scores)
print("Avg CV Score: ",scores.mean())
print("Number of CV Scores used in Avg: ",len(scores))

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score
x,y = datasets.load_iris(return_X_y=True)
clf = DecisionTreeClassifier(random_state=42)
sk_folds = StratifiedKFold(n_splits = 5)
scores = cross_val_score(clf,X,y,cv = sk_folds)
print("Cross Validation Scores: ",scores)
print("Avg CV Score: ",scores.mean())
print("Number of CV Scores used in Avg: ",len(scores))

from sklearn.model_selection import LeaveOneOut,cross_val_score
loo = LeaveOneOut()
scores = cross_val_score(clf,X,y,cv = loo)
print("Cross Validation Scores: ",scores)
print("Avg CV Score: ",scores.mean())
print("Number of CV Scores used in Avg: ",len(scores))

from sklearn.model_selection import LeavePOut
lpo = LeavePOut(p=2)
scores = cross_val_score(clf,X,y,cv = lpo)
print("Cross Validation Scores: ",scores)
print("Avg Cv Score; ",scores.mean())
print("Number of Cv Scores used in Avg: ",len(scores))

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit,cross_val_score
X,y = datasets.load_iris(return_X_y = True)
clf = DecisionTreeClassifier(random_state=42)
ss = ShuffleSplit(train_size=0.6,test_size = 0.3,n_splits = 5)
scores = cross_val_score(clf,X,y,cv = ss)
print("Cross Validation Scores: ",scores)
print("Avg CV Score: ",scores.mean())
print("Number of CV Scores used in Avg: ",len(scores))

"""All codes regarding CROSS VALIDATION are completed here. """


# In[83]:


"""All codes written below are regarding AUC-ROC CURVE. """


import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,roc_curve
n = 10000
ratio = .95
n1 = int((1-ratio) * n)
n2 = int(ratio * n)
y = np.array([0] * n1 + [1] * n2)
y_proba = np.array([1]*n)
y_pred = y_proba > .5
print(f'Accuracy Score: {accuracy_score(y,y_pred)}')
cf_mat = confusion_matrix(y,y_pred)
print("Confusion Matrix")
print(cf_mat)
print(f'Class 1 accuracy: {cf_mat[0][0]/n1}')
print(f'Class 2 frequency: {cf_mat[1][1]/n2}')

import matplotlib.pyplot as plt
def plot_roc_curve(true_y,y_prob):
    fpr,tpr,thresholds = roc_curve(true_y,y_prob)
    plt.plot(fpr,tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
plot_roc_curve(y,y_proba)
print(f'model 1 AUC Score: {roc_auc_score(y,y_proba)}')

y_proba_2 = np.array(
    np.random.uniform(0,.7,n1).tolist() +
    np.random.uniform(.3,1,n2).tolist()
)
y_pred_2 = y_proba_2 > .5
print(f'Accuracy Score: {accuracy_score(y,y_pred_2)}')
cf_mat = confusion_matrix(y,y_pred_2)
print('Confusion Matrix')
print(cf_mat)
print(f'Class o Accuracy: {cf_mat[0][0]/n1}')
print(f'Class 1 Accuracy: {cf_mat[1][1]/n2}')
plot_roc_curve(y,y_proba_2)
print(f'Model 2 AUC Score: {roc_auc_score(y,y_proba_2)}')

import numpy as np
n = 10000
y = np.array([0] * n + [1] * n)
y_prob_1 = np.array(
    np.random.uniform(.25, .5, n//2).tolist() +
    np.random.uniform(.3, .7, n).tolist() +
    np.random.uniform(.5, .75, n//2).tolist()
)
y_prob_2 = np.array(
    np.random.uniform(0, .4, n//2).tolist() +
    np.random.uniform(.3, .7, n).tolist() +
    np.random.uniform(.6, 1, n//2).tolist()
)
print(f'Model 1 Accuracy Score: {accuracy_score(y,y_prob_1 > .5)}')
print(f'Model 2 Accuracy Score: {accuracy_score(y,y_prob_2 > .5)}')
print(f'Model 1 AUC Score: {roc_auc_score(y,y_prob_1)}')
print(f'Model 2 AUC Score: {roc_auc_score(y,y_prob_2)}')

plot_roc_curve(y,y_prob_1)

plot_roc_curve(y,y_prob_2)

"""All codes regarding AUC ROC CURVE are completed here. """


# In[115]:


"""All codes written below are regarding K-NEAREST NEIGHBORS. """


import matplotlib.pyplot as plt
x = [4,5,10,4,3,11,14,8,10,12]
y = [21,19,24,17,16,25,24,22,21,21]
classes = [0,0,1,0,0,1,1,0,1,1]
plt.scatter(x,y,c = classes)
plt.show()

from sklearn.neighbors import KNeighborsClassifier
data = list(zip(x,y))
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(data,classes)
new_x = 8
new_y = 21
new_point = [(new_x,new_y)]
prediction = knn.predict(new_point)
plt.scatter(x+[new_x],y + [new_y],c=classes + [prediction[0]])
plt.text(x = new_x-1.7, y=new_y-0.7,s = f"new point, class: {prediction[0]}")
plt.show()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(data,classes)
prediction = knn.predict(new_point)
plt.scatter(x + [new_x],y + [new_y],c = classes + [prediction[0]])
plt.text(x=new_x-1.7,y = new_y-0.7,s = f"New Point,class: {prediction[0]}")
plt.show()

"""All codes regarding KNN are completed here. """
"""All codes regarding MACHINE LEARNING are completed here. """


# In[ ]:




