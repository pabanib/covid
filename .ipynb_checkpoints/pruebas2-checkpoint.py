# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 16:04:32 2021

@author: paban
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import sklearn as sk

# lectura de datos
covid = gpd.read_file('datos/covid_periodos.shp', index = True)
covid = covid.set_index(['link','mes']).sort_index(level = 0)

# Separamos los campos geometricos del dataframe
geo = covid.loc[pd.IndexSlice[:,1],'geometry']
codiprov = covid.loc[pd.IndexSlice[:,1],['codpcia','departamen','provincia']
                    ]
#  dividimos la cantidad de casos y fallecidos por la población
columnas = ['clasificac', 'fallecido']
covid2 = covid.copy()
for i in columnas:
    covid2[i] = covid[i]/covid.personas*100
covid2 = covid2[columnas]
casos = covid2.iloc[:,0].to_numpy().reshape(525,12) #se selecciona solo una columna del df

# Variables acumuladas a partir del mes que todas tienen al menos 1 

from lq import *

covid_acum = covid[columnas].groupby(covid.index.get_level_values(0)).cumsum()
# buscamos el mes en que todos los dptos tienen al menos 1 contagio
mes = 1
valor = True
while valor == True:
    valor = np.any(covid_acum.loc[pd.IndexSlice[:,mes],'clasificac'] == 0)
    mes +=1
print("El mes desde el cuál todos los dptos tienen al menos 1 contagiado es: "+str(mes))

covid2 = covid_acum.loc[pd.IndexSlice[:,mes:],:]
lq_fall_conf = lq(covid2,'fallecido','clasificac')[2]

## uso de tslearn

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

lq_fall_conf = lq_fall_conf.to_numpy().reshape(525,9,1) 
x = TimeSeriesScalerMeanVariance().fit_transform(lq_fall_conf)

km = TimeSeriesKMeans(n_clusters = 10, metric ='dtw') 

km.fit(x)

from sklearn.metrics import silhouette_score 

silhouette_score(x.reshape(525,9), km.labels_)

covid_rdos = pd.DataFrame(lq_fall_conf.reshape(525,9))

covid_rdos['kmdw'] = km.labels_

pd.DataFrame(covid_rdos.groupby('kmdw').mean().loc[:,0:8]).T.plot()

covid_rdos.groupby('kmdw').describe()

#%%

## Uso de autoencodes

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from sklearn.preprocessing import Normalizer 

norm = Normalizer('l1')
xx = norm.fit_transform(x.reshape(525,9))


pd.DataFrame(xx.mean(axis = 1)).plot(kind = 'hist')
pd.DataFrame(xx.mean(axis = 1)).describe()
# el promedio es 0 en todas las variables, lo que haría que no se agrupen bien

from sklearn.decomposition import PCA
pca = PCA(5)

result = pca.fit_transform(xx)
pca.score(xx)

km.fit(xx)
silhouette_score(xx, km.labels_)

covid_rdos['kmpca'] = km.labels_

pd.DataFrame(covid_rdos.groupby('kmpca').mean().loc[:,0:8]).T.plot()
covid_rdos.groupby('kmpca').count()

covid_rdos.query("kmpca == 2").loc[:,0:8].T.plot(legend = False)
#%%
## autoencoder y kmeans aplicado a los mismos

entrada = layers.Input(shape = (9,))
flat = layers.Flatten(input_shape= [1,9])(entrada)

encoder = layers.Dense(50, activation = 'relu')(entrada)
encoder = layers.Dense(5, activation= 'relu')(encoder)
decoder = layers.Dense(9, activation = 'softmax')(encoder)
model = Model(inputs = entrada, outputs = decoder)
model.compile(optimizer = 'sgd', loss = 'mse')
model.summary()
model.fit(xx.reshape(525,1,9),xx.reshape(525,1,9), epochs = 30)

model.predict(xx.reshape(525,1,9))[0]

enco = Model(entrada, encoder)
codif = enco.predict(xx.reshape(525,1,9))

km.fit(codif)
silhouette_score(codif.reshape(525,5), km.labels_)

covid_rdos['kmauto'] = km.labels_
pd.DataFrame(covid_rdos.groupby('kmpca').mean().loc[:,0:8]).T.plot()
covid_rdos.groupby('kmpca').count()
covid_rdos.query("kmpca == 8").loc[:,0:8].T.plot(legend = False)

#%%
# Trasnformar las variables

dife = np.diff(lq_fall_conf.reshape(525,9))

x = lq_fall_conf
x[0]
dife[0]

c = np.array([3,2,1])
w = np.tri(3)
for i in w:
    print(i.T*c)

(w*c).T

(np.flip(np.tri(3).T*c, axis = 1)).T

(np.tri(len(dife[0]))*dife[0])[:,0]

for i in range(1,len(c)):
    j = -i
    print(c[j])

np.flip(c @ np.eye(3,k= -1))

v = np.array([1,2,4,8])

def matriz_dif(v):
    # v debe ser  la serie temporal
    # calcula una matriz lxl siendo l el largo de la serie completando con las 
    #l-t diferencias en cada periodo t
    l = len(v)
    v = v.reshape(l,)
    d = np.diff(v)
    m = []
    for i in range(l):
        k = (l-1-i)
        vv = d @ np.eye(l-1, k = k)
        m.append(vv)
    m = np.array(m)
    m = np.c_[m,v.reshape(-1,1)]
    return m    


matriz_dif(x[0]).flatten()

x.shape


import tensorflow as tf

class dife_vector(layers.Layer):
    def call(self, x):
        vector = x.numpy()
        vec = matriz_dif(vector)
        
        return tf.constant(vec, dtype = 'float32')
    
pre_layer = dife_vector()

X = []
for i in range(len(x)):
    X.append(matriz_dif(x[i]).flatten())

X = np.array(X).reshape(525,81,1)
X_train = X[:480]
X_valid = X[480:]

entrada = layers.Input(shape = (81,))
#pre = pre_layer()(entrada)
encoder = layers.Dense(50, activation = 'relu')(entrada)
encoder = layers.Dense(9, activation = 'relu')(encoder)
decoder = layers.Dense(50, activation = 'relu')(encoder)
decoder = layers.Dense(81, activation = 'softmax')(decoder)

modelo = Model(inputs = entrada,outputs =  decoder)
modelo.compile(optimizer = 'adam', loss = "mse", metrics=['mse'] ) 

modelo.fit(X_train,X_train, epochs = 30, validation_data = (X_valid,X_valid))

enco = Model(inputs = entrada, outputs = encoder)

X_enco = enco.predict(X)

km = sk.cluster.KMeans(n_clusters = 10)
km.fit(X_enco)

km.inertia_
km.labels_

covid_rdos['kmauto'] = km.labels_
pd.DataFrame(covid_rdos.groupby('kmauto').mean().loc[:,0:8]).T.plot()
covid_rdos.groupby('kmauto').count()
covid_rdos.query("kmauto == 7").loc[:,0:8].T.plot(legend = False)
