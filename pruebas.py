# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 22:22:44 2021

@author: paban
"""

import pandas as pd
import sklearn as sk
import geopandas as gpd 
import pysal as ps 
import numpy as np
import matplotlib.pyplot as plt
import pickle 

#covid = pd.read_csv('D:/MisDocumentos/Documentos/Bases de datos/Covid19Casos/Covid19Casos.csv')

Arg = gpd.read_file('shape_files/pxdptodatosok.shp')

with open("datos/covid_geo.pickle", "rb") as f:
    covid_geo = pickle.load(f)

covid_geo.dtypes
#%%
Arg.plot()
Arg.query("provincia == 'Tierra del Fuego'").departamen

Arg.query("provincia == 'Tierra del Fuego'")[['departamen','link']]
Arg = Arg.drop(Arg.query("link in('94028','94021')").index, axis = 0) 

Arg.plot()

Arg['link'] = Arg['link'].astype(int)
Arg['mujeres'] = Arg['mujeres'].astype(int) 
Arg['varones'] = Arg['varones'].astype(int)
Arg['personas'] = Arg['personas'].astype(int)
Arg['hogares'] = Arg['hogares'].astype(int)
Arg['viv_part'] = Arg['viv_part'].astype(int)
Arg['viv_part_h'] = Arg['viv_part_h'].astype(int)

claves = ['residencia_dpto','fecha_apertura']
atributos = ['clasificacion_resumen_Confirmado','fallecido']


# Separar por periodos

agrupa = covid_geo.groupby(['residencia_dpto', covid_geo.fecha_apertura.dt.month])

covid_geo = agrupa[atributos].sum()

with open("datos/covid_geo_agrup.pickle", "wb") as f:
    pickle.dump(covid_geo, f)

index = covid_geo.index
idx = pd.IndexSlice

covid_periodos = {}

for i in index.get_level_values(1):
    dic = {}
    df = pd.merge(covid_geo.loc[idx[:,i],:], Arg['link'],how= 'right' ,left_on= 'residencia_dpto', right_on = 'link')
    df = df.fillna(0)
    df['link'] = df['link'].astype(int)
    df = pd.merge(df, Arg,how= 'right' ,left_on= 'link', right_on = 'link')
    dic['df'] = gpd.GeoDataFrame(df)
    covid_periodos[i] = dic
    del(dic)

covid_periodos.keys()
for k in covid_periodos.keys():
    print(len(covid_periodos[k]['df']))

for k in covid_periodos.keys():
    print(covid_periodos[k]['df'].isnull().values.any())

with open("datos/covid_periodos.pickle", "wb") as f:
    pickle.dump(covid_periodos, f)

#%% coeficiente de localización

import scipy.stats as st
def lq(datos, campo, total):
    ind = datos[campo]/datos[total]
    indg = datos[campo].sum()/datos[total].sum()
    return [ind,indg,ind/indg]

def intervalos(data, campo, total, a = .95):
    indices = lq(data, campo,total) 
    pi = indices[0]
    p = indices[1]
    ni = data[total]
    n = data[total].sum()
    
    var = (pi*(1-pi)/(ni*p**2))+((pi**2)*(1-p)/(n*p**3))+(2*(pi**2)*(1-pi)/(n*p**3))
    sd = np.sqrt(var)
    sd
    np.mean(indices[2])
    sdnorm = st.t.ppf((1+a)/2,2)*sd
    #sdnorm = st.norm.ppf((1+a)/2)*sd
    
    return pd.DataFrame([indices[2]-sdnorm,indices[2]+sdnorm.T,sdnorm,indices[2]]).T 

for k in covid_periodos.keys():
    df = covid_periodos[k]['df']
    #df_agrup = df.groupby('reg_esp').sum()
    covid_periodos[k]['lq'] = lq(df,'fallecido','personas')[2].sort_values()
    covid_periodos[k]['intervalos'] = intervalos(df,'fallecido','personas').round(2)


for k in covid_periodos.keys():
    print(covid_periodos[k]['lq'].describe())

lq(covid_periodos[10]['df'], 'fallecido', 'personas')

covid_periodos[10]['df']['lq_fallecidos'] = covid_periodos[10]['lq']
datos = covid_periodos[10]['df']

datos.plot(column = 'lq_fallecidos',   scheme = 'Quantiles', legend = False,k = 10 )

largo = 2# len(covid_periodos.keys())
fig, ax = plt.subplots(1,largo, figsize = (10,8))
j = 0
for k in [10,11]:# covid_periodos.keys():
    interv = covid_periodos[k]['intervalos']
    ax[j].errorbar(interv.index, interv[[3]].values, yerr= interv[[2]].values.reshape(525), fmt= ".", label = 'Fallecidos')
    ax[j].grid()
    ax[j].legend()
    ax[j].set_title('Intervalos de confianza LQ {}'.format(k))
    ax[j].set_xlabel('N° región')
    ax[j].set_ylabel('LQ')
    j += 1



#%%
for k in covid_periodos.keys():
    df = covid_periodos[k]['df']
    #df_agrup = df.groupby('reg_esp').sum()
    df['lq_confirmados'] = lq(df,'clasificacion_resumen_Confirmado','personas')[2]
    df['lq_fallecidos'] = lq(df,'fallecido','personas')[2]
    covid_periodos[k]['df'] = df

covid_periodos[1]['df']

covid = []
for k in covid_periodos.keys():
    covid_periodos[k]['df']['mes'] = k
    covid.append(covid_periodos[k]['df'])

covid = pd.concat(covid)
covid = covid.set_index(['link','mes'])

index = covid.index

covid = covid.join(index.to_frame())
covid.to_file('datos/covid_periodos.shp', index = True)
# comportamiento del coeficiente de localización por departamento
covid.groupby(index.get_level_values(1)).max()[['lq_confirmados', 'lq_fallecidos']].plot(title = 'Valor máximo de lq por mes')
covid.groupby(index.get_level_values(1)).mean()[['lq_confirmados', 'lq_fallecidos']].plot(title = 'Valor promedio de lq por mes')

# casos sobre población por mes
covid_sum = covid.groupby(index.get_level_values(1))[['clasificacion_resumen_Confirmado','fallecido','personas']].sum()
(covid_sum[['clasificacion_resumen_Confirmado', 'fallecido']]/covid_sum.personas.iloc[0]).plot(kind = 'bar')
(covid_sum['fallecido']/covid_sum.personas.iloc[0]).plot()

# lq agrupado por zona geográfica
pd.DataFrame(covid.groupby(index.get_level_values(0))[['lq_confirmados','lq_fallecidos']].mean()).plot(kind = 'hist', bins = 20)
pd.DataFrame(covid.groupby(index.get_level_values(0))[['lq_confirmados','lq_fallecidos']].max()).plot(kind = 'hist', bins = 20)

covid_prom = covid.groupby(index.get_level_values(0))[['lq_confirmados','lq_fallecidos']].mean()
covid_prom['geometry'] = covid.groupby(index.get_level_values(0))['geometry'].unique()
covid_prom.plot(column = 'lq_confirmados')

#%%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
pipeline = Pipeline(
    [('escalado', StandardScaler())]
    )

kmeans = sk.cluster.KMeans(n_clusters = 150)

 
class clustering():
    def __init__(self,model, metric):
        self.model = model
        self.metric = metric
     
    def fit(self, data):
        try:
            self.attributes = data.columns
        except:
            self.attributes = None
        data_transf = pipeline.fit_transform(data)
        self.data_transf = data_transf
        self.result = self.model.fit(data_transf)
        
        return self.metric(data_transf, self.model.labels_)

mod1 = clustering(kmeans, sk.metrics.silhouette_score)

datos = np.array([covid['lq_confirmados'], index.get_level_values(1)]).reshape(-1,2)
mod1.fit(datos)
mod1.model.inertia_

covid['kmeans'] = mod1.model.labels_
pd.DataFrame(covid.groupby('kmeans')['lq_confirmados'].mean()).plot()
pd.DataFrame(covid.groupby('kmeans')['lq_confirmados'].count())#.plot(kind = 'bar')


#%%

# matriz espacial
from libpysal.weights import Queen
for k in covid_periodos.keys():
    df = covid_periodos[k]['df']
    w_queen = Queen.from_dataframe(df)
    covid_periodos[k]['w_queen'] = w_queen


aglo_esp = sk.cluster.AgglomerativeClustering(linkage = 'ward',
                                              connectivity= covid_periodos[10]['w_queen'].sparse,
                                              n_clusters = 8
                                              )

mod2 = clustering(aglo_esp,sk.metrics.silhouette_score )

datos = (covid_periodos[10]['df']['fallecido']/covid_periodos[10]['df']['personas']).values.reshape(-1,1)
mod1.fit(datos)
mod2.fit(datos)
df = covid_periodos[10]['df']
df['aglo_esp'] = mod2.model.labels_

df.plot(column = 'aglo_esp')

df.groupby('aglo_esp').describe()['fallecido']
lq(df.groupby('aglo_esp').sum(),'fallecido','personas')[2]

df.query("aglo_esp == 3")['link']

Arg.query("link == 58021")[['departamen','provincia','personas']]

int = intervalos(df.groupby('aglo_esp').sum(),'fallecido','personas')
int.sort_values(1)

covid_periodos[10]['w_queen']


def isdisjunt(maxA, minB):
    if maxA < minB:
        return True
    else:
        return False

ord = int.sort_values(1)[[0,1]]

int = int[[0,1]]
list = []
for i in int.index:
    
    for j in int.index[i+1:]:
        #print(int.iloc[i],int.iloc[j])
        list.append(isdisjunt( max(int.iloc[i],), min(int.iloc[j]))) 
int.iloc[0]
int
sum(list)/len(list)

solos = []
for i in int.index:
    s = 1
    for j in int.index[i:]:
        if j == i:
            pass
        else:
            d =isdisjunt( max(int.iloc[i],), min(int.iloc[j]))
            s = s*d
    solos.append(s)

solos
int
