# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 10:33:45 2021

El archvo lee los datos del covid realiza la trasnformaciones de las variables
necesarias para trabajarlas. 
Agrega los datos geográficos y guarda un archivo shp listo para trabajar

@author: paban
"""

import pandas as pd 
import geopandas as gpd
import os

dir_principal = os.getcwd()
dir_datos = dir_principal+'\\datos'
covid = pd.read_csv(dir_datos+'/Covid19Casos.csv')

# Tranforma los datos para poder trabajarlos
import preproceso as pre 
gen_datos = pre.preparar_datos(covid)
del(covid)
covid_geo = gen_datos.fit()

import pickle 
with open("datos/covid_geo.pickle", "wb") as f:
    pickle.dump(covid_geo, f)

# Agrega condiciones espaciales a los datos
Arg = gpd.read_file(dir_principal+'/shape_files/pxdptodatosok.shp')

Arg = Arg.drop(Arg.query("link in('94028','94021')").index, axis = 0) 
Arg['link'] = Arg['link'].astype(int)
Arg['mujeres'] = Arg['mujeres'].astype(int) 
Arg['varones'] = Arg['varones'].astype(int)
Arg['personas'] = Arg['personas'].astype(int)
Arg['hogares'] = Arg['hogares'].astype(int)
Arg['viv_part'] = Arg['viv_part'].astype(int)
Arg['viv_part_h'] = Arg['viv_part_h'].astype(int)

claves = ['residencia_dpto','fecha_apertura']
atributos = ['clasificacion_resumen_Confirmado','fallecido']

periodos = pd.PeriodIndex(covid_geo.fecha_apertura, freq = 'm')
agrupa = covid_geo.groupby(['residencia_dpto', periodos])
covid_geo = agrupa[atributos].sum()

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

# controla que no quede ningún valor nulo
for k in covid_periodos.keys():
    print(covid_periodos[k]['df'].isnull().values.any())

peri = pd.PeriodIndex(covid_periodos.keys())[pd.PeriodIndex(covid_periodos.keys()) > pd.Period('2019-12')]
peri = peri[peri < pd.Period('2021-08')]

covid = []
for k in peri:
    covid_periodos[k]['df']['mes'] = k
    covid.append(covid_periodos[k]['df'])
covid = pd.concat(covid)
covid['mes'] = covid.mes.astype(str)
covid = covid.set_index(['link','mes'])

index = covid.index

covid.to_file('datos/covid_periodos.shp', index = True)

