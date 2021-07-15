# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 10:33:45 2021

@author: paban
"""

import pandas as pd 
import geopandas as gpd

covid = pd.read_csv('D:/MisDocumentos/Documentos/Bases de datos/Covid19Casos/Covid19Casos.csv')

#Arg = gpd.read_file('shape_files/pxdptodatosok.shp')

import preproceso as pre 

gen_datos = pre.preparar_datos(covid)
del(covid)

covid_geo = gen_datos.fit()
covid_geo.dtypes

import pickle 

with open("datos/covid_geo.pickle", "wb") as f:
    pickle.dump(covid_geo, f)
