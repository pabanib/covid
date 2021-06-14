# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 10:33:45 2021

@author: paban
"""

import pandas as pd 

covid = pd.read_csv('D:/MisDocumentos/Documentos/Bases de datos/Covid19Casos/Covid19Casos.csv')
dptos_censo = pd.read_csv('D:/MisDocumentos/Documentos/Bases de datos/Censo 2010/DPTO.csv',delimiter = ';')

covid.head()
covid.columns
covid[covid['fecha_apertura'] == '2021-06-03']
covid_dpt = covid[covid['residencia_departamento_nombre'] == 'Guaymallén']

def completar(n):
    return str(n).zfill(3)

covid['residencia_provincia_id'] = covid['residencia_provincia_id'].astype(str)
covid['residencia_departamento_id'] = covid['residencia_departamento_id'].apply(completar) 
covid['residencia_dpto'] = (covid.residencia_provincia_id+covid.residencia_departamento_id).astype(int)

dptos_censo['DPTO'] = dptos_censo['DPTO'].astype(str)

cruce = pd.merge(covid, dptos_censo, how= "outer", right_on= 'DPTO',left_on = 'residencia_dpto')

cruce.residencia_dpto

no_encontrados = cruce[cruce.NOMDPTO.isna()]

no_encontrados.residencia_departamento_nombre

covid.to_csv('datos/covid.csv', index = False)
