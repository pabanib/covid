# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 19:18:49 2021

@author: Pablo
"""
import numpy as np
import elegir_modelo as em
import geopandas as gpd
import os
from lq import *


dir_principal = os.getcwd()
dir_datos = dir_principal+'\\datos'

covid = gpd.read_file(dir_datos+'/covid_periodos.shp', index = True)
covid = covid.set_index(['link','mes']).sort_index(level = 0)
covid = covid.loc[pd.IndexSlice[:,'2020-03':],:]
covid = covid.to_crs('POSGAR94')

columnas = ['clasificac', 'fallecido']

# Variables acumuladas a partir del mes que todas tienen al menos 1 

covid_acum = covid[columnas].groupby(covid.index.get_level_values(0)).cumsum()
# buscamos el mes en que todos los dptos tienen al menos 1 contagio
mes = 0
valor = True
while valor == True:
    Mes = covid.index.get_level_values(1).unique()[mes]
    valor = np.any(covid_acum.loc[pd.IndexSlice[:,Mes],'clasificac'] == 0)
    mes +=1
print("El mes desde el cuál todos los dptos tienen al menos 1 contagiado es: "+str(Mes))
covid_acum['personas'] = covid.personas
covid_ult_mes = covid_acum.loc[pd.IndexSlice[:,'2021-07'],:]
covid_ult_mes = covid_ult_mes.reset_index(level = 'mes', drop = True)
geo = covid.loc[pd.IndexSlice[:,'2021-01'],'geometry']
geo = geo.reset_index(level = 'mes', drop = True)
codiprov = covid.loc[pd.IndexSlice[:,'2021-01'],['codpcia','departamen','provincia']]

import procesos
peri = procesos.peri_columna()

def datos_periodos(bd):
    gdf = gpd.GeoDataFrame(peri.fit_transform(bd))
    meses = []
    for i in list(gdf.columns):
        m = 'mes'+str(i)
        meses.append(m)
    gdf.columns = meses
    gdf = gdf.set_index(codiprov.index.get_level_values(0))
    gdf['personas'] = covid_ult_mes.personas
    gdf['geometry'] = geo
    
    return gdf
  

falle = datos_periodos(covid_acum['fallecido'])

grupos = gpd.read_file('G:/My Drive/Tesis/covid/covid2/df_R/positiv.shp')
grupos = grupos.set_index('link')

falle['sk'] = grupos['sk']

sk = falle.groupby('sk').sum()

s, l, lq_ = lq2(sk, 'mes16', 'personas')

#%%
class region():
    
    def __init__(self, df_reg, poblacion, grupo):
        
        if 'geometry' in df_reg.columns:
            geo = df_reg.geometry
            df_reg = df_reg.drop('geometry', axis = 1)
        self.grupo = df_reg[grupo].unique
        self.poblacion = poblacion
        self.region = df_reg.drop(grupo, axis = 1)
        self.variables =  list(self.region.columns)       
        self.variables.remove(poblacion) 

    def calc_lq(self):
        lambdas = []
        lqs = []
        for i in self.variables:
            s,l,lq_ = lq2(self.region, i, self.poblacion)
            lambdas.append(l)
            lqs.append(lq_)
        lambdas = np.array(lambdas).T
        lqs = np.array(lqs).T
        
        return lambdas, lqs

    def evaluar_var(self):
        self.lambdas, self.lqs = self.calc_lq()
        reg = self.region[self.variables].values
        ssd = ((reg-self.lambdas )**2).mean(axis= 0)
        tssd = ssd.mean()
        return ssd,tssd
        
    def evaluar_prom(self):
        self.lambdas, self.lqs = self.calc_lq()
        reg = self.region[self.variables].values
        ssd = (reg.mean(axis = 1)-self.lambdas.mean(axis = 1))**2
        tssd = ssd.mean()
        return ssd, tssd
        
        
r = region(falle.query("sk == 5"), 'personas','sk')        

l,lq_ = r.calc_lq()

r.evaluar_var()
r.evaluar_prom()

falle_dif = falle.iloc[:,:17].T.diff().T
falle_dif['mes0'] = falle['mes0']
falle_dif['personas'] = falle.personas
falle_dif['sk'] = falle.sk

r = region(falle_dif.query("sk == 3"),'personas','sk')
r.evaluar_prom()
r.evaluar_var()
#%%

total = region(falle_dif, 'personas','sk')

total.evaluar_var()
total.evaluar_prom()

falle_prov = falle_dif.drop('sk', axis = 1)

falle_prov['pcia'] = codiprov['provincia'].values

mza = region(falle_prov.query("pcia == 'Mendoza'"), 'personas', 'pcia')
mza.evaluar_var()
mza.evaluar_prom()
mza.lqs.max()

positiv = datos_periodos(covid_acum['clasificac'])
#positiv = positiv[['mes14']]
positiv = positiv.iloc[:,:17].T.diff().T
positiv = positiv.drop('mes0', axis = 1)
positiv['personas'] = falle.personas
positiv['sk'] = falle.sk

total = region(positiv, 'personas','sk')
total.evaluar_var()
total.evaluar_prom()

r = region(positiv.query("sk== 1"), 'personas','sk')
r.evaluar_var()
r.evaluar_prom()
r.lambdas[0]
r.lqs[0]
r.region.iloc[0]
