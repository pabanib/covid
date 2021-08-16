# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 11:20:40 2021

@author: paban
"""
import sklearn as sk
import numpy as np

class peri_columna(sk.base.BaseEstimator, sk.base.TransformerMixin):
    "Convierte el dataframe en un array de n filas y t columnas"
    def fit(self, x, y = None):
        return self
    def transform(self, x, y= None):
        return x.unstack().to_numpy()        

class agrega_centroides(sk.base.BaseEstimator, sk.base.TransformerMixin):
    "Agrega los centroides al dataframe"
    def __init__(self, centroides):
        self.centroides = centroides
    
    def coordenadas(self, centroides):
        "centroides debe ser un serie geopandas"
        l = list(centroides.apply(lambda x: list(x.coords)[0]))
        lista = []
        for i in range(len(l)):
            ll = np.array(l[i])
            lista.append(ll)
        coord = np.array(lista)
        return coord
    def fit(self, x, y = None):
        return self
    def transform(self, x, y = None):
        coord = self.coordenadas(self.centroides)
        x = np.c_[x,coord]
        return x
    