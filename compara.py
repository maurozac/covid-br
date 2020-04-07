#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Modelagem em tempo real | COVID-19 no Brasil
--------------------------------------------
"""

import pandas as pd


__author__ = "Mauro Zackiewicz"
__copyright__ = "Copyright 2020"
__license__ = "New BSD License"
__version__ = "1.0.1"
__email__ = "maurozac@gmail.com"
__status__ = "Experimento"


# 1 - preparacao dos dados

# raw = pd.read_csv("https://covid.ourworldindata.org/data/ecdc/total_deaths.csv").fillna(0.0)
raw = pd.read_csv("https://covid.ourworldindata.org/data/ecdc/new_deaths.csv").fillna(0.0)
raw = raw.drop(columns='date')

# ancorar todos no primeiro dia com 3 mortes => no tamanho do Brasil
# pq usar numero baixo introduz muito ruido
inicio = raw.ge(3).idxmax()

br = raw['Brazil'][inicio['Brazil']:]  # Series
br_n = br.shape[0]

br = pd.DataFrame({'Brazil':br})

for k in inicio.keys():
    if k == "Brazil": continue
    if inicio[k] == 0 or inicio[k] > inicio["Brazil"]: continue
    C = raw[k][inicio[k]:inicio[k]+br_n]
    br[k] = C.values

br.corr()
