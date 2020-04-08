#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Modelagem em tempo real | COVID-19 no Brasil
--------------------------------------------
"""

import pandas as pd
import requests

__author__ = "Mauro Zackiewicz"
__copyright__ = "Copyright 2020"
__license__ = "New BSD License"
__version__ = "1.0.1"
__email__ = "maurozac@gmail.com"
__status__ = "Experimento"


# 1 - preparacao dos dados

raw = pd.read_csv("https://covid.ourworldindata.org/data/ecdc/new_deaths.csv").fillna(0.0)
# raw = pd.read_csv("https://covid.ourworldindata.org/data/ecdc/total_deaths.csv").fillna(0.0)
raw = raw.drop(columns='date')
popu = pd.read_csv("https://covid.ourworldindata.org/data/ecdc/locations.csv").set_index('countriesAndTerritories')


# dados do SEADE => CSV mal formado, url sem ceritifcado valido, sem serie historica
# oh shit, so para dar mais trabalho
# 
# seade = "http://www.seade.gov.br/wp-content/uploads/2020/04/Dados-covid-19-Est-SP.csv"
# sp = pd.read_csv(requests.get(seade, verify=False).text, sep=';', decimal=',')
# a = requests.get(seade, verify=False).text
# sp = pd.read_csv(a, sep=';', decimal=',', lineterminator="\r\n")

# ancorar todos no primeiro dia com 3 mortes => no tamanho do Brasil
# pq usar numero baixo introduz muito ruido
inicio = raw.ge(3).idxmax()

br = raw['Brazil'][inicio['Brazil']:]  # Series
br_n = br.shape[0]

data = pd.DataFrame({'Brazil':br})
data['Brazil'] = data['Brazil'] * (10**6) / popu['population']['Brazil']

for k in inicio.keys():
    if k == "Brazil": continue
    if k not in popu.index: continue
    if inicio[k] == 0 or inicio[k] > inicio["Brazil"]: continue
    C = raw[k][inicio[k]:inicio[k]+br_n]
    data[k] = C.values * (10**6) / popu['population'][k]

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
data.corr()
