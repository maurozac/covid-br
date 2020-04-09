#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Modelagem em tempo real | COVID-19 no Brasil
--------------------------------------------
"""

import pandas as pd
import requests
from io import StringIO

__author__ = "Mauro Zackiewicz"
__copyright__ = "Copyright 2020"
__license__ = "New BSD License"
__version__ = "1.0.4"
__email__ = "maurozac@gmail.com"
__status__ = "Experimental"


# 1 - preparacao dos dados

raw = pd.read_csv("https://covid.ourworldindata.org/data/ecdc/new_deaths.csv").fillna(0.0)
# raw = pd.read_csv("https://covid.ourworldindata.org/data/ecdc/total_deaths.csv").fillna(0.0)
raw = raw.drop(columns='date')

popu = pd.read_csv("https://covid.ourworldindata.org/data/ecdc/locations.csv").set_index('countriesAndTerritories')
popu["paises"] = [_.replace('_', ' ').replace('United States of America', 'United States') for _ in popu.index]
popu = popu.set_index("paises")

# dados do SEADE => CSV mal formado, url sem ceritifcado valido, sem serie historica
# oh shit, so para dar mais trabalho
#
# seade = "http://www.seade.gov.br/wp-content/uploads/2020/04/Dados-covid-19-Est-SP.csv"
# seade_get = requests.get(seade, verify=False)
# seade_file = StringIO(seade_get.text)
# seade_data = pd.read_csv(seade_file, sep=';', decimal=',')

# 2 - trazendo para a mesma referencia

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

# 3 dados para SP e cidade de SP

sp = pd.read_csv("https://brasil.io/dataset/covid19/caso?state=SP&format=csv")

sp_estado = sp.loc[lambda df: df['place_type'] == "state", :]
SP_estado = list(sp_estado['deaths'].head(br_n + 1).fillna(0.0))
SP_estado = [SP_estado[i] - SP_estado[i+1] for i in range(len(SP_estado)-1)]
SP_estado.reverse()
SP_estado_popu = sp_estado['estimated_population_2019'].max()  # 45919049
data['SP'] = pd.Series(SP_estado).values * (10**6) / SP_estado_popu

br_ex_sp = [x[0]-x[1] for x in zip(list(br), SP_estado)]
data['Brazil ex-SP'] = pd.Series(br_ex_sp).values * (10**6) / (popu['population']['Brazil'] - SP_estado_popu)


sp_city = sp.loc[lambda df: df['city'] == u"São Paulo", :]
SP_city = list(sp_city['deaths'].head(br_n + 1).fillna(0.0))
SP_city = [SP_city[i] - SP_city[i+1] for i in range(len(SP_city)-1)]
SP_city.reverse()
SP_city_popu = sp_city['estimated_population_2019'].max()  # 12252023
data['SP_City'] = pd.Series(SP_city).values * (10**6) / SP_city_popu


# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html
pearson = data.corr()
print(pearson['Brazil'].sort_values(ascending=False))
print(pearson['Brazil ex-SP'].sort_values(ascending=False))
print(pearson['SP'].sort_values(ascending=False))
print(pearson['SP_City'].sort_values(ascending=False))
