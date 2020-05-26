#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Modelo SEIR | Evolução do R0 | COVID-19 no Brasil
--------------------------------------------

Ideias e modelagens desenvolvidas pela trinca:
. Mauro Zackieiwicz
. Luiz Antonio Tozi
. Rubens Monteiro Luciano

Sobre o modelo SEIR
https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model



"""

import datetime

import requests
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
sns.set()
# no ipython usar este comando antes de rodar => %matplotlib osx
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


__author__ = "Mauro Zackiewicz"   # codigo
__copyright__ = "Copyright 2020"
__license__ = "New BSD License"
__version__ = "2.0.1"
__email__ = "maurozac@gmail.com"
__status__ = "Experimental"


def preparar_dados(uf="SP", cidade=u"São Paulo"):
    u"""Busca dados e organiza tabela "data" com os dados de referência para a
    modelagem.
    Fontes:
    . Mundo: https://covid.ourworldindata.org
    . Brasil: https://brasil.io

    Retorna:
    raw <DataFrame> | Série completa do número de mortes/dia por país, sem trans-
        posição temporal
    data <DataFrame> | Série de número de mortes/dia por país trazendo para o
        zero (index 0) o primeiro dia em que ocorrem pelo menos p1 mortes
        (ver macro parâmetros). Isto reduz a quantidade de países para o grupo
        que está à frente ou pareado ao Brazil. A partir do index 0 é possível
        comparar a evolução dos casos entre os países.
    nbr <int> | Número de dias da série de dados para o Brasil

    """
    # ◔◔ {usamos as mortes diárias por parecer ser o dado mais confiável}
    raw = pd.read_csv("https://covid.ourworldindata.org/data/ecdc/new_deaths.csv").fillna(0.0)
    # ◔◔ {o link abaixo carrega o acumulado de mortes, não usamos pq a soma vai alisando a série}
    # raw_soma = pd.read_csv("https://covid.ourworldindata.org/data/ecdc/total_deaths.csv").fillna(0.0)
    tempo = raw['date']
    raw = raw.drop(columns='date')
    raw = raw.drop(columns='World')
    raw['Brasil'] = raw['Brazil']
    # contruir base para a tabela "data"
    inicio = raw.ge(1).idxmax()  # ◔◔ {encontra os index de qdo cada pais alcança p1}
    data = pd.DataFrame({'Brasil':raw['Brasil'][inicio['Brasil']:]}).reset_index().drop(columns='index')
    nbr = data.shape[0]
    dia_0 = tempo[inicio['Brasil']]
    dti = datetime.datetime(2020, int(dia_0[5:7]), int(dia_0[8:10]))
    # dados Brasil
    estados = [
        'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS',
        'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC',
        'SP', 'SE', 'TO',
        ]

    if uf not in estados or type(uf) is not str:
        uf = "SP"
        print(uf, u": UF inválida, calculando para 'SP'")

    popu = {
        "Brasil": 211.0 * 10**6,
    }
    # ◔◔ {já baixamos filtrado para uf, mas pode se usar outros estados}
    uf_data = pd.read_csv("https://brasil.io/dataset/covid19/caso?state="+uf+"&format=csv")

    # adicionar dados da uf
    uf_select = uf_data.loc[lambda df: df['place_type'] == "state", :]
    popu[uf] = uf_select['estimated_population_2019'].tail(1).values[0]
    uf_mortes = list(uf_select['deaths'].head(nbr + 1).fillna(0.0))
    uf_mortes = [uf_mortes[i] - uf_mortes[i+1] for i in range(len(uf_mortes)-1)]
    uf_mortes += [0 for _ in range(nbr-len(uf_mortes))]  # corrigir tamanho
    uf_mortes.reverse()
    data[uf] = pd.Series(uf_mortes).values

    # adicionar dados da cidade
    cidade_select = uf_data.loc[lambda df: df['city'] == cidade, :]
    if cidade_select.shape[0] > 0:
        popu[cidade] = cidade_select['estimated_population_2019'].tail(1).values[0]
        cidade_mortes = list(cidade_select['deaths'].head(nbr + 1).fillna(0.0))
        cidade_mortes = [cidade_mortes[i] - cidade_mortes[i+1] for i in range(len(cidade_mortes)-1)]
        cidade_mortes += [0 for _ in range(nbr-len(cidade_mortes))]  # corrigir tamanho
        cidade_mortes.reverse()
        if sum(cidade_mortes):
            data[cidade] = pd.Series(cidade_mortes).values
        else:
            print(u"AVISO: a cidade " + cidade + " não possui mortes confirmadas")
    else:
        print(u"AVISO: a cidade " + cidade + " não consta nos dados para esta UF")
        print(u'Utilize uma das cidades disponíveis para o terceiro gráfico:')
        for d in set(uf_data['city']):
            print(d)

    refs = ['Brasil', uf, cidade] # as referencias validas...
    # adicionar dados dos países à frente ou pareados ao Brasil
    for k in inicio.keys():
        if k == "Brasil": continue
        if inicio[k] == 0 or inicio[k] > inicio["Brasil"]: continue
        C = raw[k][inicio[k]:inicio[k]+nbr]
        data[k] = C.values

    return raw, data, nbr, refs, dti, popu


#########################   SEIR   ########################################

# https://towardsdatascience.com/social-distancing-to-slow-the-coronavirus-768292f04296

def base_seir_model(init_vals, params, t):
    S_0, E_0, I_0, R_0 = init_vals
    S, E, I, R = [S_0], [E_0], [I_0], [R_0]
    alpha, beta, gamma = params
    """
    α is the inverse of the incubation period (1/t_incubation)
    β is the average contact rate in the population
    γ is the inverse of the mean infectious period (1/t_infectious)
    """
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_S = S[-1] - (beta * S[-1] * I[-1]) * dt
        next_E = E[-1] + (beta * S[-1] * I[-1] - alpha * E[-1]) * dt
        next_I = I[-1] + (alpha * E[-1] - gamma * I[-1]) * dt
        next_R = R[-1] + (gamma * I[-1]) * dt
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)

    return np.stack([S, E, I, R]).T


def ajustar(B, popu, real):
    # parametros
    letal = 0.011  # taxa de mortalidade
    beta = B[0]
    N = popu   # populacao
    alpha = 0.2  # Incubation period = 5 days
    gamma = 0.5  # 1/γ value of 2 days, so γ = 0.5 => vem de um paper: https://arxiv.org/pdf/2002.06563.pdf
    params = (alpha, beta, gamma)

    # valores iniciais para S, E, I, R [normalizados]
    # algumas suposições simplificadoras ocorrem aqui => é uma parte sensivel do modelo
    m = 0  # mortes acumuladas até o dia i
    # m muito baixo => muito ruido na serie inicial de dados
    # m muito alto => suposicoes de partida para os coeficiente do modelo ficam ruins
    for i in range(1, len(real)):
        m = real.head(i).sum()
        if m > N * 0.000001:  # >1 morte por milhão de habitantes
            break

    S_0 = 1 - (m/letal)/N   # susceptiveis: todos - E [aproximado por todos - I]
    E_0 = (m/letal)/N * (beta/gamma)   # E > I => E ~ I*R0
    I_0 = (m/letal)/N   # infectados: mortos/letalidade
    R_0 = 0       # recuperados: ~ 0
    init_vals = (S_0, E_0, I_0, R_0)

    # eixo X
    t_max = 200
    dt = 1
    t = np.linspace(i, t_max, int(t_max/dt) + 1)

    results = base_seir_model(init_vals, params, t)

    M = [x[2]*N*letal for x in results]
    d = len(real) - i

    return abs(sum(real[i:])-sum(M[:d]))
    # return abs(sum(real[i:].rolling(7).mean().fillna(0.0))-sum(M[1:d+1]))

# 
# args = popu['Brasil'], data['Brasil']
# b = minimize(ajustar, np.array([1.75]), args=args, bounds=[(0.1,5)])
# b.x[0]/gamma


def beta_evolution(popu, real):
    rs = []
    for t in range(-30,0,1):
        args = popu, real.head(t)
        b = minimize(ajustar, np.array([1.75]), args=args, bounds=[(0.1,5)])
        rs.append(b.x[0])
    return rs

# beta_evolution(popu['Brasil'], data['Brasil'])


def projetar_SEIR(beta, popu, real, ref):
    # parametros do modelo
    N = popu   # populacao
    letal = 0.011  # taxa de letalidade 1.1% segundo informed guess
    alpha = 0.2  # Incubation period = 5 days
    gamma = 0.5  # 1/γ value of 2 days, so γ = 0.5 => vem de um paper: https://arxiv.org/pdf/2002.06563.pdf
    params = (alpha, beta, gamma)
    # R0 = beta/gamma

    # valores iniciais para S, E, I, R [normalizados]
    # algumas suposições simplificadoras ocorrem aqui => é uma parte sensivel do modelo
    m = 0
    for i in range(1, len(real)):
        m = real.head(i).sum()
        if m > N * 0.000001:
            break

    S_0 = 1 - (m/letal)/N   # susceptiveis: todos - E [aproximado por todos - I]
    E_0 = (m/letal)/N * (beta/gamma)   # E > I => E ~ I*R0
    I_0 = (m/letal)/N   # infectados: mortos/letalidade
    R_0 = 0       # recuperados: 0
    init_vals = (S_0, E_0, I_0, R_0)

    # eixo x
    t_max = 200     # dias
    dt = 1
    t = np.linspace(i, t_max, int(t_max/dt) + 1)

    results = base_seir_model(init_vals, params, t)

    # mortes via modelo => Infectados * popu * letalidade
    M = [x[2]*N*letal for x in results]
    d = len(real) - i
    # qualidade do ajuste
    par = pd.DataFrame([
        M[:d],
        real[i:]
    ])
    co = par.T.corr()[1][0]
    print("[*] Correlação entre Modelo e dados reais (" + ref + "):", co)

    return np.hstack((np.zeros(i) + np.nan, np.array(M)))

## GRAF

def gerar_fig_relatorio(uf, cidade):
    """Roda vários cenários e monta mosaico de gráficos + notas."""
    alisa = 7  # alisamento com média móvel para dados reais
    gamma = 0.5
    notas = u"""

    Fontes dos dados:
        https://covid.ourworldindata.org
        https://brasil.io
    """

    equipe = u'  M.Zac | L.Tozi | R.Luciano || https://github.com/Maurozac/covid-br/blob/master/covid_br_SEIR.py'

    totais = u"""
    Mortes estimadas (no dia)"""

    hoje = str(datetime.datetime.now())[:16]
    fig, ax = plt.subplots(1, 3, figsize=(12, 7), sharex=True, sharey=True)
    fig.suptitle(u"Projeção da epidemia Covid-19" +  " | " + hoje, fontsize=12)
    fig.subplots_adjust(bottom=0.5)
    fig.text(0.33, 0.42, notas, fontsize=7, verticalalignment='top')
    fig.text(0.33, 0.02, equipe, family="monospace", fontsize='6', color='#ff003f', horizontalalignment='left')

    print('[-~-] Coletando dados atualizados das fontes')
    raw, data, nbr, refs, dti, popu = preparar_dados(uf, cidade)
    dtf = dti + datetime.timedelta(days=200)

    print("[-~-] Rodando modelo SEIR (aguarde)")
    for i in [0, 1, 2]:
        if refs[i] == 'n/d':
            ax[i].set_title(u"Dados não disponíveis", fontsize=8)
            break

        ref = refs[i]
        ax[i].set_title(ref, fontsize=8)
        ax[i].set_xlabel(str(dti)[:10]+" (0) a "+str(dtf)[:10]+" (200)", fontsize=8)
        ax[i].set_xlim(0, 220)
        ax[i].set_ylabel(u'Mortes no dia', fontsize=8)
        ax[i].tick_params(labelsize=8)

        # dados modelo SEIR
        args = popu[ref], data[ref]
        b = minimize(ajustar, np.array([1.75]), args=args, bounds=[(0.1,5)], tol=0.01)
        beta = b.x[0]
        Rzero = "R0: "+str(round(beta/gamma, 2))
        M = projetar_SEIR(beta, popu[ref], data[ref], ref)
        ax[i].plot(M, linewidth=3, color="#ff7c7a")
        ax[i].text(10, 2000, Rzero, fontsize=8, verticalalignment="bottom")
        projes = {
            str(datetime.datetime.now()+datetime.timedelta(days=0))[:10]:int(M[nbr-1]),
            str(datetime.datetime.now()+datetime.timedelta(days=7))[:10]:int(M[nbr+7-1]),
            str(datetime.datetime.now()+datetime.timedelta(days=14))[:10]:int(M[nbr+14-1]),
            str(datetime.datetime.now()+datetime.timedelta(days=21))[:10]:int(M[nbr+21-1]),
            str(datetime.datetime.now()+datetime.timedelta(days=28))[:10]:int(M[nbr+28-1]),
            str(datetime.datetime.now()+datetime.timedelta(days=35))[:10]:int(M[nbr+35-1]),
        }
        totais += "\n\n    " + ref + "\n" + "\n".join(["    " + x[0] + ": " + str(x[1]) for x in projes.items()])
        # dados reais
        ax[i].plot(data[ref].rolling(alisa).mean(), linewidth=5, color="#1f78b4")
        # projeções
        for o in [0, 7, 14, 21, 28]:
            ax[i].plot(nbr+o-1, M[nbr+o-1], 'o', markersize=5.0, color="whitesmoke", markeredgecolor="#1f78b4")

    fig.text(0.12, 0.42, totais, fontsize=7, verticalalignment='top', color="#1f78b4")
    # Rs
    axR = fig.add_subplot(position=[0.68, 0.12, 0.22, 0.22])  # [left, bottom, width, height]
    axR.set_title("Evolução do R-zero", fontsize=8)
    axR.set_xlabel("Últimos 30 dias", fontsize=8)
    axR.tick_params(labelsize=8)
    axR.set_xticklabels([str(x) for x in range(35,-1,-5)])

    print("[-~-] Calculando evolução de R-zero (aguarde)")
    paleta = ["#9EC5DE", "#5E9EC9", "#1f78b4"]
    for ref in refs:
        print("[*] "+ref)
        cor = paleta.pop()
        rs = beta_evolution(popu[ref], data[ref])
        Rs = np.array([x/gamma for x in rs])
        axR.plot(Rs, linewidth=2, color=cor)
        axR.text(0, Rs[0], ref, fontsize=8, verticalalignment="top", color=cor)

    return fig

# 
# gerar_fig_relatorio("SP", "São Paulo")
# gerar_fig_relatorio('RJ', "Rio de Janeiro")
# gerar_fig_relatorio('AM', "Manaus")
# 

#########################   RELATORIO   ########################################

def relatorio_hoje(uf, cidade, my_path):
    """Calcula tudo e gera um relatorio em pdf."""
    # gera o dash do dia
    dashboard = gerar_fig_relatorio(uf, cidade)
    # salva em um arquivo pdf
    hoje = str(datetime.datetime.now())[:10]
    pp = PdfPages(my_path+"covid_dashboard_"+uf+"_"+cidade+"_"+hoje+".pdf")
    dashboard.savefig(pp, format='pdf')
    pp.close()


# acerte o caminho para o seu ambiente... esse aí é o meu :-)
my_path = "/Users/tapirus/Desktop/"

# relatorio_hoje("AM", "Manaus", my_path)
relatorio_hoje("SP", "São Paulo", my_path)
