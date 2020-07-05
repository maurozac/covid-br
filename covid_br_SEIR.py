#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Modelo SEIR | Evolução do R | COVID-19 no Brasil
--------------------------------------------

Ideias e modelagens desenvolvidas pela trinca:
. Mauro Zackieiwicz
. Luiz Antonio Tozi
. Rubens Monteiro Luciano

Sobre o modelo SEIR
https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology#The_SEIR_model

"""

import datetime
from io import StringIO, BytesIO

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
    . Mundo: https://covid.ourworldindata.org  # usa dados federais DEPRECATED
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
    raw['Brasil'] = raw['Brazil']
    # contruir base para a tabela "data"
    inicio = raw.ge(1).idxmax()  # ◔◔ {encontra os index de qdo cada pais alcança p1}
    data = pd.DataFrame({'Brasil':raw['Brasil'][inicio['Brasil']:]}).reset_index().drop(columns='index')
    nbr = data.shape[0]
    dia_0 = tempo[inicio['Brasil']]
    dti = datetime.datetime(2020, int(dia_0[5:7]), int(dia_0[8:10]))
    print("[*] BRASIL total de mortes (ourworldindata, federal):", data['Brasil'].sum())

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
    # down = requests.get("https://brasil.io/dataset/covid19/caso/?state="+uf+"&format=csv")
    # if down.status_code == 200:
    #     uf_data = pd.read_csv(StringIO(down.text))
    # else:
    #     print('[*] FALHA no download de dados | brasil.io')

    # ◔◔ {NEW baixar full, todos os estados}
    # down = requests.get("https://brasil.io/dataset/covid19/caso/?format=csv")  # DEPRECATED
    down = requests.get("https://data.brasil.io/dataset/covid19/caso.csv.gz")
    if down.status_code == 200:
        full_data = pd.read_csv(BytesIO(gzip.decompress(down.content)))
    else:
        print('[*] FALHA no download de dados | brasil.io')

    # dados BRASIL vindo da soma dos estados
    all_select = full_data.loc[lambda df: df['place_type'] == "state", :]

    # adicionar dados da uf
    # uf_select = uf_data.loc[lambda df: df['place_type'] == "state", :]
    uf_select = all_select.loc[lambda df: df['state'] == uf, :]
    popu[uf] = uf_select['estimated_population_2019'].tail(1).values[0]
    uf_mortes = list(uf_select['deaths'].head(nbr + 1).fillna(0.0))
    uf_mortes = [uf_mortes[i] - uf_mortes[i+1] for i in range(len(uf_mortes)-1)]
    uf_mortes += [0 for _ in range(nbr-len(uf_mortes))]  # corrigir tamanho
    uf_mortes.reverse()
    data[uf] = pd.Series(uf_mortes).values

    ### HACK: recalcula data['Brasil'] com dados dos estados ###
    # popu Brasil => já tem
    br_mortes = pd.Series(uf_mortes).values  # começa carregado com os dados da uf
    for u in estados:
        if u == uf: continue  # pula uf
        u_select = all_select.loc[lambda df: df['state'] == u, :]
        u_mortes = list(u_select['deaths'].head(nbr + 1).fillna(0.0))
        u_mortes = [u_mortes[i] - u_mortes[i+1] for i in range(len(u_mortes)-1)]
        u_mortes += [0 for _ in range(nbr-len(u_mortes))]  # corrigir tamanho
        u_mortes.reverse()
        br_mortes += pd.Series(u_mortes).values

    data["Brasil"] = br_mortes   ## em raw permanece o dado federal
    print("[*] BRASIL total de mortes (brasil.io, soma dos estados):", br_mortes.sum())
    #### feito em 08 de junho 2020  ####

    print("[*]", uf, "total de mortes (brasil.io):", data[uf].sum())

    # adicionar dados da cidade
    # cidade_select = uf_data.loc[lambda df: df['city'] == cidade, :]
    cidade_select = full_data.loc[lambda df: df['city'] == cidade, :]
    if cidade_select.shape[0] > 0:
        popu[cidade] = cidade_select['estimated_population_2019'].tail(1).values[0]
        cidade_mortes = list(cidade_select['deaths'].head(nbr + 1).fillna(0.0))
        cidade_mortes = [cidade_mortes[i] - cidade_mortes[i+1] for i in range(len(cidade_mortes)-1)]
        cidade_mortes += [0 for _ in range(nbr-len(cidade_mortes))]  # corrigir tamanho
        cidade_mortes.reverse()
        if sum(cidade_mortes):
            data[cidade] = pd.Series(cidade_mortes).values
            print("[*]", cidade, "total de mortes (brasil.io):", data[cidade].sum())
        else:
            print(u"[*] AVISO: a cidade " + cidade + " não possui mortes confirmadas")
    else:
        print(u"[*] AVISO: a cidade " + cidade + " não consta nos dados para esta UF")

    refs = ['Brasil', uf, cidade] # as referencias validas...

    return raw, data, nbr, refs, dti, popu


# raw, data, nbr, refs, dti, popu = preparar_dados()

#########################   SEIR   ########################################

"""
Parâmetros GLOBAIS do MODELO SEIR

Gamma vem de um paper: https://arxiv.org/pdf/2002.06563.pdf
Letalidade estimada por países que testaram exaustivamente: 1,1%

"""

ALPHA = 0.2  # 1/α Incubation period = 5 days
GAMMA = 0.5  # 1/γ value of 2 days, so γ = 0.5
LETAL = 0.006  # taxa de letalidade 0.7% a 1.1%, sem colapso
SUB = 0.15   # taxa de mortes não notificadas => real = dado/(1-sub) [% por dentro!]
ISOLA = 0.45  # taxa de isolamento => 0: sem isolamento; 1: lockdown completo => reduz população Susceptível
# Não usar ISOLA = 1 => quebra o modelo (div by zero)
relatorio_hoje("SP", "São Paulo", my_path)


def base_seir_model(init_vals, params, t):
    """
    RODA incrementos discretos no sistema de equações diferenciais
    RETORNA numpy stack com shape: (t, 4), com t: número de incrementos (dias) # -*- coding: utf-8 -*-
    para as curvas S, E, I, R.
    Código segundo proposta de:
    https://towardsdatascience.com/social-distancing-to-slow-the-coronavirus-768292f04296
    """
    S_0, E_0, I_0, R_0 = init_vals
    S, E, I, R = [S_0], [E_0], [I_0], [R_0]
    ALPHA, beta, GAMMA = params
    """
    α is the inverse of the incubation period (1/t_incubation)
    β is the average contact rate in the population
    γ is the inverse of the mean infectious period (1/t_infectious)
    """
    dt = t[1] - t[0]
    for _ in t[1:]:
        next_S = S[-1] - (beta * S[-1] * I[-1]) * dt
        next_E = E[-1] + (beta * S[-1] * I[-1] - ALPHA * E[-1]) * dt
        next_I = I[-1] + (ALPHA * E[-1] - GAMMA * I[-1]) * dt
        next_R = R[-1] + (GAMMA * I[-1]) * dt
        S.append(next_S)
        E.append(next_E)
        I.append(next_I)
        R.append(next_R)

    return np.stack([S, E, I, R]).T


def rodar_SEIR(beta, init, popu, real):
    """Prepara dados de entrada do modelo SEIR e o executa.
    Modelos exponenciais são muito sensíveis às condições iniciais,
    as hipóteses e aproximações assumidas aqui são importantes.
    RETORNA: resultado do modelo SEIR
    """
    # parametros
    N = popu * (1-ISOLA)  # populacao, descontados os isolados
    real = real/(1-SUB)     # ajuste para compensar subnotificao das mortes
    params = (ALPHA, beta, GAMMA)
    # valores iniciais para S, E, I, R [normalizados]
    # algumas suposições simplificadoras ocorrem aqui => é uma parte sensivel do modelo
    m = 0  # mortes acumuladas até o dia i
    # m muito baixo => muito ruido na serie inicial de dados
    # m muito alto => suposicoes de partida para os coeficiente do modelo ficam ruins
    for i in range(1, len(real)):
        m = real.head(i).sum()
        if m > N * (init * 0.000001):  # mortes por milhão de habitantes
            break

    S_0 = 1 - (m/LETAL)/N * (beta/GAMMA) - (m/LETAL)/N  # S = N - E - I - R
    E_0 = (m/LETAL)/N * (beta/GAMMA)   # E > I => E ~ I*R
    I_0 = (m/LETAL)/N   # infectados: ~mortos/letalidade
    R_0 = 0       # recuperados: ~0
    init_vals = (S_0, E_0, I_0, R_0)
    # eixo x
    t_max = 200     # dias
    dt = 1
    t = np.linspace(i, t_max, int(t_max/dt) + 1)
    # seir => np.array([S,E,I,R])
    seir = base_seir_model(init_vals, params, t)

    return seir, i


def projetar_curvas_com_SEIR(beta, init, popu, real, ref):
    """Roda modelo e avalia qualidade do ajuste com dados reais
    RETORNA np array com curva modelada para mortes, a correlacao e a curva para
    infectados
    """
    seir, i = rodar_SEIR(beta, init, popu, real)
    # i: dia em que a condição inicial é observada, dia em que começa a simular o SEIR
    # mortes via modelo => Infectados * (popu * isolamento) * letalidade
    M = [x[2]*popu*(1-ISOLA)*LETAL for x in seir]
    I = [x[2]*popu*(1-ISOLA) for x in seir]
    d = len(real) - i  # comprimento dos dados sobrepostos
    # qualidade do ajuste => calculado sobre o trecho sobreposto
    par = pd.DataFrame([
        M[:d],
        real[i:]/(1-SUB),   # nao precisaria corrigir por sub, é transformação linear
    ])
    co = par.T.corr()[1][0]
    print("[*] " + ref)
    print("[.] beta: " + str(round(beta, 4)) +" init: " + str(round(init, 4)))
    print("[.] Correlação entre modelo e dados reais:", round(co, 4))
    # print(i, d, len(M), len(M[:d]), len(real), len(real[i:]))

    # adiciona i células vazias (NAN) no início para alinhar corretamente no eixo x
    mortes_teoricas = np.hstack((np.zeros(i) + np.nan, np.array(M)))
    infect_teorico = np.hstack((np.zeros(i) + np.nan, np.array(I)))

    return mortes_teoricas, co, infect_teorico


def ajustar(B, popu, real):
    """Função no formato esperado por scipy.optimize.minimize para rodar o
    modelo SEIR e buscar ajuste.
    A variável a ser minimizada é a diferença absoluta entre o total de mortes
    observados e previstos => deve tender a zero
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """
    # parametros
    beta = B[0]  # scipy.optimize.minimize requer entrada de np.array([p0,...])
    init = B[1]  # scipy.optimize.minimize requer entrada de np.array([p0,...])
    seir, i = rodar_SEIR(beta, init, popu, real)
    # mortes via modelo => Infectados * popu * letalidade * isolamento
    M = [x[2]*popu*(1-ISOLA)*LETAL for x in seir]
    d = len(real) - i
    real = real/(1-SUB)     # ajuste para compensar subnotificao das mortes
    # variaval a minimizar: soma das mortes no trecho sobreposto e a correlacao
    mini = abs(sum(real[i:]) - sum(M[:d]))
    return mini


# args = popu['Brasil'], data['Brasil']
# b = minimize(ajustar, np.array([1.2, .1]), args=args, bounds=[(0.1,5), (0.01,100)])
# b = minimize(ajustar, np.array([1.2, .1]), args=args, method='Nelder-Mead')
# b.x[0]/GAMMA


def beta_evolution(popu, real, k=30):
    """Roda sucessivos ajuste para determinar beta nos últimos k dias.
    k padrão é 30
    Retorna List com k betas.
    """
    rs = []
    for t in range(-k, 0, 1):
        args = popu, real.head(t)
        b = minimize(ajustar, np.array([1.2, 1.]), args=args, bounds=[(0.1,5), (0.01,100)])
        # b = minimize(ajustar, np.array([1.2, 1.]), args=args, method='Nelder-Mead')
        rs.append(b.x[0])
    return rs


####################   DASHBORD   ################

def gerar_fig_relatorio(uf, cidade):
    """Roda vários cenários e monta mosaico de gráficos + notas."""
    alisa = 7  # alisamento com média móvel para dados reais ficarem mais apresentáveis
    notas = u"""
    Projeções obtidas por modelo epidemiológico (SEIR | Susceptíveis, Expostos,
    Infectados, Recuperados)

    • taxa de isolamento: """+str(int(ISOLA*100))+"""%
    • taxa de letalidade: """+str(round(LETAL*100, 2))+"""%
    • mortes não contabilizadas: """+str(int(SUB*100))+"""%

    Modelo epidemiológico ajustado sobre os últimos dados reais disponíveis para
    estimar o parâmetro R (taxa de transmissão) corrente.

    As projeções dependem do valor de R. Valores de R em queda indicam tendência
    de queda na intensidade das projeções para os próximos dias. E vice-versa.

    Fontes dos dados:
        https://covid.ourworldindata.org
        https://brasil.io
    """

    equipe = u'  M.Zac | L.Tozi | R.Luciano || veja mais em: https://github.com/Maurozac/covid-br/blob/master/covid_br_SEIR.py'

    totais = u"""
    Mortes estimadas (no dia)"""

    hoje = str(datetime.datetime.now())[:16]
    fig, ax = plt.subplots(1, 3, figsize=(12, 7), sharex=True, sharey=True)
    fig.suptitle(u"Projeção da epidemia Covid-19" +  " | " + hoje, fontsize=12)
    fig.subplots_adjust(bottom=0.5)
    fig.text(0.3, 0.42, notas, fontsize=7, verticalalignment='top')
    fig.text(0.3, 0.02, equipe, family="monospace", fontsize='6', color='#ff003f', horizontalalignment='left')

    print('[-~-] Coletando dados atualizados das fontes')
    raw, data, nbr, refs, dti, popu = preparar_dados(uf, cidade)
    dtf = dti + datetime.timedelta(days=200)

    print("[-~-] Ajustando modelo SEIR (aguarde)")
    for i in [0, 1, 2]:
        if refs[i] == 'n/d':
            ax[i].set_title(u"Dados não disponíveis", fontsize=8)
            break

        ref = refs[i]
        ax[i].set_title(ref, fontsize=8)
        ax[i].set_xlabel(str(dti)[:10]+" (0) a "+str(dtf)[:10]+" (200)", fontsize=8)
        ax[i].set_ylabel(u'Mortes no dia', fontsize=8)
        ax[i].tick_params(labelsize=8)

        # dados modelo SEIR
        args = popu[ref], data[ref]
        b = minimize(ajustar, np.array([1.2, 1.]), args=args, bounds=[(0.1,5), (0.01,100)])
        # b = minimize(ajustar, np.array([1.2, 1.]), args=args, method='Nelder-Mead')
        beta = b.x[0]
        init = b.x[1]
        Rzero = "R: "+str(round(beta/GAMMA, 2))
        M, co, I = projetar_curvas_com_SEIR(beta, init, popu[ref], data[ref], ref)

        # preparar e exportar CSV de infectados
        index = pd.to_datetime(dti) + pd.to_timedelta(np.arange(len(I)), 'D')
        pd.Series(I, index=index).to_csv(my_path+ref+"_infectados.csv", header=False)

        # graficos
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
        reais = data[ref]/(1-SUB)     # ajuste para compensar subnotificao das mortes
        ax[i].plot(reais.rolling(alisa).mean(), linewidth=5, color="#1f78b4")
        # projeções
        for o in [0, 7, 14, 21, 28, 35]:
            ax[i].plot(nbr+o-1, M[nbr+o-1], 'o', markersize=5.0, color="whitesmoke", markeredgecolor="#1f78b4")

    fig.text(0.12, 0.42, totais, fontsize=7, verticalalignment='top', color="#1f78b4")
    # Rs
    axR = fig.add_subplot(position=[0.68, 0.12, 0.22, 0.22])  # [left, bottom, width, height]
    axR.set_title("Evolução do R", fontsize=8)
    axR.set_xlabel("Últimos 30 dias", fontsize=8)
    axR.tick_params(labelsize=8)
    axR.set_xticklabels([str(x) for x in range(35,-1,-5)])

    print("[-~-] Calculando evolução do R (aguarde)")
    paleta = ["#9EC5DE", "#5E9EC9", "#1f78b4"]
    for ref in refs:
        print("[*] "+ref)
        cor = paleta.pop()
        rs = beta_evolution(popu[ref], data[ref])
        Rs = np.array([x/GAMMA for x in rs])
        axR.plot(Rs, linewidth=2, color=cor)
        axR.text(1, Rs[0], ref, fontsize=8, verticalalignment="top", color=cor)

    return fig

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
my_path = "/Users/tapirus/Desktop/covid/"

relatorio_hoje("AM", "Manaus", my_path)
relatorio_hoje("PE", "Recife", my_path)
relatorio_hoje("RJ", "Rio de Janeiro", my_path)
relatorio_hoje("PA", "Belém", my_path)
relatorio_hoje("SP", "São Paulo", my_path)
relatorio_hoje("RS", "Porto Alegre", my_path)
relatorio_hoje("CE", "Fortaleza", my_path)
