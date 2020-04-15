#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Modelagem em tempo real | COVID-19 no Brasil
--------------------------------------------

Ideias e modelagens desenvolvidas pela trinca:
. Mauro Zackieiwicz
. Luiz Antonio Tozi
. Rubens Monteiro Luciano

Esta modelagem possui as seguintes características:

a) NÃO seguimos modelos paramétricos => Não existem durante a epidemia dados
suficientes ou confiáveis para alimentar modelos epidemiológicos como a excelente
calaculadora http://gabgoh.github.io/COVID/index.html (ela serve para gerar cená-
rios e para modelar a epidemia DEPOIS que ela passar). Além disso, a natureza
exponencial das curvas as torna extremamente sensíveis aos parâmetros que a defi-
nem. Isso faz com que a confiabilidade preditiva desses modelos seja ilusória.

b) A evolução epidemia no Brasil começou depois da de outros países. Nossa mode-
lagem se apoia nesse fato. Com os dados disponíveis, procuramos no instante pre-
sente determinar quem estamos seguindo, ou seja, que países mais se pareceram
conosco passado o mesmo período de disseminação. A partir do que aconteceu nesses
países projetamos o que pode acontecer aqui.

c) Esta conta é refeita dia a dia. Dependendo de nossa competência em conter ou
não a disseminação do Covid-19 nos aproximaremos dos países que melhor ou pior
lidaram com a epidemia e a projeção refletirá essa similaridade.

d) As decisões de modelagem são indicadas no código com os zoinhos: # ◔◔ {...}
São pontos de partida para discutir a modelagem e propor alternativas.

"""

import datetime

import numpy as np
import pandas as pd
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
__version__ = "1.3.1"
__email__ = "maurozac@gmail.com"
__status__ = "Experimental"


def preparar_dados(p1, p4):
    u"""Busca dados e organiza tabela "data" com os dados de referência para a
    modelagem.
    Fontes:
    . Mundo: https://covid.ourworldindata.org
    . Brasil: https://brasil.io

    Retorna:
    raw <DataFrame> | Série completa do número de mortes/dia por país, sem trans-
        posição temporal
    inicio <Series> | Referência dos indexes em raw para justapor o início das
        curvas dos diferentes países
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
    raw_soma = pd.read_csv("https://covid.ourworldindata.org/data/ecdc/total_deaths.csv").fillna(0.0)
    # tempo = raw['date']  # ◔◔ {não usamos as datas}
    raw = raw.drop(columns='date')

    # correcao de subnotificacao Brazil:
    raw['Brazil'] = raw['Brazil'] * p4

    # dados da população mundo
    # popu = pd.read_csv("https://covid.ourworldindata.org/data/ecdc/locations.csv").set_index('countriesAndTerritories')
    # popu["paises"] = [_.replace('_', ' ').replace('United States of America', 'United States') for _ in popu.index]
    # popu = popu.set_index("paises")

    # dados Brasil
    # ◔◔ {já baixamos filtrado para SP, mas pode se usar outros estados}
    sp = pd.read_csv("https://brasil.io/dataset/covid19/caso?state=SP&format=csv")

    # contruir base para a tabela "data"
    inicio = raw.ge(p1).idxmax()  # ◔◔ {encontra os index de qdo cada pais alcança 3}
    data = pd.DataFrame({'Brazil':raw['Brazil'][inicio['Brazil']:]}).reset_index().drop(columns='index')
    nbr = data.shape[0]

    # adicionar dados de SP
    sp_estado = sp.loc[lambda df: df['place_type'] == "state", :]
    SP_estado = list(sp_estado['deaths'].head(nbr + 1).fillna(0.0))
    SP_estado = [SP_estado[i] - SP_estado[i+1] for i in range(len(SP_estado)-1)]
    SP_estado.reverse()
    # SP_estado_popu = sp_estado['estimated_population_2019'].max()  # 45919049
    data['SP'] = pd.Series(SP_estado).values * p4

    # adicionar dados da cidade de SP
    sp_city = sp.loc[lambda df: df['city'] == u"São Paulo", :]
    SP_city = list(sp_city['deaths'].head(nbr + 1).fillna(0.0))
    SP_city = [SP_city[i] - SP_city[i+1] for i in range(len(SP_city)-1)]
    SP_city.reverse()
    # SP_city_popu = sp_city['estimated_population_2019'].max()  # 12252023
    data['SP_City'] = pd.Series(SP_city).values * p4

    # adicionar dados do Brasil sem SP
    br_ex_sp = [x[0]-x[1] for x in zip(list(data['Brazil']), SP_estado)]
    data['Brazil_sem_SP'] = pd.Series(br_ex_sp).values

    # adicionar dados dos países à frente ou pareados ao Brasil
    for k in inicio.keys():
        if k == "Brazil": continue
        if inicio[k] == 0 or inicio[k] > inicio["Brazil"]: continue
        C = raw[k][inicio[k]:inicio[k]+nbr]
        data[k] = C.values

    return raw, inicio, data, nbr


def rodar_modelo(raw, inicio, data, nbr, p2, p3, ref):
    """
    Usa os dados preparados para gerar dados para visualização e a projeção da
    evoluação da epidemia.

    Retorna:
    correlacionados <list>: Países mais correlacionados, usados para a projeção
    calibrados <DataFrame>: Série alisada de mortes por dia com dados de ref e
        países correlacionados
    projetado <Array>: Série estimada para a evoluação da epidemia em ref
    infos <dict>: informações sobre o pico estimado da epidemia

    """

    # ◔◔ {Optamos por não alisar dados antes de calcular a correlação. Sabemos
    # que a qualidade do report dos dados é variável, mas assumimos que o ruído
    # é aleatório e por isso não é preciso alisar para que a correlação seja
    # válida. Ao contrário, a correlação "bruta" seria a mais verossível}

    # ◔◔ {mas caso você ache que vale a pena alisar antes, use o codigo abaixo}
    # alisamento para os casos de morte reportados (média móvel)
    # data = data.rolling(5).mean()

    # calcular a matriz de correlações:
    pearson = data.corr()
    # ◔◔ {o default do método usa a correlação de Pearson, cf. ref abaixo}
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html

    # ◔◔ {se quiser ver as prévias ...}
    # print(pearson['Brazil'].sort_values(ascending=False))
    # print(pearson['Brazil_sem_SP'].sort_values(ascending=False))
    # print(pearson['SP'].sort_values(ascending=False))
    # print(pearson['SP_City'].sort_values(ascending=False))

    # ◔◔ { não incluir os casos locais para evitar endogeneidade}
    out = ['Brazil', 'Brazil_sem_SP', 'SP', 'SP_City',]  # nao misturar com os demais cortes locais

    # selecionar os p2 países que melhor se correlacionam com a ref
    correlacionados = [_ for _ in pearson[ref].sort_values(ascending=False).keys() if _ not in out][:p2]

    # criar tabela, começa com dados da ref
    calibrados = pd.DataFrame({ref:data[ref]})

    # preencher com os dados dos países correlacionados
    for k in correlacionados:
        # ◔◔ {pega os dados em raw pq agora usaremos todos os dados disponíveis para o país}
        C = raw[k][inicio[k]:]
        additional = pd.DataFrame({k: C.values})  # array
        calibrados = pd.concat([calibrados, additional], axis=1)

    # ◔◔ {aqui usamos um alisamento p3 de dias para deixar a visualização melhor}
    calibrados = calibrados.rolling(p3).mean()

    # ◔◔ {a projeção usa os dados alisados}
    # ◔◔ {como é feita a projeção:
    # 1. cada país correlacionado terá um peso, proporcianal a quanto se correlaciona
    # .. soma dos pesos = 1
    # .. quanto mais correlacionado, maior o peso }
    pesos = [pearson[ref][c] for c in correlacionados]  # melhor corr pesa mais
    pesos = [pesos[i]/sum(pesos) for i in range(len(pesos))]  # pesos normalizados
    pesos = dict(zip(correlacionados, pesos))  # num dict para facilitar

    # proj <list>: vai ter ao final o tamanho da maior serie em calibrados
    proj = [np.nan for _ in range(nbr)]  # começa com nan onde já temos os dados da ref
    proj[-1] =  calibrados[ref][nbr - 1] # primeiro valor coincide com último de ref
    # será a partir daí que começa a projeção

    # ◔◔ {a projeção segue dia a dia as variações dos países correlacionado}
    for d in range(nbr, calibrados.shape[0]):
        x = 0  # incremento estimado para o dia
        for c in correlacionados:
            if not np.isnan(calibrados[c][d]):
                # adiciona o incremento % do país ponderado por seu peso
                x += (calibrados[c][d]/calibrados[c][d-1]) * pesos[c]
            else:
                # ◔◔ {qdo acabam os dados de um país ele pára de influenciar a taxa}
                x += 1 * pesos[c]
            # print(d, c, x)
        # a série da projeção é construída aplicando o incremento estimado ao dia anterior
        proj.append(proj[-1] * x)

    # projetado <Array>
    projetado = np.array(proj)

    # ◔◔ {informações adicionais}
    # pico => valor máximo da série projetada
    pico = np.nan_to_num(projetado).max()  # float
    # mortes valor absoluto
    mortes_no_pico = str(int(pico))  # str
    ix_do_pico = proj.index(np.nan_to_num(projetado).max())  # int => index
    # no caso do pico já ter passado
    if calibrados[ref].max() > pico:
        pico = calibrados[ref].max()
        mortes_no_pico = str(int(pico))
        ix_do_pico = list(calibrados[ref]).index(pico)

    # dia em que acontece o pico
    dia_do_pico = str(datetime.datetime.now() + datetime.timedelta(days=ix_do_pico-nbr))[:10] # str

    # mortes totais: hoje mais tres semanas
    ix_hoje = list(calibrados[ref]).index(calibrados[ref][nbr - 1])
    mortes_totais = {
        str(datetime.datetime.now())[:10]: int(calibrados[ref].sum()),
        str(datetime.datetime.now() + datetime.timedelta(days=7))[:10]: int(calibrados[ref].sum()+projetado[26+1:26+1+7].sum()),
        str(datetime.datetime.now() + datetime.timedelta(days=14))[:10]: int(calibrados[ref].sum()+projetado[26+1:26+1+14].sum()),
        str(datetime.datetime.now() + datetime.timedelta(days=21))[:10]: int(calibrados[ref].sum()+projetado[26+1:26+1+21].sum()),
    }

    # consolidado para output
    infos = {
        "mortes_no_pico": mortes_no_pico,
        "dia_do_pico": dia_do_pico,
        "pico": pico,
        "index": ix_do_pico,
        "mt": mortes_totais,
    }

    return correlacionados, calibrados, projetado, infos


def gerar_grafico(correlacionados, calibrados, projetado, infos):
    """
    Paleta: https://encycolorpedia.com/
    #1f78b4 base: azul #7ba3cd white shade
    #111111 branco
    #ff003f vermelho #ff7c7a white shade
    #000000 preto
    """
    fig, ax = plt.subplots()
    hoje = str(datetime.datetime.now())[:16]
    ax.set_title(u"Evolução da Covid-19 | " + ref + " | " + hoje, fontsize=10)
    ax.set_xlabel(u'Dias desde ' + str(p1) + ' mortes em um dia', fontsize=8)
    ax.set_xlim(0, calibrados.shape[0]+20)
    ax.set_ylabel(u'Mortes por dia', fontsize=8)
    for c in correlacionados:
        ax.plot(calibrados[c], linewidth=3, color="#ff7c7a")
        lvi = calibrados[c].last_valid_index()
        if c == "China": nome = "Wuhan"
        else: nome = c
        ax.text(lvi+1, calibrados[c][lvi], nome, fontsize=6, verticalalignment="center")
    ax.plot(calibrados[ref], linewidth=3, color="#1f78b4")
    ax.plot(projetado, linewidth=2, linestyle=":", color="#1f78b4")
    lvi = pd.Series(projetado).last_valid_index()
    ax.text(lvi+1, projetado[lvi], ref, fontsize=6, verticalalignment="center")
    # ax.legend(calibrados, fontsize=8)
    ax.plot(infos["index"], infos["pico"], '^', markersize=6.0, color="#1f78b4")
    msg = "PICO ~" + infos["mortes_no_pico"] + " mortes em " + infos["dia_do_pico"] + " (subnotificação x" + str(p4) +")"
    ax.text(infos["index"]-2, infos["pico"]+25, msg, fontsize=7, color="#1f78b4")
    fig.text(0.99, 0.01, u'M.Zac | L.Tozi | R.Luciano', family="monospace", fontsize='6', color='#ff003f', horizontalalignment='right')


def gerar_fig_relatorio(p1, p2, p3, p4):
    """Roda vários cenários e monta mosaico de gráficos + notas."""
    # parametros padrao, precisa mudar aqui localmente
    # p1 = 20  # mortes no dia para iniciar série
    # p2 = 3  # número de países mais correlacionados
    # p3 = 7  # alisamento para o gráfico (média móvel)
    # p4 = 12  # correcao por subnotificacao nos dados brasileiros
    ref = ["Brazil", "SP", "SP_City"]

    notas = u"""
    Sobre o modelo e as estimativas:

    As projeções para Brasil, São Paulo e cidade de São Paulo são obtidas a partir da trajetória observada nos três países que melhor
    se correlacionem com a evolução dos nossos dados. O desenho da curva projetada (pontilhada) é reflexo do comportamento observado
    nos países seguidos. Conforme a epidemia avança, em função do nosso desempenho, esse referencial pode mudar (podemos passar a se-
    guir lugares com melhor ou pior desempenho e isso mudará as projeções).

    Os dados brasileiros estão corrigidos pela melhor estimativa disponível para as subnotificações. Esse parâmetro será alterado sempre
    que surgirem estimativas mais precisas. A referência atual de subnotificação de mortes (12 vezes) foi obtida a partir de:
    https://saude.estadao.com.br/noticias/geral,em-um-mes-brasil-tem-alta-de-2239-mortes-por-problemas-respiratorios,70003268759

    Outros parâmetros relevantes:
    => as curvas dos diferentes lugares são emparelhadas a partir do dia em que ocorrem 20 ou mais mortes (metedologia usada pelo El País).
    => as curvas são alisadas com média móvel de 7 dias, por isso não iniciam no dia zero. O alisamento permite melhor visualização das curvas.
    => as projeções são recalculadas diariamente e podem sofrer alterações significativas em função das novas informações incorporadas e do
    aprendizado acumulado.

    Todo o código para gerar este reletório está aberto em: https://github.com/Maurozac/covid-br/blob/master/compara.py
    Contribuições são bem vindas (e você pode regerar as projeções com outros parâmetros para explorar as características do modelo e obter
    projeções para cenários baseados em outras premissas).
    """

    totais = u"""
    Mortes estimadas (acumulado)"""

    hoje = str(datetime.datetime.now())[:16]
    fig, ax = plt.subplots(1, 3, figsize=(12, 6), sharex=True, sharey=True)
    fig.suptitle(u"Projeção da epidemia Covid-19" +  " | " + hoje, fontsize=12)
    fig.subplots_adjust(bottom=0.5)
    fig.text(0.33, 0.42, notas, fontsize=7, verticalalignment='top')
    fig.text(0.33, 0.02, u'  M.Zac | L.Tozi | R.Luciano', family="monospace", fontsize='6', color='#ff003f', horizontalalignment='left')

    for i in [0, 1, 2]:
        raw, inicio, data, nbr = preparar_dados(p1, p4)
        correlacionados, calibrados, projetado, infos = rodar_modelo(raw, inicio, data, nbr, p2, p3, ref[i])

        ax[i].set_title(ref[i], fontsize=8)
        ax[i].set_xlabel(u'Dias desde ' + str(p1) + ' mortes em um dia', fontsize=8)
        ax[i].set_xlim(0, calibrados.shape[0]+20)
        ax[i].set_ylabel(u'Mortes por dia', fontsize=8)
        for c in correlacionados:
            ax[i].plot(calibrados[c], linewidth=3, color="#ff7c7a")
            lvi = calibrados[c].last_valid_index()
            if c == "China": nome = "Wuhan"
            else: nome = c
            ax[i].text(lvi+1, calibrados[c][lvi], nome, fontsize=6, verticalalignment="center")
        ax[i].plot(calibrados[ref[i]], linewidth=3, color="#1f78b4")
        ax[i].plot(projetado, linewidth=2, linestyle=":", color="#1f78b4")
        lvi = pd.Series(projetado).last_valid_index()
        ax[i].text(lvi+1, projetado[lvi], ref[i], fontsize=6, verticalalignment="center")
        # ax.legend(calibrados, fontsize=8)
        ax[i].plot(infos["index"], infos["pico"], '^', markersize=5.0, color="1", markeredgecolor="#1f78b4")
        msg = "PICO ~" + infos["mortes_no_pico"] + " mortes em " + infos["dia_do_pico"] + " (sub x" + str(p4) +")"
        ax[i].text(infos["index"]-2, infos["pico"]+35, msg, fontsize=7, color="#1f78b4")
        totais += "\n\n    " + ref[i] + "\n" + "\n".join(["    " + x[0] + ": " + str(x[1]) for x in infos['mt'].items()])

    fig.text(0.12, 0.42, totais, fontsize=7, verticalalignment='top', color="#1f78b4")

    return fig


#########################   Subnotificações   ##################################

"""
◔◔ {cada fonte abaixo implica em um valor para o coeficiente p4 de ajuste
pela ordem, infos mais recentes ao final

ref: https://noticias.uol.com.br/saude/ultimas-noticias/redacao/2020/04/09/covid-19-declaracoes-de-obito-apontam-48-mais-mortes-do-que-dado-oficial.htm}
p4 = 1.48

https://saude.estadao.com.br/noticias/geral,em-um-mes-brasil-tem-alta-de-2239-mortes-por-problemas-respiratorios,70003268759
extrapolação => 2239 mortes por covid em março nao contabilizadas, de modo que o total ao final do mês
seria de 201 (covid oficial) + 2239 (potencialmente no pior cenário) = 2440
p4 = 12

"""

################   Para estudar e calibrar o modelo   ##########################

# Macro parâmetros
p1 = 20  # mortes no dia para iniciar série
p2 = 3  # número de países mais correlacionados
p3 = 7  # alisamento para o gráfico (média móvel)
p4 = 12  # correcao por subnotificacao nos dados brasileiros
# ◔◔ {ref: https://noticias.uol.com.br/saude/ultimas-noticias/redacao/2020/04/09/covid-19-declaracoes-de-obito-apontam-48-mais-mortes-do-que-dado-oficial.htm}
ref = "Brazil"  # escolher um entre: "SP_City", "SP", "Brazil", "Brazil_sem_SP"

raw, inicio, data, nbr = preparar_dados(p1, p4)
correlacionados, calibrados, projetado, infos = rodar_modelo(raw, inicio, data, nbr, p2, p3, ref)
gerar_grafico(correlacionados, calibrados, projetado, infos)

#########################   RELATORIO   ########################################

# acerte o caminho para o seu ambiente... esse aí é o meu :-)
hoje = str(datetime.datetime.now())[:10]
my_path = "/Users/tapirus/Desktop/covid_dashboard_"+hoje+".pdf"

# gera o dash do dia
dashboard = gerar_fig_relatorio(p1, p2, p3, p4)

# salva em um arquivo pdf
pp = PdfPages(my_path)
dashboard.savefig(pp, format='pdf')
pp.close()
