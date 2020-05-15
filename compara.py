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

import requests
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
__version__ = "1.5.2"
__email__ = "maurozac@gmail.com"
__status__ = "Experimental"


def preparar_dados(p1, uf="SP", cidade=u"São Paulo"):
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
    # raw_soma = pd.read_csv("https://covid.ourworldindata.org/data/ecdc/total_deaths.csv").fillna(0.0)
    # tempo = raw['date']  # ◔◔ {não usamos as datas}
    raw = raw.drop(columns='date')
    raw = raw.drop(columns='World')

    # para ver tbem os dados "oficias"
    para_oficial = raw['Brazil']

    # correcao de subnotificacao Brasil:
    sub, hip = estimar_subnotificacao('Brasil')
    p4br = ((sub + raw['Brazil'].sum()) / raw['Brazil'].sum())
    raw['Brasil'] = raw['Brazil'] * p4br

    # dict subs usa mesmas refs como chave => para reportar nos graficos
    subs = {"Brasil": str(round(p4br, 1)) + " (" + hip + ")"}

    # contruir base para a tabela "data"
    inicio = raw.ge(p1).idxmax()  # ◔◔ {encontra os index de qdo cada pais alcança p1}
    data = pd.DataFrame({'Brasil':raw['Brasil'][inicio['Brasil']:]}).reset_index().drop(columns='index')
    nbr = data.shape[0]

    oficial = pd.DataFrame({'Brasil':para_oficial[inicio['Brasil']:]}).reset_index().drop(columns='index')

    # dados Brasil
    estados = [
        'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS',
        'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC',
        'SP', 'SE', 'TO',
        ]

    if uf not in estados or type(uf) is not str:
        uf = "SP"
        print(uf, u": UF inválida, usando 'SP'")

    # ◔◔ {já baixamos filtrado para uf, mas pode se usar outros estados}
    uf_data = pd.read_csv("https://brasil.io/dataset/covid19/caso?state="+uf+"&format=csv")

    # adicionar dados da uf
    uf_select = uf_data.loc[lambda df: df['place_type'] == "state", :]
    uf_mortes = list(uf_select['deaths'].head(nbr + 1).fillna(0.0))
    uf_mortes = [uf_mortes[i] - uf_mortes[i+1] for i in range(len(uf_mortes)-1)]
    uf_mortes += [0 for _ in range(nbr-len(uf_mortes))]  # corrigir tamanho
    uf_mortes.reverse()
    oficial[uf] = pd.Series(uf_mortes).values

    sub_uf, hip_uf = estimar_subnotificacao(uf)
    p4uf = ((sub_uf + pd.Series(uf_mortes).values.sum())/pd.Series(uf_mortes).values.sum())
    data[uf] = pd.Series(uf_mortes).values * p4uf
    subs[uf] = str(round(p4uf, 1)) + " (" + hip_uf + ")"

    # adicionar dados da cidade
    cidade_select = uf_data.loc[lambda df: df['city'] == cidade, :]
    if cidade_select.shape[0] > 0:
        cidade_mortes = list(cidade_select['deaths'].head(nbr + 1).fillna(0.0))
        cidade_mortes = [cidade_mortes[i] - cidade_mortes[i+1] for i in range(len(cidade_mortes)-1)]
        cidade_mortes += [0 for _ in range(nbr-len(cidade_mortes))]  # corrigir tamanho
        cidade_mortes.reverse()
        if sum(cidade_mortes):
            # subnotificacao para cidade => aprox pela do estado
            oficial[cidade] = pd.Series(cidade_mortes).values
            data[cidade] = pd.Series(cidade_mortes).values * p4uf
            subs[cidade] = str(round(p4uf, 1)) + " (" + hip_uf + ")"
        else:
            subs["n/d"] = ""
            print(u"AVISO: a cidade " + cidade + " não possui mortes confirmadas")
    else:
        subs["n/d"] = ""
        print(u"AVISO: a cidade " + cidade + " não consta nos dados para esta UF")
        print(u'Utilize uma das cidades disponíveis para o terceiro gráfico:')
        for d in set(uf_data['city']):
            print(d)

    refs = list(subs.keys())  # as referencias validas...
    # adicionar dados dos países à frente ou pareados ao Brasil
    for k in inicio.keys():
        if k == "Brasil": continue
        if inicio[k] == 0 or inicio[k] > inicio["Brasil"]: continue
        C = raw[k][inicio[k]:inicio[k]+nbr]
        data[k] = C.values

    return raw, inicio, data, nbr, subs, refs, oficial



def rodar_modelo(raw, inicio, data, nbr, p2, p3, ref, refs):
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
    try: data = data.drop(columns='Brazil')
    except: pass

    # calcular a matriz de correlações:
    pearson = data.corr()
    # ◔◔ {o default do método usa a correlação de Pearson, cf. ref abaixo}
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html

    # ◔◔ { não incluir os casos locais para evitar endogeneidade}
    out = refs # nao misturar com os demais cortes locais

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
    # dia em que acontece o pico [! soma 1 no index pq projetado sobrepoe o primeiro valor]
    dia_do_pico = str(datetime.datetime.now() + datetime.timedelta(days=ix_do_pico-nbr+1))[:10] # str
    # no caso do pico já ter passado
    if calibrados[ref].max() > pico:
        pico = calibrados[ref].max()
        mortes_no_pico = str(int(pico))
        ix_do_pico = list(calibrados[ref]).index(pico)
        dia_do_pico = str(datetime.datetime.now() + datetime.timedelta(days=ix_do_pico-nbr))[:10] # str

    # mortes totais: hoje mais tres semanas
    ix_hoje = list(calibrados[ref]).index(calibrados[ref][nbr - 1])
    mortes_totais = {
        str(datetime.datetime.now())[:10]: int(calibrados[ref].sum()),
        str(datetime.datetime.now() + datetime.timedelta(days=7))[:10]: int(calibrados[ref].sum()+projetado[nbr+1:nbr+1+7].sum()),
        str(datetime.datetime.now() + datetime.timedelta(days=14))[:10]: int(calibrados[ref].sum()+projetado[nbr+1:nbr+1+14].sum()),
        str(datetime.datetime.now() + datetime.timedelta(days=21))[:10]: int(calibrados[ref].sum()+projetado[nbr+1:nbr+1+21].sum()),
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


def gerar_fig_relatorio(p1, p2, p3, uf, cidade):
    """Roda vários cenários e monta mosaico de gráficos + notas."""

    notas = u"""
    Sobre o modelo e as estimativas:

    As projeções são obtidas a partir da trajetória observada nos três países que melhor se correlacionem com a evolução dos dados do Brasil e localidades.
    O desenho da curva projetada (pontilhada) é reflexo do comportamento observado nos países seguidos. Conforme a epidemia avança a referência pode mudar.

    Outros parâmetros relevantes:
        • os valores são corrigidos por uma estimativa de subnotificação (s) calculado para duas situações:
            (a) mortes suspeitas aguardando confirmação e ainda não notificadas
            (b) mortes potencialmente devido à Covid-19 notificadas como devidas a outras causas
        • as curvas dos diferentes lugares são emparelhadas a partir do dia em que ocorrem N ou mais mortes (eixo x).
        • as curvas são alisadas (médias móveis), por isso não iniciam no dia zero. O alisamento permite melhor visualização das curvas mas pode gerar algum
        desvio com relação aos número diários absolutos.
        • as projeções são recalculadas diariamente e podem sofrer alterações significativas em função das novas informações incorporadas.

    Fontes dos dados:
        https://covid.ourworldindata.org
        https://brasil.io
        https://transparencia.registrocivil.org.br
    """

    equipe = u'  M.Zac | L.Tozi | R.Luciano || https://github.com/Maurozac/covid-br/blob/master/compara.py'

    totais = u"""
    Mortes estimadas (acumulado)"""

    hoje = str(datetime.datetime.now())[:16]
    fig, ax = plt.subplots(1, 3, figsize=(12, 6), sharex=True, sharey=True)
    fig.suptitle(u"Projeção da epidemia Covid-19" +  " | " + hoje, fontsize=12)
    fig.subplots_adjust(bottom=0.5)
    fig.text(0.33, 0.42, notas, fontsize=7, verticalalignment='top')
    fig.text(0.33, 0.02, equipe, family="monospace", fontsize='6', color='#ff003f', horizontalalignment='left')
    raw, inicio, data, nbr, subs, refs, oficial = preparar_dados(p1, uf, cidade)

    for i in [0, 1, 2]:
        if refs[i] == 'n/d':
            ax[i].set_title(u"Dados não disponíveis", fontsize=8)
            break
        correlacionados, calibrados, projetado, infos = rodar_modelo(raw, inicio, data, nbr, p2, p3, refs[i], refs)
        ax[i].set_title(refs[i], fontsize=8)
        ax[i].set_xlabel(u'Dias desde ' + str(p1) + ' mortes em um dia', fontsize=8)
        ax[i].set_xlim(0, calibrados.shape[0]+25)
        ax[i].set_ylabel(u'Mortes por dia', fontsize=8)
        for c in correlacionados:
            ax[i].plot(calibrados[c], linewidth=3, color="#ff7c7a")
            lvi = calibrados[c].last_valid_index()
            ax[i].text(lvi+1, calibrados[c][lvi], c, fontsize=6, verticalalignment="center")
        ax[i].plot(calibrados[refs[i]], linewidth=3, color="#1f78b4")
        ax[i].plot(projetado, linewidth=2, linestyle=":", color="#1f78b4")
        lvi = pd.Series(projetado).last_valid_index()
        ax[i].text(lvi+1, projetado[lvi], refs[i], fontsize=6, verticalalignment="center")
        ax[i].plot(infos["index"], infos["pico"], '^', markersize=5.0, color="1", markeredgecolor="#1f78b4")
        msg = "PICO ~" + infos["mortes_no_pico"] + " mortes em " + infos["dia_do_pico"] + " s=" + subs[refs[i]]
        ax[i].text(infos["index"]-1, infos["pico"]-120, msg, fontsize=7, color="#1f78b4", verticalalignment='top')
        ax[i].plot(oficial[refs[i]], linewidth=1, linestyle="--", color="#1f78b4")
        ax[i].text(oficial.shape[0]+1, list(oficial[refs[i]])[-1], 'oficial', fontsize=6, verticalalignment="center")
        totais += "\n\n    " + refs[i] + "\n" + "\n".join(["    " + x[0] + ": " + str(x[1]) for x in infos['mt'].items()])

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
=> DEPRECATED: esta situação ocorreu apenas pontualmente durante o mes de março,
mudamos a metodologia para calcular esse parâmetro on-fly (cf. função abaixo)

p4 foi para dentro das funções

"""


def estimar_subnotificacao(ref):
    u"""Usa dados do Portal da Transparencia do Registro Civil do Brasil para estimar
    a subnotificação de casos de Covid-19.

    https://transparencia.registrocivil.org.br/especial-covid

    Este portal nos premite ver diferença entre a ocorrência de mortes atribuídas
    à insuficiência respiratória e pneumonia em 2019 e 2020.

    O PROBLEMA => sabemos que há subnoticações de mortes por COVID por pelo menos duas causas:
    a) demora na confirmação de casos, testados ou não
    b) casos não são nem testados e são notificados com causa mortis distinta

    Para estimar a subnotificação adotamos as seguintes hipóteses:

    I) mortes por pneumonia e insuficiencia_respiratoria devereriam ser APROXIMADAMENTE iguais em
    2019 e 2020

    II) caso a) => por causa da demora na confirmacao a morte não é notificada e os números
    de mortes por pneumonia ou insuficiencia_respiratoria para 2020 aparecem menores do que 2019.
    Essa diferença seria igual ao número máximo de mortes por covid ainda não confirmadas. Esse
    número corresponde ao número de mortes ainda no "limbo", sem causa morte determinada.

    III) caso b) => por causa de notificação errada/incompleta + mortes colaterais, o número de 2020 fica maior:
    a diferença sendo atribuída ao covid, direta ou indiretamente.

    IV) os casos a) e b) seriam estratégias deliberadas e, portanto, não se ocorreriam simultaneamente

    V) ok, mortes colaterais podem puxar estimativas para baixo no caso a); mas por enquanto não há
    muito o que fazer, são ESTIMATIVAS. Fica a ressalva que s(a) pode estar sendo subestimado.

    Como as bases de dados são dinâmicas e os números vão mudando conforme confirmações vão
    sendo computadas, inclusive retroativamente. Portanto, o coeficiente de subnotificação (s) precisa ser
    recalculado diariamente.

    Inputs:
    .ref => sigla do estado ou calcula para Brasil
    .total => soma das mortes para ref

    Retorna: tupla
    . sub: número de casos potencialmente subnotificados
    . hip: hipóte 'a': casos não notificados; 'b': casos notificados com outra causa
    """

    sub, hip = 1, "ø"
    estados = [
        'AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS',
        'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC',
        'SP', 'SE', 'TO',
        ]
    if ref not in estados:
        ref = "all"

    hoje = str(datetime.datetime.now())[:10]

    api = "https://transparencia.registrocivil.org.br/api/covid?"
    api += "data_type=data_ocorrido"
    api += "&search=death-respiratory"
    api += "&state=" + ref
    api += "&start_date=2020-03-16"
    api += "&end_date=" + hoje

    call_1 = api + "&causa=insuficiencia_respiratoria"
    call_2 = api + "&causa=pneumonia"

    try:
        c1 = requests.get(call_1).json()
        c2 = requests.get(call_2).json()

        m19 = c1['chart']['2019'] + c2['chart']['2019']
        m20 = c1['chart']['2020'] + c2['chart']['2020']

        if m20 <= m19:   # caso a
            sub = m19 - m20
            hip = "a"
        else:           # caso b
            sub = m20 - m19
            hip = "b"
    except:
        print("[!] FALHA em registrocivil.org.br")

    return sub, hip


#########################   RELATORIO   ########################################

def relatorio_hoje(p1, p2, p3, uf, cidade, my_path):
    """Calcula tudo e gera um relatorio em pdf."""
    # gera o dash do dia
    dashboard = gerar_fig_relatorio(p1, p2, p3, uf, cidade)
    # salva em um arquivo pdf
    hoje = str(datetime.datetime.now())[:10]
    pp = PdfPages(my_path+"covid_dashboard_"+uf+"_"+cidade+"_"+hoje+".pdf")
    dashboard.savefig(pp, format='pdf')
    pp.close()


# acerte o caminho para o seu ambiente... esse aí é o meu :-)
my_path = "/Users/tapirus/Desktop/"
# parametros do modelo: mortes para parear séries, países comparados, alisamento
p1, p2, p3 = 15, 3, 7

relatorio_hoje(p1, p2, p3, "SP", "São Paulo", my_path)
relatorio_hoje(p1, p2, p3, "AM", "Manaus", my_path)

