
'''
ANALISIS RED ELÉCTRICA DE ALEMANIA
'''

'''
Apartado 1

Segun se indique por parte de los profesores en cada caso, se podr´a usar la red obtenida en el primer proyecto
o bien una red obtenida de un repositorio p´ublico.
'''


from math import factorial
import random
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from collections import Counter
import collections
import community
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import naive_greedy_modularity_communities 

vertices_def = pd.read_csv('vertices_def.csv', delimiter=',')
enlaces_def = pd.read_csv('enlaces_def.csv')


'''
CREACION DE LA RED
'''

G = nx.Graph()

nodos = []
nodos = list(vertices_def.iloc[:, 1])
nodos_r = np.linspace(1, len(nodos), len(nodos))
i = 0
for i in range(len(nodos)):
    G.add_node(int(nodos[i]))
enlaces = []
for i in range(len(enlaces_def)):
    enlaces.append((int(enlaces_def.iloc[i, 2]), int(enlaces_def.iloc[i, 3])))
a = []
for item in enlaces:
    if item not in a:
        a.append(item)
for i in range(len(a)):
    G.add_edge(a[i][0], a[i][1])


# PARAMETROS DE LA RED

G_size = G.number_of_nodes()
G_edges = G.number_of_edges()
degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
mean_degree = statistics.mean(degree_sequence)
n_comp = nx.number_connected_components(G)

Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
Gcc_size = Gcc.number_of_nodes()
Gcc_edges = Gcc.number_of_edges()
degree_sequence_Gcc = sorted((d for n, d in Gcc.degree()), reverse=True)
mean_degree_Gcc = statistics.mean(degree_sequence_Gcc)
D_Gcc = nx.diameter(Gcc)

def distribution_degree(G):
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    fig = plt.figure("Degree of a random graph", figsize=(16, 16))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(6, 6)
    ax1 = fig.add_subplot(axgrid[:, :3])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")
    ax2 = fig.add_subplot(axgrid[:, 3:])
    ax2.plot(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Number of Nodes")
    fig.tight_layout()
    plt.show()
    return(degree_sequence)

# DEFINIMOS FUNCION PARA VISUALIZACION
def VIS(G, txt):
    V = G
    degree_sequence = sorted((d for n, d in V.degree()), reverse=True)
    dmax = max(degree_sequence)
    fig = plt.figure("Degree of a random graph", figsize=(16, 16))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)
    ax0 = fig.add_subplot(axgrid[0:3, :])
    pos = nx.spring_layout(V, seed=10396953)
    nx.draw_networkx_nodes(V, pos, ax=ax0, node_size=10)
    nx.draw_networkx_edges(V, pos, ax=ax0, alpha=0.4)
    ax0.set_title("All components of G")
    ax0.set_axis_off()
    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")
    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Number of Nodes")
    ruta = "Imagenes/"+ txt +".jpg"
    plt.savefig(ruta)
    fig.tight_layout()
    plt.show()
    # PARAMETROS DE LA RED
    Gp = V
    G_size = Gp.number_of_nodes()
    G_edges = Gp.number_of_edges()
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    mean_degree = statistics.mean(degree_sequence)
    n_comp = nx.number_connected_components(Gp)
    GccP = Gp.subgraph(sorted(nx.connected_components(Gp), key=len, reverse=True)[0])
    Gcc_size = GccP.number_of_nodes()
    Gcc_edges = GccP.number_of_edges()
    degree_sequence_Gcc = sorted((d for n, d in GccP.degree()), reverse=True)
    mean_degree_Gcc = statistics.mean(degree_sequence_Gcc)
    D_Gcc = nx.diameter(GccP)
    print("Grado medio: ", mean_degree)
    print("Numero de componentes: ", n_comp)
    print("Tamaño componente gigante: ", Gcc_size)
    print("Grado medio componente gigante: ", mean_degree_Gcc)
    print("Diametro componente gigante: ", D_Gcc)

VIS(Gcc, "G_Real")



'''
APARTADO 2

Se consideraran, adem´as de la red bajo estudio, un conjunto de redes obtenidas por m´etodos aleatorios,
usando los modelos de Erd˝os-Renyi (entre 3 y 5 redes) y de configuraci´on (entre 7 y 10 redes), con los mismos
parametros que la red de referencia.
'''


'''
CREACION DE REDES ALEATORIAS
'''

''' MODELO DE ERDOS-RENYI '''

# Parametros de nuestra red escogida
n = Gcc.number_of_nodes()
e = Gcc.number_of_edges()

# Calculo del %  de aristas de la red seleccionada
total_aristas_posibles = (factorial(n))/(factorial(2)*factorial(n-2))
porcentaje_de_aristas = Gcc.number_of_edges() / (total_aristas_posibles)

def ER1(N, p):
    G_ER = nx.Graph()
    nodos = np.array(range(N))
    i = 0
    for i in range(len(nodos)):
        G_ER.add_node(nodos[i])
    for i in range(N):
        for j in range(N):
            if i < j:
                P = np.random.rand()
                if P < p:
                    G_ER.add_edge(i, j)
    print(G_ER.number_of_nodes())
    print(G_ER.number_of_edges())
    return (G_ER)

def ER2(N, e):
    G_ER = nx.Graph()
    nodos = np.array(range(N))
    i = 0
    for i in range(len(nodos)):
        G_ER.add_node(nodos[i])
    for i in range(e):
        node1 = np.random.choice(nodos)
        node2 = np.random.choice(nodos)
        pos = int(list(np.where(nodos == node2))[0])
        if node1 == node2:
            pos = pos + 1
            node2 = nodos[pos]
        G_ER.add_edge(node1, node2)
    print(G_ER.number_of_nodes())
    print(G_ER.number_of_edges())
    return (G_ER)

# GRAFOS DE ERDOS-RENYI

G_ER1 = ER1(n, porcentaje_de_aristas)
G_ER2 = ER1(n, porcentaje_de_aristas)
G_ER3 = ER2(n, e)
G_ER4 = ER2(n, e)

degree_G_ER1 = distribution_degree(G_ER1)
degree_G_ER2 = distribution_degree(G_ER2)
degree_G_ER3 = distribution_degree(G_ER3)
degree_G_ER4 = distribution_degree(G_ER4)

#VISUALIZACION
VIS(G_ER1, "G_ER1")
VIS(G_ER2, "G_ER2")
VIS(G_ER3, "G_ER3")
VIS(G_ER4, "G_ER4")



''' MODELO DE CONFIGURACION '''

def CONFIG(G):
    G_C = nx.Graph()
    nodos = list(G.nodes())
    for i in range(len(nodos)):
        G_C.add_node(int(nodos[i]))
    pool = nodos
    pool_aux = np.array(pool)
    for p in nodos:
        p = int(p)
        while G_C.degree(p) < G.degree(p):
            for i in pool_aux:
                if G_C.degree(int(i)) == G.degree(int(i)):
                    pos = int(list(np.where(pool_aux == i))[0])
                    pool_aux = np.delete(pool_aux, pos)
            node_in = int(np.random.choice(pool_aux))
            if node_in == p:
                node_in = int(np.random.choice(pool_aux))
            if node_in == p:
                node_in = int(np.random.choice(pool_aux))
            if node_in == p:
                node_in = int(np.random.choice(pool_aux))
            if G_C.degree(node_in) < G.degree(node_in):
                G_C.add_edge(p, node_in)
    return(G_C)


'''
G_C = nx.Graph()
nodos = list(G.nodes())
for i in range(len(nodos)):
    G_C.add_node(int(nodos[i]))
pool = nodos
pool_aux = np.array(pool)
for p in nodos:
    p = int(p)
    while G_C.degree(p) < G.degree(p):
        for i in pool_aux:
            if G_C.degree(int(i)) == G.degree(int(i)):
                pos = int(list(np.where(pool_aux == i))[0])
                pool_aux = np.delete(pool_aux, pos)
        node_in = int(np.random.choice(pool_aux))
        if G_C.degree(node_in) < G.degree(node_in):
            G_C.add_edge(p, node_in)
        if G_C.degree(node_in) > G.degree(node_in):
            dif = G_C.degree(node_in) - G.degree(node_in) - 1
            for j in range(dif):
                G_C.remove_edge(p, node_in)
                

No penaliza los bucles pero el grafo resultante no tiene exactamente la misma distribucion de nodos
'''

# GRAFOS DE MODELO DE CONFIGURACION

G_C1 = CONFIG(Gcc)
G_C2 = CONFIG(Gcc)
G_C3 = CONFIG(Gcc)
G_C4 = CONFIG(Gcc)
G_C5 = CONFIG(Gcc)
G_C6 = CONFIG(Gcc)
G_C7 = CONFIG(Gcc)

# DEGREE
degree_G_C1 = distribution_degree(G_C1)
degree_G_C2 = distribution_degree(G_C2)
degree_G_C3 = distribution_degree(G_C3)
degree_G_C4 = distribution_degree(G_C4)
degree_G_C5 = distribution_degree(G_C5)
degree_G_C6 = distribution_degree(G_C1)
degree_G_C7 = distribution_degree(G_C2)

# VISUALIZACION
VIS(G_C1, "G_C1")
VIS(G_C2, "G_C2")
VIS(G_C3, "G_C3")
VIS(G_C4, "G_C4")
VIS(G_C5, "G_C5")
VIS(G_C6, "G_C6")
VIS(G_C7, "G_C7")



'''
APARTADO 3

Se comparar´an un m´ınimo de 3 ´ındices de centralidad y los coeficientes de clustering, as´ı como la distribuci´on
de nodos en componentes conexas (componente mayor vs otras componentes), de la red de referencia y las
redes aleatorias, comparando e interpretando los resultados.
'''

# INDICES DE CENTRALIDAD

#DEGREE
degree_sequence_Real = sorted((d for n, d in Gcc.degree()), reverse=True)
degree_sequence_ER1 = sorted((d for n, d in G_ER1.degree()), reverse=True)
degree_sequence_ER2 = sorted((d for n, d in G_ER2.degree()), reverse=True)
degree_sequence_ER3 = sorted((d for n, d in G_ER3.degree()), reverse=True)
degree_sequence_ER4 = sorted((d for n, d in G_ER4.degree()), reverse=True)
degree_sequence_C1 = sorted((d for n, d in G_C1.degree()), reverse=True)
degree_sequence_C2 = sorted((d for n, d in G_C2.degree()), reverse=True)
degree_sequence_C3 = sorted((d for n, d in G_C3.degree()), reverse=True)
degree_sequence_C4 = sorted((d for n, d in G_C4.degree()), reverse=True)
degree_sequence_C5 = sorted((d for n, d in G_C5.degree()), reverse=True)
degree_sequence_C6 = sorted((d for n, d in G_C6.degree()), reverse=True)
degree_sequence_C7 = sorted((d for n, d in G_C7.degree()), reverse=True)

#EIGENVECTOR
E_Real = nx.eigenvector_centrality(Gcc, max_iter = 1000)
E_ER1 = nx.eigenvector_centrality(G_ER1, max_iter = 1000)
E_ER2 = nx.eigenvector_centrality(G_ER2, max_iter = 1000)
E_ER3 = nx.eigenvector_centrality(G_ER3, max_iter = 1000)
E_ER4 = nx.eigenvector_centrality(G_ER4, max_iter = 1000)
E_C1 = nx.eigenvector_centrality(G_C1, max_iter = 1000)
E_C2 = nx.eigenvector_centrality(G_C2, max_iter = 1000)
E_C3 = nx.eigenvector_centrality(G_C3, max_iter = 1000)
E_C4 = nx.eigenvector_centrality(G_C4, max_iter = 1000)
E_C5 = nx.eigenvector_centrality(G_C5, max_iter = 1000)
E_C6 = nx.eigenvector_centrality(G_C6, max_iter = 1000)
E_C7 = nx.eigenvector_centrality(G_C7, max_iter = 1000)

#PAGERANK
PR_Real = nx.pagerank(Gcc, max_iter = 1000)
PR_ER1 = nx.pagerank(G_ER1, max_iter = 1000)
PR_ER2 = nx.pagerank(G_ER2, max_iter = 1000)
PR_ER3 = nx.pagerank(G_ER3, max_iter = 1000)
PR_ER4 = nx.pagerank(G_ER4, max_iter = 1000)
PR_C1 = nx.pagerank(G_C1, max_iter = 1000)
PR_C2 = nx.pagerank(G_C2, max_iter = 1000)
PR_C3 = nx.pagerank(G_C3, max_iter = 1000)
PR_C4 = nx.pagerank(G_C4, max_iter = 1000)
PR_C5 = nx.pagerank(G_C5, max_iter = 1000)
PR_C6 = nx.pagerank(G_C6, max_iter = 1000)
PR_C7 = nx.pagerank(G_C7, max_iter = 1000)

#CLOSENESS
C_Real = nx.closeness_centrality(Gcc)
C_ER1 = nx.closeness_centrality(G_ER1)
C_ER2 = nx.closeness_centrality(G_ER2)
C_ER3 = nx.closeness_centrality(G_ER3)
C_ER4 = nx.closeness_centrality(G_ER4)
C_C1 = nx.closeness_centrality(G_C1)
C_C2 = nx.closeness_centrality(G_C2)
C_C3 = nx.closeness_centrality(G_C3)
C_C4 = nx.closeness_centrality(G_C4)
C_C5 = nx.closeness_centrality(G_C5)
C_C6 = nx.closeness_centrality(G_C6)
C_C7 = nx.closeness_centrality(G_C7)

#BETWENESS
B_Real = nx.betweenness_centrality(Gcc)
B_ER1 = nx.betweenness_centrality(G_ER1)
B_ER2 = nx.betweenness_centrality(G_ER2)
B_ER3 = nx.betweenness_centrality(G_ER3)
B_ER4 = nx.betweenness_centrality(G_ER4)
B_C1 = nx.betweenness_centrality(G_C1)
B_C2 = nx.betweenness_centrality(G_C2)
B_C3 = nx.betweenness_centrality(G_C3)
B_C4 = nx.betweenness_centrality(G_C4)
B_C5 = nx.betweenness_centrality(G_C5)
B_C6 = nx.betweenness_centrality(G_C6)
B_C7  = nx.betweenness_centrality(G_C7)


# COEFICIENTE DE CLUSTERING
Ct_Real = nx.clustering(Gcc)
Ct_ER1 = nx.clustering(G_ER1)
Ct_ER2 = nx.clustering(G_ER2)
Ct_ER3 = nx.clustering(G_ER3)
Ct_ER4 = nx.clustering(G_ER4)
Ct_C1 = nx.clustering(G_C1)
Ct_C2 = nx.clustering(G_C2)
Ct_C3 = nx.clustering(G_C3)
Ct_C4 = nx.clustering(G_C4)
Ct_C5 = nx.clustering(G_C5)
Ct_C6 = nx.clustering(G_C6)
Ct_C7 = nx.clustering(G_C7)



# RESPRESENTACION GRAFICA

#Formato lista
def VIS_1(data, txt):
    plt.figure()
    plt.hist(x=data, bins = 80)
    plt.title( txt , size = 8 ) 
    plt.xlim(0, max(data)+max(data)/10)
    plt.ylabel( "Numbero de nodos" , size = 8 ) 
    ruta = "Imagenes/"+ txt +".jpg"
    plt.savefig(ruta)
    plt.show() 
    pos = degree_sequence_Real.index(max(degree_sequence_Real))
    print("Maximo: ", degree_sequence_Real[pos])

VIS_1(degree_sequence_Real, "degree_sequence_Real")
VIS_1(degree_sequence_ER1, "degree_sequence_ER1")
VIS_1(degree_sequence_ER2, "degree_sequence_ER2")
VIS_1(degree_sequence_ER3, "degree_sequence_ER3")
VIS_1(degree_sequence_ER4, "degree_sequence_ER4")
VIS_1(degree_sequence_C1, "degree_sequence_C1")
VIS_1(degree_sequence_C2, "degree_sequence_C2")
VIS_1(degree_sequence_C3, "degree_sequence_C3")
VIS_1(degree_sequence_C4, "degree_sequence_C4")
VIS_1(degree_sequence_C5, "degree_sequence_C5")
VIS_1(degree_sequence_C6, "degree_sequence_C6")
VIS_1(degree_sequence_C7, "degree_sequence_C7")

# Formato diccionario
def VIS_2(data, txt):
    xlabel = "Coeficiente de clustering: "
    dataB = list(data.values())
    plt.figure()
    plt.hist(x=dataB, bins = 80)
    plt.title( xlabel + txt, size = 8 ) 
    plt.xlim(0, max(dataB)+max(dataB)/10)
    plt.ylabel( "Numbero de nodos" , size = 8 ) 
    ruta = "Imagenes/"+ txt +".jpg"
    plt.savefig(ruta)
    plt.show() 
    list_of_key = list(data.keys())
    list_of_value = list(data.values())
    pos = list_of_value.index(max(list_of_value))
    print("Maximo: ", list_of_value[pos])
    print("Nodo: ", list_of_key[pos])

# PAGERANK
VIS_2(PR_Real, "PR_Real")
VIS_2(PR_ER1, "PR_ER1")
VIS_2(PR_ER2, "PR_ER2")
VIS_2(PR_ER3, "PR_ER3")
VIS_2(PR_ER4, "PR_ER4")
VIS_2(PR_ER4, "PR_ER4")
VIS_2(PR_C1, "PR_C1")
VIS_2(PR_C2, "PR_C2")
VIS_2(PR_C3, "PR_C3")
VIS_2(PR_C4, "PR_C4")
VIS_2(PR_C5, "PR_C5")
VIS_2(PR_C6, "PR_C6")
VIS_2(PR_C7, "PR_C7")

#CLOSENESS
VIS_2(C_Real, "C_Real")
VIS_2(C_ER1, "C_ER1")
VIS_2(C_ER2, "C_ER2")
VIS_2(C_ER3, "C_ER3")
VIS_2(C_ER4, "C_ER4")
VIS_2(C_ER4, "C_ER4")
VIS_2(C_C1, "C_C1")
VIS_2(C_C2, "C_C2")
VIS_2(C_C3, "C_C3")
VIS_2(C_C4, "C_C4")
VIS_2(C_C5, "C_C5")
VIS_2(C_C6, "C_C6")
VIS_2(C_C7, "C_C7")

#BETWENESS
VIS_2(B_Real, "B_Real")
VIS_2(B_ER1, "B_ER1")
VIS_2(B_ER2, "B_ER2")
VIS_2(B_ER3, "B_ER3")
VIS_2(B_ER4, "B_ER4")
VIS_2(B_ER4, "B_ER4")
VIS_2(B_C1, "B_C1")
VIS_2(B_C2, "B_C2")
VIS_2(B_C3, "B_C3")
VIS_2(B_C4, "B_C4")
VIS_2(B_C5, "B_C5")
VIS_2(B_C6, "B_C6")
VIS_2(B_C7, "B_C7")


#CLUSTERING
VIS_2(Ct_Real, "Ct_Real")
VIS_2(Ct_ER1, "Ct_ER1")
VIS_2(Ct_ER2, "Ct_ER2")
VIS_2(Ct_ER3, "Ct_ER3")
VIS_2(Ct_ER4, "Ct_ER4")
VIS_2(Ct_ER4, "Ct_ER4")
VIS_2(Ct_C1, "Ct_C1")
VIS_2(Ct_C2, "Ct_C2")
VIS_2(Ct_C3, "Ct_C3")
VIS_2(Ct_C4, "Ct_C4")
VIS_2(Ct_C5, "Ct_C5")
VIS_2(Ct_C6, "Ct_C6")
VIS_2(Ct_C7, "Ct_C7")


# DISTRIBUCION DE NODOS

GC2 = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[10])
GC3 = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[15])
GC4 = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[20])

nodesC2 = GC2.nodes()
nodesC3 = GC3.nodes()
nodesC4 = GC4.nodes()

nodesC = nodesC4
for j in nodesC:
    print('########################################################')
    print('Modelo: G_Real') #G_Real = G
    print('Componente grupo: GC4')
    print(vertices_def.iloc[j])



'''
APARTADO 4

Se aplicar´an algoritmos de distribuci´on en comunidades, tanto de la red de referencia como de las aleatorias,
comparando e interpretando los resultados.
'''

Gbp = G_C7
graf = "_C7"
partition = community.best_partition(Gbp)

#Calcula la partición de los nodos del gráfico que maximiza la modularidad usando las heurísticas de Louvain
#Esta es la partición de mayor modularidad, es decir, la partición más alta del dendrograma generado por el algoritmo de Louvain.

# VISUALIZACION
size = float(len(set(partition.values())))
pos = nx.spring_layout(Gbp)
count = 0.
for com in set(partition.values()) :
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(Gbp, pos, list_nodes, node_size = 20,
                                node_color = str(count / size))
nx.draw_networkx_edges(Gbp,pos, alpha=0.5)
txt = "Comunidades_Spring"
ruta = "Imagenes/"+ txt +graf + ".jpg"
plt.savefig(ruta)
plt.show()

# VISUALIZACION
size = float(len(set(partition.values())))
pos = nx.kamada_kawai_layout(Gbp)
count = 0.
for com in set(partition.values()) :
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(Gbp, pos, list_nodes, node_size = 20,
                                node_color = str(count / size))
nx.draw_networkx_edges(Gbp,pos, alpha=0.5)
txt = "Comunidades_Kamada_kawai"
ruta = "Imagenes/"+ txt + graf + ".jpg"
plt.savefig(ruta)
plt.show()

# DISTRIBUCION DE LOS NODOS A CADA COMUNIDAD
C = []
GRAFOS = [Gcc, G_ER1, G_ER2, G_ER3, G_ER4, G_C1, G_C2, G_C3, G_C4, G_C5, G_C6, G_C7]
GRAFOS_txt = ["Gcc", "G_ER1", "G_ER2", "G_ER3", "G_ER4", "G_C1", "G_C2", "G_C3", "G_C4", "G_C5", "G_C6", "G_C7"]

for g in range(len(GRAFOS)):
    partition = community.best_partition(GRAFOS[g])
    list_of_key = np.array(list(partition.keys()))
    list_of_value = np.array(list(partition.values()))
    comunities = []
    list_of_value_unique = np.unique(list_of_value)
    for i in list_of_value_unique:
        pos = list(np.where(list_of_value == i))[0]
        comunities.append(pos)
    for j in range(len(comunities)):
        for i in comunities[j]:
            print('########################################################')
            print('Modelo: ', GRAFOS_txt[g])
            print('Comunidad grupo: ', j)
            print(vertices_def.iloc[i])
    C.append(comunities)


comunities = C[0]
for i in comunities[4]:
    print('########################################################')
    print('Modelo: ', GRAFOS_txt[g])
    print('Comunidad grupo: ', 4)
    print(vertices_def.iloc[i])
for i in comunities[6]:
    print('########################################################')
    print('Modelo: ', GRAFOS_txt[g])
    print('Comunidad grupo: ', 6)
    print(vertices_def.iloc[i])



'''
APARTADO 5

Se realizar´an simulaciones aleatorias de percolaci´on de nodos, siguiendo un modelo uniforme, estudiando como
afecta la misma a las componentes (si se mantiene una componente gigante), tanto para la red de referencia
como las aleatorias. Se representar´a, en media, como afecta la probabilidad de ocupaci´on a la medida de la
componente mayor.
'''

#Utilizaremos varios porcentajes de ocupamiento [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

percolacion = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


# PERCOLACION ALEATORIA (REALIZAMOS 10 PERCOLACIONES DE LAS CUALES CALCULAREMOS EL TAMAÑO MEDIO QUE TOMAREMOS COMO DATO MAS OBJETIVO)
def PERCOLACION_UNIFORME(G_aux):
    percolacion = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    s_total = []
    n_nodes_total = []
    nodes = list(G_aux.nodes())
    for i in range(10):
        s = []
        n_nodes = []
        for j in percolacion:
            umbral = j
            G_per = G_aux.copy()
            for i in nodes:
                if np.random.random() > umbral:
                    G_per.remove_node(i)
            G_per_cc = G_per.subgraph(sorted(nx.connected_components(G_per), key=len, reverse=True)[0])
            s.append(G_per_cc.number_of_nodes())   
            n_nodes.append(G_per.number_of_nodes())
        s_total.append(s)
        n_nodes_total.append(n_nodes)
    # PROMEDIO PARA LAS 10 SIMULACIONES DE PERCOLACION REALIZADAS
    promedio_s = [0]
    promedio_n = [0]
    S = []
    for j in range(len(s_total[0])):
        per_sum = 0
        for i in s_total:
            per_sum = per_sum + i[j]
        promedio_s.append(per_sum/len(s_total))
    for j in range(len(n_nodes_total[0])):
        per_sum_n = 0
        for i in n_nodes_total:
            per_sum_n = per_sum_n + i[j]
        promedio_n.append(per_sum_n/len(n_nodes_total))
    for i in range(len(promedio_n)):
        if promedio_n[i] == 0:
            S = [0]
        else:
            S.append(promedio_s[i]/promedio_n[i])
    return(promedio_s, promedio_n, S )

s_Real, n_Real, S_Real = PERCOLACION_UNIFORME(Gcc)
s_ER1, n_ER1, S_ER1 = PERCOLACION_UNIFORME(G_ER1)
s_ER2, n_ER2, S_ER2 = PERCOLACION_UNIFORME(G_ER2)
s_ER3, n_ER3, S_ER3 = PERCOLACION_UNIFORME(G_ER3)
s_ER4, n_ER4, S_ER4 = PERCOLACION_UNIFORME(G_ER4)
s_C1, n_C1, S_C1 = PERCOLACION_UNIFORME(G_C1)
s_C2, n_C2, S_C2 = PERCOLACION_UNIFORME(G_C2)
s_C3, n_C3, S_C3 = PERCOLACION_UNIFORME(G_C3)
s_C4, n_C4, S_C4 = PERCOLACION_UNIFORME(G_C4)
s_C5, n_C5, S_C5 = PERCOLACION_UNIFORME(G_C5)
s_C6, n_C6, S_C6 = PERCOLACION_UNIFORME(G_C6)
s_C7, n_C7, S_C7 = PERCOLACION_UNIFORME(G_C7)


# PLOT DE LA EVOLUCION DE LA COMPONENTE GIGANTE
x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
plt.figure()
plt.plot(x, S_Real, color = 'green')
plt.plot(x, S_ER1, color = 'blue')
plt.plot(x, S_ER2, color = 'blue')
plt.plot(x, S_ER3, color = 'blue')
plt.plot(x, S_ER4, color = 'blue')
plt.plot(x, S_C1, color = 'red')
plt.plot(x, S_C2, color = 'red')
plt.plot(x, S_C3, color = 'red')
plt.plot(x, S_C4, color = 'red')
plt.plot(x, S_C5, color = 'red')
plt.plot(x, S_C6, color = 'red')
plt.plot(x, S_C7, color = 'red')
plt.xlabel('% ocupacion')
plt.ylabel('S')
plt.xlim(0, 1)
plt.ylim(0, 1)
txt = "Percolacion_Uniforme"
plt.title('Evolucion de S')
ruta = "Imagenes/"+ txt +".jpg"
plt.savefig(ruta)
plt.show()


# VISUALIZACION DE LA EVOLUCION DE LA PERCOLACION EN EL GRAFO
txt = "Percolacion_Uniforme_C7"
txtcc = "Percolacion_Uniforme_C7_CC"
G_aux = G_C7
for j in percolacion:
    umbral = j
    G_per = G_aux.copy()
    nodes = list(G_aux.nodes())
    for i in nodes:
        if np.random.random() > umbral:
            G_per.remove_node(i)            
    G_per_cc = G_per.subgraph(sorted(nx.connected_components(G_per), key=len, reverse=True)[0])
    # VISUALIZACION G_PER
    degree_sequence = sorted((d for n, d in G_per.degree()), reverse=True)
    dmax = max(degree_sequence)
    fig = plt.figure("Degree of a random graph", figsize=(16, 16))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)
    ax0 = fig.add_subplot(axgrid[0:3, :])
    pos = nx.spring_layout(G_per, seed=10396953)
    nx.draw_networkx_nodes(G_per, pos, ax=ax0, node_size=10)
    nx.draw_networkx_edges(G_per, pos, ax=ax0, alpha=0.4)
    ax0.set_title("All components of G")
    ax0.set_axis_off()
    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")
    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Number of Nodes")
    txt2 = str(j)
    plt.title('Evolucion de S')
    ruta = "Imagenes/"+ txt + '_' + txt2 + ".jpg"
    plt.savefig(ruta)
    fig.tight_layout()
    plt.show()
    #VISUALIZACION G_PER_CC
    degree_sequence = sorted((d for n, d in G_per_cc.degree()), reverse=True)
    dmax = max(degree_sequence)
    fig = plt.figure("Degree of a random graph", figsize=(16, 16))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)
    ax0 = fig.add_subplot(axgrid[0:3, :])
    pos = nx.spring_layout(G_per_cc, seed=10396953)
    nx.draw_networkx_nodes(G_per_cc, pos, ax=ax0, node_size=10)
    nx.draw_networkx_edges(G_per_cc, pos, ax=ax0, alpha=0.4)
    ax0.set_title("All components of G")
    ax0.set_axis_off()
    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")
    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Number of Nodes")
    txt2 = str(j)
    ruta = "Imagenes/"+ txtcc + '_' + txt2 +".jpg"
    plt.savefig(ruta)
    fig.tight_layout()
    plt.show()
    
    
    
    
'''
APARTADO 6

Se repetir´a el apartado anterior considerando percolaci´on no uniforme, donde se desocupan todos los nodos
con grado mayor que un valor dado. El an´alisis se har´a en funci´on de este grado.
'''

percolacion = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

G_aux = G_ER1

degree = list(G_aux.degree())
node = list((n for n, d in G_aux.degree()))
degree_sequence = list((d for n, d in G_aux.degree()))
arch = pd.DataFrame(list(zip(node,degree_sequence)), columns = ['Node', 'Degree'])
arch.to_excel('apoyo_percolacion.xlsx')
#Ordenamos en el excel y volvemos a cargar / No encuentro forma facil de hacerlo en python y esto es mucho mas rapido
final_Real = pd.read_excel('apoyo_percolacion_Real.xlsx')
final_ER1 = pd.read_excel('apoyo_percolacion_ER1.xlsx')
final_ER2 = pd.read_excel('apoyo_percolacion_ER2.xlsx')
final_ER3 = pd.read_excel('apoyo_percolacion_ER3.xlsx')
final_ER4 = pd.read_excel('apoyo_percolacion_ER4.xlsx')
final_C = pd.read_excel('apoyo_percolacion_C1.xlsx')


#PERCOLACION POR GRADO 

#SE ELIMINAN NODOS EN FUNCION DEL GRADO CON ORDEN DESCENDIENTE

def PERCOLACION_2(G_aux, final):
    nodes = final['Node']
    s = [0]
    n_nodes = [0]
    S = [0]
    for j in percolacion:
        umbral = j
        n_node_delete = (1-umbral) * len(nodes)
        G_per = G_aux.copy()
        for i in range(int(n_node_delete)):
                G_per.remove_node(nodes[i])
        G_per_cc = G_per.subgraph(sorted(nx.connected_components(G_per), key=len, reverse=True)[0])
        s.append(G_per_cc.number_of_nodes())   
        n_nodes.append(G_per.number_of_nodes())
        S.append(G_per_cc.number_of_nodes()/G_per.number_of_nodes())
    return(S)

Sp2_Real = PERCOLACION_2(Gcc, final_Real)
Sp2_ER1 = PERCOLACION_2(G_ER1, final_ER1)
Sp2_ER2 = PERCOLACION_2(G_ER2, final_ER2)
Sp2_ER3 = PERCOLACION_2(G_ER3, final_ER3)
Sp2_ER4 = PERCOLACION_2(G_ER4, final_ER4)
Sp2_C1 = PERCOLACION_2(G_C1, final_C)
Sp2_C2 = PERCOLACION_2(G_C2, final_C)
Sp2_C3 = PERCOLACION_2(G_C3, final_C)
Sp2_C4 = PERCOLACION_2(G_C4, final_C)
Sp2_C5 = PERCOLACION_2(G_C5, final_C)
Sp2_C6 = PERCOLACION_2(G_C6, final_C)
Sp2_C7 = PERCOLACION_2(G_C7, final_C)

# PLOT DE LA EVOLUCION DE LA COMPONENTE GIGANTE
x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
plt.figure()
plt.plot(x, Sp2_Real, color = 'green')
plt.plot(x, Sp2_ER1, color = 'blue')
plt.plot(x, Sp2_ER2, color = 'blue')
plt.plot(x, Sp2_ER3, color = 'blue')
plt.plot(x, Sp2_ER4, color = 'blue')
plt.plot(x, Sp2_C1, color = 'red')
plt.plot(x, Sp2_C2, color = 'red')
plt.plot(x, Sp2_C3, color = 'red')
plt.plot(x, Sp2_C4, color = 'red')
plt.plot(x, Sp2_C5, color = 'red')
plt.plot(x, Sp2_C6, color = 'red')
plt.plot(x, Sp2_C7, color = 'red')
plt.xlabel('% ocupacion')
plt.ylabel('S')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('Evolucion de S')
txt = "Percolacion_No Uniforme_GradDesc"
ruta = "Imagenes/"+ txt +".jpg"
plt.savefig(ruta)
plt.show()

# VISUALIZACION DE LA EVOLUCION DE LA PERCOLACION EN EL GRAFO
txt = "Percolacion_No_Uniforme_Desc_C7"
txtcc = "Percolacion_No_Uniforme_Desc_C7_CC"
G_aux = G_C7
nodes = final_C['Node']
for j in percolacion:
    umbral = j
    n_node_delete = (1-umbral) * len(nodes)
    G_per = G_aux.copy()
    for i in range(int(n_node_delete)):
            G_per.remove_node(nodes[i])           
    G_per_cc = G_per.subgraph(sorted(nx.connected_components(G_per), key=len, reverse=True)[0])
    # VISUALIZACION G_PER
    degree_sequence = sorted((d for n, d in G_per.degree()), reverse=True)
    dmax = max(degree_sequence)
    fig = plt.figure("Degree of a random graph", figsize=(16, 16))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)
    ax0 = fig.add_subplot(axgrid[0:3, :])
    pos = nx.spring_layout(G_per, seed=10396953)
    nx.draw_networkx_nodes(G_per, pos, ax=ax0, node_size=10)
    nx.draw_networkx_edges(G_per, pos, ax=ax0, alpha=0.4)
    ax0.set_title("All components of G")
    ax0.set_axis_off()
    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")
    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Number of Nodes")
    txt2 = str(j)
    plt.title('Evolucion de S')
    ruta = "Imagenes/"+ txt + '_' + txt2 + ".jpg"
    plt.savefig(ruta)
    fig.tight_layout()
    plt.show()
    #VISUALIZACION G_PER_CC
    degree_sequence = sorted((d for n, d in G_per_cc.degree()), reverse=True)
    dmax = max(degree_sequence)
    fig = plt.figure("Degree of a random graph", figsize=(16, 16))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)
    ax0 = fig.add_subplot(axgrid[0:3, :])
    pos = nx.spring_layout(G_per_cc, seed=10396953)
    nx.draw_networkx_nodes(G_per_cc, pos, ax=ax0, node_size=10)
    nx.draw_networkx_edges(G_per_cc, pos, ax=ax0, alpha=0.4)
    ax0.set_title("All components of G")
    ax0.set_axis_off()
    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")
    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Number of Nodes")
    txt2 = str(j)
    ruta = "Imagenes/"+ txtcc + '_' + txt2 +".jpg"
    plt.savefig(ruta)
    fig.tight_layout()
    plt.show()




#SE ELIMINAN NODOS EN FUNCION DEL GRADO CON ORDEN ASCENDIENTE

def PERCOLACION_3(G_aux, final):
    nodes = final['Node']
    s = [0]
    n_nodes = [0]
    S = [0]
    for j in percolacion:
        umbral = j
        n_node_delete = (1-umbral) * len(nodes)
        G_per = G_aux.copy()
        for i in range(int(n_node_delete)):
            pos = len(nodes)-i-1
            G_per.remove_node(nodes[pos])
        G_per_cc = G_per.subgraph(sorted(nx.connected_components(G_per), key=len, reverse=True)[0])
        s.append(G_per_cc.number_of_nodes())   
        n_nodes.append(G_per.number_of_nodes())
        S.append(G_per_cc.number_of_nodes()/G_per.number_of_nodes())
    return(S)

Sp3_Real = PERCOLACION_3(Gcc, final_Real)
Sp3_ER1 = PERCOLACION_3(G_ER1, final_ER1)
Sp3_ER2 = PERCOLACION_3(G_ER2, final_ER2)
Sp3_ER3 = PERCOLACION_3(G_ER3, final_ER3)
Sp3_ER4 = PERCOLACION_3(G_ER4, final_ER4)
Sp3_C1 = PERCOLACION_3(G_C1, final_C)
Sp3_C2 = PERCOLACION_3(G_C2, final_C)
Sp3_C3 = PERCOLACION_3(G_C3, final_C)
Sp3_C4 = PERCOLACION_3(G_C4, final_C)
Sp3_C5 = PERCOLACION_3(G_C5, final_C)
Sp3_C6 = PERCOLACION_3(G_C6, final_C)
Sp3_C7 = PERCOLACION_3(G_C7, final_C)

# PLOT DE LA EVOLUCION DE LA COMPONENTE GIGANTE
x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
plt.figure()
plt.plot(x, Sp3_Real, color = 'green')
plt.plot(x, Sp3_ER1, color = 'blue')
plt.plot(x, Sp3_ER2, color = 'blue')
plt.plot(x, Sp3_ER3, color = 'blue')
plt.plot(x, Sp3_ER4, color = 'blue')
plt.plot(x, Sp3_C1, color = 'red')
plt.plot(x, Sp3_C2, color = 'red')
plt.plot(x, Sp3_C3, color = 'red')
plt.plot(x, Sp3_C4, color = 'red')
plt.plot(x, Sp3_C5, color = 'red')
plt.plot(x, Sp3_C6, color = 'red')
plt.plot(x, Sp3_C7, color = 'red')
plt.xlabel('% ocupacion')
plt.ylabel('S')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title('Evolucion de S')
txt = "Percolacion_No Uniforme_GradAesc"
ruta = "Imagenes/"+ txt +".jpg"
plt.savefig(ruta)
plt.show()

# VISUALIZACION DE LA EVOLUCION DE LA PERCOLACION EN EL GRAFO - REPARAR O HACER A MANO, NO SALEN BIEN
txt = "Percolacion_No_Uniforme_Asc_C7"
txtcc = "Percolacion_No_Uniforme_Asc_C7_CC"
G_aux = G_C7
nodes = final_C['Node']
for j in percolacion:
    umbral = j
    n_node_delete = (1-umbral) * len(nodes)
    G_per = G_aux.copy()
    for i in range(int(n_node_delete)):
        pos = len(nodes)-i-1
        G_per.remove_node(nodes[pos])
    G_per_cc = G_per.subgraph(sorted(nx.connected_components(G_per), key=len, reverse=True)[0])          
    # VISUALIZACION G_PER
    degree_sequence = sorted((d for n, d in G_per.degree()), reverse=True)
    dmax = max(degree_sequence)
    fig = plt.figure("Degree of a random graph", figsize=(16, 16))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)
    ax0 = fig.add_subplot(axgrid[0:3, :])
    pos = nx.spring_layout(G_per, seed=10396953)
    nx.draw_networkx_nodes(G_per, pos, ax=ax0, node_size=10)
    nx.draw_networkx_edges(G_per, pos, ax=ax0, alpha=0.4)
    ax0.set_title("All components of G")
    ax0.set_axis_off()
    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")
    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Number of Nodes")
    txt2 = str(j)
    plt.title('Evolucion de S')
    ruta = "Imagenes/"+ txt + '_' + txt2 + ".jpg"
    plt.savefig(ruta)
    fig.tight_layout()
    plt.show()
    #VISUALIZACION G_PER_CC
    degree_sequence = sorted((d for n, d in G_per_cc.degree()), reverse=True)
    dmax = max(degree_sequence)
    fig = plt.figure("Degree of a random graph", figsize=(16, 16))
    # Create a gridspec for adding subplots of different sizes
    axgrid = fig.add_gridspec(5, 4)
    ax0 = fig.add_subplot(axgrid[0:3, :])
    pos = nx.spring_layout(G_per_cc, seed=10396953)
    nx.draw_networkx_nodes(G_per_cc, pos, ax=ax0, node_size=10)
    nx.draw_networkx_edges(G_per_cc, pos, ax=ax0, alpha=0.4)
    ax0.set_title("All components of G")
    ax0.set_axis_off()
    ax1 = fig.add_subplot(axgrid[3:, :2])
    ax1.plot(degree_sequence, "b-", marker="o")
    ax1.set_title("Degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")
    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_title("Degree histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Number of Nodes")
    txt2 = str(j)
    ruta = "Imagenes/"+ txtcc + '_' + txt2 +".jpg"
    plt.savefig(ruta)
    fig.tight_layout()
    plt.show()