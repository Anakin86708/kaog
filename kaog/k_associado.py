from functools import cached_property
from statistics import mean
from typing import Dict, Set, Union, List

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from kaog import NOME_COLUNA_Y
from kaog.distancias import Distancias


class KAssociado:

    def __init__(self, k: int, data: pd.DataFrame):
        """
        Cada instância de `data` é representada como um vértice, que será conectado a todos seus `k` vizinhos mais próximos que pertencerem a mesma classe.
        Diferente do KNN, o grafo K-associado só permite conexões de mesma classe e conexões múltiplas são mantidas.

        :param k: Quantidade máxima de conexões dos vértices.
        :type k: int
        :param data: Conjunto de dados com classe associada.
        :type data: pd.DataFrame
        """
        self._k = k
        self._data: pd.DataFrame = data.copy()
        self.distancias = Distancias(k, self.x)

        vizinhos = self._determinar_vizinhos()

        self._edgelist = self._create_edgelist(vizinhos)
        self.grafo = self._criar_grafo()

    @property
    def k(self):
        return self._k

    @property
    def data(self) -> pd.DataFrame:
        return self._data.copy()

    @cached_property
    def x(self) -> pd.DataFrame:
        return self.data.drop(NOME_COLUNA_Y, axis=1)

    @cached_property
    def y(self) -> pd.Series:
        return self.data[NOME_COLUNA_Y]

    @property
    def componentes(self) -> List[frozenset[int]]:
        return list(map(frozenset, nx.algorithms.weakly_connected_components(self.grafo)))

    def draw(self):
        nx.draw(self.grafo, with_labels=True)
        plt.show()

    def pureza(self, componente: Union[int, Set[int], frozenset[int]]) -> float:
        """
        Calcula a pureza do componente ao qual o vértice pertence.

        :param componente: Vértice contido no componente a ser calculado ou todo o componente.
        :type componente: Union[int, Set[int]]
        :return: Valor da pureza do componente.
        :rtype: float
        :raises ValueError: Se o componente ou vértice não pertence ao grafo.
        :raises TypeError: Se o tipo do argumento `componente` não for int ou Set[int].
        """
        if isinstance(componente, set) or isinstance(componente, frozenset):
            if componente not in self.componentes:
                raise ValueError(f'O componente {componente} não pertence ao grafo.')
            vertice_pertencente = next(iter(componente))
        elif isinstance(componente, int):
            vertice_pertencente = componente
        else:
            raise TypeError(f'O argumento `componente` deve ser int, Set[int] ou frozenset[int].')

        componente = self.obter_componentes_contendo(vertice_pertencente)
        return self._obter_media_grau_componente(componente) / (2 * self.k)

    def media_grau_componentes(self) -> float:
        """
        Calcula a média do grau de **todos** os componentes contidos no grafo.

        :return: Média do grau dos componentes.
        :rtype: float
        """
        # Obter todos os componentes
        gen_comp = nx.algorithms.weakly_connected_components(self.grafo)
        graus = []
        for comp in list(gen_comp):
            graus.append(sum(map(self.grafo.degree, comp)))
        return mean(graus)

    def obter_componentes_contendo(self, vertice: int) -> frozenset[int]:
        """
        Obtém o componente conectado ao vértice.

        :param vertice: Vértice a ser buscado.
        :type vertice: int
        :return: Componente conectado ao vértice.
        :rtype: Set[int]
        :raises ValueError: Se o vértice não estiver conectado ao grafo.
        """
        gen_comp = nx.algorithms.weakly_connected_components(self.grafo)
        for comp in gen_comp:
            if vertice in comp:
                return frozenset(comp)
        raise ValueError(f'O vértice {vertice} não pertence a nenhuma componente.')

    def adicionar_arestas(self, novas_arestas):
        self.grafo.add_edges_from(novas_arestas)

    def _obter_media_grau_componente(self, componente: Union[Set[int], frozenset[int]]) -> float:
        """
        Obtém a média do grau dos vértices do componente conectado ao vértice.

        :param componente: Vértice a ser buscado.
        :type componente: Set[int]
        :return: Média do grau dos vértices do componente conectado ao vértice.
        :rtype: float
        :raises ValueError: Se o vértice não estiver conectado ao grafo.
        """
        graus = [self.grafo.degree(i) for i in componente]
        return mean(graus)
    # noinspection PyTypeChecker

    def _determinar_vizinhos(self) -> Dict[int, pd.Index]:
        """
        Determina os `k` vizinhos mais próximos de cada vértice, apenas se foram de mesma classe.

        :return: Dicionário com os `k` vizinhos mais próximos de cada vértice.
        :rtype: dict
        """
        vizinhos: Dict[int, pd.Index] = {}
        for idx, row in self.x.iterrows():
            vizinhos_ = self.distancias.vizinhos_mais_proximos_de(idx)
            # Verificar as classes
            y_vizinhos = self.y[vizinhos_]
            vizinhos_ = y_vizinhos.where(y_vizinhos == self.y[idx]).dropna().index
            vizinhos[idx] = vizinhos_
        return vizinhos

    @staticmethod
    def _create_edgelist(vizinhos):
        return [(x, y) for x in vizinhos for y in vizinhos[x]]

    def _criar_grafo(self):
        graph = nx.DiGraph()
        graph.add_edges_from(self._edgelist)
        return graph
