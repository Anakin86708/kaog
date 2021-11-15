from functools import cached_property
from typing import Dict

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
        self.grafo = nx.from_edgelist(self._edgelist)

    @property
    def data(self) -> pd.DataFrame:
        return self._data.copy()

    @cached_property
    def x(self) -> pd.DataFrame:
        return self.data.drop(NOME_COLUNA_Y, axis=1)

    @cached_property
    def y(self) -> pd.Series:
        return self.data[NOME_COLUNA_Y]

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

    def _create_edgelist(self, vizinhos):
        return [(x, y) for x in vizinhos for y in vizinhos[x]]
