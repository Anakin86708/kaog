from functools import cached_property
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class Distancias:
    METRIC = 'euclidean'

    def __init__(self, k: int, x: pd.DataFrame):
        self._k = k
        self.x = x.copy()
        self.index_map = self._create_map_pandas_to_numpy()
        self._distancias, self._vizinhos = self._calcular_distancias_e_vizinhos()

    @cached_property
    def distancias(self):
        """Compensar pelo primeiro valor que corresponde ao mesmo ponto"""
        return self._distancias[:, 1:]

    @cached_property
    def vizinhos(self) -> np.ndarray:
        """Compensar pelo primeiro valor que corresponde ao mesmo ponto"""
        return self._vizinhos[:, 1:]

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, k: int):
        self._k = k
        self._distancias, self._vizinhos = self._calcular_distancias_e_vizinhos()

    def _create_map_pandas_to_numpy(self) -> Dict[int, int]:
        """
        Mapear cada índice da coluna do DataFrame para um número inteiro respectivo a linha da matriz.
        :return: Dicionário com o índice do DataFrame como chave e o índice da matriz como valor.
        :rtype: Dict[int, int]
        """
        return {k: v for v, k in enumerate(self.x.index)}

    def _calcular_distancias_e_vizinhos(self) -> (np.ndarray, np.ndarray):
        """
        Cria uma matriz de distâncias entre os pontos de `self.x`, assim como os índices dos vizinhos mais próximos.

        :return: Matriz de distâncias e matriz com os vizinhos mais próximos.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """
        # TODO: implementar o HEOM
        # Compensação para o primeiro vizinho que corresponde ao próprio ponto.
        k_ = self.k + 1
        nn = NearestNeighbors(n_neighbors=k_, metric=self.METRIC, n_jobs=-1).fit(self.x)
        return nn.kneighbors(self.x)

    def vizinhos_mais_proximos_de(self, indice: int) -> np.ndarray:
        """
        Com base no índice do pandas e fazendo uso do mapa de índices, retorna os vizinhos mais próximos de um
        determinado ponto.

        :param indice: Índice do pandas.
        :type indice: int
        :return: Índices dos vizinhos mais próximos.
        :rtype: numpy.ndarray
        """
        return self.vizinhos[self.index_map[indice]]
