from functools import cached_property
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class Distancias:
    METRIC = 'euclidean'

    def __init__(self, k: int, x: pd.DataFrame):
        """
        Calcula as distâncias e vizinhos mais próximos de cada ponto utilizando a métrica da classe.

        :param k: Quantidade de vizinhos mais próximos.
        :type k: int
        :param x: Dados de entrada.
        :type x: pandas.DataFrame
        """
        self._k = k
        self.x = x.copy()
        self.index_map = self._create_map_pandas_to_numpy()
        self._distancias, self._vizinhos = self._calcular_distancias_e_vizinhos()

    @property
    def distancias(self):
        return self._distancias

    @property
    def vizinhos(self) -> np.ndarray:
        """Apenas os índices para os vizinhos, **sem considerar** as informações de classes!"""
        return self._vizinhos

    @cached_property
    def rever_index_max(self):
        return {v: k for k, v in self.index_map.items()}

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, k: int):
        self._k = k
        self._distancias, self._vizinhos = self._calcular_distancias_e_vizinhos()

    def vizinhos_mais_proximos_de(self, indice: int) -> np.ndarray:
        """
        Com base no índice do pandas e fazendo uso do mapa de índices, retorna os vizinhos mais próximos de um
        determinado ponto.

        :param indice: Índice do pandas.
        :type indice: int
        :return: Índices dos vizinhos mais próximos.
        :rtype: numpy.ndarray
        """
        numpy_indice_ = self.vizinhos[self.index_pandas_to_numpy(indice)]
        return np.array(list(map(self.index_numpy_to_pandas, numpy_indice_)))

    def index_pandas_to_numpy(self, index: int) -> int:
        return self.index_map[index]

    def index_numpy_to_pandas(self, index: int) -> int:
        return self.rever_index_max[index]

    def _create_map_pandas_to_numpy(self) -> Dict[int, int]:
        """
        Mapear cada índice da coluna do DataFrame para um número inteiro respectivo a linha da matriz.
        :return: Dicionário com o índice do DataFrame como chave e o índice da matriz como valor.
        :rtype: Dict[int, int]
        """
        return {k: v for v, k in enumerate(self.x.index)}

    def _calcular_distancias_e_vizinhos(self) -> (np.ndarray, np.ndarray):
        """
        Cria um array de **todas** as distâncias entre os pontos de `self.x`, assim como os índices dos vizinhos mais
        próximos. Não leva em consideração as classes, apenas as distâncias.

        :return: Array de distâncias e array com os vizinhos mais próximos.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """
        # TODO: implementar o HEOM
        k_ = self.x.shape[0]
        nn = NearestNeighbors(n_neighbors=k_, metric=self.METRIC, n_jobs=-1, algorithm='ball_tree').fit(self.x)

        distances, kneighbors = nn.kneighbors(n_neighbors=self.k, return_distance=True)
        for i, (d, k) in enumerate(zip(distances, kneighbors)):
            sort_idx = np.lexsort((k, d))
            distances[i] = d[sort_idx]
            kneighbors[i] = k[sort_idx]

        return distances, kneighbors

    def _remover_distancia_vizinho_mesmo_ponto(self, k_, distances_, kneighbors_):
        replace = -1
        distances = np.zeros((k_, k_ - 1), dtype=float)
        kneighbors = np.zeros((k_, k_ - 1), dtype=int)
        for idx, line in enumerate(kneighbors_):
            idx_buscado = np.where(line == idx)[0][0]  # Encontra o índice do ponto em questão.
            line[idx_buscado] = replace
            kneighbors[idx] = np.array(list(filter(lambda x: x != replace, line)))
            distances_[idx][idx_buscado] = replace
            distances[idx] = np.array(list(filter(lambda x: x != replace, distances_[idx])))
        return distances, kneighbors
