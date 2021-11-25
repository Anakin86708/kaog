from functools import cached_property
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class Distancias:
    METRIC = 'euclidean'

    def __init__(self, x: pd.DataFrame):
        """
        Calcula as distâncias e vizinhos mais próximos de cada ponto utilizando a métrica da classe.

        :param x: Dados de entrada, sem informação de classes.
        :type x: pandas.DataFrame
        """
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

    def k_vizinhos_mais_proximos_de(self, indice: int, k: int = None) -> np.ndarray:
        """
        Com base no índice do pandas e fazendo uso do mapa de índices, retorna os k-vizinhos mais próximos de um
        determinado ponto.

        :param indice: Índice do pandas.
        :type indice: int
        :param k: Quantidade de vizinhos mais próximos. Por padrão, retorna todos.
        :type k: int
        :return: Índices dos vizinhos mais próximos.
        :rtype: numpy.ndarray
        """
        if k is None:
            k = self.x.shape[0] - 1
        numpy_indice_ = self.vizinhos[self.index_pandas_to_numpy(indice)][:k]
        return np.array(list(map(self.index_numpy_to_pandas, numpy_indice_)))

    def distancias_de(self, indice: int) -> np.ndarray:
        """
        Retorna as distâncias de um ponto para todos os outros.

        :param indice: Índice do pandas.
        :type indice: int
        :return: Array com as distâncias.
        :rtype: numpy.ndarray
        """
        return self.distancias[self.index_pandas_to_numpy(indice)]

    def distancia_entre(self, indice_1: int, indice_2: int) -> float:
        """
        Retorna a distância entre dois pontos.

        :param indice_1: Índice do pandas.
        :type indice_1: int
        :param indice_2: Índice do pandas.
        :type indice_2: int
        :return: Distância entre os dois pontos.
        :rtype: float
        """
        vizinhos_1 = self.k_vizinhos_mais_proximos_de(indice_1)
        pos = np.where(vizinhos_1 == indice_2)[0][0]
        return self.distancias_de(indice_1)[pos]

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
        k = self.x.shape[0]
        nn = NearestNeighbors(n_neighbors=k, metric=self.METRIC, n_jobs=-1, algorithm='ball_tree').fit(self.x)

        # Decrementa k para considerar o próprio ponto
        distances, kneighbors = nn.kneighbors(n_neighbors=k - 1, return_distance=True)
        self._ordenar(distances, kneighbors)

        return distances, kneighbors

    @staticmethod
    def _ordenar(distances, kneighbors):
        """
        Ordena os arrays de distâncias e vizinhos mais próximos, de forma que estejam ordenados de forma crescente pela
        distância e em seguida ordenados pelos índices dos vizinhos.
        Isso garante que se houver empate no valor da distância, será considerado a ordem dos índices dos vizinhos.

        :param distances: Array com as distâncias entre os pontos.
        :type distances: numpy.ndarray
        :param kneighbors: Array com os índices dos vizinhos mais próximos.
        :type kneighbors: numpy.ndarray
        """
        for i, (d, k_) in enumerate(zip(distances, kneighbors)):
            sort_idx = np.lexsort((k_, d))
            distances[i] = d[sort_idx]
            kneighbors[i] = k_[sort_idx]
