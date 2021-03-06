import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Callable, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class Distancias:
    """Cálculo de distancias entre pontos e os vizinhos mais próximos.

    **Distancias**

    Método de cálculo de distâncias entre pontos por meio da métrica definida em `METRIC`.

    **Atributos**
    METRIC
        Métrica de cálculo de distâncias. Pode ser definida como uma função ou como um nome de métrica reconhecida pelo
        NearestNeighbors.

    """
    METRIC: Union[str, Callable] = 'euclidean'

    def __init__(self, x: pd.DataFrame, colunas_categoricas: pd.Index = pd.Index([])):
        """
        Recebe o DataFrame com os pontos que serão calculadas as distâncias.

        :param x: Dados de entrada, sem informação de classes.
        :type x: pandas.DataFrame
        """
        self.x = x.copy()
        self.cat_cols = colunas_categoricas.copy()
        self.index_map = self._create_map_pandas_to_numpy()
        self._distancias, self._vizinhos = self._calcular_distancias_e_vizinhos(self.x)

    @property
    def distancias(self):
        """Array com as distâncias entre os pontos."""
        return self._distancias

    @property
    def vizinhos(self) -> np.ndarray:
        """Apenas os índices para os vizinhos, **sem considerar** as informações de classes!"""
        return self._vizinhos

    @property
    def rever_index_max(self):
        """Possibilita a conversão de índice da matriz para índice do DataFrame."""
        return {v: k for k, v in self.index_map.items()}

    def k_vizinhos_mais_proximos_de(self, instancia: Union[pd.Series, int], k: int = None) -> np.ndarray:
        """
        Com base no índice do pandas e fazendo uso do mapa de índices, retorna os k-vizinhos mais próximos de um
        determinado ponto.

        :param instancia: Instância sendo buscada. Pode representar um `pd.Series` ou um índice. Caso seja uma `pd.Series`, o índice deve estar em `name`.
        :type instancia: Union[pd.Series, int]
        :param k: Quantidade de vizinhos mais próximos. Por padrão, retorna todos.
        :type k: int
        :return: Índices dos vizinhos mais próximos.
        :rtype: numpy.ndarray
        """
        indice = self._determinar_indice(instancia)
        k = self._determinar_k(k)
        try:
            # Obter os vizinhos mais próximos, considerando que o ponto buscado está incluso.
            numpy_indice_ = self.vizinhos[self.index_pandas_to_numpy(indice)][:k]
        except (IndexError, KeyError):
            # Recalcular distâncias e vizinhos, agora com a instancia buscada.
            numpy_indice_ = self._vizinhos_com_instancia(instancia, k)

        return np.array(list(map(self.index_numpy_to_pandas, numpy_indice_)))

    def distancias_de(self, indice: Union[pd.Series, int]) -> np.ndarray:
        """
        Retorna as distâncias de um ponto para todos os outros.

        :param instancia: Instância sendo buscada. Pode representar um `pd.Series` ou um índice. Caso seja uma `pd.Series`, o índice deve estar em `name`.
        :type indice: Union[pd.Series, int]
        :return: Array com as distâncias.
        :rtype: numpy.ndarray
        """
        indice = self._determinar_indice(indice)
        return self.distancias[self.index_pandas_to_numpy(indice)]

    def distancia_entre(self, indice_1: Union[pd.Series, int], indice_2: Union[pd.Series, int]) -> float:
        """
        Retorna a distância entre dois pontos.

        :param indice_1: Instância sendo buscada. Pode representar um `pd.Series` ou um índice. Caso seja uma `pd.Series`, o índice deve estar em `name`.
        :type indice_1: Union[pd.Series, int]
        :param indice_2: Instância sendo buscada. Pode representar um `pd.Series` ou um índice. Caso seja uma `pd.Series`, o índice deve estar em `name`.
        :type indice_2: Union[pd.Series, int]
        :return: Distância entre os dois pontos.
        :rtype: float
        """
        indice_1 = self._determinar_indice(indice_1)
        indice_2 = self._determinar_indice(indice_2)

        vizinhos_1 = self.k_vizinhos_mais_proximos_de(indice_1)
        pos = np.where(vizinhos_1 == indice_2)[0][0]
        return self.distancias_de(indice_1)[pos]

    @staticmethod
    def _determinar_indice(instancia: Union[pd.Series, int]) -> int:
        """
        Determina se o índice já está representado em um inteiro ou precisa ser extraido de um `pd.Series`.

        :param instancia: Instância sendo buscada. Pode representar um `pd.Series` ou um índice. Caso seja uma `pd.Series`, o índice deve estar em `name`.
        :type instancia: Union[pd.Series, int]
        :return: Índice da instância.
        :rtype: int
        """
        if isinstance(instancia, pd.Series):
            indice = instancia.name
        else:
            indice = instancia
        return indice

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

    def _vizinhos_com_instancia(self, instancia, k):
        """
        Recalcula os vizinhos mais próximos, agora com a instancia buscada. Usado quando o ponto buscado não está em `self.x`.

        :param instancia: Instância sendo buscada.
        :type instancia: pd.Series
        :param k: Quantidade de vizinhos mais próximos a serem retornados.
        :type k: int
        :return: Índices dos vizinhos mais próximos.
        :rtype: numpy.ndarray
        :raises: AttributeError: Se instância não for um `pd.Series`.
        """
        try:
            _, vizinhos = self._calcular_distancias_e_vizinhos(pd.concat([self.x, instancia.to_frame().T]))
            numpy_indice_ = vizinhos[-1][:k]  # Ao usar o concat, o índice é o último.
        except AttributeError as e:
            # Se não for um Series, é um inteiro.
            logging.error('É necessário passar um Series como parâmetro quando o índice não está presente em `x`.')
            raise e
        return numpy_indice_

    def _determinar_k(self, k: Union[int, None]) -> int:
        """
        Determina o valor de `k` a ser usado. Caso seja None, retorna o tamanho do DataFrame menos 1, compensando o próprio ponto.

        :param k: Número de vizinhos mais próximos a ser calculado.
        :type k: Union[int, None]
        :return: Valor de `k` a ser usado.
        :rtype: int
        """
        if k is None:
            k = self.x.shape[0] - 1
        return k

    def index_pandas_to_numpy(self, index: int) -> int:
        """Converte o índice do pandas para o índice da matriz."""
        return self.index_map[index]

    def index_numpy_to_pandas(self, index: int) -> int:
        """Converte o índice da matriz para o índice do pandas."""
        return self.rever_index_max[index]

    def _create_map_pandas_to_numpy(self) -> Dict[int, int]:
        """
        Mapear cada índice da coluna do DataFrame para um número inteiro respectivo a linha da matriz.

        :return: Dicionário com o índice do DataFrame como chave e o índice da matriz como valor.
        :rtype: Dict[int, int]
        :raises ValueError: Se existirem índices repetidos sem `self.x`.
        """
        if self.x.index.duplicated().any():
            raise ValueError('Os índices não podem ser duplicados.')
        return {k: v for v, k in enumerate(self.x.index)}

    def _calcular_distancias_e_vizinhos(self, data: pd.DataFrame) -> (np.ndarray, np.ndarray):
        """
        Cria um array de **todas** as distâncias entre os pontos de `self.x`, assim como os índices dos vizinhos mais
        próximos. Não leva em consideração as classes, apenas as distâncias.

        :param data: DataFrame com os pontos.
        :type data: pd.DataFrame
        :return: Array de distâncias e array com os vizinhos mais próximos.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """
        k = data.shape[0]
        x = self._categoricos_para_numericos(data)
        logging.debug('Calculando distâncias e vizinhos...')
        # distances = self._computar_distancias(x)
        nn = NearestNeighbors(n_neighbors=k, metric=self.METRIC, n_jobs=1, algorithm='ball_tree').fit(x)

        # Decrementa k para considerar o próprio ponto
        distances, kneighbors = nn.kneighbors(n_neighbors=k - 1, return_distance=True)
        logging.debug('Distâncias calculadas.')
        self._ordenar(distances, kneighbors)
        return distances, kneighbors

    def _categoricos_para_numericos(self, x: pd.DataFrame):
        """
        Converte os valores que estão em colunas categóricas para valores numéricos, permitindo que seja aplicada a
        distância.

        :param x: Conjunto de dados, sem informação de classes.
        :type x: pandas.DataFrame
        :return: Conjunto de dados, com valores numéricos.
        :rtype: pandas.DataFrame
        """
        x = x.copy()
        logging.debug('Realizando factorize dos dados...')
        for col in self.cat_cols:
            x[col] = pd.factorize(x[col])[0] + 1

        logging.debug('Dados convertidos.')
        return x

    def _computar_distancias(self, x):

        def computar_distancias_da_linha(distances, i, range_, x):
            for j in range(i + 1, range_):
                idx_i = self.index_numpy_to_pandas(i)
                idx_j = self.index_numpy_to_pandas(j)
                distances[i, j] = self.METRIC(x.loc[idx_i], x.loc[idx_j])

        # Usar a conversao de indice de numpy para pandas
        range_ = x.shape[0]
        distances = np.zeros((range_, range_))
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i in range(range_):
                # Passar da posicao i ate o range_
                futures.append(
                    executor.submit(computar_distancias_da_linha, distances, i, range_, x)
                )

            logging.debug("Aguardando futures...")
            executor.shutdown(wait=True)
            logging.debug("Futures finalizados")

        return distances
