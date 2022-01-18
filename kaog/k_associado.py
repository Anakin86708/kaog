from statistics import mean
from typing import Dict, Set, Union, List

import networkx as nx
import pandas as pd

from kaog.distancias import Distancias
from kaog.util import NOME_COLUNA_Y
from kaog.util.draw import DrawableGraph


class KAssociado(DrawableGraph):
    """Representação de um grafo k-associado.

    **KAssociado**

    Grafo k-associado é um grafo que possui um número máximo de k ligações originando de cada vértice. Essa ligação é
    feita considerando os vértices mais próximos e a conexão só é feita se a classe dos dois vértices for igual.
    Sendo assim, para as distâncias, não importa as classes dos vértices. A classe só é utilizada para a conexão.
    """

    def __init__(self, k: int, data: pd.DataFrame, colunas_categoricas: pd.Index = pd.Index([])):
        """
        Cada instância de `data` é representada como um vértice, que será conectado a todos seus `k` vizinhos mais
        próximos, se pertencerem a mesma classe.
        Diferente do KNN, o grafo K-associado só permite conexões de mesma classe e conexões múltiplas são mantidas.

        :param k: Quantidade máxima de conexões dos vértices.
        :type k: int
        :param data: Conjunto de dados com classe associada.
        :type data: pd.DataFrame
        """
        self._k = k
        self._data: pd.DataFrame = data.copy()
        self.distancias = Distancias(self.x, colunas_categoricas)

        vizinhos = self._determinar_vizinhos()

        self._edgelist = self._create_edgelist(vizinhos)
        self._grafo = self._criar_grafo()

    @property
    def grafo(self):
        """Grafo k-associado gerado."""
        return self._grafo

    @property
    def k(self):
        """Valor de k do grafo em questão."""
        return self._k

    @property
    def data(self) -> pd.DataFrame:
        """Conjunto de dados, com classe associada."""
        return self._data.copy()

    @property
    def x(self) -> pd.DataFrame:
        """Dados sem classe associada."""
        return self.data.drop(NOME_COLUNA_Y, axis=1, errors='ignore')

    @property
    def y(self) -> pd.Series:
        """Classe de cada vértice."""
        return self.data[NOME_COLUNA_Y]

    @property
    def componentes(self) -> List[frozenset[int]]:
        """Componentes do grafo."""
        return list(map(frozenset, self._gen_componentes()))

    @staticmethod
    def _create_edgelist(vizinhos):
        """Cria a lista de arestas do grafo, considerando os vizinhos."""
        return [(x, y) for x in vizinhos for y in vizinhos[x]]

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
        vertice_pertencente = self._sanitize_pureza(componente)

        componente = self.obter_componentes_contendo(vertice_pertencente)
        pureza = self._obter_media_grau_componente(componente) / (2 * self.k)
        if not 0 <= pureza <= 1:
            raise RuntimeError(f'O valor da pureza do componente {componente} é {pureza}, fora do intervalo [1,0].')
        return pureza

    def draw(self, title=None, color_by_component=False):
        """
        Desenha o grafo. Por padrão, a cor de cada vértice é a sua classe.

        :param title: Título do gráfico.
        :type title: str
        :param color_by_component: Se deve colorir os vértices por componente ao invés das classes.
        :type color_by_component: bool
        """
        if title is None:
            title = f'{self.k}-associado'
        super().draw(title=title, color_by_component=color_by_component)

    def _gen_componentes(self):
        """Gerador para os componentes do grafo."""
        return nx.algorithms.weakly_connected_components(self.grafo)

    def _sanitize_pureza(self, componente: Union[int, Set[int], frozenset[int]]):
        """
        Com base no tipo do componente, retorna ao menos um vértice pertencente ao componente.

        :param componente: Podem ser um vértice ou um conjunto de vértices.
        :type componente: Union[int, Set[int]]
        :return: Vértice pertencente ao componente.
        :rtype: int
        :raises ValueError: Se o componente ou vértice não pertence ao grafo.
        :raises TypeError: Se o tipo do argumento `componente` não for int, Set[int] ou frozenset[int].
        """
        if isinstance(componente, set) or isinstance(componente, frozenset):
            if componente not in self.componentes:
                raise ValueError(f'O componente {componente} não pertence ao grafo.')
            vertice_pertencente = next(iter(componente))
        elif isinstance(componente, int):
            vertice_pertencente = componente
        else:
            raise TypeError(f'O argumento `componente` deve ser int, Set[int] ou frozenset[int].')
        return vertice_pertencente

    # noinspection PyTypeChecker

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
        :rtype: frozenset[int]
        :raises ValueError: Se o vértice não estiver conectado ao grafo.
        """
        for comp in self._gen_componentes():
            if vertice in comp:
                return frozenset(comp)
        raise ValueError(f'O vértice {vertice} não pertence a nenhum componente.')

    def adicionar_arestas(self, novas_arestas):
        """Adiciona as arestas ao grafo."""
        self.grafo.add_edges_from(novas_arestas)

    def _determinar_vizinhos(self) -> Dict[int, pd.Index]:
        """
        Determina os `k` vizinhos mais próximos de cada vértice, apenas se foram de mesma classe.

        :return: Dicionário com os `k` vizinhos mais próximos de cada vértice.
        :rtype: dict
        """
        vizinhos: Dict[int, pd.Index] = {}
        for idx, row in self.x.iterrows():
            vizinhos_ = self.distancias.k_vizinhos_mais_proximos_de(row, self.k)
            # Verificar as classes
            y_vizinhos = self.y[vizinhos_]
            # Manter apenas os vizinhos que pertençam a mesma classe
            vizinhos_ = y_vizinhos.where(y_vizinhos == self.y[idx]).dropna().index
            vizinhos[idx] = vizinhos_
        return vizinhos

    # noinspection PyTypeChecker

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

    def _criar_grafo(self):
        """Cria o grafo a partir dos dados do dataset e a lista de arestas."""
        graph = nx.DiGraph()
        graph.add_nodes_from(self.x.index)
        graph.add_edges_from(self._edgelist)
        return graph
