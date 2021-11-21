from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd

from kaog import NOME_COLUNA_Y
from kaog._grafo_otimo import GrafoOtimo
from kaog.k_associado import KAssociado
from kaog.util.draw import DrawableGraph


class KAOG(DrawableGraph):
    def __init__(self, data: pd.DataFrame):
        self._data = data.copy()

        self.grafos_associados: Dict[int, KAssociado] = {}
        self.componentes_otimos: Dict[frozenset[int], int] = {}  # Mapeia o valor de k do componente escolhido
        self._criar_kaog()

    @property
    def data(self):
        return self._data.copy()

    @property
    def x(self):
        return self.data.drop(NOME_COLUNA_Y, axis=1)

    @property
    def y(self):
        return self.data[NOME_COLUNA_Y]

    @property
    def grafo(self):
        return self.grafo_otimo

    @property
    def componentes(self):
        return self.grafo_otimo.componentes

    def _criar_kaog(self):
        k = 1
        self._iniciar_grafo_otimo()
        while 1:
            ultima_taxa = self._calcular_ultima_taxa()
            k += 1
            grafo_k = self._criar_grafo_associado(k)

            # Iterar por todos os novos componentes do grafo k-associado
            for componente_k in grafo_k.componentes:
                pureza_k = grafo_k.pureza(componente_k)
                componentes_otimo = self._obter_componentes_otimos(componente_k)
                purezas_componentes_otimos = self._calcular_pureza_componentes_otimos(componentes_otimo)
                if (pureza_k >= purezas_componentes_otimos).all():
                    self._inserir_novo_componente_otimo(k, grafo_k.grafo.subgraph(componente_k))

            if self._calcular_ultima_taxa() < ultima_taxa:
                break

    def _calcular_pureza_componentes_otimos(self, componentes_otimo: List[frozenset[int]]) -> np.ndarray:
        """
        Calcula a pureza de cada um dos componentes ótimos.

        :param componentes_otimo: Componentes pertencentes ao grafo ótimo.
        :type componentes_otimo: List[frozenset[int]]
        :return: Array com a pureza de cada um dos componentes ótimos.
        :rtype: np.ndarray
        """
        return np.array([self.grafo_otimo.pureza(comp) for comp in componentes_otimo])

    def _obter_componentes_otimos(self, componente_k: frozenset[int]) -> List[frozenset[int]]:
        """
        Com base no `componente_k`, encontrar todos os outros componentes que estão no grafo ótimo e que foram unidos
        para formar o `componente_k`.

        :param componente_k: Componente a ser buscado os componentes ótimos.
        :type componente_k: frozenset[int]
        :return: Lista contendo dos componentes ótimos que foram unidos para formar o `componente_k`.
        :rtype: List[frozenset[int]]
        """
        componentes_otimo = []
        vertices_usados = set()  # Usado para saber quais vertices ja foram usados
        for vertice in componente_k:
            if vertice not in vertices_usados:
                # Se o vertice não foi usado ainda, entao ele é um vertice de um dos componentes ótimos
                componentes_otimo.append(self.grafo_otimo.obter_componente_contendo(vertice))
                vertices_usados = vertices_usados.union(componentes_otimo[-1])
        return componentes_otimo

    def _iniciar_grafo_otimo(self):
        """Inicia o grafo ótimo como um grafo 1-associado."""
        k = 1
        k_associado = self._criar_grafo_associado(k)
        nodes = k_associado.grafo.nodes()
        edges = k_associado.grafo.edges()
        self.grafo_otimo = GrafoOtimo(nodes, edges)

    def _criar_grafo_associado(self, k: int):
        """
        Cria um novo grafo k-associado com base em `k` e armazena nos grafos criados.

        :param k: Valor de k para o grafo.
        :type k: int
        :return: Novo grafo k-associado.
        """
        k_associado = KAssociado(k, self.data)
        self.grafos_associados[k] = k_associado
        return k_associado

    def _calcular_ultima_taxa(self) -> float:
        """
        Calcula a última taxa do último grafo k-associado.

        :return: Valor da taxa.
        :rtype: float
        """
        k = max(self.grafos_associados)
        ultimo_grafo = self.grafos_associados[k]
        return ultimo_grafo.media_grau_componentes() / k

    def _inserir_novo_componente_otimo(self, k: int, componente_k: nx.DiGraph):
        """
        Adiciona o novo componente ótimo ao grafo ótimo.

        :param k: Valor de k associado ao componente em questão.
        :type k: int
        :param componente_k: Componente ótimo a ser adicionado ao grafo ótimo.
        :type componente_k: nx.DiGraph
        """
        self.grafo_otimo.adicionar_componente_otimo(novo_componente=componente_k, k=k)
