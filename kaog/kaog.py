from typing import Dict

import pandas as pd

from kaog.k_associado import KAssociado


class KAOG:
    def __init__(self, data: pd.DataFrame):
        self._data = data.copy()

        self.grafos_associados: Dict[int, KAssociado] = {}
        self.grafo_otimo = self._criar_kaog()

    @property
    def data(self):
        return self._data.copy()

    def _criar_kaog(self):
        k = 1
        grafo_otimo = self._criar_grafo_associado(k)
        while 1:
            ultima_taxa = self._calcular_ultima_taxa()
            k += 1
            grafo_k = self._criar_grafo_associado(k)
            for componente_k in grafo_k.componentes:
                pureza_k = grafo_k.pureza(componente_k)

                componentes_otimo = self._obter_componentes_otimo(componente_k, grafo_otimo)

            break

    @staticmethod
    def _obter_componentes_otimo(componente_k: frozenset[int], grafo_otimo):
        componentes_otimo = []
        vertices_usados = set()  # Usado para saber quais vertices ja foram usados
        for vertice in componente_k:
            if vertice not in vertices_usados:
                # Se o vertice não foi usado ainda, entao ele é um vertice de um dos componentes ótimos
                componentes_otimo.append(grafo_otimo.obter_componentes_contendo(vertice))
                vertices_usados = vertices_usados.union(componentes_otimo[-1])
        return componentes_otimo

    def _criar_grafo_associado(self, k):
        k_associado = KAssociado(k, self.data)
        self.grafos_associados[k] = k_associado
        return k_associado

    def _calcular_ultima_taxa(self):
        k = len(self.grafos_associados)
        ultimo_grafo = self.grafos_associados[k]
        return ultimo_grafo.media_grau_componentes() / k
