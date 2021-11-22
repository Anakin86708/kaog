from statistics import mean
from typing import List, Dict, Set, Union

import networkx as nx


class GrafoOtimo(nx.DiGraph):
    def __init__(self, nodes, edges, **attr):
        super().__init__(**attr)
        self.add_nodes_from(nodes)
        self.add_edges_from(edges)

        self._componente_e_k: Dict[frozenset[int], int] = {k: 1 for k in self.componentes}

    @property
    def componentes(self) -> List[frozenset[int]]:
        return list(map(frozenset, self._gen_componentes()))

    def _gen_componentes(self):
        """Gerador para os componentes do grafo."""
        return nx.algorithms.weakly_connected_components(self)

    def obter_k_de_componente(self, componente: frozenset[int]) -> int:
        """
        Obtém o valor de k do grafo associado do qual o componente foi obtido.

        :param componente: Conjunto de vértices do componente.
        :type componente: frozenset[int]
        :return: Valor de k.
        :rtype: int
        """
        return self._componente_e_k[componente]

    def pureza(self, componente: frozenset[int]) -> float:
        """
        Calcula a pureza de determinado componente ótimo.

        :param componente: Deve pertencer aos componentes ótimos.
        :return: Valor da pureza
        """
        k = self.obter_k_de_componente(componente)
        pureza = self._obter_media_grau_componente(componente) / (2 * k)
        if not 0 <= pureza <= 1:
            raise RuntimeError(f'O valor da pureza do componente {componente} é {pureza}, fora do intervalo [1,0].')
        return pureza

    def obter_componente_contendo(self, vertice: int) -> frozenset[int]:
        """
        Retorna o componente conectado ao vértice.

        :param vertice: Vértice a ser buscado.
        :type vertice: int
        :return: Componente conectado ao vértice.
        :rtype: frozenset[int]
        """
        for comp in self._gen_componentes():
            if vertice in comp:
                return frozenset(comp)
        raise ValueError(f'O vértice {vertice} não pertence a nenhum componente.')

    def adicionar_componente_otimo(self, novo_componente: nx.DiGraph, k: int):
        """
        Método que deve ser invocado quando necessário adicionar um novo componente ótimo, já que faz a remoção dos
        anteriores e a associação do valor de k.

        :param novo_componente: Subgrafo indicando aquele componente ótimo.
        :param k: Valor de k do qual o componente foi tirado.
        :return:
        """
        # Iterar todos os componentes ótimos e remover os que estão no novo componente
        novo_componente_ = frozenset(list(nx.algorithms.weakly_connected_components(novo_componente))[0])
        to_remove = []
        for componente_otimo in self._componente_e_k.keys():
            if componente_otimo.issubset(novo_componente_):
                to_remove.append(componente_otimo)

        for item in to_remove:
            del self._componente_e_k[item]

        # Adicionar o novo componente ótimo
        self._componente_e_k[novo_componente_] = k

        # Atualizar o grafo ótimo
        # self.remove_edges_from(self.subgraph(novo_componente_).edges)
        self.add_nodes_from(novo_componente.nodes)
        self.add_edges_from(novo_componente.edges)

    def _obter_media_grau_componente(self, componente: Union[Set[int], frozenset[int]]) -> float:
        """
        Obtém a média do grau dos vértices do componente conectado ao vértice.

        :param componente: Vértice a ser buscado.
        :type componente: Set[int]
        :return: Média do grau dos vértices do componente conectado ao vértice.
        :rtype: float
        :raises ValueError: Se o vértice não estiver conectado ao grafo.
        """
        graus = [self.degree(i) for i in componente]
        return mean(graus)
