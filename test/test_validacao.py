import unittest

import networkx as nx
import pandas as pd
from sklearn.datasets import load_iris

from kaog import KAOG


class ValidacaoTest(unittest.TestCase):
    """
    Testes voltados para a validação do algoritmo, com base na Tese do Prof. Dr. João Roberto Bertini Jr.
    """

    def setUp(self) -> None:
        """Tratamento de dados para testes, conforme especificado na tese."""
        iris = load_iris()
        x = pd.DataFrame(iris.data, columns=iris.feature_names)
        x = x.drop(columns=['sepal width (cm)', 'petal length (cm)'])
        y = pd.DataFrame(iris.target, columns=['target'])
        data = pd.concat([x, y], axis=1)
        data = data.drop_duplicates()

        self.kaog = KAOG(data)

    def test_k1(self):
        k = 1
        grafo = self.kaog.grafos_associados[k]
        self.assertEqual(k, grafo.k)

        # Comparar alguns componentes do grafo conhecidos
        comp_15 = frozenset({15, 18, 14})
        pureza_15 = 1.0
        edges_15 = nx.OutEdgeView([(18, 15), (14, 18), (15, 18)])
        self.assertEqual(comp_15, grafo.obter_componentes_contendo(15))
        self.assertEqual()


if __name__ == '__main__':
    unittest.main()
