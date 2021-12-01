import unittest

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
        edges_15 = [(18, 15), (14, 18), (15, 18)]
        self.assertEqual(comp_15, grafo.obter_componentes_contendo(15))
        self.assertEqual(pureza_15, grafo.pureza(15))
        self.assertEqual(edges_15, list(grafo.grafo.subgraph(comp_15).edges()))

    def test_k6(self):
        k = 6
        grafo = self.kaog.grafos_associados[k]
        self.assertEqual(k, grafo.k)

        # Comparar a distancia
        dist = grafo.distancias
        self.assertAlmostEqual(0.2, dist.distancia_entre(43, 26))
        self.assertAlmostEqual(0.223607, dist.distancia_entre(43, 21), places=5)
        self.assertLess(dist.distancia_entre(43, 23), dist.distancia_entre(43, 21))
        self.assertLess(dist.distancia_entre(43, 26), dist.distancia_entre(43, 21))
        self.assertLessEqual(dist.distancia_entre(43, 40), dist.distancia_entre(43, 60))
        self.assertLessEqual(dist.distancia_entre(43, 45), dist.distancia_entre(43, 60))

        # Comparar alguns componentes do grafo conhecidos
        self.assertNotIn((43, 60), grafo.grafo.edges)
        self.assertIn((43, 23), grafo.grafo.edges)
        self.assertIn((43, 21), grafo.grafo.edges)
        self.assertIn((43, 26), grafo.grafo.edges)
        self.assertEqual(1, grafo.pureza(43))

    def test_grafo_otimo(self):
        otimo = self.kaog.grafo_otimo
        pureza_15 = 1.0
        k_comp_15 = 7
        vertices_pertencentes = [13, 43, 21, 11, 17, 1, 5, 32]

        comp_15 = otimo.obter_componente_contendo(15)
        for vertice in vertices_pertencentes:
            with self.subTest(vertice=vertice):
                self.assertIn(vertice, comp_15)
        self.assertEqual(pureza_15, otimo.pureza(comp_15))
        self.assertEqual(k_comp_15, otimo.obter_k_de_componente(comp_15))


if __name__ == '__main__':
    unittest.main()
