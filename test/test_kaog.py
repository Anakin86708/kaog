import unittest

import pandas as pd

from kaog import KAOG, KAssociado, NOME_COLUNA_Y


class KAOGTest(unittest.TestCase):

    def setUp(self) -> None:
        x = [
            (-1, -1),
            (-2, -1),
            (-3, -2),
            (1, 1),
            (2, 1),
            (3, 2),
            (0, -1),
            (1, -3),
            (2, -2),
        ]
        self.x = pd.DataFrame(x)
        self.y = pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 0], index=self.x.index, name=NOME_COLUNA_Y)
        self.data = pd.concat([self.x, self.y], axis=1)

    def test_kaog(self):
        instance = KAOG(self.data.copy())
        self.assertIsInstance(instance, KAOG)

    def test_criar_grafo_associado(self):
        instance = KAOG(self.data.copy())
        k = 2
        k_associado = instance._criar_grafo_associado(k)
        self.assertIsInstance(k_associado, KAssociado)
        self.assertEqual(k, k_associado.k)

    def test_calcular_ultima_taxa(self):
        data = self.data.iloc[:7].copy()
        instance = KAOG(data.copy())
        k = 2
        mock_grafos_associados = {k: KAssociado(k, data), k + 1: None}
        assert len(mock_grafos_associados) == k
        instance.grafos_associados = mock_grafos_associados

        taxa = instance._calcular_ultima_taxa()
        expected = 7
        self.assertEqual(expected, taxa)

    def test_obter_componentes_otimo(self):
        instance = KAOG(self.data)
        k = 1
        grafo_otimo = KAssociado(k, self.data)
        componente_k = frozenset({0, 1, 2, 6, 7, 8})
        expected = [frozenset({0, 1, 2, 6}), frozenset({8, 7})]
        componentes_otimo = instance._obter_componentes_otimo(componente_k, grafo_otimo)
        self.assertEqual(expected, componentes_otimo)


if __name__ == '__main__':
    unittest.main()
