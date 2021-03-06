import unittest

import pandas as pd

from kaog import KAOG, KAssociado
from kaog.util import ColunaYSingleton


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
        self.y = pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 0], index=self.x.index, name=ColunaYSingleton().NOME_COLUNA_Y)
        self.data = pd.concat([self.x, self.y], axis=1)

    def test_instance_kaog(self):
        instance = KAOG(self.data.copy())
        self.assertIsInstance(instance, KAOG)

    def test_criar_kaog(self):
        pass

    def test_calcular_pureza_componentes_otimos(self):
        pass

    def test_obter_componentes_otimos(self):
        pass

    def test_iniciar_grafo_otimo(self):
        pass

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
        mock_grafos_associados = {k: KAssociado(k, data), k - 1: None}
        assert len(mock_grafos_associados) == k
        instance.grafos_associados = mock_grafos_associados

        taxa = instance._calcular_ultima_taxa()
        expected = 7
        self.assertEqual(expected, taxa)

    def test_inserir_novo_componente_otimo(self):
        data = self.data.copy()
        instance = KAOG(self.data.copy())
        pass


if __name__ == '__main__':
    unittest.main()
