import unittest

import numpy as np
import pandas as pd

from kaog import NOME_COLUNA_Y
from kaog.k_associado import KAssociado


class KAssociadoTest(unittest.TestCase):

    def setUp(self) -> None:
        self.k = 2
        x = [
            (-1, -1),
            (-2, -1),
            (-3, -2),
            (1, 1),
            (2, 1),
            (3, 2),
            (0, -1),
        ]
        self.x = pd.DataFrame(x)
        self.y = pd.Series([0, 0, 0, 1, 1, 1, 0], index=self.x.index, name=NOME_COLUNA_Y)
        self.data = pd.concat([self.x, self.y], axis=1)

    def test_k_associado(self):
        k, x = self.k, self.x.copy()
        x.set_index(np.random.randint(0, 255, size=x.shape[0]), inplace=True)
        y = pd.Series([0, 0, 0, 1, 1, 1, 0], index=x.index, name=NOME_COLUNA_Y)
        instance = KAssociado(k, pd.concat([x, y], axis=1))
        self.assertIsInstance(instance, KAssociado)

    def test_data(self):
        instance = self._create_new_instance()
        pd.testing.assert_frame_equal(self.data, instance.data)

    def test_x(self):
        k, x = self.k, self.x.copy()
        x.columns = pd.Index([0, 1], dtype='object')
        y = pd.Series([0, 0, 0, 1, 1, 1, 0], index=x.index, name=NOME_COLUNA_Y)
        data = pd.concat([x, y], axis=1)
        instance = KAssociado(k, data)
        pd.testing.assert_frame_equal(x, instance.x)

    def test_y(self):
        instance = self._create_new_instance()
        pd.testing.assert_series_equal(self.y, instance.y)

    def test_determinar_vizinhos(self):
        instance = self._create_new_instance()
        expected = {
            0: pd.Index([1, 6]),
            1: pd.Index([0, 2]),
            2: pd.Index([1, 0]),
            3: pd.Index([4, 5]),
            4: pd.Index([3, 5]),
            5: pd.Index([4, 3]),
            6: pd.Index([0, 1])
        }
        vizinhos = instance._determinar_vizinhos()
        for idx, expec in expected.items():
            pd.testing.assert_index_equal(expec, vizinhos[idx])

    def test_obter_componentes_contendo(self):
        instance = self._create_new_instance()
        expected = {2: {0, 1, 2, 6}, 3: {3, 4, 5}}
        for idx, expec in expected.items():
            self.assertEqual(expec, instance._obter_componentes_contendo(idx))

    def test_obter_media_grau_componente(self):
        instance = self._create_new_instance()
        expected = {frozenset({0, 1, 2, 6}): 4, frozenset({3, 4, 5}): 4}
        for comp, expec in expected.items():
            self.assertEqual(expec, instance._obter_media_grau_componente(comp))

    def test_pureza(self):
        instance = self._create_new_instance()
        expected = {0: 1, 3: 1}
        for idx, expec in expected.items():
            self.assertEqual(expec, instance.pureza(idx))

    def test_media_grau_componentes(self):
        instance = self._create_new_instance()
        expected = 14
        self.assertEqual(expected, instance.media_grau_componentes())

    ################
    # Util methods #
    ################

    def _create_new_instance(self):
        k, x, y, data = self.k, self.x.copy(), self.y.copy(), self.data.copy()
        instance = KAssociado(k, data)
        return instance


if __name__ == '__main__':
    unittest.main()
