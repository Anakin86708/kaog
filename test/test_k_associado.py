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

    def test_k_associado(self):
        k, x = self.k, self.x.copy()
        x.set_index(np.random.randint(0, 255, size=x.shape[0]), inplace=True)
        y = pd.Series([0, 0, 0, 1, 1, 1, 0], index=x.index, name=NOME_COLUNA_Y)
        instance = KAssociado(k, pd.concat([x, y], axis=1))
        self.assertIsInstance(instance, KAssociado)

    def test_data(self):
        k, x = self.k, self.x.copy()
        y = pd.Series([0, 0, 0, 1, 1, 1, 0], index=x.index, name=NOME_COLUNA_Y)
        data = pd.concat([x, y], axis=1)
        instance = KAssociado(k, data)
        pd.testing.assert_frame_equal(data, instance.data)

    def test_x(self):
        k, x = self.k, self.x.copy()
        x.columns = pd.Index([0, 1], dtype='object')
        y = pd.Series([0, 0, 0, 1, 1, 1, 0], index=x.index, name=NOME_COLUNA_Y)
        data = pd.concat([x, y], axis=1)
        instance = KAssociado(k, data)
        pd.testing.assert_frame_equal(x, instance.x)

    def test_y(self):
        k, x = self.k, self.x.copy()
        y = pd.Series([0, 0, 0, 1, 1, 1, 0], index=x.index, name=NOME_COLUNA_Y)
        data = pd.concat([x, y], axis=1)
        instance = KAssociado(k, data)
        pd.testing.assert_series_equal(y, instance.y)

    def test_determinar_vizinhos(self):
        k, x = self.k, self.x.copy()
        y = pd.Series([0, 0, 0, 1, 1, 1, 0], index=x.index, name=NOME_COLUNA_Y)
        data = pd.concat([x, y], axis=1)
        instance = KAssociado(k, data)
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


if __name__ == '__main__':
    unittest.main()
