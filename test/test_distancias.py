import unittest

import numpy as np
import pandas as pd

from kaog.distancias import Distancias


class DistanciasTest(unittest.TestCase):

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

    def test_distancias(self):
        k, x = self.k, self.x
        instance = Distancias(k, x)
        self.assertIsInstance(instance, Distancias)

    def test_create_map_pandas_to_numpy(self):
        k, x = self.k, self.x.copy()
        x.set_index(np.random.randint(0, 255, size=x.shape[0]), inplace=True)
        instance = Distancias(k, x)
        map_idx = instance._create_map_pandas_to_numpy()
        for expected_idx, idx in enumerate(x.index):
            self.assertEqual(expected_idx, map_idx[idx])

    def test_calcular_distancias_e_vizinhos(self):
        k, x = self.k, self.x.copy()
        instance = Distancias(k, x)

        for k in range(self.k, x.shape[0]):
            instance.k = k
            distancias, vizinhos = instance._calcular_distancias_e_vizinhos()
            self.assertEqual(x.shape[0], vizinhos.shape[0])
            self.assertEqual(x.shape[0], distancias.shape[0])
            self.assertEqual(x.shape[0] - 1, vizinhos.shape[1])
            self.assertEqual(x.shape[0] - 1, distancias.shape[1])

    def test_vizinhos_mais_proximos_de(self):
        k, x = self.k, self.x.copy()
        instance = Distancias(k, x)

        proximos = instance.vizinhos_mais_proximos_de(6)
        expected = np.array([0, 1])
        np.testing.assert_array_equal(expected, proximos)

        if __name__ == '__main__':
            unittest.main()
