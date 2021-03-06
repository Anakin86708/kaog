import unittest
from math import sqrt

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
        instance = Distancias(x)
        self.assertIsInstance(instance, Distancias)

    def test_create_map_pandas_to_numpy(self):
        for _ in range(50):
            with self.subTest():
                k, x = self.k, self.x.copy()
                x.set_index(np.random.randint(0, 255, size=x.shape[0]), inplace=True)
                if x.index.duplicated().any():
                    self.assertRaises(ValueError, Distancias, x)
                else:
                    instance = Distancias(x)
                    map_idx = instance._create_map_pandas_to_numpy()
                    for expected_idx, idx in enumerate(x.index):
                        self.assertEqual(expected_idx, map_idx[idx])

    def test_calcular_distancias_e_vizinhos(self):
        k, x = self.k, self.x.copy()
        instance = Distancias(x)

        shape_ = x.shape[0] - 1  # Desconsidera o próprio elemento
        for k in range(self.k, x.shape[0]):
            instance.k = k
            distancias, vizinhos = instance._calcular_distancias_e_vizinhos(x)
            self.assertEqual(x.shape[0], vizinhos.shape[0])
            self.assertEqual(x.shape[0], distancias.shape[0])
            self.assertEqual(shape_, vizinhos.shape[1])
            self.assertEqual(k, vizinhos[:, :k].shape[1])
            self.assertEqual(shape_, distancias.shape[1])
            self.assertEqual(k, distancias[:, :k].shape[1])

    def test_k_vizinhos_mais_proximos_de(self):
        """Deve ter apenas os k-vizinhos mais próximos de um elemento."""
        k, x = self.k, self.x.copy()
        instance = Distancias(x)

        proximos = instance.k_vizinhos_mais_proximos_de(6, k)
        expected = np.array([0, 1])
        np.testing.assert_array_equal(expected, proximos)

    def test_all_k_vizinhos_mais_proximos_de(self):
        """Deve ter todos os vizinhos mais proximos de um elemento."""
        k, x = self.k, self.x.copy()
        instance = Distancias(x)
        input_ = pd.Series(name=6)

        proximos = instance.k_vizinhos_mais_proximos_de(input_)
        expected = np.array([0, 1, 3, 4, 2, 5])
        np.testing.assert_array_equal(expected, proximos)

    def test_distancia_entre(self):
        """Distância entre dois elementos, considerando a distância euclidiana."""
        k, x = self.k, self.x.copy()
        instance = Distancias(x)

        distancia = instance.distancia_entre(0, 1)
        self.assertEqual(sqrt(1), distancia)

    def test_distancias_is_sorted(self):
        k, x = self.k, self.x.copy()
        instance = Distancias(x)

        distancias = instance.distancias
        self.assertTrue(np.all(np.diff(distancias) >= 0))

        if __name__ == '__main__':
            unittest.main()
