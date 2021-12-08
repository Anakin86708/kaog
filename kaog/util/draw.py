from abc import ABC, abstractmethod
from itertools import cycle
from random import shuffle

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from kaog.util import NOME_COLUNA_Y


class DrawableGraph(ABC):
    """Adiciona métodos para desenhar grafos.

    **DrawableGraph**

    Classe abstrata que adiciona a possibilidade de desenhar grafos.
    """

    @property
    @abstractmethod
    def grafo(self) -> nx.DiGraph:
        """Representação do grafo."""
        raise NotImplementedError

    @property
    @abstractmethod
    def componentes(self):
        """Componentes pertencentes ao grafo."""
        raise NotImplementedError

    @property
    @abstractmethod
    def data(self):
        """Dados do grafo, com informação de classes."""
        raise NotImplementedError

    @property
    @abstractmethod
    def x(self):
        """Dados do grafo, sem classes."""
        raise NotImplementedError

    @property
    @abstractmethod
    def y(self):
        """Classe associada aos dados do grafo."""
        raise NotImplementedError

    def draw(self, title=None, color_by_component=False):
        """
        Realiza o plot do grafo. Para dados em `self.x` que estejam em 2D, o plot é feito sem t-SNE, enquanto maiores
        dimensões são plotados com t-SNE.
        Por padrão, a cor dos vértices é a cor da classe associada.

        :param color_by_component: Se `True`, a cor dos vértices é a cor do componente conectado ao vértice, ao invés da classe.
        :type color_by_component: object
        """
        scala_fig = 2.5
        scale_coords = 1
        fig = plt.figure(figsize=(6.4 * scala_fig, 4.8 * scala_fig))
        ax = fig.gca()
        if title is not None:
            ax.set_title(title)
        pos = nx.kamada_kawai_layout(self.grafo)
        markers = [*Line2D.markers.keys()][3:-4]
        y_unique = self.y.unique()
        replace = {k: v for k, v in zip(y_unique, cycle(markers))}
        scaller = MinMaxScaler()
        scaller.fit(y_unique.reshape(-1, 1))

        # Obeter coordenadas utilizando t-SNE
        cords = self._get_coords(scale_coords)

        if color_by_component:
            self._draw_color_by_component(ax, cords)
        else:
            self._draw_color_by_class(ax, cords, replace, scaller, y_unique)

    def _draw_color_by_component(self, ax, cords):
        """Colorir os vértices de acordo com o componente conectado ao vértice."""
        colors = list(range(len(self.componentes)))
        shuffle(colors)
        for i, componente in zip(colors, self.componentes):
            vertices = componente

            nx.draw(
                self.grafo,
                nodelist=vertices,
                with_labels=True,
                ax=ax,
                pos=cords,
                node_color=[i] * len(vertices),
                cmap=plt.cm.get_cmap('hsv'),
                vmin=0,
                vmax=len(self.componentes),
            )
        plt.show()

    def _draw_color_by_class(self, ax, cords, replace, scaller, y_unique):
        """Colorir os vértices de acordo com a classe."""
        # Iterar os vértices de cada classe
        for i, classe in enumerate(y_unique):
            vertices = self.data[self.data[NOME_COLUNA_Y] == classe].index
            # color = scaller.transform(np.array([1]).reshape(1, -1))[0][0]

            nx.draw(
                self.grafo,
                nodelist=vertices,
                with_labels=True,
                ax=ax,
                pos=cords,
                node_shape=replace[classe],
                node_color=[i] * len(vertices),
                cmap=plt.cm.get_cmap('plasma'),
                vmin=scaller.data_min_[0],
                vmax=scaller.data_max_[0],
            )
        plt.show()

    def _get_coords(self, scale_coords=1):
        """Obter coordenadas utilizando t-SNE, se necessário."""
        if self.x.shape[1] == 2:
            # 2D, não precisa de t-SNE
            cords = self.x.to_dict(orient='index')
        else:
            # Precisa de t-SNE
            tsne = TSNE(init='pca', learning_rate='auto', n_jobs=-1)
            tsne_cords = pd.DataFrame(tsne.fit_transform(self.x), index=self.x.index)
            cords = tsne_cords.to_dict(orient='index')

        # Remover os dicionários internos e utilizar listas
        for k, v in cords.items():
            cords[k] = np.array([x for x in v.values()]) * scale_coords
        return cords
