# %%
# Mesmo conjunto de dados utilizado no artigo para validar
# Utilizar atributos 1 e 4 do Iris dataset
import pandas as pd
from sklearn.datasets import load_iris

from kaog import KAOG

iris = load_iris()
x = pd.DataFrame(iris.data, columns=iris.feature_names)
x = x.drop(columns=['sepal width (cm)', 'petal length (cm)'])
y = pd.DataFrame(iris.target, columns=['target'])
data = pd.concat([x, y], axis=1)
data = data.drop_duplicates()

kaog = KAOG(data)
# %%
kaog.draw()

# %%
kaog.draw(color_by_component=True)

# %%
for k in range(1, 7):
    kaog.grafos_associados[k].draw()
    kaog.grafos_associados[k].draw(color_by_component=True)
