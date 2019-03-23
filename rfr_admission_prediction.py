# Importar librerias necesarias
from typing import List, Any, Tuple

import numpy as np
import pandas as pd
from time import time

import seaborn as sns
from IPython.display import display

# Importar aquí librerias de sklearn y fastai que se consideren necesarias
import matplotlib.pyplot as plt
import os

from pandas import DataFrame
from scipy.constants import alpha
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


os.chdir("/Users/ebalboa/Documents/AI-SATURDAYS/20190316/rfr-university-admission-predictor/")
# Crear dataframe a partir de .csv
input_data = pd.read_csv("data/admission.csv", sep=",")

# Mostrar número de filas y columnas del dataframe
display(input_data.shape)

# Mostrar las primeras 10 filas
display(input_data.head(10))

display(input_data.describe())

# El número de serie del alumno no se considera una variable importante por lo que la eliminaremos del dataset
raw_data = input_data.drop(['Serial No.'], axis=1)
display(raw_data.head(2))

# Simplificamos nombres de columnas para que sea mas fácil.
renamed_df = raw_data.rename(index=str,
                             columns={"GRE Score": "GRE", "TOEFL Score": "TOEFL", "University Rating": "URat",
                                      "LOR ": "LOR", "Chance of Admit ": "Chance"}).drop(columns={"URat"})

display(renamed_df.head(5))

# Hallar número de valores únicos en cada columna

display(renamed_df.nunique())

display(renamed_df.SOP.unique())

display(renamed_df.LOR.unique())

display(renamed_df.columns)

# Comprobar la existencia de valores nulos en el dataset
print("Nulos por columnas: " + '\n' + str(renamed_df.isnull().any()))

# Mostrar información general del dataframe
display(renamed_df.info())

# Descripción analítica básica del dataframe
display(renamed_df.describe())

# Mostrar matriz de correlación de variables
# Pista: explore plt.matshow y corr() de un dataframe
corr_matrix = renamed_df.corr()
# plot the heatmap
sns.heatmap(corr_matrix)
    #(corr_matrix, cmap='BuGn')


#display(plt.matshow(corr_matrix), interpolation='nearest')


ax1 = plt.figure().add_subplot(111)
plt.figure().colorbar(ax1.imshow(corr_matrix, interpolation="nearest"), ticks=[.4, 0.5, .6, .75, .8, .85, .90, .95, 1])
plt.title('Correlation Matrix')
labels = ['GRE', 'TOEFL', 'URat', 'SOP', 'LOR', 'CGPA', 'Research', 'Chance']

ax1.set_xticklabels(labels, fontsize=8)
ax1.set_yticklabels(labels, fontsize=8)
plt.show()

sns.heatmap(corr_matrix,
        xticklabels=corr_matrix.columns,
        yticklabels=corr_matrix.columns)

# Mostrar correlaciones como una función discreta entre las diferentes variables con una matriz
# útil para apreciar relaciones lineales
pd.plotting.scatter_matrix(renamed_df, diagonal='kde')
plt.show()

# Pista: explore pd.plotting.scatter_matrix

# Crear un dataframe solo con la columna de la variable dependiente

dep_df = renamed_df.Chance

# Crear un dataframe con las variables independientes
indep_df = renamed_df.drop('Chance', axis=1)
display(indep_df.head(5))

# Definir un RF con diferentes hiperparámetros (¡experimentar!)
modelRfC: RandomForestRegressor = RandomForestRegressor(criterion="mse", max_depth=10,random_state=42,n_estimators=100)

# Entrenar un RF con la totalidad del dataset
trained_data: object = modelRfC.fit(indep_df, dep_df)

trained_data

###### Snippet para imprimir resultados, df es la variable que refiere
###### al dataframe y clf al regresor, cambiarlas si es necesario


feature_list: List[Tuple[Any, Any]] = list(zip(renamed_df.columns.values, modelRfC.feature_importances_))
sorted_by_importance = sorted(feature_list, key=lambda x: x[1], reverse=True)

for feat, value in sorted_by_importance:
    print(feat, value)

# Partir el test en cierta proporción (¡experimentar!)
X_train, X_test, y_train, y_test = train_test_split(indep_df, dep_df, test_size=0.2, random_state=42)

###### Snippet para imprimir resultados, X_train es la variable que refiere
###### a la porcion de entrenamiento y X_test a la de test

print("El dataset de training tiene {} elementos.".format(X_train.shape[0]))
print("El dataset de testing tiene {} elementos.".format(X_test.shape[0]))

# Definir un rfregressor
from sklearn.ensemble import RandomForestRegressor
rfregressor = RandomForestRegressor(min_samples_leaf=12, n_estimators=100)


# Entrenar el regresor con el dataset de train
rfregressor.fit(X_train, y_train)

# Predecir valores para las variables independientes de test
y_predict = rfregressor.predict(X_test)


# Calcular la precisión
# Pista: explorar sklearn.metrics
from sklearn.metrics import r2_score
sco = rfregressor.score(X_test, y_test)
R2 = r2_score(y_test, y_predict)
print("R2 : ", R2)
print("Score : ", sco)

print("Precision global: " + str(metrics.r2_score(y_test, y_predict)))
print("MSE: " + str(metrics.mean_squared_error(y_test, y_predict)))

# podemos quitar variables como la de distribucion no lineal

