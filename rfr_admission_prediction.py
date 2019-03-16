# Importar librerias necesarias
import numpy as np
import pandas as pd
from time import time
from IPython.display import display

# Importar aquí librerias de sklearn y fastai que se consideren necesarias
import matplotlib.pyplot as plt
import os

from pandas import DataFrame
from scipy.constants import alpha

os.chdir("/Users/ebalboa/Documents/AI-SATURDAYS/20190316/rfr-university-admission-predictor/")
# Crear dataframe a partir de .csv
input_data = pd.read_csv("data/admission.csv", sep=",")

# Mostrar número de filas y columnas del dataframe
display(input_data.info())

# Mostrar las primeras 10 filas
display(input_data.head(10))

display(input_data.describe())

# El número de serie del alumno no se considera una variable importante por lo que la eliminaremos del dataset
raw_data = input_data.drop(['Serial No.'], axis=1)
display(raw_data.head(2))

# Simplificamos nombres de columnas para que sea mas fácil.
renamed_df = raw_data.rename(index=str,
                             columns={"GRE Score": "GRE", "TOEFL Score": "TOEFL", "University Rating": "URat",
                                      "LOR ": "LOR", "Chance of Admit ": "Chance"})

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
matriz_corr: DataFrame = renamed_df.corr()

display(plt.matshow(matriz_corr), interpolation='nearest')



fig = plt.figure()
ax = fig.add_subplot(111)
data = np.random.random((4,4))
cax = ax.matshow(data, interpolation='nearest')
fig.colorbar(cax)
plt.show()

# Mostrar correlaciones como una función discreta entre las diferentes variables con una matriz
# útil para apreciar relaciones lineales

# Pista: explore pd.plotting.scatter_matrix

# Crear un dataframe solo con la columna de la variable dependiente

# Crear un dataframe con las variables independientes


# Definir un RF con diferentes hiperparámetros (¡experimentar!)


# Entrenar un RF con la totalidad del dataset


###### Snippet para imprimir resultados, df es la variable que refiere
###### al dataframe y clf al regresor, cambiarlas si es necesario
