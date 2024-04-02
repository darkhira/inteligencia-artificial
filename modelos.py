import joblib as jb
# Importando las bibliotecas necesarias para los 3 metodos
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import ipywidgets as widgets
import matplotlib.pyplot as plt

# Cargando el conjunto de datos desde una URL
url = 'https://raw.githubusercontent.com/adiacla/bigdata/master/DatosEmpresaChurn.csv'
df = pd.read_csv(url)

# Eliminando las columnas 'ANTIG', 'CATEG', 'Unnamed: 0' y 'VISIT'
df.drop(['ANTIG', 'CATEG', 'Unnamed: 0', 'VISIT'], axis=1, inplace=True)

# Imputando los valores NaN con la media de la columna
df = df.replace(',', '.', regex=True)
df = df.astype(float)
imputer = SimpleImputer(strategy='mean')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Codificando las columnas categóricas
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == object:
        df[column] = le.fit_transform(df[column])

# Normalizando el conjunto de datos
scaler = StandardScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Dividiendo el conjunto de datos en conjuntos de entrenamiento y prueba
X = df.drop('TARGET CLASS', axis=1)
y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Convirtiendo y_train a enteros
y_train = y_train.astype(int)

# Creando y entrenando el modelo de Naive Bayes Gaussiano
modeloNB = GaussianNB()
modeloNB.fit(X_train, y_train)

# Haciendo predicciones con el modelo
y_pred_NB = modeloNB.predict(X_test)

# Creando y entrenando el modelo de árbol de decisión
modeloArbol = DecisionTreeClassifier(random_state=0)
modeloArbol.fit(X_train, y_train)

# Haciendo predicciones con el modelo de árbol de decisión
y_pred_arbol = modeloArbol.predict(X_test)

# Creando y entrenando el modelo de Random Forest
modeloBosque = RandomForestClassifier(n_estimators=10, criterion="gini", bootstrap=True, max_features="sqrt",
                                      max_samples=0.8, oob_score=True, random_state=0)
modeloBosque.fit(X_train, y_train)

# Haciendo predicciones con el modelo de Random Forest
y_pred_bosque = modeloBosque.predict(X_test)

# Guardando los modelos entrenados
jb.dump(modeloNB, 'modeloNB.bin')
jb.dump(modeloArbol, 'ModeloArbol.bin')
jb.dump(modeloBosque, 'ModeloBosque.bin')
