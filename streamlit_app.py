import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import seaborn as sns
import pickle
from sklearn.impute import SimpleImputer
import altair as alt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree




# Cargar el conjunto de datos desde una URL
url = 'https://raw.githubusercontent.com/adiacla/bigdata/master/DatosEmpresaChurn.csv'
df = pd.read_csv(url)

# Convertir etiquetas de clase a valores enteros binarios
le = LabelEncoder()
df['TARGET CLASS'] = le.fit_transform(df['TARGET CLASS'])
y = df['TARGET CLASS']  # Definir y después de la conversión

# Sidebar para aceptar parámetros de entrada
with st.sidebar:
    st.header('Descargar CSV')
    st.write('Haz clic en el siguiente botón para descargar el archivo CSV.')
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("DatosEmpresaChurn", csv_data, mime='text/csv', file_name='DatosEmpresaChurn.csv')



with st.expander('Naive Bayes'):
 if 'df' in locals():
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

    # Convertir etiquetas de clase a valores enteros binarios
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Creando y entrenando el modelo de Naive Bayes Gaussiano
    modeloNB = GaussianNB()
    modeloNB.fit(X_train, y_train)

    # Haciendo predicciones con el modelo
    y_pred = modeloNB.predict(X_test)

    # Creando una matriz de confusión para evaluar las predicciones
    cm = confusion_matrix(y_test, y_pred)

    # Visualizando la matriz de confusión
    st.subheader("Confusion Matrix")
    st.write(cm)

    # Calculando la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)

    # Imprimiendo la precisión del modelo
    st.subheader("Model Accuracy")
    st.write(f'La precisión del modelo es: {accuracy}')

    # Haciendo predicciones con el modelo
    y_pred_proba_NB = modeloNB.predict_proba(X_test)[::, 1]

    # Calculando y dibujando la curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_NB)
    auc_score = auc(fpr, tpr)
    st.subheader("ROC Curve")
    st.write(f"AUC = {auc_score:.3f}")

    # Mostrar la curva ROC
    st.altair_chart(alt.Chart(pd.DataFrame({'false_positive_rate': fpr, 'true_positive_rate': tpr})).mark_line().encode(
        x='false_positive_rate',
        y='true_positive_rate'
    ).properties(
        width=500,
        height=300
    ))
with st.expander('Arboles de decision'):
 if 'df' in locals():
    #Arboles de Decision
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

    # Convertir etiquetas de clase a valores enteros binarios
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    # Creando y entrenando el modelo de árbol de decisión
    modeloArbol = DecisionTreeClassifier(random_state=0)
    modeloArbol.fit(X_train, y_train)

    # Haciendo predicciones con el modelo
    y_pred = modeloArbol.predict(X_test)

    # Creando una matriz de confusión para evaluar las predicciones
    cm = confusion_matrix(y_test, y_pred)

    # Visualizando la matriz de confusión
    st.subheader("Confusion Matrix")
    st.write(cm)

    # Calculando la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)

    # Imprimiendo la precisión del modelo
    st.subheader("Model Accuracy")
    st.write(f'La precisión del modelo es: {accuracy}')

    # Haciendo predicciones con el modelo
    y_pred_proba_arb = modeloArbol.predict_proba(X_test)[::, 1]

    # Calculando y dibujando la curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_arb)
    auc_score = auc(fpr, tpr)
    st.subheader("ROC Curve")
    st.write(f"AUC = {auc_score:.3f}")

    # Mostrar la curva ROC
    st.altair_chart(alt.Chart(pd.DataFrame({'false_positive_rate': fpr, 'true_positive_rate': tpr})).mark_line().encode(
        x='false_positive_rate',
        y='true_positive_rate'
    ).properties(
        width=500,
        height=300
    ))

with st.expander('Bosques Aleatorios'):
    if 'df' in locals():
        # Preprocesamiento de datos
        df = df.replace(',', '.', regex=True)
        df = df.astype(float)
        imputer = SimpleImputer(strategy='mean')
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

        # Convertir las etiquetas de clase a valores enteros
        le = LabelEncoder()
        df['TARGET CLASS'] = le.fit_transform(df['TARGET CLASS'])

        # Normalización de los datos
        scaler = StandardScaler()
        X = scaler.fit_transform(df.drop('TARGET CLASS', axis=1))
        y = df['TARGET CLASS']

        # División del conjunto de datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Creación y entrenamiento del modelo de Bosques Aleatorios
        modeloBosque = RandomForestClassifier(n_estimators=10,
                                               criterion="gini",
                                               bootstrap=True,
                                               max_features="sqrt",
                                               max_samples=0.8,
                                               oob_score=True,
                                               random_state=0)
        modeloBosque.fit(X_train, y_train)

        # Predicciones con el modelo
        y_pred = modeloBosque.predict(X_test)

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)

        # Visualización de la matriz de confusión
        st.subheader("Confusion Matrix")
        st.write(cm)

        # Precisión del modelo
        accuracy = accuracy_score(y_test, y_pred)

        # Imprimiendo la precisión del modelo
        st.subheader("Model Accuracy")
        st.write(f'La precisión del modelo es: {accuracy}')

        # Predicciones de probabilidad
        y_pred_proba_bos = modeloBosque.predict_proba(X_test)[::,1]

        # Calculando y dibujando la curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_bos)
        auc_score = auc(fpr, tpr)
        st.subheader("ROC Curve")
        st.write(f"AUC = {auc_score:.3f}")

        # Mostrar la curva ROC
        st.altair_chart(alt.Chart(pd.DataFrame({'false_positive_rate': fpr, 'true_positive_rate': tpr})).mark_line().encode(
            x='false_positive_rate',
            y='true_positive_rate'
        ).properties(
            width=500,
            height=300
        ))

        # Visualizando algunos (NO TODOS) árboles del bosque aleatorio
        st.subheader("Visualización de algunos árboles del Bosque Aleatorio")
        for i, tree_in_forest in enumerate(modeloBosque.estimators_):
            if i < 5:  # Visualizar solo los primeros 5 árboles
                fig, ax = plt.subplots(figsize=(10, 7))  # Crear una nueva figura
                plot_tree(tree_in_forest, filled=True, feature_names=df.columns[:-1], ax=ax)  # Pasar ax=ax
                st.pyplot(fig)  # Pasar la figura a st.pyplot()
