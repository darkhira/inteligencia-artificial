
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import joblib as jb


@st.cache_resource
def load_models():
  modeloNB=jb.load('modeloNB.bin')
  modeloArbol=jb.load('ModeloArbol.bin')
  modeloBosque=jb.load('ModeloBosque.bin')
  return modeloNB,modeloArbol,modeloBosque


modeloNB,modeloArbol,modeloBosque= load_models()

st.title("Aplicación de predicción")
st.header('Trabajo-Data-Science', divider='rainbow')
st.subheader('Ejemplo en los modelos :blue[Arbol de Decisión, Bosque Aleatorio y Naive Bayes]')


with st.container( border=True):
  st.subheader("Modelo Machine Learning para predecir la deserción de clientes")
  st.write("""

**Introducción**
cliente de nuestros sueños es el que permanece fiel a la empresa, comprando siempre sus productos o servicios. Sin embargo, en la realidad,
los clientes a veces deciden alejarse de la empresa para probar o empezar a comprar otros productos o servicios y esto puede ocurrir en
cualquier fase del customer journey. Sin embargo, existen varias medidas para prevenir o gestionar mejor esta circunstancia. Por eso lo mejor
es tener una herramienta predictiva que nos indique el estado futuro de dichos clientes usando inteligencia artificial, tomar las acciones
de retenció necesaria. Constituye pues esta aplicación una herramienta importante para la gestión del marketing.

Los datos fueron tomados con la Información de la base de datos CRM de la empresa ubicada en Bucaramanfa,donde se
preparó 3 modelos de machine Learnig para predecir la deserció de clientes, tanto actuales como nuevos.

Datos Actualizados en la fuente: 20 de Marzo del 2024


Se utilizó modelos supervidados de clasificacion  tanto Naive Bayes, Arboles de decisión y Bosques Aleatorios
entendiendo que hay otras técnicas, es el resultado de la aplicacion practico del curso de inteligencia artificial en estos modelos
revisado en clase. Aunqe la aplicación final sería un solo modelo, aqui se muestran los tres modelos para
comparar los resultados.

 """)



modeloA=['Naive Bayes', 'Arbol de Decisión', 'Bosque Aleatorio']

churn = {1 : 'Cliente se retirará', 0 : 'Cliente No se Retirará' }


styleimagen ="<style>[data-testid=stSidebar] [data-testid=stImage]{text-align: center;display: block;margin-left: auto;margin-right: auto;width: 100%;}</style>"
st.sidebar.markdown(styleimagen, unsafe_allow_html=True)


styletexto = "<style>h2 {text-align: center;}</style>"
st.sidebar.markdown(styletexto, unsafe_allow_html=True)
st.sidebar.header('Seleccione los datos de entrada')


def seleccionar(modeloL):


  st.sidebar.subheader('Selector de Modelo')
  modeloS=st.sidebar.selectbox("Modelo",modeloL)

  st.sidebar.subheader('Seleccione la COMP')
  COMPS=st.sidebar.slider("Seleccion",4000,18000,8000,100)

  st.sidebar.subheader('Selector del PROM')
  PROMS=st.sidebar.slider("Seleccion",   0.7, 9.0,5.0,.5)

  st.sidebar.subheader('Selector de COMINT')
  COMINTS=st.sidebar.slider("Seleccione",1500,58000,12000,100)

  st.sidebar.subheader('Selector de COMPPRES')
  COMPPRESS=st.sidebar.slider('Seleccione', 17000,90000,25000,100)

  st.sidebar.subheader('Selector de RATE')
  RATES=st.sidebar.slider("Seleccione",0.5,4.2,2.0,0.1)

  st.sidebar.subheader('Selector de DIASSINQ')
  DIASSINQS=st.sidebar.slider("Seleccione", 270,1800,500,10)

  st.sidebar.subheader('Selector de TASARET')
  TASARETS=st.sidebar.slider("Seleccione",0.3,1.9,0.8,.5)

  st.sidebar.subheader('Selector de NUMQ')
  NUMQS=st.sidebar.slider("Seleccione",3.0,10.0,4.0,0.5)

  st.sidebar.subheader('Selector de RETRE entre 3 y 30')

  RETRES=st.sidebar.number_input("Ingrese el valor de RETRE", value=3.3, placeholder="Digite el numero...")

  return modeloS,COMPS, PROMS, COMINTS ,COMPPRESS, RATES, DIASSINQS,TASARETS, NUMQS, RETRES



modelo,COMP, PROM, COMINT ,COMPPRES, RATE, DIASSINQ,TASARET, NUMQ, RETRE=seleccionar(modeloA)


with st.container(border=True):
  st.subheader("Predición")
  st.title("Predicción de Churn")
  st.write(""" El siguiente es el pronóstico de la deserción usando el modelo
           """)
  st.write(modelo)
  st.write("Se han seleccionado los siguientes parámetros:")
  # Presento los parámetros seleccionados en el slidder
  lista=[[COMP, PROM, COMINT ,COMPPRES, RATE, DIASSINQ,TASARET, NUMQ, RETRE]]
  X_predecir=pd.DataFrame(lista,columns=['COMP', 'PROM', 'COMINT', 'COMPPRES', 'RATE', 'DIASSINQ','TASARET', 'NUMQ', 'RETRE'])
  st.dataframe(X_predecir)




  if modelo=='Naive Bayes':
      y_predict=modeloNB.predict(X_predecir)
      probabilidad=modeloNB.predict_proba(X_predecir)
      importancia=pd.DataFrame()
  elif modelo=='Arbol de Decisión':
      y_predict=modeloArbol.predict(X_predecir)
      probabilidad=modeloArbol.predict_proba(X_predecir)
      importancia=modeloArbol.feature_importances_
      features=modeloArbol.feature_names_in_
  else :
      y_predict=modeloBosque.predict(X_predecir)
      probabilidad=modeloBosque.predict_proba(X_predecir)
      importancia=modeloBosque.feature_importances_
      features=modeloBosque.feature_names_in_


  styleprediccion= '<p style="font-family:sans-serif; color:Green; font-size: 42px;">La predicción es</p>'
  st.markdown(styleprediccion, unsafe_allow_html=True)
  prediccion='Resultado: '+ str(y_predict[0])+ "    - en conclusion :"+churn[y_predict[0]]
  st.header(prediccion+'   :warning::warning::warning:')

  st.write("Con la siguiente probabilidad")

  col1, col2= st.columns(2)
  col1.metric(label="Probalidad de NO :", value="{0:.2%}".format(probabilidad[0][0]),delta=" ")
  col2.metric(label="Probalidad de SI:", value="{0:.2%}".format(probabilidad[0][1]),delta=" ")

  st.write("La importancia de cada Factor en el modelo es:")
  if modelo!='Naive Bayes':
    importancia=pd.Series(importancia,index=features)
    st.bar_chart(importancia)
  else:
    st.write("Naive Bayes no tiene parámetro de importancia de los features")
