# Importa las bibliotecas necesarias
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="App de Árbol de Decisión", layout="wide")
st.title("Aplicación de Árbol de Decisión 🌳")
st.write("Esta app te permite entrenar, validar y visualizar un árbol de decisión usando tus propios datos.")

# --- Sección de Carga de Datos ---
st.header("1. Carga tus Datos")
st.markdown("---")

uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

df = None
if uploaded_file is not None:
    # Lee el archivo CSV en un DataFrame de pandas
    try:
        df = pd.read_csv(uploaded_file)
        st.success("¡Archivo cargado con éxito!")
        st.write("Datos cargados:")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"Error al leer el archivo. Asegúrate de que sea un archivo CSV válido. Error: {e}")

# Solo continúa si se ha cargado un archivo
if df is not None:
    # --- Sección de Configuración de Columnas ---
    st.header("2. Configuración del Conjunto de Datos")
    st.markdown("---")

    all_columns = df.columns.tolist()
    
    # Selector para la variable objetivo (y)
    target_column = st.selectbox("Selecciona la columna objetivo (y)", all_columns)

    # Selector para las características (X)
    feature_columns = st.multiselect("Selecciona las características (X)", 
                                     [col for col in all_columns if col != target_column])

    if not feature_columns:
        st.warning("Por favor, selecciona al menos una característica.")
    else:
        # Asegúrate de que los datos son numéricos para el modelo
        try:
            X = df[feature_columns]
            y = df[target_column]
            
            # Convierte las columnas a tipos numéricos, si es posible
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            y = pd.to_numeric(y, errors='coerce')
            
            # Elimina filas con valores nulos generados por la conversión
            data = pd.concat([X, y], axis=1).dropna()
            X = data[feature_columns]
            y = data[target_column]

            if len(X) == 0:
                st.error("No hay datos numéricos válidos en las columnas seleccionadas para continuar.")
            else:
                st.info(f"Usando **{len(X)}** filas de datos válidos después de la limpieza.")
                
                # --- Sección de División de Datos ---
                st.header("3. División de los Datos")
                st.markdown("---")
                test_size = st.slider("Porcentaje de datos para el conjunto de prueba", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                st.write(f"Conjunto de entrenamiento: {len(X_train)} muestras")
                st.write(f"Conjunto de prueba: {len(X_test)} muestras")
                
                # --- Sección de Entrenamiento del Árbol de Decisión ---
                st.header("4. Entrenamiento del Árbol de Decisión")
                st.markdown("---")
                max_depth = st.slider("Profundidad máxima del árbol", min_value=1, max_value=15, value=5, step=1)
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                st.write(f"Entrenando un **Árbol de Decisión** con una profundidad máxima de **{max_depth}**.")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.success(f"Modelo entrenado con éxito. Precisión en el conjunto de prueba: **{accuracy:.2f}**")

                # --- Sección de Evaluación del Modelo ---
                st.header("5. Evaluación y Métricas")
                st.markdown("---")
                st.subheader("Matriz de Confusión")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu", ax=ax)
                plt.xlabel('Predicho')
                plt.ylabel('Real')
                st.pyplot(fig)

                st.subheader("Reporte de Clasificación")
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
                
                # --- Sección de Visualización del Árbol ---
                st.header("6. Visualización del Árbol")
                st.markdown("---")
                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(model,
                          feature_names=X.columns,
                          class_names=[str(c) for c in y.unique()],
                          filled=True,
                          fontsize=8,
                          ax=ax)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error al procesar los datos. Asegúrate de que las columnas seleccionadas sean numéricas. Error: {e}")
