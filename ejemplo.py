# Importa las bibliotecas necesarias
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="App de ML con Streamlit", layout="wide")
st.title("Aplicación de Aprendizaje Automático con Streamlit 🤖")
st.write("Esta app simula un conjunto de datos y permite entrenar, validar y visualizar modelos de ML supervisados.")

# --- Sección de Generación de Datos ---
st.header("1. Generación de Datos Simulados")
st.markdown("---")

# Slider para la cantidad de muestras
n_samples = st.slider("Número de muestras", min_value=100, max_value=5000, value=1000, step=100)

@st.cache_data
def generate_data(n):
    """Genera un conjunto de datos simulados para clasificación."""
    np.random.seed(42)
    X = np.random.rand(n, 2) * 10
    y = (X[:, 0] + X[:, 1] > 10).astype(int)  # Una regla simple para la clasificación
    noise = np.random.rand(n)
    y[noise > 0.8] = 1 - y[noise > 0.8]  # Introduce algo de ruido
    df = pd.DataFrame(X, columns=['Caracteristica_1', 'Caracteristica_2'])
    df['Clase'] = y
    return df

df = generate_data(n_samples)
st.write(f"Se generaron {n_samples} muestras con 2 características y una clase binaria.")

# --- Sección de Análisis Exploratorio de Datos (EDA) ---
st.header("2. Análisis Exploratorio de Datos (EDA) 🔍")
st.markdown("---")

# Mostrar las primeras filas del DataFrame
st.subheader("Vista previa de los datos")
st.dataframe(df.head())

# Estadísticas descriptivas
st.subheader("Estadísticas Descriptivas")
st.write(df.describe())

# Conteo de valores de la clase objetivo
st.subheader("Distribución de la Clase")
class_counts = df['Clase'].value_counts()
st.bar_chart(class_counts)
st.write(class_counts)

# Visualización de la relación entre características
st.subheader("Visualización de las Características")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='Caracteristica_1', y='Caracteristica_2', hue='Clase', data=df, palette='viridis', ax=ax)
ax.set_title('Distribución de Clases en las Características')
st.pyplot(fig)

# --- Sección de División de Datos ---
st.header("3. División de los Datos")
st.markdown("---")

test_size = st.slider("Porcentaje de datos para el conjunto de prueba", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
X = df.drop('Clase', axis=1)
y = df['Clase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

st.write(f"Conjunto de entrenamiento: {len(X_train)} muestras")
st.write(f"Conjunto de prueba: {len(X_test)} muestras")

# --- Sección de Entrenamiento y Predicción ---
st.header("4. Entrenamiento del Modelo")
st.markdown("---")

model_choice = st.selectbox("Selecciona un modelo", ["Regresión Logística", "Random Forest"])

if model_choice == "Regresión Logística":
    model = LogisticRegression()
else:
    model = RandomForestClassifier(n_estimators=100, random_state=42)

st.write(f"Entrenando el modelo: **{model_choice}**")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"Modelo entrenado con éxito. Precisión en el conjunto de prueba: **{accuracy:.2f}**")

# --- Sección de Evaluación del Modelo ---
st.header("5. Evaluación y Métricas")
st.markdown("---")

st.subheader("Matriz de Confusión")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
plt.xlabel('Predicho')
plt.ylabel('Real')
st.pyplot(fig)

st.subheader("Reporte de Clasificación")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)

# --- Sección de Visualización de Límites de Decisión ---
st.header("6. Visualización del Límite de Decisión")
st.markdown("---")

def plot_decision_boundary(X_train, y_train, model):
    """Grafica el límite de decisión del modelo."""
    x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
    y_min, y_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    sns.scatterplot(x=X_test.iloc[:, 0], y=X_test.iloc[:, 1], hue=y_test, style=y_pred, ax=ax, palette='viridis', markers=['o', 'X'], s=100)
    ax.set_title("Límite de Decisión del Modelo")
    ax.set_xlabel("Caracteristica_1")
    ax.set_ylabel("Caracteristica_2")
    return fig

st.pyplot(plot_decision_boundary(X_train, y_train, model))
