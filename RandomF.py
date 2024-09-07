# Evidencia 2 Redes neuronales
# Leonardo Cossío Dinorín

# RandomF:
# Código que utiliza el modelo de Random Forest para clasificar
# noticias verdaderas y falsas.

# Dataset obtenido de:
# https://github.com/lutzhamel/fake-news

# Librerías
import pandas as pd # Visualización y manipulación de datos
import seaborn as sns # Graficas y visualización de datos
import matplotlib.pyplot as plt # Gráficas y visualización de datos
import numpy as np # Arrays y operaciones matemáticas

from sklearn.model_selection import train_test_split # Separación del dataset
from sklearn.feature_extraction.text import TfidfVectorizer # Conversión de texto a número
from sklearn.ensemble import RandomForestClassifier # Clasificador
from sklearn.pipeline import Pipeline # Constructor de pipeline
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, confusion_matrix # Métricas

# Cargar datos
data = pd.read_csv("fake_or_real_news.csv")

# Preprocesamiento de los datos
data["fake"] = data["label"].apply(lambda x: 0 if x == "REAL" else 1) # One-Hot encoding
data = data.drop("label", axis=1) # Elimina la columna de label
X, y = data["text"], data["fake"] # Separa los datos de entrada y de salida

# Dividir el dataset en train, validation y test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 20% test del dataset completo
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42) # 20% de train será para validation

# Crear el pipeline
pipeline = Pipeline([
    # Convierte el texto en una matriz numérica utilizando el 
    # valor de TF-IDF (Term Frequency-Inverse Document Frequency),
    # que refleja la importancia de cada término en un documento en 
    # relación con el conjunto de documentos.
    ('tfidf', TfidfVectorizer(stop_words="english", max_df=0.7)),  # TF-IDF
    ('rf', RandomForestClassifier(n_estimators=120, criterion="gini", class_weight="balanced", random_state=42))  # Clasificador seleccionado
])

# Entrenamiento del modelo
pipeline.fit(X_train, y_train)

# -------- EVALUACIÓN DEL MODELO ---------------------------------

# Evaluación en el conjunto de validación
y_valid_pred = pipeline.predict(X_valid) # Predicciones
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
valid_f1 = f1_score(y_valid, y_valid_pred)
valid_recall = recall_score(y_valid, y_valid_pred)
valid_precision = precision_score(y_valid, y_valid_pred)

print("\n***************** VALIDATION SCORE ********************")
print(f"Accuracy on validation set: {valid_accuracy:.2f}")
print(f"Recall on validation set: {valid_recall:.2f}")
print(f"Precision on validation set: {valid_precision:.2f}")
print(f"F1 score on validation set: {valid_f1:.2f}")
print("*******************************************************\n")

# Evaluación en el conjunto de prueba
y_test_pred = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)

print("\n***************** TEST SCORE **************************")
print(f"Accuracy on test set: {test_accuracy:.2f}")
print(f"Recall on test set: {test_recall:.2f}")
print(f"Precision on test set: {test_precision:.2f}")
print(f"F1 score on test set: {test_f1:.2f}")
print("*******************************************************\n")

# Matriz de confusión (Test)
cm = confusion_matrix(y_test, y_test_pred)

# Convertir la matriz de confusión a un DataFrame
cm_df = pd.DataFrame(cm, index=['REAL', 'FAKE'], columns=['Predicción REAL', 'Predicción FAKE'])

# Crear el gráfico
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5, linecolor='black')

# Añadir títulos y etiquetas
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')

# Mostrar el gráfico
plt.show()


# --------- PREDICCIONES INDIVIDUALES ----------------------------

# Seleccionar una fila aleatoria del conjunto de datos de prueba
random_index = np.random.randint(0, len(X_test))
sample_text = X_test.iloc[random_index]

# Transformar el texto usando el mismo vectorizador
sample_vectorized = pipeline.named_steps['tfidf'].transform([sample_text])

# Hacer una predicción con el modelo entrenado
predicted_class = pipeline.named_steps['rf'].predict(sample_vectorized)[0]

# Obtener la etiqueta real para comparación
true_class = y_test.iloc[random_index]

# Mostrar el resultado
print(f"Índice de la muestra: {random_index}")
#print(f"Texto de la muestra: {sample_text}")
print(f"\nClase real: {true_class}")
print(f"Clase predicha: {predicted_class}")
