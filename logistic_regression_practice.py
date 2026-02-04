import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 3. Datos: Horas de estudio vs Aprobación (0=Falla, 1=Pasa)
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

# 4. División de datos (80% entrenamiento, 20% prueba)
X = df[['StudyHours']]
y = df['Pass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Entrenamiento del modelo Logístico
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Predicción y Evaluación
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:\n", conf_matrix)

# 8. Visualización de la Curva Sigmoide
# Creamos un rango de horas (de 1 a 10) para dibujar la curva suavemente
study_hours_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
# Obtenemos la probabilidad de "Pasar" (columna 1) para cada hora del rango
y_prob = model.predict_proba(study_hours_range)[:, 1]

plt.scatter(X, y, color='blue', label='Datos Reales')
plt.plot(study_hours_range, y_prob, color='red', label='Curva Sigmoide')
plt.axhline(0.5, color='gray', linestyle='--', label='Umbral de Decisión (50%)')
plt.xlabel('Horas de Estudio')
plt.ylabel('Probabilidad de Aprobar')
plt.title('Regresión Logística: Horas vs Probabilidad')
plt.legend()
plt.show()

# Fin de la práctica