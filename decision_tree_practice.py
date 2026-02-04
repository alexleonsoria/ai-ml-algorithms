import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn import tree

# 3. Preparación del Dataset (Horas de estudio, Nota anterior y Resultado)
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'PrevExamScore': [30, 40, 45, 50, 60, 65, 70, 75, 80, 85],
    'Pass': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Falla, 1 = Pasa
}

df = pd.DataFrame(data)

# 4. División de datos
X = df[['StudyHours', 'PrevExamScore']] # Variables predictoras
y = df['Pass'] # Variable objetivo

# Dividimos 80% para entrenar y 20% para probar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Entrenamiento del modelo inicial (sin límites)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print(f"--- Modelo Inicial ---")
print(f"Profundidad del árbol: {model.get_depth()}")
print(f"Número de hojas: {model.get_n_leaves()}")

# 6 & 7. Predicciones y Evaluación
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))

# 8. Visualización del Árbol de Decisión

plt.figure(figsize=(12,8))
tree.plot_tree(model, 
               feature_names=['StudyHours', 'PrevExamScore'], 
               class_names=['Fail', 'Pass'], 
               filled=True)
plt.title('Árbol de Decisión: Clasificación de Aprobados')
plt.show()

# 9. Tuning: Limitando la profundidad para evitar Overfitting
# Aquí aplicamos la "poda" limitando el crecimiento a 3 niveles
model_tuned = DecisionTreeClassifier(max_depth=3, random_state=42)
model_tuned.fit(X_train, y_train)
y_pred_tuned = model_tuned.predict(X_test)

print(f"\n--- Modelo Optimizado (Tuned) ---")
print(f"Accuracy (Tuned): {accuracy_score(y_test, y_pred_tuned)}")