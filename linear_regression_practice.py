import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 3. Creación del dataset (Basado en la salida de la guía)
data = {
    'SquareFootage': [1500, 1800, 2400, 3000, 3500],
    'Price': [200000, 250000, 300000, 350000, 400000]
}
df = pd.DataFrame(data)

# 4. Preparación y división de datos
X = df[['SquareFootage']] 
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Entrenamiento del modelo
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predicciones
y_pred = model.predict(X_test)

# 7. Evaluación
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# 8. Visualización

plt.scatter(X, y, color='blue', label='Datos Reales')
plt.plot(X, model.predict(X), color='red', label='Línea de Regresión')
plt.xlabel('Pies Cuadrados (Square Footage)')
plt.ylabel('Precio (Price)')
plt.title('Predicción de Precios de Casas')
plt.legend()
plt.show()