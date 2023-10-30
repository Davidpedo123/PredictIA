import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


sales_df = pd.read_csv('Datos de Heladoss/Datoshelados.csv')
plt.style.use('ggplot')
# Gráfico de dispersión
plt.figure()
plt.scatter(sales_df['temperatura'], sales_df['Ganancias'])
plt.xlabel('Temperatura')
plt.ylabel('Ganancias')
plt.title('Gráfico de Dispersión')
plt.savefig('scatter_plot.png')  # Guardar la figura
plt.show()

# Crear el modelo
x_train, x_test, y_train, y_test = train_test_split(sales_df['temperatura'], sales_df['Ganancias'], test_size=0.2, random_state=42)

# Crear el modelo
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

# Entrenar el modelo en el conjunto de entrenamiento

# Entrenar el modelo
epochs_hist = model.fit(x_train, y_train, epochs=300)
keys = epochs_hist.history.keys()
loss_test = model.evaluate(x_test, y_test)
print(f"Pérdida en el conjunto de prueba: {loss_test}")

# Gráfico de pérdida durante el entrenamiento
plt.figure()
plt.plot(epochs_hist.history['loss'])
plt.title("Progreso de pérdida durante el entrenamiento")
plt.xlabel('Épochs')
plt.ylabel('Pérdida de entrenamiento')# Guardar la figura
plt.show()

# Predicciones y gráfico de barras
plt.figure()
Temp = int(input("Ingrese la Temperatura: "))
Ganancias = model.predict([Temp])
print("La ganancia será", Ganancias)

# Gráfico de barras
plt.figure()
plt.scatter(sales_df['temperatura'], sales_df['Ganancias'], marker='o', color='blue', alpha=0.7)
plt.xlabel('Temperatura [Grados Celsius]')
plt.ylabel('Ganancias [Dólares]')
plt.title('Relación entre Temperatura y Ganancias')
 # Guardar la figura
plt.show()
