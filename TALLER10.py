import numpy as np
from sklearn.svm import SVC

# ==============================================================================
# 1. EJECUCIÓN INICIAL Y COMPROBACIÓN DE VECTORES DE SOPORTE (Puntos 1 y 2)
# ==============================================================================
X = np.array([[2, 2], [3, 3], [4, 2], [6, 6], [7, 8], [8, 7]])
Y = np.array([0, 0, 0, 1, 1, 1])

modelo_lineal = SVC(kernel='linear')
modelo_lineal.fit(X, Y)

print("--- MODELO LINEAL INICIAL ---")
print("Vectores de Soporte descubiertos:\n", modelo_lineal.support_vectors_)

# Predicción del punto [5, 4]
nuevo_punto = np.array([[5, 4]])
pred_inicial = modelo_lineal.predict(nuevo_punto)
print("El punto [5, 4] pertenece a la clase:", pred_inicial[0])


# ==============================================================================
# 2. INCLUSIÓN DE PUNTO CONFLICTIVO Y PRUEBA CON KERNEL LINEAL (Puntos 2 y 3)
# ==============================================================================
# Agregamos el nuevo punto [5, 5] etiquetado como Clase A (0)
X_conflictivo = np.vstack([X, [5, 5]])
Y_conflictivo = np.append(Y, 0)

modelo_lineal_fail = SVC(kernel='linear')
modelo_lineal_fail.fit(X_conflictivo, Y_conflictivo)

print("\n--- MODELO LINEAL CON PUNTO CONFLICTIVO [5, 5] ---")
print("Predicción para [5, 4] con Kernel Lineal:", modelo_lineal_fail.predict(nuevo_punto)[0])


# ==============================================================================
# 3. SOLUCIÓN CON KERNEL RBF (Punto 4)
# ==============================================================================
modelo_rbf = SVC(kernel='rbf', gamma='scale')
modelo_rbf.fit(X_conflictivo, Y_conflictivo)

print("\n--- MODELO NO LINEAL (KERNEL RBF) ---")
print("Predicción para [5, 4] con Kernel RBF:", modelo_rbf.predict(nuevo_punto)[0])