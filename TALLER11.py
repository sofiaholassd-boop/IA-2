import numpy as np

# 1. Definir la Función de Activación (Escalón)
def funcion_escalon(z):
    if z >= 0:
        return 1
    else:
        return 0

# 2. Definir la Estructura del Perceptrón
def perceptron(X, W, b):
    # Combinación lineal / Producto Punto (Z = X * W + b)
    Z = np.dot(X, W) + b
    # Activación
    salida = funcion_escalon(Z)
    return salida

# ==============================================================================
# 3. VALORES AJUSTADOS PARA RESOLVER LA COMPUERTA OR
# ==============================================================================
pesos_OR = np.array([0.5, 0.5])  # Vector W
sesgo_OR = -0.2                   # Constante b (Sesgo ajustado)

# Probar los 4 casos de la tabla de verdad OR
casos = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

print("--- COMPROBACIÓN DE LA COMPUERTA OR ---")
for X in casos:
    res = perceptron(X, pesos_OR, sesgo_OR)
    print(f"Entradas: {X}  --->  Salida Perceptrón: {res}")