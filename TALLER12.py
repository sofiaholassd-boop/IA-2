import numpy as np

# Función de Activación: Sigmoide (devuelve probabilidades entre 0 y 1)
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# ==============================================================================
# PUNTO 3: ENTRADA DE LOTE (BATCH DE 2 CLIENTES SIMULTÁNEOS) - Matriz de 2x3
# ==============================================================================
# Cliente 1: [0.5, 0.8, 0.2]
# Cliente 2: [0.1, 0.9, 0.9]
X_batch = np.array([
    [0.5, 0.8, 0.2],
    [0.1, 0.9, 0.9]
])

# ==============================================================================
# PARAMETROS DE LA RED NEURONAL (PESOS Y SESGOS)
# ==============================================================================
# Capa Oculta (3 entradas x 4 neuronas)
W1 = np.array([
    [0.1,  0.2, 0.3,  0.4],
    [-0.5, 0.6, 0.7, -0.8],
    [0.9, -0.1, 0.2,  0.3]
])
b1 = np.array([0.1, 0.2, 0.3, 0.4])  # 4 Sesgos

# Capa de Salida (4 neuronas ocultas x 1 neurona final)
W2 = np.array([0.5, 0.6, 0.7, 0.8])
b2 = np.array([-0.1])

# ==============================================================================
# PROPAGACIÓN HACIA ADELANTE (FORWARD PASS EN BATCH)
# ==============================================================================
# 1. Proceso Capa Oculta
Z1 = np.dot(X_batch, W1) + b1
A1 = sigmoide(Z1)  # Activación de la capa oculta

# 2. Proceso Capa Final
Z2 = np.dot(A1, W2) + b2
Salida_Final = sigmoide(Z2)

# ==============================================================================
# MOSTRAR RESULTADOS (PUNTOS 2 Y 4 DEL LABORATORIO)
# ==============================================================================
print("--- ANÁLISIS DE LA CAPA OCULTA (Z1 vs A1) ---")
print("Z1 (Valores brutos de combinación lineal):\n", Z1)
print("\nA1 (Valores activados por Sigmoide entre 0 y 1):\n", A1)

print("\n--- PREDICCIÓN FINAL EN LOTE (BATCH) ---")
for i, prob in enumerate(Salida_Final):
    print(f"Cliente {i+1} - Probabilidad calculada: {np.round(prob, 4)}")