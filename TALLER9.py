import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# 1. Dataset ampliado de 10 clientes (3 dimensiones: [Edad, Salario, Num_Hijos])
X_entrenamiento = np.array([
    [20, 30, 0],  # Cliente 1 -> NO COMPRA
    [40, 50, 2],  # Cliente 2 -> COMPRA
    [35, 45, 1],  # Cliente 3 -> COMPRA
    [18, 22, 0],  # Cliente 4 -> NO COMPRA
    [22, 28, 0],  # Cliente 5 -> NO COMPRA
    [50, 80, 3],  # Cliente 6 -> COMPRA
    [45, 75, 2],  # Cliente 7 -> COMPRA
    [28, 35, 1],  # Cliente 8 -> NO COMPRA
    [60, 90, 1],  # Cliente 9 -> COMPRA
    [25, 32, 0]   # Cliente 10 -> NO COMPRA
])

# Etiquetas binarias (0 = NO COMPRA, 1 = COMPRA)
Y_entrenamiento = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0])

# Nuevo cliente a clasificar: [Edad: 30, Salario: 40, Num_Hijos: 1]
nuevo_cliente = np.array([[30, 40, 1]])

# 2. Escalado de características (Pone todas las variables en la misma magnitud)
scaler = StandardScaler()
X_escalado = scaler.fit_transform(X_entrenamiento)
nuevo_cliente_escalado = scaler.transform(nuevo_cliente)

# 3. Clasificación con K = 1 sobre datos escalados
modelo_k1 = KNeighborsClassifier(n_neighbors=1)
modelo_k1.fit(X_escalado, Y_entrenamiento)
pred_k1 = modelo_k1.predict(nuevo_cliente_escalado)[0]

texto_k1 = "COMPRA (1)" if pred_k1 == 1 else "NO COMPRA (0)"
print("Predicción con K=1:", texto_k1)

# 4. Clasificación con K = 5 sobre datos escalados
modelo_k5 = KNeighborsClassifier(n_neighbors=5)
modelo_k5.fit(X_escalado, Y_entrenamiento)
pred_k5 = modelo_k5.predict(nuevo_cliente_escalado)[0]

texto_k5 = "COMPRA (1)" if pred_k5 == 1 else "NO COMPRA (0)"
print("Predicción con K=5:", texto_k5)