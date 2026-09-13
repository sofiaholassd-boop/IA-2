import cv2
import numpy as np

# 1. Cargar la imagen a color
ruta_imagen = '/workspaces/IA-2/imagen taller 6/image.png'
imagen_color = cv2.imread(ruta_imagen)

if imagen_color is None:
    print(f"Error: No se pudo cargar la imagen en {ruta_imagen}")
else:
    # Copia para dibujar resultados
    resultado = imagen_color.copy()

    # 2. Pipeline completo de procesamiento
    # Paso A: Conversión a escala de grises
    gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

    # Paso B: Binarización con método de Otsu
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Paso C: Limpieza Morfológica (Apertura)
    kernel = np.ones((3, 3), np.uint8)
    limpia = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

    # Paso D: Extracción de Contornos
    contornos, _ = cv2.findContours(limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Objetos totales detectados: {len(contornos)}")
    umbral_area = 1500  # Valor límite para clasificar tamaño

    for i, cnt in enumerate(contornos):
        # 3. Calcular métricas
        area = cv2.contourArea(cnt)
        
        # Filtrar pequeñas partículas de ruido
        if area > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            print(f"Objeto #{i+1}: Área = {area} px")

            # 4. Lógica de clasificación por área
            if area > umbral_area:
                color_box = (255, 0, 0)  # Azul (BGR): Objeto grande
                etiqueta = "Grande"
            else:
                color_box = (0, 0, 255)  # Rojo (BGR): Objeto pequeño
                etiqueta = "Pequeno"

            # Dibujar Bounding Box y etiqueta
            cv2.rectangle(resultado, (x, y), (x + w, y + h), color_box, 2)
            cv2.putText(resultado, etiqueta, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 1)

            # Calcular Centroide
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(resultado, (cx, cy), 4, (0, 255, 0), -1)

    # 5. Guardar la imagen procesada
    
    # 5. Guardar la imagen procesada
    cv2.imwrite('/workspaces/IA-2/imagen taller 6/image.png', resultado)
    print("Procesamiento finalizado. Resultado guardado en '/workspaces/IA-2/imagen taller 6/image.png'.")