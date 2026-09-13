import cv2
import numpy as np

# 1. Cargar la imagen en escala de grises
ruta_imagen = '/workspaces/IA-2/imagen taller 5/image.png'
imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

if imagen is None:
    print(f"Error: No se pudo cargar la imagen en {ruta_imagen}")
else:
    # 2. Detección de Bordes con Sobel
    sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)

    # Convertir valores absolutos a 8-bit (0-255)
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)

    # 3. Detección de Bordes con Canny usando distintas configuraciones de umbral
    canny_optimo = cv2.Canny(imagen, 50, 150)   # Umbral estándar equilibrado
    canny_sensible = cv2.Canny(imagen, 10, 50)   # Umbral bajo (capta mucho detalle y ruido)
    canny_estricto = cv2.Canny(imagen, 200, 250) # Umbral alto (solo conserva bordes muy fuertes)

    # 4. Guardar imágenes en archivos para visualización en GitHub Codespaces
    # 4. Guardar imágenes en archivos para visualización en GitHub Codespaces
    cv2.imwrite('/workspaces/IA-2/taller5_sobel_x.png', sobel_x_abs)
    cv2.imwrite('/workspaces/IA-2/taller5_sobel_y.png', sobel_y_abs)
    cv2.imwrite('/workspaces/IA-2/taller5_canny_optimo.png', canny_optimo)
    cv2.imwrite('/workspaces/IA-2/taller5_canny_sensible.png', canny_sensible)
    cv2.imwrite('/workspaces/IA-2/taller5_canny_estricto.png', canny_estricto)

    print("¡Procesamiento exitoso del Taller 5!")
    print("Archivos generados:")
    print(" - taller5_sobel_x.png")
    print(" - taller5_sobel_y.png")
    print(" - taller5_canny_optimo.png")
    print(" - taller5_canny_sensible.png")
    print(" - taller5_canny_estricto.png")