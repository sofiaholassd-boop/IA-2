import cv2
import numpy as np

# 1. Cargar la imagen previa
ruta_imagen = '/workspaces/IA-2/imagen taller 4/image.png'
imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print(f"Error: No se pudo cargar la imagen en {ruta_imagen}")
else:
    # Generar ruido sintético de "Sal y Pimienta" sobre la imagen para la prueba
    ruidosa = imagen.copy()
    num_ruido = 5000
    
    # Agregar sal (puntos blancos)
    coords_sal = [np.random.randint(0, i - 1, num_ruido) for i in imagen.shape[:2]]
    ruidosa[tuple(coords_sal)] = 255
    
    # Agregar pimienta (puntos negros)
    coords_pimienta = [np.random.randint(0, i - 1, num_ruido) for i in imagen.shape[:2]]
    ruidosa[tuple(coords_pimienta)] = 0

    # 2. Aplicar los tres filtros con un Kernel de 7x7
    ksize = 7
    blur_media = cv2.blur(ruidosa, (ksize, ksize))
    blur_gauss = cv2.GaussianBlur(ruidosa, (ksize, ksize), 0)
    blur_mediana = cv2.medianBlur(ruidosa, ksize)

    # 3. Guardar las imágenes procesadas en la carpeta del proyecto
    cv2.imwrite('/workspaces/IA---2/taller4_1_ruidosa.png', ruidosa)
    cv2.imwrite('/workspaces/IA---2/taller4_2_media.png', blur_media)
    cv2.imwrite('/workspaces/IA---2/taller4_3_gaussiano.png', blur_gauss)
    cv2.imwrite('/workspaces/IA---2/taller4_4_mediana.png', blur_mediana)

    print("¡Procesamiento exitoso del Taller 4!")
    print("Se han generado las imágenes:")
    print(" - taller4_1_ruidosa.png")
    print(" - taller4_2_media.png")
    print(" - taller4_3_gaussiano.png")
    print(" - taller4_4_mediana.png")