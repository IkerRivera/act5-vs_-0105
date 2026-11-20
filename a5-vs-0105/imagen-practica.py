
import cv2
import numpy as np
import os

# ==========================================
# 1. CONFIGURACION INICIAL
# ==========================================

ruta_imagen = "rayo.webp"
carpeta_salida = "resultados"

# Crear carpeta para guardar resultados
os.makedirs(carpeta_salida, exist_ok=True)

# Cargar imagen
imagen = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR)

# Verificar que la imagen se haya cargado
if imagen is None:
    print("ERROR: No se pudo cargar rayo.webp")
    print("Verifica que la imagen este en la misma carpeta.")
    exit()

print("Imagen cargada correctamente.")

# Obtener dimensiones
alto, ancho = imagen.shape[:2]

# ==========================================
# 2. IMAGEN ORIGINAL CON NOMBRE
# ==========================================

imagen_original = imagen.copy()

cv2.putText(
    imagen_original,
    "Iker Montoya 0105",
    (20, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2,
    cv2.LINE_AA
)

cv2.imwrite(
    carpeta_salida + "/original_con_nombre.jpg",
    imagen_original
)

cv2.imshow("1. Imagen Original - Iker Montoya 0105",
           imagen_original)

# ==========================================
# 3. CONVERTIR A ESCALA DE GRISES
# ==========================================

imagen_gris = cv2.cvtColor(
    imagen,
    cv2.COLOR_BGR2GRAY
)

cv2.imwrite(
    carpeta_salida + "/gris.jpg",
    imagen_gris
)

cv2.imshow("2. Escala de Grises", imagen_gris)

# ==========================================
# 4. CAMBIAR TAMANO DE IMAGEN
# ==========================================

imagen_redimensionada = cv2.resize(
    imagen,
    (300, 200)
)

cv2.imwrite(
    carpeta_salida + "/redimensionada.jpg",
    imagen_redimensionada
)

cv2.imshow("3. Imagen Redimensionada",
           imagen_redimensionada)

# ==========================================
# 5. ROTAR IMAGEN 90 GRADOS
# ==========================================

imagen_rotada = cv2.rotate(
    imagen,
    cv2.ROTATE_90_CLOCKWISE
)

cv2.imwrite(
    carpeta_salida + "/rotada.jpg",
    imagen_rotada
)

cv2.imshow("4. Imagen Rotada", imagen_rotada)

# ==========================================
# 6. APLICAR DESENFOQUE GAUSSIANO
# ==========================================

imagen_desenfocada = cv2.GaussianBlur(
    imagen,
    (15, 15),
    0
)

cv2.imwrite(
    carpeta_salida + "/desenfocada.jpg",
    imagen_desenfocada
)

cv2.imshow("5. Desenfoque Gaussiano",
           imagen_desenfocada)

# ==========================================
# 7. DETECCION DE BORDES CANNY
# ==========================================

bordes = cv2.Canny(
    imagen_gris,
    100,
    200
)

cv2.imwrite(
    carpeta_salida + "/bordes_canny.jpg",
    bordes
)

cv2.imshow("6. Deteccion de Bordes Canny",
           bordes)

# ==========================================
# 8. DIBUJAR RECTANGULO Y AGREGAR TEXTO
# ==========================================

imagen_anotada = imagen.copy()

# Dibujar rectangulo
cv2.rectangle(
    imagen_anotada,
    (20, 60),
    (min(200, ancho - 1), min(200, alto - 1)),
    (255, 0, 0),
    3
)

# Agregar texto
cv2.putText(
    imagen_anotada,
    "OpenCV",
    (20, 45),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.imwrite(
    carpeta_salida + "/anotada.jpg",
    imagen_anotada
)

cv2.imshow("7. Rectangulo y Texto",
           imagen_anotada)

# ==========================================
# 9. UMBRALIZACION BINARIA
# ==========================================

_, imagen_umbralizada = cv2.threshold(
    imagen_gris,
    127,
    255,
    cv2.THRESH_BINARY
)

cv2.imwrite(
    carpeta_salida + "/umbralizada.jpg",
    imagen_umbralizada
)

cv2.imshow("8. Umbralizacion",
           imagen_umbralizada)

# ==========================================
# 10. DETECCION DE CONTORNOS
# ==========================================

contornos, jerarquia = cv2.findContours(
    bordes,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

imagen_contornos = imagen.copy()

cv2.drawContours(
    imagen_contornos,
    contornos,
    -1,
    (0, 255, 0),
    2
)

cv2.imwrite(
    carpeta_salida + "/contornos.jpg",
    imagen_contornos
)

cv2.imshow("9. Deteccion de Contornos",
           imagen_contornos)

# ==========================================
# 11. INFORMACION DE LA IMAGEN
# ==========================================

print("\n========== INFORMACION DE LA IMAGEN ==========")

print("Nombre: rayo.webp")
print("Autor: Iker Montoya 0105")
print("Ancho:", ancho, "pixeles")
print("Alto:", alto, "pixeles")
print("Canales:", imagen.shape[2])
print("Contornos detectados:", len(contornos))

print("\nTodas las imagenes se guardaron en:")
print(carpeta_salida)

# ==========================================
# 12. CERRAR VENTANAS
# ==========================================

print("\nPresiona cualquier tecla para cerrar las ventanas.")

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nPrograma finalizado correctamente.")
