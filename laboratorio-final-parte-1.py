# Author: Oscar Viudez Cuevas
import numpy as np

from geometria_proyectiva import *

# Utilizando únicamente la imagen datos\imagenes\img_001.jpg:

def primer_punto(frame):
    # (1) Calcular la homografía H entre el plano que contiene al tablero de ajedrez y el plano de la pantalla, situando
    # el origen de coordenadas del plano que contiene al tablero de ajedrez en la esquina inferior interna izquierda del
    # tablero (ver Figura 1) y teniendo en cuenta que el lado de cada casilla del tablero mide 4 cm.

    # Obtener los puntos fuente y destino para calcular la homografía
    src_points, dst_points, pattern_found = get_src_and_dst_points(frame)

    # Comprobar si se ha encontrado el patrón
    if not pattern_found:
        print("No se ha podido encontrar el tablero de ajedrez en la imagen.")
        return None

    # Calcular la homografía
    return get_homography(src_points, dst_points)

def segundo_punto(frame, H):
    # (2) Utilizar la homografía H obtenida en el apartado anterior, para proyectar un cuadrado de 8 cm de lado
    # sobre el tablero. El cuadrado debe estar situado sobre 4 casillas.

    # Definir los puntos del cuadrado en el sistema de coordenadas del tablero
    square_points = np.array([(4, 4), (4, 12), (12, 4), (12, 12)], dtype=np.float32)  # Cuadrado de 8 cm sobre 4 casillas

    # Transformar los puntos del cuadrado al sistema de coordenadas de la imagen
    pixel_points = [transform_2d_point_to_pixel(square_point, H) for square_point in square_points]

    # Dibujar el cuadrado en la imagen
    draw_square_on_frame(frame, pixel_points, (0, 180, 0))

def tercer_punto(frame, H):
    # (3) Utilizar la homografía H obtenida en el apartado anterior, para proyectar un cuadrado de 8 cm de lado
    # sobre el tablero. En este caso, los vértices del cuadrado deben estar situados en el centro de 4 casillas.

    # Definir los puntos del cuadrado en el sistema de coordenadas del tablero
    square_points = np.array([(6, 6), (6, 14), (14, 6), (14, 14)], dtype=np.float32)  # Cuadrado de 8 cm sobre 4 casillas

    # Transformar los puntos del cuadrado al sistema de coordenadas de la imagen
    pixel_points = [transform_2d_point_to_pixel(square_point, H) for square_point in square_points]

    # Dibujar el cuadrado en la imagen
    draw_square_on_frame(frame, pixel_points, (0, 180, 0))

def main():
    print("Laboratorio Final - Parte 1: Geometría Proyectiva")

    FRAME_PATH = "resources/datos/imagenes/img_001.jpg"
    HOMOGRAPHY_PATH = "resultados/parte1_H.txt"
    FIRST_IMAGE_PATH = "resultados/parte1_2.png"
    SECOND_IMAGE_PATH = "resultados/parte1_3.png"

    FRAME = get_frame(FRAME_PATH)

    H = primer_punto(FRAME)
    # Guardar la homografía en un fichero
    np.savetxt(HOMOGRAPHY_PATH, H)

    second_frame = FRAME.copy()
    segundo_punto(second_frame, H)
    save_frame(FIRST_IMAGE_PATH, second_frame)

    third_frame = FRAME.copy()
    tercer_punto(third_frame, H)
    save_frame(SECOND_IMAGE_PATH, third_frame)

if __name__ == "__main__": main()