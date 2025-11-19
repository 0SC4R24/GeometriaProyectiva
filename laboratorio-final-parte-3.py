# Author: Oscar Viudez Cuevas

import os

import numpy as np

from geometria_proyectiva import *

# Dada la componente intrínseca o matriz de calibración de la cámara K, definida en el fichero datos\K.txt y la
# matriz de homografía H del apartado (1) de la Parte 1:

def primer_punto(K, H):
    # (1) Calcular los parámetros extrínsecos de la cámara R y t que se obtienen al situar el origen de coordenadas
    # en la esquina inferior interna izquierda del tablero (ver Figura 4). Comprobar si la matriz de rotación obtenida
    # R es una verdadera matriz de rotación.

    # Utilizar la función get_extrinsic_parameters definida en geometria_proyectiva.py
    # get_extrinsic_parameters ya realiza la comprobación de sí R es una verdadera matriz de rotación
    return get_extrinsic_parameters(K, H)

def segundo_punto(R, t):
    # (2) Si la matriz de rotación obtenida R en el apartado (1) no es una verdadera matriz de rotación, calcular de
    # nuevo los parámetros extrínsecos de la cámara R y t utilizando un metodo que asegure que es una verdadera
    # matriz de rotación.

    # Utilizar la función singular_value_decomposition definida en geometria_proyectiva.py
    U, _, Vt = singular_value_decomposition(R)
    R_corrected = U @ Vt

    # Asegurarse de que R_corrected es una verdadera matriz de rotación
    if np.linalg.det(R_corrected) < 0:
        R_corrected = -R_corrected

    # Devolver la matriz de rotación corregida y el vector de traslación original
    return R_corrected, t

def tercer_punto(frame, K, R, t):
    # (3) Utilizar la función drawFrameAxes de OpenCV y los resultados del apartado (2) para dibujar los ejes en
    # el origen de coordenadas sobre la imagen datos\imagenes\img_001.jpg.

    # Dibujar los ejes del sistema de coordenadas en la imagen
    cv2.drawFrameAxes(frame, K, None, R, t, 8)

    # Devolver la imagen con los ejes dibujados
    return frame

def main():
    print("Laboratorio Final - Parte 3: Geometría Proyectiva")

    K_MATRIX_PATH = "resources/datos/K.txt"
    HOMOGRAPHY_PATH = "resultados/parte1_H.txt"
    FRAME = get_frame("resources/datos/imagenes/img_001.jpg")

    # Cargar la matriz de calibración K
    K = np.loadtxt(K_MATRIX_PATH) if os.path.exists(K_MATRIX_PATH) else None

    # Si el archivo de la matriz de calibración no existe, lanzar un error
    if K is None:
        raise FileNotFoundError(f"No se pudo encontrar el archivo de la matriz de calibración K en {K_MATRIX_PATH}")

    # Cargar la homografía H del apartado (1) de la Parte 1
    H = np.loadtxt(HOMOGRAPHY_PATH) if os.path.exists(HOMOGRAPHY_PATH) else None

    # Si el archivo de la homografía no existe, calcularla
    if H is None:
        print("Calculando la homografía H del apartado (1) de la Parte 1...")
        H = get_homography(*get_src_and_dst_points(FRAME)[0:2])

    R, t = primer_punto(K, H)
    np.savetxt("resultados/parte3_1_R.txt", R)
    np.savetxt("resultados/parte3_1_t.txt", t)

    R, t = segundo_punto(R, t)
    np.savetxt("resultados/parte3_2_R.txt", R)
    np.savetxt("resultados/parte3_2_t.txt", t)

    frame_axes = tercer_punto(FRAME, K, R, t)
    save_frame("resultados/parte3.png", frame_axes)

if __name__ == "__main__": main()