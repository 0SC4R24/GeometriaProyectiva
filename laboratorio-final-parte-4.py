# Author: Oscar Viudez Cuevas

from geometria_proyectiva import *

def primer_punto(frame, k = "resources/datos/K.txt"):
    # Calcular la matriz de proyección P = K[R | t] para cada uno de los fotogramas img_XXX.jpg de la carpeta
    # datos\imagenes y utilizarla para proyectar un cubo de lado 8 cm sobre 4 casillas del tablero de ajedrez. Generar
    # un video a 30 fotogramas por segundo con las imágenes resultantes. Nota: En el caso de que no sea posible
    # detectar el tablero de ajedrez en alguno de los fotogramas, incluir el fotograma original sin el cubo.

    # Calcular la matriz de proyección P = K[R | t]
    P = get_projection_matrix(frame, k)

    # Comprobar si se ha podido calcular la matriz de proyección
    if P is None:
        print("No se ha podido encontrar el tablero de ajedrez en la imagen.")
        return frame # Devolver el fotograma original si no se encuentra el patrón

    # Definir los puntos 3D del cubo en el sistema de coordenadas del tablero
    cube_points = [(8, 8, 0), (8, 16, 0), (16, 8, 0), (16, 16, 0),
                   (8, 8, 8), (8, 16, 8), (16, 8, 8), (16, 16, 8)]

    # Dibujar el cubo en la imagen
    draw_cube_on_frame(frame, cube_points, P, (0, 180, 0))

    # Devolver el fotograma procesado
    return frame

def main():
    print("Laboratorio Final - Parte 4: Geometría Proyectiva")

    # Nombre del video de salida
    VIDEO_NAME = "resultados/parte4.mp4"

    # Cargar todos los fotogramas
    frames = [get_frame(f"resources/datos/imagenes/img_{i:03d}.jpg") for i in range(1, 736)]

    # Procesar cada fotograma
    processed_frames = [primer_punto(frame) for frame in frames]

    # Guardar el video
    out_mp4 = cv2.VideoWriter(VIDEO_NAME, cv2.VideoWriter_fourcc(*"mp4v"), 30, frames[0].shape[1::-1])
    for frame in processed_frames:
        out_mp4.write(frame)
    out_mp4.release()

if __name__ == "__main__": main()