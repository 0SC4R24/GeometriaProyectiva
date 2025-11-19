# Author: Oscar Viudez Cuevas

from geometria_proyectiva import *

def primer_punto(frame):
    # Replicar el proceso de la Parte 1 para cada uno de los fotogramas img_XXX.jpg de la carpeta datos\imagenes
    # y generar un video a 30 fotogramas por segundo con las imágenes resultantes. Nota: En el caso de que no sea
    # posible detectar el tablero de ajedrez en alguno de los fotogramas, incluir el fotograma original sin el cuadrado.

    # Obtener los puntos fuente y destino para calcular la homografía
    src_points, dst_points, pattern_found = get_src_and_dst_points(frame)

    # Comprobar si se ha encontrado el patrón
    if not pattern_found:
        print("No se ha podido encontrar el tablero de ajedrez en la imagen.")
        return frame # Devolver el fotograma original si no se encuentra el patrón

    # Calcular la homografía
    H = get_homography(src_points, dst_points)

    # Definir los puntos del cuadrado en el sistema de coordenadas del tablero
    square_1_points = np.array([(8, 8), (8, 16), (16, 8), (16, 16)], dtype="float32")
    square_2_points = np.array([(18, 10), (18, 18), (26, 10), (26, 18)], dtype="float32")

    # Transformar los puntos del cuadrado al sistema de coordenadas de la imagen
    pixel_square_1_points = [transform_2d_point_to_pixel(square_point, H) for square_point in square_1_points]
    pixel_square_2_points = [transform_2d_point_to_pixel(square_point, H) for square_point in square_2_points]

    # Dibujar el cuadrado en la imagen
    draw_square_on_frame(frame, pixel_square_1_points, (0, 180, 0))
    draw_square_on_frame(frame, pixel_square_2_points, (0, 180, 0))

    # Devolver el fotograma procesado
    return frame

def main():
    print("Laboratorio Final - Parte 2: Geometría Proyectiva")

    # Nombre del video de salida
    VIDEO_NAME = "resultados/parte2.mp4"

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