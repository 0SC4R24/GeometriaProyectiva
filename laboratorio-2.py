import cv2
import numpy as np

def main():
    pts = np.loadtxt("resources/pts_porteria.txt")
    utad_logo = cv2.imread("resources/u-tad-logo.jpg")
    src_points = np.array([[0,0], [utad_logo.shape[1], 0], [utad_logo.shape[1], utad_logo.shape[0]], [0, utad_logo.shape[0]]], dtype="float32")

    #Se comprueba la ubicación de cada punto anotando un círculo de diferentes colores.
    #Se averigua que los puntos "dibujan" una "c" invertida, empezando por arriba a la izquierda..
    # cv2.circle(frame_aux, (round(pts[0,0]), round(pts[0,1])), 5, (0,0,255), -1)
    # cv2.circle(frame_aux, (round(pts[1,0]), round(pts[1,1])), 5, (0,255,255), -1)
    # cv2.circle(frame_aux, (round(pts[2,0]), round(pts[2,1])), 5, (255,0,0), -1)
    # cv2.circle(frame_aux, (round(pts[3,0]), round(pts[3,1])), 5, (0,255,0), -1)

    # cv2.imshow("Frame", frame_aux)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cap = cv2.VideoCapture(0)
    # if not cap.isOpened():
    #     print("No se pudo abrir la cámara")
    # else:
    #     cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)
    #
    #     while True:
    #         success, cam_frame = cap.read()
    #         if not success:
    #             break
    #         cv2.imshow("Camera Preview", cam_frame)
    #         if cv2.waitKey(15):
    #             break
    #
    #     cap.release()
    #     cv2.destroyWindow("Camera Preview")

    frame_aux = cv2.imread(f"resources/frames/frame_001.jpg")
    size = (frame_aux.shape[1], frame_aux.shape[0])

    out_mp4 = cv2.VideoWriter("video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, size)

    print(pts.shape)
    frames = []
    for i in (range(0, 129)):
        frames.append(cv2.imread(f"resources/frames/frame_{i+1:03d}.jpg"))
    for j in (range(0, pts.shape[1], 2)):
        dst_points = np.array([pts[0, j:j+2], pts[1, j:j+2], pts[2, j:j+2], pts[3, j:j+2]])
        out_mp4.write(process_frame(frames[j//2], utad_logo, (src_points, dst_points)))

    out_mp4.release()

    # cv2.imshow("Partidito", final_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def process_frame(frame, image, points):
    homography_mat = cv2.findHomography(points[0], points[1], cv2.RANSAC, 5.0)
    perspective = cv2.warpPerspective(image, homography_mat[0], (frame.shape[1], frame.shape[0]))
    return cv2.addWeighted(frame, 1, perspective, 0.5, 0)

if __name__ == "__main__":
    main()