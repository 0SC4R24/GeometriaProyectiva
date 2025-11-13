import cv2

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara")
    else:
        cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)

        while True:
            success, frame = cap.read()
            if not success:
                break

            edge = cv2.Canny(frame, 100, 200)
            inverse = cv2.bitwise_not(frame)
            cv2.imshow("Camera Preview", edge)

            if cv2.waitKey(15) & 0xFF == 27:  # Presiona 'Esc' para salir
                break

        cap.release()
        cv2.destroyWindow("Camera Preview")

if __name__ == "__main__":
    main()