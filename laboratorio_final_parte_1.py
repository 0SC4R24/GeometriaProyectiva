import cv2
import numpy as np

def get_homography(src_points, dst_points):
    """Calculate homography matrix."""
    return cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)[0]

def main():
    # Get the K matrix from the camera
    k_matrix = np.loadtxt("resources/datos/K.txt")

    # Get first frame
    frame_1 = cv2.imread("resources/datos/imagenes/img_001.jpg")

    # Create parameters for chessboard detection
    pattern_size = (9, 6) # Number of inner corners per chessboard column and row
    tile_size = 40 # Size of a single tile in mm

    # Find chessboard corners
    pattern_found = cv2.findChessboardCorners(frame_1, pattern_size)

    # If found plot the corners
    corners = pattern_found[1] if pattern_found[0] else []

    # Get specific corner points
    indexes = [45, 27, 47, 29] # Corresponding to the four corners of the chessboard
    dst_points = np.array([[corners[i][0][0], corners[i][0][1]] for i in indexes], dtype="float32")

    # Compute the homography
    src_points = np.array([[tile_size, 0], [0, 0], [tile_size, tile_size], [0, tile_size]], dtype="float32") # Size of a single tile in mm
    homography = get_homography(src_points, dst_points)
    print(homography)

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)] # Red, Green, Blue, Yellow
    # Transform and plot points of a 40x40 mm square grid
    for x in range(0, tile_size + 1, tile_size):
        for y in range(0, tile_size + 1, tile_size):
            # Transform the point using the homography
            point = np.array([[x], [y], [1]])
            transformed_point = homography @ point
            transformed_point /= transformed_point[2]
            print(f"Transformed point ({x},{y}):", transformed_point[:2].flatten())

            # Draw the point on the frame
            x_i, y_i = transformed_point[:2, 0].astype(int)
            print((y // tile_size) + (2 if x == tile_size else 0))
            cv2.circle(frame_1, (int(x_i), int(y_i)), 5, colors[(y // tile_size) + (2 if x == tile_size else 0)], -1)

    # Show the first frame
    cv2.imshow("frame_1", frame_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()