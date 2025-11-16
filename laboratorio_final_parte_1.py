import cv2
import numpy as np

def singular_value_decomposition(A):
    """Perform Singular Value Decomposition on matrix A."""
    U, S, Vt = np.linalg.svd(A)

    # U = np.zeros((A.shape[0], A.shape[0]))
    # S = np.zeros((A.shape[0], A.shape[1]))
    # Vt = np.zeros((A.shape[1], A.shape[1]))
    #
    # AtA = A.T @ A
    # eigenvalues, eigenvectors = np.linalg.eig(AtA)
    # idx = eigenvalues.argsort()[::-1]
    # eigenvalues = eigenvalues[idx]
    # eigenvectors = eigenvectors[:, idx]
    #
    # for i in range(len(eigenvalues)):
    #     S[i, i] = np.sqrt(eigenvalues[i])
    #
    # Vt = eigenvectors.T
    #
    # for i in range(len(eigenvalues)):
    #     if S[i, i] > 1e-10:
    #         U[:, i] = (A @ Vt[i, :]) / S[i, i]

    return U, S, Vt

def get_homography(src_points, dst_points):
    """Calculate homography matrix."""
    if len(src_points) < 4 or len(dst_points) < 4:
        raise ValueError("At least 4 correspondences are required to compute homography.")

    A = []
    for i in range(len(src_points)):
        u_a, v_a = src_points[i][0], src_points[i][1]
        u_b, v_b = dst_points[i][0], dst_points[i][1]
        A.append([u_a, v_a, 1, 0, 0, 0, -u_a * u_b, -v_a * u_b, -u_b])
        A.append([0, 0, 0, u_a, v_a, 1, -u_a * v_b, -v_a * v_b, -v_b])
    A = np.array(A)

    U, S, Vt = singular_value_decomposition(A)
    H = Vt[-1].reshape(3, 3)
    H /= H[2, 2]

    # Optional: Validate with OpenCV's findHomography
    # H_cv2 = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)[0]

    return H

def transform_3d_point_to_pixel(point_3d, homography):
    """Transform a 3D point to pixel coordinates using the homography matrix."""
    point = np.array([[point_3d[0]], [point_3d[1]], [1]])
    transformed_point = homography @ point
    transformed_point /= transformed_point[2]
    x_i, y_i = transformed_point[:2, 0].astype(int)
    return int(x_i), int(y_i)

def draw_square_on_frame(frame, square_points, color=(0, 255, 0)):
    """Draw a square on the frame given its corner points."""
    cv2.line(frame, square_points[0], square_points[1], color, 3)
    cv2.line(frame, square_points[0], square_points[2], color, 3)
    cv2.line(frame, square_points[1], square_points[3], color, 3)
    cv2.line(frame, square_points[2], square_points[3], color, 3)

def process_frame(frame, frame_number = 1, debug = False):
    """Process a single frame to detect chessboard and draw axes and squares."""
    # Create parameters for chessboard detection
    pattern_size = (9, 6) # Number of inner corners per chessboard column and row
    tile_size = 4 # Size of a single tile in cm

    # Find chessboard corners
    pattern_found = cv2.findChessboardCorners(frame, pattern_size)

    # If pattern not found, return the original frame
    if not pattern_found[0]:
        return frame

    # If found plot the corners
    corners = pattern_found[1] if pattern_found[0] else []

    # Draw and display the corners if debug is True
    if debug:
        cv2.drawChessboardCorners(frame, pattern_size, corners, pattern_found[0])
        # Write the frame number on the frame
        cv2.putText(frame, f"Frame: {frame_number:03d}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Get destination points from detected corners in pixel space
    indexes = range(len(corners))
    dst_points = np.array([[corners[i][0][0], corners[i][0][1]] for i in indexes], dtype="float32")

    # Get corresponding source points of destination points in chessboard space
    src_points = np.array([(x, y) for y in range((pattern_size[1] - 1) * tile_size, -1, -tile_size) for x in range(0, pattern_size[0] * tile_size, tile_size)], dtype="float32")

    # Compute the homography
    homography = get_homography(src_points, dst_points)
    # print(homography)

    # Define the points of the chessboard plane axes
    points = [(0, 0), (0, tile_size * 2), (tile_size * 2, 0)]

    # Define the points in the pixel space plane
    pixel_points = [transform_3d_point_to_pixel((x, y), homography) for x, y in points]

    # Draw the axes on the frame
    cv2.line(frame, pixel_points[0], pixel_points[1], (0, 255, 0), 3)
    cv2.line(frame, pixel_points[0], pixel_points[2], (0, 0, 255), 3)

    # Set square points in chessboard space
    square_1_points = np.array([(8, 8), (8, 16), (16, 8), (16, 16)], dtype="float32")
    square_2_points = np.array([(18, 10), (18, 18), (26, 10), (26, 18)], dtype="float32")

    # Transform square points to pixel coordinates
    square_1_pixel_points = [transform_3d_point_to_pixel((x, y), homography) for x, y in square_1_points]
    square_2_pixel_points = [transform_3d_point_to_pixel((x, y), homography) for x, y in square_2_points]

    # Draw squares on the frame
    draw_square_on_frame(frame, square_1_pixel_points, (0, 180, 0))
    draw_square_on_frame(frame, square_2_pixel_points, (0, 180, 0))

    return frame

def main():
    DEBUG_FRAME = False
    GET_ONE_FRAME = True
    FRAME_INDEX = 157

    if GET_ONE_FRAME:
        frame = cv2.imread(f"resources/datos/imagenes/img_{FRAME_INDEX:03d}.jpg")
        processed_frame = process_frame(frame, FRAME_INDEX, DEBUG_FRAME)

        cv2.imshow("Processed Frame", processed_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        frames = [cv2.imread(f"resources/datos/imagenes/img_{i:03d}.jpg") for i in range(1, 736)]
        processed_frames = [process_frame(frame, index + 1, DEBUG_FRAME) for index, frame in enumerate(frames)]

        out_mp4 = cv2.VideoWriter("chessboard_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, frames[0].shape[1::-1])
        for frame in processed_frames:
            out_mp4.write(frame)
        out_mp4.release()

if __name__ == "__main__":
    main()