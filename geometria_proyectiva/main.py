# Author: Oscar Viudez Cuevas
# Description: Script utilizado para la realización de la parte de geometría proyectiva del laboratorio final.

import cv2
import numpy as np

def get_frame(path):
    """
    Read an image from the given path.
    Args:
        path (str): Path to the image file.
    Returns:
        np.ndarray: Loaded image.
    """

    return cv2.imread(path)

def save_frame(path, frame):
    """
    Save an image to the given path.
    Args:
        path (str): Path to save the image file.
        frame (np.ndarray): Image to save.
    Returns:
        bool: True if the image was saved successfully, False otherwise.
    """

    return cv2.imwrite(path, frame)

def singular_value_decomposition(A):
    """
    Perform Singular Value Decomposition on matrix A.
    Args:
        A (np.ndarray): Input matrix to decompose.
    Returns:
        (np.ndarray, np.ndarray, np.ndarray): U, S, Vt matrices from SVD.
    """

    return np.linalg.svd(A)

def get_homography(src_points, dst_points):
    """
    Calculate homography matrix.
    Args:
        src_points (np.ndarray): Source points in chessboard space.
        dst_points (np.ndarray): Destination points in pixel space.
    Returns:
        np.ndarray: Homography matrix H.
    """

    # Ensure there are at least 4 correspondences
    if len(src_points) < 4 or len(dst_points) < 4:
        raise ValueError("At least 4 correspondences are required to compute homography.")

    # Construct matrix A based on point correspondences
    A = []
    for i in range(len(src_points)):
        u_a, v_a = src_points[i][0], src_points[i][1]
        u_b, v_b = dst_points[i][0], dst_points[i][1]
        A.append([u_a, v_a, 1, 0, 0, 0, -u_a * u_b, -v_a * u_b, -u_b])
        A.append([0, 0, 0, u_a, v_a, 1, -u_a * v_b, -v_a * v_b, -v_b])
    A = np.array(A)

    # Perform SVD on A
    U, S, Vt = singular_value_decomposition(A)

    # If S is diagonal with positive values in descending order, the last column of V is equal to h
    if not np.isclose(S, np.diag(np.sort(np.abs(S))[::-1])).any():
        raise ValueError("Singular values are not in the expected format.")

    # Reshape h to get the homography matrix H
    H = Vt[-1].reshape(3, 3)
    H /= H[2, 2]

    # H = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)[0]

    return H

def transform_2d_point_to_pixel(point_2d, homography):
    """
    Transform a 2D point to pixel coordinates using the homography matrix.
    Args:
        point_2d (tuple): 3D point in chessboard space (x, y).
        homography (np.ndarray): Homography matrix H.
    Returns:
        (int, int): Transformed pixel coordinates (x_i, y_i).
    """

    # Convert point to homogeneous coordinates
    point = np.array([[point_2d[0]], [point_2d[1]], [1]])

    # Apply homography transformation
    transformed_point = homography @ point

    # Normalize to get pixel coordinates
    transformed_point /= transformed_point[2]

    # Convert to integer pixel coordinates
    x_i, y_i = transformed_point[:2, 0].astype(int)

    # Return pixel coordinates
    return int(x_i), int(y_i)

def transform_3d_point_to_pixel(point_3d, projection_matrix):
    """
    Transform a 3D point to pixel coordinates using the projection matrix.
    Args:
        point_3d (np.ndarray): 3D point in chessboard space (x, y, z).
        projection_matrix (np.ndarray): Projection matrix P.
    Returns:
        (int, int): Transformed pixel coordinates (x_i, y_i).
    """

    # Convert point to homogeneous coordinates
    point = np.array([[point_3d[0]], [point_3d[1]], [point_3d[2]], [1]])

    # Apply homography transformation
    transformed_point = projection_matrix @ point

    # Normalize to get pixel coordinates
    transformed_point /= transformed_point[2]

    # Convert to integer pixel coordinates
    x_i, y_i = transformed_point[:2, 0].astype(int)

    # Return pixel coordinates
    return int(x_i), int(y_i)

def draw_square_on_frame(frame, square_points, color=(0, 255, 0)):
    """
    Draw a square on the frame given its corner points.
    Args:
        frame (np.ndarray): The image frame to draw on.
        square_points (list): List of four corner points of the square in pixel coordinates.
        color (tuple): Color of the square lines in BGR format.
    """

    cv2.line(frame, square_points[0], square_points[1], color, 3)
    cv2.line(frame, square_points[0], square_points[2], color, 3)
    cv2.line(frame, square_points[1], square_points[3], color, 3)
    cv2.line(frame, square_points[2], square_points[3], color, 3)

def draw_cube_on_frame(frame, cube_points, projection_matrix, color=(0, 255, 0)):
    """
    Draw a cube on the frame given its 3D corner points and projection matrix.
    Args:
        frame (np.ndarray): The image frame to draw on.
        cube_points (list): List of eight corner points of the cube in chessboard space.
        projection_matrix (np.ndarray): Projection matrix P.
        color (tuple): Color of the cube lines in BGR format.
    """

    # Transform cube points to pixel coordinates
    pixel_points = [transform_3d_point_to_pixel(point, projection_matrix) for point in cube_points]

    # Draw bottom square
    draw_square_on_frame(frame, pixel_points[0:4], color)

    # Draw top square
    draw_square_on_frame(frame, pixel_points[4:8], color)

    # Draw vertical lines
    for i in range(4):
        cv2.line(frame, pixel_points[i], pixel_points[i + 4], color, 3)

def get_extrinsic_parameters(K, H):
    """
    Calculate extrinsic parameters (rotation and translation) from intrinsic matrix K and homography H.
    Args:
        K (np.ndarray): Intrinsic camera matrix.
        H (np.ndarray): Homography matrix.
    Returns:
        (np.ndarray, np.ndarray): Rotation matrix R and translation vector t.
    """

    # Inverse of intrinsic matrix
    K_inv = np.linalg.inv(K)

    # Compute matrix M
    M = K_inv @ H

    # Compute M's column vectors
    M_ = M[:, :2]

    # Compute scale factor
    scale = 1 / np.linalg.norm(M_[:, 0])

    # Compute rotation vectors
    r1 = scale * M_[:, 0]
    r2 = scale * M_[:, 1]
    r3 = np.cross(r1, r2)

    # Form rotation matrix
    R = np.column_stack((r1, r2, r3))

    # Enforce valid rotation matrix using SVD
    U, _, Vt = singular_value_decomposition(R)
    R = U @ Vt

    # Ensure R is a proper rotation matrix
    if np.isclose(np.linalg.det(R), -1):
        R = -R

    # Compute translation vector
    t = scale * M[:, 2].reshape(3, 1)

    # Return rotation matrix and translation vector
    return R, t

def get_src_and_dst_points(frame, frame_number = 1, debug = False, pattern_size = (9, 6), tile_size = 4):
    """
    Get source and destination points for homography calculation.
    Args:
        frame (np.ndarray): The image frame to process.
        frame_number (int): Frame number for debugging purposes.
        debug (bool): Whether to draw detected corners on the frame.
        pattern_size (tuple): Number of inner corners per chessboard column and row.
        tile_size (int): Size of a single tile in cm.
    Returns:
        (np.ndarray, np.ndarray, bool): Source points, destination points, and pattern found flag.
    """

    # Find chessboard corners
    pattern_found = cv2.findChessboardCorners(frame, pattern_size)

    # If pattern not found, return the original frame
    if not pattern_found[0]:
        return [], [], pattern_found[0]

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

    # Return source and destination points
    return src_points, dst_points, pattern_found[0]

def process_frame(frame, frame_number = 1, debug = False):
    """
    Process a single frame to detect chessboard and draw axes and squares.
    Args:
        frame (np.ndarray): The image frame to process.
        frame_number (int): Frame number for debugging purposes.
        debug (bool): Whether to draw detected corners on the frame.
    Returns:
        np.ndarray: Processed frame with drawn axes and squares.
    """

    # Create parameters for chessboard detection
    pattern_size = (9, 6) # Number of inner corners per chessboard column and row
    tile_size = 4 # Size of a single tile in cm

    # Get source and destination points
    src_points, dst_points, pattern_found = get_src_and_dst_points(frame, frame_number, debug, pattern_size, tile_size)

    # If pattern not found, return the original frame
    if not pattern_found:
        print(f"Frame {frame_number:03d} - Chessboard pattern not found.")
        return frame

    # Compute the homography
    homography = get_homography(src_points, dst_points)

    # Define the points of the chessboard plane axes
    points = [(0, 0), (0, tile_size * 2), (tile_size * 2, 0)]

    # Define the points in the pixel space plane
    pixel_points = [transform_2d_point_to_pixel((x, y), homography) for x, y in points]

    # Draw the axes on the frame
    cv2.line(frame, pixel_points[0], pixel_points[1], (0, 255, 0), 3)
    cv2.line(frame, pixel_points[0], pixel_points[2], (0, 0, 255), 3)

    # Set square points in chessboard space
    square_1_points = np.array([(8, 8), (8, 16), (16, 8), (16, 16)], dtype="float32")
    square_2_points = np.array([(18, 10), (18, 18), (26, 10), (26, 18)], dtype="float32")

    # Transform square points to pixel coordinates
    square_1_pixel_points = [transform_2d_point_to_pixel((x, y), homography) for x, y in square_1_points]
    square_2_pixel_points = [transform_2d_point_to_pixel((x, y), homography) for x, y in square_2_points]

    # Draw squares on the frame
    draw_square_on_frame(frame, square_1_pixel_points, (0, 180, 0))
    draw_square_on_frame(frame, square_2_pixel_points, (0, 180, 0))

    # Return the processed frame
    return frame

def get_projection_matrix(frame, k_path = "../resources/datos/K.txt", frame_number = 1, debug = False):
    """
    Calculate projection matrix using intrinsic and extrinsic parameters.
    Args:
        frame (np.ndarray): The image frame to process.
        k_path (str): Path to the intrinsic camera matrix K file.
        frame_number (int): Frame number for debugging purposes.
        debug (bool): Whether to draw detected corners on the frame.
    """

    # Load intrinsic camera matrix K
    K = np.loadtxt(k_path)

    # Create parameters for chessboard detection
    pattern_size = (9, 6)  # Number of inner corners per chessboard column and row
    tile_size = 4  # Size of a single tile in cm

    # Get source and destination points
    src_points, dst_points, pattern_found = get_src_and_dst_points(frame, frame_number, debug, pattern_size, tile_size)

    # If pattern not found, return
    if not pattern_found:
        return None

    # Compute the homography
    H = get_homography(src_points, dst_points)

    # Get extrinsic parameters
    R, t = get_extrinsic_parameters(K, H)

    # Draw the frame axes using the first rotation and translation
    cv2.drawFrameAxes(frame, K, None, R, t, 8)

    # Calculate projection matrix P
    P = K @ np.hstack((R, t))

    # Return projection matrix
    return P

def calculate_cube_in_frame(frame, frame_number = 1, debug = False):
    """
    Calculate and draw a cube in the frame using the projection matrix.
    Args:
        frame (np.ndarray): The image frame to process.
        frame_number (int): Frame number for debugging purposes.
        debug (bool): Whether to draw detected corners on the frame.
    Returns:
        np.ndarray: Frame with drawn cube.
    """

    # Calculate projection matrix
    P = get_projection_matrix(frame, frame_number, debug)

    # If projection matrix is None, return the original frame
    if P is None:
        return frame

    # Draw a cube on the frame
    cube_points = [(8, 8, 0), (8, 16, 0), (16, 8, 0), (16, 16, 0),
                   (8, 8, 8), (8, 16, 8), (16, 8, 8), (16, 16, 8)]
    draw_cube_on_frame(frame, cube_points, P, (0, 180, 0))

    # Return the frame with drawn cube
    return frame

def main():
    DEBUG_FRAME = False # Set to True to draw detected corners and frame number
    GET_ONE_FRAME = True # Set to True to process a single frame, False to process all frames and create video
    FRAME_INDEX = 1 # Frame index to process if GET_ONE_FRAME is True
    DRAW_SQUARES = False # Set to True to draw squares and axes, False to calculate and draw cube
    VIDEO_NAME = "../resultados/chessboard_cube.mp4" # Output video name

    if GET_ONE_FRAME:
        frame = cv2.imread(f"../resources/datos/imagenes/img_{FRAME_INDEX:03d}.jpg")

        processed_frame = process_frame(frame, FRAME_INDEX, DEBUG_FRAME) if DRAW_SQUARES else calculate_cube_in_frame(frame, FRAME_INDEX, DEBUG_FRAME)

        cv2.imshow("Processed Frame", processed_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        frames = [cv2.imread(f"../resources/datos/imagenes/img_{i:03d}.jpg") for i in range(1, 736)]

        processed_frames = [process_frame(frame, index + 1, DEBUG_FRAME) if DRAW_SQUARES else calculate_cube_in_frame(frame, index + 1, DEBUG_FRAME) for index, frame in enumerate(frames)]

        out_mp4 = cv2.VideoWriter(VIDEO_NAME, cv2.VideoWriter_fourcc(*"mp4v"), 30, frames[0].shape[1::-1])
        for frame in processed_frames:
            out_mp4.write(frame)
        out_mp4.release()

if __name__ == "__main__": main()