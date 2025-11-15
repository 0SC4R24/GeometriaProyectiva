import cv2
import numpy as np

def get_homography(src_points, dst_points):
    """Calculate homography matrix."""
    return cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)[0]

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
    GET_ONE_FRAME = False
    FRAME_INDEX = 1

    if GET_ONE_FRAME:
        frame = cv2.imread(f"resources/datos/imagenes/img_{FRAME_INDEX:03d}.jpg")
        processed_frame = process_frame(frame, FRAME_INDEX, DEBUG_FRAME)

        cv2.imshow("Processed Frame", processed_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        frames = [(cv2.imread(f"resources/datos/imagenes/img_{i:03d}.jpg"), i) for i in range(1, 736)]
        processed_frames = [process_frame(*frame, DEBUG_FRAME) for frame in frames]

        out_mp4 = cv2.VideoWriter("chessboard_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, frames[0][0].shape[1::-1])
        for frame in processed_frames:
            out_mp4.write(frame)
        out_mp4.release()

if __name__ == "__main__":
    main()