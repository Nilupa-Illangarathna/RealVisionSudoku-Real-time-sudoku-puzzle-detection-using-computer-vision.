# Lib imports
import cv2
import numpy as np

color_yellow = (0, 255, 255)  # Yellow
color_purple = (128, 0, 128)  # Purple
color_orange = (0, 165, 255)  # Orange
color_green = (0, 255, 0)  # Green
color_red = (0, 0, 255)  # Red
color_blue = (255, 0, 0)  # Blue
color_light_blue = (255, 165, 0)  # Light Blue
color_pink = (147, 20, 255)  # Pink
color_teal = (128, 128, 0)  # Teal
color_gray = (169, 169, 169)  # Gray

# Global variables
initial_desctiprion_reported = False # if False, it waiting to print them for once, else it will not print again
total_iterations = 0
total_vertical_lines_length = 0
puzzle_size = None

# Video capture
def initialize_video_capture(device_index=0):
    """
    Initialize video capture from the webcam.
    :param device_index: Index of the webcam device.
    :return: VideoCapture object.
    """
    return cv2.VideoCapture(device_index)

# Basic Image Processing
def image_processing(frame):
    """
    Apply image processing steps to detect edges in the frame.
    :param frame: Input frame.
    :return: Edges detected in the frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

# Finding Contours
def find_contours(edges):
    """
    Find contours in the edged image.
    :param edges: Edges detected in the frame.
    :return: Contours found in the edged image.
    """
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_contour(frame, contour):
    """
    Draw the contour on the original frame.
    :param frame: Original frame.
    :param contour: Contour to be drawn.
    """
    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)


#/////////////////////// Seperate vertical and horizontal Hough Lines (to size prediction) ////////////////////
def cluster_lines(lines):
    """
    Cluster lines into approximately vertical and horizontal lines.
    :param lines: List of lines in the format (x1, y1, x2, y2).
    :return: Two clusters - vertical_lines and horizontal_lines.
    """
    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        x1, y1, x2, y2 = line
        # Check the slope of the line
        if abs(x2 - x1) > abs(y2 - y1): # The diff of X coords and Y coords are considered to get the vertial or horizontal decision.
            horizontal_lines.append(line)
        else:
            vertical_lines.append(line)

    return vertical_lines, horizontal_lines


def draw_lines_on_image(image, lines, window_name, color, window_scale):
    """
    Draw lines on the image.
    :param image: Input image.
    :param lines: List of lines in the format (x1, y1, x2, y2).
    :param window_name: Name of the window to display the image.
    """
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(image, (x1, y1), (x2, y2), color, 1)

    cv2.imshow(window_name, cv2.resize(image, (0, 0), fx=window_scale, fy=window_scale))


#/////////////////////// Sort the lines in each axis for further processing (to size prediction) ////////////////////
def sort_lines(lines, orientation='vertical'):
    """
    Sort lines based on their coordinates (x for vertical, y for horizontal).
    :param lines: List of lines in the format (x1, y1, x2, y2).
    :param orientation: 'vertical' or 'horizontal'.
    :return: Sorted list of lines.
    """
    if orientation == 'vertical':
        # Sort lines based on x coordinates
        sorted_lines = sorted(lines, key=lambda line: (line[0] + line[2]) / 2)
    elif orientation == 'horizontal':
        # Sort lines based on y coordinates
        sorted_lines = sorted(lines, key=lambda line: (line[1] + line[3]) / 2)
    else:
        raise ValueError("Invalid orientation. Use 'vertical' or 'horizontal'.")

    return sorted_lines

# Making sure only one line is there per each edge detected
def check_and_update_gaps(lines, gap_threshold):
    """
    Check and update gaps between adjacent lines in the list.
    :param lines: Sorted list of lines.
    :param gap_threshold: Threshold to consider lines related to the same edge.
    :return: Updated list of lines.
    """
    updated_lines = []

    if lines:
        updated_lines.append(lines[0])

        for i in range(1, len(lines)):
            # Check the gap between adjacent lines
            gap = abs((lines[i - 1][0] + lines[i - 1][2]) / 2 - (lines[i][0] + lines[i][2]) / 2)

            if gap < gap_threshold:
                # Update the current line to the average of the two adjacent lines
                updated_lines[-1] = (
                    int((lines[i - 1][0] + lines[i][0]) / 2),
                    int((lines[i - 1][1] + lines[i][1]) / 2),
                    int((lines[i - 1][2] + lines[i][2]) / 2),
                    int((lines[i - 1][3] + lines[i][3]) / 2)
                )
            else:
                # Add the current line if the gap is larger than the threshold
                updated_lines.append(lines[i])

    return updated_lines



def average_lines(lines):
    """
    Calculate the average line from a list of lines.
    :param lines: List of lines in the format (x1, y1, x2, y2).
    :return: Average line.
    """
    avg_line = (
        int(sum(line[0] for line in lines) / len(lines)),
        int(sum(line[1] for line in lines) / len(lines)),
        int(sum(line[2] for line in lines) / len(lines)),
        int(sum(line[3] for line in lines) / len(lines))
    )

    return avg_line

# Single iteration of the overall process
def sudoku_puzzle_verification(frame, approx):
    global total_iterations, total_vertical_lines_length, puzzle_size, initial_desctiprion_reported


    if len(approx) == 4:
        pts_dst = np.array([[460, 460], [0, 460], [0, 0], [460, 0]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(approx.reshape(4, 2).astype(np.float32), pts_dst)
        warped = cv2.warpPerspective(frame, matrix, (460, 460))
        # warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        warped = cv2.flip(warped, 0)
        warped_edges = cv2.Canny(warped, 50, 150)
        warped_lines = cv2.HoughLinesP(warped_edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=100)

        if warped_lines is not None:
            # Cluster lines into vertical and horizontal
            vertical_lines, horizontal_lines = cluster_lines(warped_lines[:, 0, :]) # Horizontal and vertical line sets are seperated

            # print(f"vertical size : {len(vertical_lines)} and horizontal size : {len(horizontal_lines)}")

            # Draw updated vertical lines on the warped image
            draw_lines_on_image(warped.copy(), vertical_lines, 'Vertical Lines', color_orange, 0.55)

            # Draw updated horizontal lines on the warped image
            draw_lines_on_image(warped.copy(), horizontal_lines, 'Horizontal Lines', color_purple, 0.55) # Yellow

            # Sort and update vertical lines
            sorted_vertical_lines = sort_lines(vertical_lines, orientation='vertical')
            updated_vertical_lines = check_and_update_gaps(sorted_vertical_lines, gap_threshold=10)

            # Sort and update horizontal lines
            sorted_horizontal_lines = sort_lines(horizontal_lines, orientation='horizontal')
            updated_horizontal_lines = check_and_update_gaps(sorted_horizontal_lines, gap_threshold=10)

            # print(f"updated_vertical_lines size : {len(updated_vertical_lines)} and updated_horizontal_lines size : {len(updated_horizontal_lines)}")

            # Draw updated vertical lines on the warped image
            draw_lines_on_image(warped.copy(), updated_vertical_lines, 'updated_vertical_lines Lines', color_orange, 0.55)

            # Draw updated horizontal lines on the warped image
            draw_lines_on_image(warped.copy(), updated_horizontal_lines, 'updated_horizontal_lines Lines', color_purple, 0.55)

            # TODO: Approch 01 - Contours to crop to small parts

            warped_contour_detected_image = warped.copy()
            # Find contours in the flattened Sudoku grid image
            cell_contours, _ = cv2.findContours(warped_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter only rectangular contours
            cell_contours = [contour for contour in cell_contours if len(contour) >= 4]

            # Draw contours in blue on the Sudoku Wrapped Image window
            cv2.drawContours(warped_contour_detected_image, cell_contours, -1, (255, 0, 0), 2)

            if warped_contour_detected_image is not None:
                cv2.imshow('warped_contour_detected_image Image',
                           cv2.resize(warped_contour_detected_image, (0, 0), fx=0.70, fy=0.70))

            if warped_lines is not None:
                for line in warped_lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(warped, (x1, y1), (x2, y2), (0, 0, 255), 1)

            # Draw updated lines on the warped image
            updated_lines = updated_vertical_lines + updated_horizontal_lines
            draw_lines_on_image(warped, warped_lines[:, 0, :], 'Warped Image', (0, 0, 255), 0.85)  # Red

            total_vertical_lines_length += len(updated_vertical_lines)
            total_iterations += 1

            # Calculate average lines and determine puzzle size after 2 seconds
            if 10 <= total_iterations <= 20 and not initial_desctiprion_reported:  # Adjust this value based on your frame rate (iterations per second)
                grid_size_approximater = total_vertical_lines_length / (total_iterations)

                if grid_size_approximater < 12:
                    puzzle_size = 9
                else:
                    puzzle_size = 16

                print(f"vertical size : {len(vertical_lines)} and horizontal size : {len(horizontal_lines)}")
                print(f"updated_vertical_lines size : {len(updated_vertical_lines)} and updated_horizontal_lines size : {len(updated_horizontal_lines)}")
                print(f"Averaged Puzzle Size: {puzzle_size}")
                print(f"total_vertical_lines_length: {total_vertical_lines_length}")
                print(f"grid_size_approximater: {grid_size_approximater}")
                print(f"total_iterations: {total_iterations}")
                initial_desctiprion_reported = True

def display_frame(frame):
    """
    Display the resulting frame with detected contours and Hough lines.
    :param frame: Frame to be displayed.
    """
    cv2.imshow('Sudoku Recognition', frame)

# main function with the loop
def main():
    """
    Main function to capture video, process frames, and display results.
    """
    # # Declare total_iterations as a global variable
    # global total_iterations
    #
    # # Initialize total_iterations
    # total_iterations = 0

    # Initialize video capture from the webcam
    cap = initialize_video_capture()

    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()

        # Apply image processing to detect edges in the frame
        edges = image_processing(frame)

        # Find contours in the edged image
        contours = find_contours(edges)

        # If contours are found
        if contours:
            # Sort contours by area and find the largest contour (presumed Sudoku puzzle)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

            # Approximate the contour to a polygon
            approx = cv2.approxPolyDP(contours[0], 0.02 * cv2.arcLength(contours[0], True), True)

            # Draw the contour on the original frame
            draw_contour(frame, approx)

            # If a Sudoku puzzle is detected (assuming 4 corners), proceed with further processing
            sudoku_puzzle_verification(frame, approx)

        # Display the resulting frame with detected contours and Hough lines
        display_frame(frame)

        # Increment total_iterations
        # total_iterations += 1

        # Break the loop when 'q' is pressed
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    # Release the webcam
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
