# Lib imports
import time

import cv2
import numpy as np
import os
import copy
from PIL import Image
import pytesseract

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'E:\Tesseract OCR\tesseract.exe'


# Ensure the "outputs" folder exists
output_folder = 'outputs'
os.makedirs(output_folder, exist_ok=True)


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
color_black = (0 , 0 , 0) # Black

# Global variables
initial_desctiprion_reported = False # if False, it waiting to print them for once, else it will not print again
total_iterations = 0
first_time=True
total_vertical_lines_length = 0
puzzle_size = None

circle_radius_ratio = 0.44

# # Video capture
# def initialize_video_capture(device_index=0):
#     """
#     Initialize video capture from the webcam.
#     :param device_index: Index of the webcam device.
#     :return: VideoCapture object.
#     """
#     return cv2.VideoCapture(device_index)


# Video capture
def initialize_video_capture(device_index=0, width=1000000000, height=1000000000):
    """
    Initialize video capture from the webcam with specified width and height.
    :param device_index: Index of the webcam device.
    :param width: Width of the video capture.
    :param height: Height of the video capture.
    :return: VideoCapture object.
    """
    cap = cv2.VideoCapture(device_index)

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


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

def draw_contour(frame, contour, color):
    """
    Draw the contour on the original frame.
    :param frame: Original frame.
    :param contour: Contour to be drawn.
    """
    cv2.drawContours(frame, [contour], -1, color, 1)


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






# Function to save cropped circles as images
def save_cropped_circles(frame, centroids, puzzle_size):
    height, width = frame.shape[:2]
    cell_height = (height / puzzle_size)
    cell_width = (width / puzzle_size)

    for idx, centroid in enumerate(centroids):
        radius = int(circle_radius_ratio * min(cell_height, cell_width))

        # Calculate the coordinates for cropping
        x_crop = max(0, centroid[0] - radius)
        y_crop = max(0, centroid[1] - radius)
        x2_crop = min(width, centroid[0] + radius)
        y2_crop = min(height, centroid[1] + radius)

        # Crop the circle region
        cropped_circle = frame[y_crop:y2_crop, x_crop:x2_crop]

        # Calculate the row and column numbers (1-indexed)
        row_number = (idx // puzzle_size) + 1
        column_number = (idx % puzzle_size) + 1

        # Save the cropped circle as an image with the specified naming convention
        output_path = os.path.join(output_folder, f'cropped_{row_number} x {column_number}.png')
        cv2.imwrite(output_path, cropped_circle)

# Modify calculate_and_draw_cells to save cropped circles
def calculate_and_draw_cells(frame, original_warped_frame, puzzle_size):
    # Make a copy of the original frame
    original_frame = copy.deepcopy(original_warped_frame)

    height, width = frame.shape[:2]
    cell_height = (height / puzzle_size)
    cell_width = (width / puzzle_size)

    centroids = []  # To store the centroids of rectangles

    for i in range(puzzle_size):
        for j in range(puzzle_size):
            x1 = int(j * cell_width)
            y1 = int(i * cell_height)
            x2 = int((j + 1) * cell_width)
            y2 = int((i + 1) * cell_height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_teal, 1)

            # Calculate and store the centroid
            centroid_x = int((x1 + x2) / 2)
            centroid_y = int((y1 + y2) / 2)
            centroids.append((centroid_x, centroid_y))

    # Draw red points at the centroids
    for centroid in centroids:
        cv2.circle(frame, centroid, 1, color_red, -1)

    # Draw circles around each centroid
    for centroid in centroids:
        radius = int(circle_radius_ratio * min(cell_height, cell_width))
        cv2.circle(frame, centroid, radius, color_purple, 1)

    # Save cropped circles as images
    save_cropped_circles(original_frame, centroids, puzzle_size)

    return frame



def apply_ocr_on_cropped_images(puzzle_size):
    # Iterate through each cell in the grid
    for i in range(1, puzzle_size + 1):
        for j in range(1, puzzle_size + 1):
            # Construct the image path based on the naming convention
            image_path = os.path.join(output_folder, f'cropped_{i} x {j}.png')

            # Read the input image using OpenCV
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Preprocess the image
            preprocessed_image = preprocess_image(image)

            # Specify OCR engine mode (6 for treating the image as sparse text)
            # and whitelist to capture only digits
            custom_config = f'--psm 6 -c tessedit_char_whitelist=0123456789'

            # Perform OCR using Tesseract with custom configuration
            text = pytesseract.image_to_string(preprocessed_image, config=custom_config)

            # Validate and save the extracted text to the sudoku_puzzle.txt file
            validate_and_save_to_file(text, i, j, puzzle_size)

def preprocess_image(image):
    # Check if the image is already grayscale
    if len(image.shape) == 2:
        return image

    # Convert to grayscale if the image has more than one channel
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresholded


def validate_and_save_to_file(text, row, column, puzzle_size):
    # Parse the detected text and convert to an integer
    try:
        number = int(text)
    except ValueError:
        # If OCR fails to detect a number, consider it as 0
        number = 0

    # Validate the number based on the puzzle size
    if 0 <= number <= puzzle_size:
        # Append the number to the sudoku_puzzle.txt file
        with open('sudoku_puzzle.txt', 'a') as file:
            file.write(f"{number} ")

        # If the column is the last one, start a new line
        if column == puzzle_size:
            with open('sudoku_puzzle.txt', 'a') as file:
                file.write('\n')


# Single iteration of the overall process
def sudoku_puzzle_verification(frame, original_frame,  approx):
    global total_iterations, total_vertical_lines_length, puzzle_size, initial_desctiprion_reported, first_time

    if len(approx) == 4:
        pts_dst = np.array([[460, 460], [0, 460], [0, 0], [460, 0]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(approx.reshape(4, 2).astype(np.float32), pts_dst)

        warped = cv2.warpPerspective(frame, matrix, (460, 460))
        # warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        warped = cv2.flip(warped, 0)
        warped_edges = cv2.Canny(warped, 50, 150)
        warped_lines = cv2.HoughLinesP(warped_edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=100)

        warped_original = cv2.warpPerspective(original_frame, matrix, (460, 460))
        # warped_original = cv2.rotate(warped_original, cv2.ROTATE_90_COUNTERCLOCKWISE)
        warped_original = cv2.flip(warped_original, 0)

        # Make a copy of the original frame
        original_warped_frame = copy.deepcopy(warped_original)

        if first_time:

            cv2.imshow('Sudoku Recognition', original_warped_frame)
            first_time=False

        if warped_lines is not None:
            vertical_lines, horizontal_lines = cluster_lines(warped_lines[:, 0, :])

            draw_lines_on_image(warped.copy(), vertical_lines, 'Vertical Lines', color_orange, 0.55)
            draw_lines_on_image(warped.copy(), horizontal_lines, 'Horizontal Lines', color_purple, 0.55)

            sorted_vertical_lines = sort_lines(vertical_lines, orientation='vertical')
            updated_vertical_lines = check_and_update_gaps(sorted_vertical_lines, gap_threshold=10)

            sorted_horizontal_lines = sort_lines(horizontal_lines, orientation='horizontal')
            updated_horizontal_lines = check_and_update_gaps(sorted_horizontal_lines, gap_threshold=10)

            draw_lines_on_image(warped.copy(), updated_vertical_lines, 'updated_vertical_lines Lines', color_orange, 0.55)
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

            updated_lines = updated_vertical_lines + updated_horizontal_lines
            draw_lines_on_image(warped, warped_lines[:, 0, :], 'Warped Image', (0, 0, 255), 0.85)

            total_vertical_lines_length += len(updated_vertical_lines)
            total_iterations += 1

            if 10 <= total_iterations <= 20 and not initial_desctiprion_reported:
                grid_size_approximater = total_vertical_lines_length / (total_iterations)

                if grid_size_approximater < 12:
                    puzzle_size = 9
                else:
                    puzzle_size = 16

                print(f"Averaged Puzzle Size: {puzzle_size}")
                print(f"total_vertical_lines_length: {total_vertical_lines_length}")
                print(f"grid_size_approximater: {grid_size_approximater}")
                print(f"total_iterations: {total_iterations}")
                initial_desctiprion_reported = True

                # Calculate cell size and draw rectangles around each cell
                calculate_and_draw_cells(warped, original_warped_frame, puzzle_size)


                # Display the resulting frame with detected contours and Hough lines
                display_frame(frame)

                # Create a new window for the wrapped image with the grid
                cv2.imshow('Wrapped Image with Grid', warped)

                # Add a delay to allow time for the window to be displayed
                cv2.waitKey(500)  # Adjust the delay time (in milliseconds) as needed

                # TODO : Add the OCR code call after this
                apply_ocr_on_cropped_images(puzzle_size)


def display_frame(frame, scale_factor=0.3):
    """
    Display the resulting frame with detected contours and Hough lines.
    :param frame: Frame to be displayed.
    :param scale_factor: Factor to scale down the frame.
    """
    resized_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    cv2.imshow('Sudoku Recognition', resized_frame)


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

        # Make a copy of the original frame
        original_frame = copy.deepcopy(frame)
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

            # Make a copy of the original frame
            original_frame = copy.deepcopy(frame)

            # Draw the contour on the original frame
            draw_contour(frame, approx, color_green) # TO FIND SIZE
            draw_contour(original_frame, approx, color_black) # TO CROP CELLS

            # If a Sudoku puzzle is detected (assuming 4 corners), proceed with further processing
            sudoku_puzzle_verification(frame, original_frame, approx)

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