

import cv2
import numpy as np

# Initialize video capture from the webcam
cap = cv2.VideoCapture(0)

# Initialize variables to store contours over multiple frames
all_cell_contours = []

# Initialize warped variable outside the loop
warped = None

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Image processing for Sudoku puzzle detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours are found
    if contours:
        # Sort contours by area and find the largest contour (presumed Sudoku puzzle)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True)

        # Draw the contour on the original frame in green
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

        # If a Sudoku puzzle is detected (assuming 4 corners), proceed with digit identification
        if len(approx) == 4:
            # Define the destination points for the corrected projective transformation
            pts_dst = np.array([[460, 460], [0, 460], [0, 0], [460, 0]], dtype=np.float32)

            # Perform projective transformation
            matrix = cv2.getPerspectiveTransform(approx.reshape(4, 2).astype(np.float32), pts_dst)
            warped = cv2.warpPerspective(frame, matrix, (460, 460))

            # Perform rotation to correct the orientation (90 degrees counterclockwise)
            warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Flip the warped image vertically (180 degrees)
            warped = cv2.flip(warped, 1)

            # Convert the warped image to grayscale
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            # Apply GaussianBlur to reduce noise
            warped_blurred = cv2.GaussianBlur(warped_gray, (5, 5), 0)

            # Apply Canny edge detector to find edges
            warped_edges = cv2.Canny(warped_blurred, 50, 150)

            # Find contours in the flattened Sudoku grid image
            cell_contours, _ = cv2.findContours(warped_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter only rectangular contours
            cell_contours = [contour for contour in cell_contours if len(contour) >= 4]

            # Draw contours in blue on the Sudoku Wrapped Image window
            cv2.drawContours(warped, cell_contours, -1, (255, 0, 0), 2)

            # Store the detected cell contours in the list
            all_cell_contours.extend(cell_contours)

    # Display the resulting frame with detected contours
    cv2.imshow('Sudoku Recognition', frame)

    # Display the Sudoku Wrapped Image for digit identification
    if warped is not None:
        cv2.imshow('Warped Image', warped)

    # Create a new window to show the detected cell contours on the wrapped image
    if all_cell_contours:
        contours_image = np.copy(warped) if warped is not None else np.copy(frame)
        cv2.drawContours(contours_image, all_cell_contours, -1, (0, 0, 255), 2)
        cv2.imshow('Detected Cell Contours', contours_image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
