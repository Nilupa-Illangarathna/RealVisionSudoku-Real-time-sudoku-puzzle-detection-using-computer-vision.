# Real-time Camera Sudoku Puzzle Recognition

## 1. Introduction

This project focuses on developing a real-time Sudoku puzzle recognition system using computer vision techniques. It analyzes live camera feeds, detects Sudoku grids, processes grid contours, and extracts individual cells for OCR-based digit recognition.

## 2. Methodology

### 2.1 Algorithmic Approach

#### 2.1.1 Video Capture
The live camera feed is continuously captured, initializing video capture from the webcam.

![Continuous video feed](docs/figures/video_feed.png)

#### 2.1.2 Image Processing
Frames undergo grayscale conversion, Gaussian blurring, and Canny edge detection.

#### 2.1.3 Contour Detection
Contours are identified in the edged image, and the largest contour (presumed to be the Sudoku puzzle) is approximated to a polygon.

![Contour detection](docs/figures/contour_detection.png)

#### 2.1.4 Perspective Transformation
The Sudoku puzzle undergoes a geometric shift to obtain a flattened, standardized 460x460 image.

#### 2.1.5 Line Detection
Using the Hough line detection algorithm, both vertical and horizontal lines within the Sudoku puzzle are identified.

![Line detection](docs/figures/line_detection.png)

#### 2.1.6 Cell Cropping
The grid is partitioned into individual cells through meticulous contour analysis.

![Cell cropping](docs/figures/cell_cropping.png)

#### 2.1.7 OCR
Optical Character Recognition (OCR) is applied to each isolated cell cropped image using Tesseract.

![OCR](docs/figures/ocr.png)

## 3. Results

The project successfully detects and solves Sudoku puzzles captured in a live feed. Tesseract OCR is used for digit recognition.

![Original puzzle detection](docs/figures/original_puzzle.png)

## 4. Limitations

While precise, Tesseract OCR faces occasional challenges in accurate digit recognition due to factors like image quality and font variations. Alternative OCR solutions may be explored for future enhancements.

## 5. Usage

Follow the steps in [SETUP.md](SETUP.md) to set up the development environment.

## 6. Acknowledgments

We acknowledge Tesseract OCR for digit recognition.

## 7. License

This project is licensed under the [MIT License](LICENSE).

---

**Note:** Please replace placeholder links and filenames with actual paths and images.
