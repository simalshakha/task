# Rectangle and Line Detection with Numbering

This project detects rectangles and lines inside those rectangles in an image, then assigns and draws numbers on the lines based on their length (shorter lines get lower numbers).

---

## Description

The script processes an input image containing rectangles and lines inside those rectangles. It:

1. Finds all contours in the image.
2. Filters contours to detect rectangles (4-sided polygons).
3. Detects line contours that lie strictly inside these rectangles.
4. Measures the length (maximum distance between any two points) of each line inside rectangles.
5. Sorts these lines by length and assigns numbers accordingly (shorter lines get lower numbers).
6. Draws the detected lines and their assigned numbers on the output image.
7. Saves the output image and displays it.

---

## Dependencies

- Python 3.x
- OpenCV (`opencv-python`)
- Matplotlib

Install the required libraries with:

```bash
pip install opencv-python matplotlib
```
---
### usuage

1.Place your input images (e.g., triangle.png, image.png) in the project directory.

2.Run the scripts using Python:

```bash

python rectangle_line_numbering.py

python rectangle_alignment.py
```
3.Outputs will be saved as image files (e.g., output_image.png, aligned_rectangle_1.png, etc.) and displayed.

---
### Functions Overview
1.Rectangle and Line Detection with Numbering
a.max_distance(points): Calculates the maximum distance between any two points in a contour.

b.find_rectangles(contours): Finds contours approximated as rectangles.

c.find_and_number_lines_inside(contours, hierarchy, rects): Finds lines inside rectangles and assigns numbers based on length.

d.draw_lines_and_numbers(img, contours, lines_inside, line_nums): Draws line contours and numbering on the image.

2.Rectangle Alignment and Extraction
a.order_points(pts): Orders four points consistently for perspective transform.
b.four_point_transform(image, pts): Straightens the rectangle using perspective transform.

---
### Approach / Methodology
#### Rectangle and Line Detection with Numbering
Initially attempted to extract all contours with parents but found it ineffective.
Improved by selecting contours with parents but no children to isolate lines inside shapes, though this included lines inside any shape, not only rectangles.
Final approach detects rectangles explicitly and checks if lines lie strictly inside these rectangles via bounding box containment — accurately filtering the target lines.

#### Rectangle Alignment and Extraction
Converted images to grayscale and applied thresholding to separate shapes from background.
Used contour detection to find shapes, approximating polygons and filtering for quadrilaterals (4-point contours) as candidate rectangles.
Applied perspective transform to straighten each detected rectangle, correcting any tilt or rotation.
Checked rectangle dimensions and rotated those with width greater than height by 90 degrees to ensure vertical (portrait) orientation.
Extracted and saved each aligned rectangle as an individual image for further processing or analysis.

