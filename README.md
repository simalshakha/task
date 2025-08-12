# Rectangle Detection and Processing Toolkit

This project contains two related scripts for detecting, processing, and analyzing rectangles and lines inside images:

1. **Rectangle and Line Detection with Numbering**  
   Detects rectangles and lines inside those rectangles, assigns numbers to lines based on length, and draws the numbering on the image.

2. **Rectangle Alignment and Extraction**  
   Detects rectangles, straightens (aligns) them via perspective transform, rotates them for vertical orientation, extracts, saves, and displays each rectangle separately.

---

## Description

### 1. Rectangle and Line Detection with Numbering

- Detects contours and filters those approximated as rectangles (4-sided polygons).  
- Identifies line contours inside these rectangles using contour hierarchy and bounding box checks.  
- Measures the maximum distance between points in each line to determine line length.  
- Sorts lines by length and assigns numbering (shorter lines get lower numbers).  
- Draws contours of lines and their assigned numbers on the original image.  
- Saves and displays the resulting image.

### 2. Rectangle Alignment and Extraction

- Converts the image to grayscale and thresholds it to create a binary image.  
- Finds contours and selects those with 4 points as rectangles.  
- Applies a perspective transform to straighten each detected rectangle.  
- Rotates rectangles to ensure a vertical (portrait) orientation.  
- Saves each aligned rectangle as an individual image.  
- Displays all aligned rectangles.

---

## Dependencies

- Python 3.x  
- OpenCV (`opencv-python`)  
- Matplotlib  

Install via:

```bash
pip install opencv-python matplotlib
