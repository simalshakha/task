import cv2
import matplotlib.pyplot as plt
from math import hypot
from itertools import combinations
import os
import numpy as np

# Quick function to get the longest distance between contour points
def max_distance(points):
    pts = points.reshape(-1, 2)
    max_d = 0
    for (x1, y1), (x2, y2) in combinations(pts, 2):
        d = hypot(x2 - x1, y2 - y1)
        if d > max_d:
            max_d = d
    return max_d

# Angle at pt0 between pt1 and pt2
def angle(pt1, pt2, pt0):
    dx1, dy1 = pt1[0] - pt0[0], pt1[1] - pt0[1]
    dx2, dy2 = pt2[0] - pt0[0], pt2[1] - pt0[1]
    dot = dx1*dx2 + dy1*dy2
    mag1 = np.hypot(dx1, dy1)
    mag2 = np.hypot(dx2, dy2)
    if mag1*mag2 == 0:
        return 0
    cos_a = max(min(dot / (mag1*mag2), 1), -1)
    return np.degrees(np.arccos(cos_a))

# Check if 4-point polygon is roughly rectangle
def is_rectangle(approx, tol=10):
    if len(approx) != 4:
        return False
    for i in range(4):
        pt0 = approx[i][0]
        pt1 = approx[(i+1)%4][0]
        pt2 = approx[(i+3)%4][0]
        ang = angle(pt1, pt2, pt0)
        if abs(ang - 90) > tol:
            return False
    return True

# Find all rectangles from contours
def find_rectangles(contours):
    rects = []
    for i, c in enumerate(contours):
        approx = cv2.approxPolyDP(c, 0.02*cv2.arcLength(c, True), True)
        if len(approx)==4 and is_rectangle(approx):
            x, y, w, h = cv2.boundingRect(c)
            rects.append((x, y, w, h, i))
    return rects

# Find lines inside rectangles and assign numbers
def find_and_number_lines(contours, hierarchy, rects):
    lines = []
    for i, c in enumerate(contours):
        parent = hierarchy[0][i][3]
        first_child = hierarchy[0][i][2]
        # Only consider contours with parent but no child
        if parent != -1 and first_child == -1:
            lx, ly, lw, lh = cv2.boundingRect(c)
            for rx, ry, rw, rh, ridx in rects:
                if rx<lx and ry<ly and (lx+lw)<(rx+rw) and (ly+lh)<(ry+rh):
                    dist = max_distance(c)
                    lines.append((dist, i))
                    break
    lines.sort(key=lambda x: x[0])
    line_nums = {idx: n+1 for n, (_, idx) in enumerate(lines)}
    return lines, line_nums

# Draw lines and numbers
def draw_lines_and_numbers(img, contours, lines, line_nums, rects):
    for rx, ry, rw, rh, ridx in rects:
        inside_lines = [idx for _, idx in lines if 
                        rx < cv2.boundingRect(contours[idx])[0] < rx+rw and
                        ry < cv2.boundingRect(contours[idx])[1] < ry+rh]
        for idx in inside_lines:
            cv2.drawContours(img, [contours[idx]], -1, (255,0,0), 2)
            txt = str(line_nums[idx])
            text_size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            tx = rx + (rw - text_size[0])//2
            ty = ry + rh + text_size[1] + 5
            cv2.putText(img, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
            
            #print(f"Line {txt} drawn inside rectangle at ({rx},{ry},{rw},{rh})")

# Main
img = cv2.imread('image.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#print(hierarchy)
output = img.copy()
rects = find_rectangles(contours)
lines, line_nums = find_and_number_lines(contours, hierarchy, rects)
draw_lines_and_numbers(output, contours, lines, line_nums, rects)

plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()


out_dir = 'task1-output'
os.makedirs(out_dir, exist_ok=True)
cv2.imwrite(os.path.join(out_dir, 'output_image.png'), output)
print("Saved output image to task1-output folder")
