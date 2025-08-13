import cv2
import matplotlib.pyplot as plt
from math import hypot
from itertools import combinations
import os
import numpy as np

# --- helper functions ---
def max_distance(points):
    pts = points.reshape(-1, 2)
    max_d = 0
    for (x1, y1), (x2, y2) in combinations(pts, 2):
        dist = hypot(x2 - x1, y2 - y1)
        if dist > max_d:
            max_d = dist
    return max_d

def angle(pt1, pt2, pt0):
    dx1 = pt1[0] - pt0[0]
    dy1 = pt1[1] - pt0[1]
    dx2 = pt2[0] - pt0[0]
    dy2 = pt2[1] - pt0[1]
    dot = dx1 * dx2 + dy1 * dy2
    mag1 = np.sqrt(dx1*dx1 + dy1*dy1)
    mag2 = np.sqrt(dx2*dx2 + dy2*dy2)
    if mag1 * mag2 == 0:
        return 0
    cos_angle = dot / (mag1 * mag2)
    cos_angle = max(min(cos_angle, 1), -1)
    ang = np.arccos(cos_angle) * 180 / np.pi
    return ang

def is_rectangle(approx, max_deviation=10):
    if len(approx) != 4:
        return False
    for i in range(4):
        pt0 = approx[i][0]
        pt1 = approx[(i + 1) % 4][0]
        pt2 = approx[(i + 3) % 4][0]
        ang = angle(pt1, pt2, pt0)
        if abs(ang - 90) > max_deviation:
            return False
    return True

def find_rectangles(contours):
    rects_found = []
    for i, cnt in enumerate(contours):
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and is_rectangle(approx):
            x, y, w, h = cv2.boundingRect(cnt)
            rects_found.append((x, y, w, h, i))
    return rects_found


import numpy as np
import cv2

import numpy as np
import cv2


def find_and_number_lines_inside(contours, hierarchy, rects):
    lines_inside = []
    for i, cnt in enumerate(contours):
        parent_idx = hierarchy[0][i][3]
        first_child = hierarchy[0][i][2]
        if parent_idx != -1 and first_child == -1:
            lx, ly, lw, lh = cv2.boundingRect(cnt)
            for rx, ry, rw, rh, ridx in rects:
                inside_rect = rx < lx and ry < ly and (lx + lw) < (rx + rw) and (ly + lh) < (ry + rh)
                if inside_rect:
                    dist = max_distance(cnt)
                    lines_inside.append((dist, i))
                    break
    lines_inside.sort(key=lambda x: x[0])
    line_nums = {idx: n+1 for n, (_, idx) in enumerate(lines_inside)}
    return lines_inside, line_nums

def draw_lines_and_numbers_below(img, contours, lines_inside, line_nums, rects):
    for rx, ry, rw, rh, ridx in rects:
        inside_lines = [idx for _, idx in lines_inside if 
                        rx < cv2.boundingRect(contours[idx])[0] and
                        ry < cv2.boundingRect(contours[idx])[1] and
                        (cv2.boundingRect(contours[idx])[0] + cv2.boundingRect(contours[idx])[2]) < (rx + rw) and
                        (cv2.boundingRect(contours[idx])[1] + cv2.boundingRect(contours[idx])[3]) < (ry + rh)]
        for idx in inside_lines:
            cv2.drawContours(img, [contours[idx]], -1, (255, 0, 0), 2)
            text = str(line_nums[idx])
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = rx + (rw - text_size[0]) // 2  
            text_y = ry + rh + text_size[1] + 5      
            cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)


img = cv2.imread('image.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
output = img.copy()

rects = find_rectangles(contours)
lines_inside, line_nums = find_and_number_lines_inside(contours, hierarchy, rects)
draw_lines_and_numbers_below(output, contours, lines_inside, line_nums, rects)


plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

output_dir = 'task1-output'
os.makedirs(output_dir, exist_ok=True)
cv2.imwrite(os.path.join(output_dir, 'output_image.png'), output)
