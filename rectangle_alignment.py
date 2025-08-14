import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def calc_angle(a, b, c):
    vec1 = a - b
    vec2 = c - b
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)


def check_rectangle(quad_pts, tol=10):
    pts = quad_pts.reshape(4, 2)
    for i in range(4):
        angle = calc_angle(pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4])
        if not (90 - tol <= angle <= 90 + tol):
            return False
    return True

img_path = 'image.png'
img = cv2.imread(img_path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


_, bw = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
save_folder = 'task2-output'
os.makedirs(save_folder, exist_ok=True)

detected_rects = []

for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx_poly = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    
    if len(approx_poly) == 4 and check_rectangle(approx_poly):
        rect = cv2.minAreaRect(approx_poly)
        box_pts = cv2.boxPoints(rect)
        box_pts = np.int0(box_pts)
        
        w, h = int(rect[1][0]), int(rect[1][1])
        cx, cy = int(rect[0][0]), int(rect[0][1])
        ang = rect[2]
        
        if w < h:
            ang += 90
            w, h = h, w
        
        # Rotate and crop
        rot_mat = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
        rotated_img = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
        cropped = rotated_img[cy - h//2 : cy + h//2, cx - w//2 : cx + w//2]
        detected_rects.append(cropped)

for idx, rect_img in enumerate(detected_rects, start=1):
    out_file = os.path.join(save_folder, f'rect_{idx}.png')
    cv2.imwrite(out_file, rect_img)
    plt.figure()
    plt.imshow(cv2.cvtColor(rect_img, cv2.COLOR_BGR2RGB))
    plt.title(f'Rectangle {idx}')
    plt.axis('off')

plt.show()
