import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to check if a polygon is rectangle
def angle_between(p1, p2, p3):
    """Calculate angle at p2 formed by points p1-p2-p3"""
    v1 = p1 - p2
    v2 = p3 - p2
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def is_rectangle(pts, angle_tolerance=10):
    """Check if 4 points form a rectangle"""
    pts = pts.reshape(4, 2)
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        p3 = pts[(i + 2) % 4]
        angle = angle_between(p1, p2, p3)
        if not (90 - angle_tolerance <= angle <= 90 + angle_tolerance):
            return False
    return True

# Load image
image = cv2.imread('image.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
output_dir = 'task2-output'
os.makedirs(output_dir, exist_ok=True)
rectangles = []

for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) == 4 and is_rectangle(approx):
        
        rect = cv2.minAreaRect(approx)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        W = int(rect[1][0])
        H = int(rect[1][1])
        center = (int(rect[0][0]), int(rect[0][1]))
        angle = rect[2]

        if W < H:
            angle += 90
            W, H = H, W  #
        

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        x, y = int(center[0] - W/2), int(center[1] - H/2)
        cropped = rotated[y:y+H, x:x+W]
        rectangles.append(cropped)

# Save or display rectangles
for i, rect_img in enumerate(rectangles):
    filename = f'rectangle_{i+1}.png'
    cv2.imwrite(os.path.join(output_dir, filename), rect_img)
    plt.figure()
    plt.title(f'Rectangle {i+1}')
    plt.imshow(cv2.cvtColor(rect_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

plt.show()
