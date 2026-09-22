## Detect-the-lines-using-Hough-Transform
Lane Detection


##  Aim
To implement a basic lane detection pipeline using OpenCV by completing missing code segments at specified locations.

## Learning Objective
Understand each stage of image processing Learn how to build a complete computer vision pipeline Practice writing code in guided sections Important Instruction: 👉 Write code ONLY in places marked as # Your Code Here 👉 Do NOT modify any other part of the code

## Software Used
Anaconda – Python 3.7 Jupyter Notebook / VS Code OpenCV (cv2) NumPy Matplotlib

## Algorithm & Explanation
```
Step 1: Import Libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
Step 2: Read the Image
image = cv2.imread("road.png")

if image is None:
    raise FileNotFoundError("Could not load 'road.png'. Check the file path.")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB
Step 3: Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

---

###  Step 4: Display Images

plt.figure(figsize=(10,5))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1) plt.imshow(image_rgb) plt.title("Original Image") plt.axis("off")

plt.subplot(1, 2, 2) plt.imshow(gray, cmap="gray") plt.title("Grayscale Image") plt.axis("off")

plt.tight_layout() plt.show()

Step 5: Thresholding
threshold = 150
_, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(6, 6))
plt.imshow(thresh, cmap="gray")
plt.title("Thresholded Image")
plt.axis("off")
plt.show()

---

###  Step 6: Region of Interest (ROI)


height, width = thresh.shape

roi_vertices = np.array([[
    (int(0.1 * width), height),
    (int(0.45 * width), int(0.6 * height)),
    (int(0.55 * width), int(0.6 * height)),
    (int(0.9 * width), height)
]], dtype=np.int32)

mask = np.zeros_like(thresh)
cv2.fillPoly(mask, roi_vertices, 255)
roi_masked = cv2.bitwise_and(thresh, mask)

plt.figure(figsize=(6, 6))
plt.imshow(roi_masked, cmap="gray")
plt.title("ROI Masked Image")
plt.axis("off")
plt.show()

---

### Step 7: Edge Detection (Canny)


edges = cv2.Canny(roi_masked, 50, 150)

plt.figure(figsize=(6, 6))
plt.imshow(edges, cmap="gray")
plt.title("Edge Detected Image")
plt.axis("off")
plt.show()

---

###  Step 8: Gaussian Blur


smoothed = cv2.GaussianBlur(edges, (5, 5), 0)

plt.figure(figsize=(6, 6))
plt.imshow(smoothed, cmap="gray")
plt.title("Smoothed (Blurred) Edge Image")
plt.axis("off")
plt.show()

---

###  Step 9: Hough Transform


# Detect lines using Hough Transform

lines = cv2.HoughLinesP(
    smoothed,
    rho=2,
    theta=np.pi / 180,
    threshold=50,
    minLineLength=40,
    maxLineGap=100
)

line_image = np.zeros_like(image)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line.flatten()
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

line_image_rgb = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 6))
plt.imshow(line_image_rgb)
plt.title("Detected Lines")
plt.axis("off")
plt.show()

---

### Step 10: Lane Detection Logic



final_output = cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)
final_output_rgb = cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 6))
plt.imshow(final_output_rgb)
plt.title("Final Lane Detection Output")
plt.axis("off")
plt.show()
```
## Expected Output
![Uploading image.png…]()

## Result
Thus, the lane detection pipeline is successfully implemented by completing the missing code sections. The system detects and highlights lane lines effectively.
