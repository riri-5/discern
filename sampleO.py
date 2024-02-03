import numpy as np
import cv2

# Define object-specific variables
dist = 0
focal = 450
pixels = 56.54
width = 2.8

# Basic constants for OpenCV functions
kernel = np.ones((3, 3), 'uint8')
font = cv2.FONT_HERSHEY_SIMPLEX
org = (0, 20)
fontScale = 0.6
color = (0, 0, 255)
thickness = 2

# Function to find the distance from the camera
def get_dist(rectangle_params, image):
    # Unpack the tuple with two values
    (center, dimensions, angle) = rectangle_params
    # Find the number of pixels covered
    pixels = dimensions[0]
    print(pixels)
    # Calculate distance
    dist = (width * focal) / pixels
    
    # Write on the image
    image = cv2.putText(image, 'Distance from Camera in CM:', org, font,  
       1, color, 2, cv2.LINE_AA)

    image = cv2.putText(image, f'{dist:.2f} cm', (110, 50), font,  
       fontScale, color, 1, cv2.LINE_AA)

    return image

# Extract Frames 
cap = cv2.VideoCapture(0)

cv2.namedWindow('Object Dist Measure', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Dist Measure', 700, 600)

# Loop to capture video frames
while True:
    ret, img = cap.read()

    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Predefined mask for blue color detection
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

    # Remove Extra garbage from the image
    d_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)

    # Find the histogram
    contours, _ = cv2.findContours(d_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    for cnt in contours:
        # Check for contour area
        if 100 < cv2.contourArea(cnt) < 306000:

            # Draw a rectangle on the contour
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect) 
            box = np.int0(box)
            cv2.drawContours(img, [box], -1, (255, 0, 0), 3)
            
            img = get_dist(rect, img)

    cv2.imshow('Object Dist Measure', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
