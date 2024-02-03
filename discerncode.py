import numpy as np
import cv2

# Define object-specific variables
dist = 0
focal = 450
pixels = 56.54
width = 2.8

# Basic constants for OpenCV functions
kern = np.ones((3, 3), 'uint8')
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
    # Calculate distance
    dist = (width * focal) / pixels
    
    # Create a black image for drawing text
    text_img = np.zeros_like(image)
    
    # Write on the black image with thicker lines to simulate bold
    text_img = cv2.putText(text_img, 'Distance from Camera in CM:', org, font,  
       1, color, 2, cv2.LINE_AA)

    text_img = cv2.putText(text_img, f'{dist:.2f} cm', (110, 50), font,  
       fontScale, color, 2, cv2.LINE_AA)

    # Add the text image to the main image
    image = cv2.addWeighted(image, 1, text_img, 1, 0)

    return image


def get_shape_contour(approx, vertices):
    num_vertices = len(approx)
    if num_vertices == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        return "Square" if 0.85 <= ar <= 1.15 else "Rectangle"
    elif num_vertices > 7:
        return "Circle"
    else:
        return "Unknown"

def main():
    webcam = cv2.VideoCapture(0)
    cv2.namedWindow('Object Dist Measure', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Object Dist Measure', 700, 600)

    while True:
        ret, frame = webcam.read()

        if not ret:
            break
        
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Predefined mask for blue color detection
        lower_blue = np.array([100, 100, 100])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

        # Remove Extra garbage from the image
        d_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=5)

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
                cv2.drawContours(frame, [box], -1, (255, 0, 0), 3)
                
                frame = get_dist(rect, frame)

        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Adjusted color ranges for red, green, and blue
        red_lower = np.array([0, 100, 100], np.uint8)
        red_upper = np.array([10, 255, 255], np.uint8)
        red_mask_1 = cv2.inRange(hsvFrame, red_lower, red_upper)

        red_lower = np.array([160, 100, 100], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)
        red_mask_2 = cv2.inRange(hsvFrame, red_lower, red_upper)

        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

        # Adjusted green and blue color ranges
        green_lower = np.array([40, 40, 40], np.uint8)
        green_upper = np.array([80, 255, 255], np.uint8)
        green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

        blue_lower = np.array([90, 40, 40], np.uint8)
        blue_upper = np.array([130, 255, 255], np.uint8)
        blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

        kernal = np.ones((5, 5), "uint8")

        red_mask = cv2.dilate(red_mask, kernal)
        res_red = cv2.bitwise_and(frame, frame, mask=red_mask)

        green_mask = cv2.dilate(green_mask, kernal)
        res_green = cv2.bitwise_and(frame, frame, mask=green_mask)

        blue_mask = cv2.dilate(blue_mask, kernal)
        res_blue = cv2.bitwise_and(frame, frame, mask=blue_mask)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            shape = get_shape_contour(approx, len(approx))
            size = cv2.contourArea(cnt)

            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.drawContours(frame, [approx], 0, (255, 255, 255), 2)  # Change color to white
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)  # Change color to white

            cv2.putText(frame, f"{shape} - Size: {size:.2f} ", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Color detection code
        contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 300:
                x, y, w, h = cv2.boundingRect(contour)
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                cv2.putText(frame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))

        contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 300:
                x, y, w, h = cv2.boundingRect(contour)
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(frame, "Green Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

        contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 300:
                x, y, w, h = cv2.boundingRect(contour)
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                cv2.putText(frame, "Blue Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))

        # Program Termination
        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
