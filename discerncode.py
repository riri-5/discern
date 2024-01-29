import cv2
import numpy as np

def get_shape_contour(approx, vertices):
    num_vertices = len(approx)

    if num_vertices == 3:
        return "Triangle"
    elif num_vertices == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        return "Square" if 0.95 <= ar <= 1.05 else "Rectangle"
    elif num_vertices == 5:
        return "Pentagon"
    elif 6 <= num_vertices <= 8:
        return "Hexagon"
    elif num_vertices > 8:
        return "Circle"
    else:
        return "Unknown"

def detect_color(frame, cnt):
    mask = np.zeros_like(frame)
    cv2.drawContours(mask, [cnt], 0, (255, 255, 255), -1)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mean_color = cv2.mean(frame, mask=mask)[:3]
    return tuple(int(val) for val in mean_color)

def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            shape = get_shape_contour(approx, len(approx))
            color = detect_color(frame, cnt)
            size = cv2.contourArea(cnt)

            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, f"{shape} - {color} - Size: {size:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
