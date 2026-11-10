import cv2
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue

# Load image function
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.add(image, image)  # Brighten the image by adding it to itself
    return image

# Display the results using Matplotlib
def display_results(original, clahe_img, binary, final):
    fig = plt.figure(figsize=(12, 12))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(clahe_img, cmap="gray")
    ax2.set_title("CLAHE Image")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(binary, cmap="gray")
    ax3.set_title("Binary Image")
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    ax4.set_title("Final Image with Path")
    ax4.axis("off")

    plt.show()

# Click event to handle the mouse clicks
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        if binary[y, x] == 0:
            points.append((x, y))
            cv2.circle(original_display, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow("Original Image", original_display)
            if len(points) == 2:
                cv2.destroyAllWindows()

# Heuristic function for the A* algorithm
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# A* search algorithm
def a_star_search(start, goal, binary):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = PriorityQueue()
    oheap.put((fscore[start], start))

    while not oheap.empty():
        current = oheap.get()[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + 1
            if (
                0 <= neighbor[0] < binary.shape[1]
                and 0 <= neighbor[1] < binary.shape[0]
            ):
                if binary[neighbor[1]][neighbor[0]] == 255:
                    continue
            else:
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [
                i[1] for i in oheap.queue
            ]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                oheap.put((fscore[neighbor], neighbor))

    return False

# Load the image
img = load_image("3.png")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
clahe_img = clahe.apply(gray_img)

# Apply thresholding (Otsu's method + Binary Inversion)
_, binary = cv2.threshold(clahe_img, 110, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

# Apply morphological operations to enhance the binary image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# Prepare for pathfinding and interaction
result_image = img.copy()
points = []
original_display = img.copy()
binary_display = binary.copy()

# Display the original image for user interaction
cv2.imshow("Original Image", original_display)
cv2.setMouseCallback("Original Image", click_event)
cv2.waitKey(0)

# Once two points are selected, find the path using A* algorithm
if len(points) == 2:
    start, goal = points
    path = a_star_search(start, goal, binary)
    
    if path:
        for point in path:
            cv2.circle(result_image, point, 1, (0, 0, 255), -1)
        cv2.imshow("Path", result_image)
        cv2.waitKey(0)
        
        # Display results: original, CLAHE, binary, and final path image
        display_results(original_display, clahe_img, binary, result_image)
    else:
        print("No path found")
else:
    print("Please click on two points")