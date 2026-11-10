import cv2
import numpy as np
from skimage.morphology import skeletonize # type: ignore
from scipy.ndimage import distance_transform_edt
import networkx as nx
import os

def process_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Skeletonize the binary image
    skeleton = skeletonize(binary)
    
    # Convert back to uint8
    skeleton = (skeleton * 255).astype(np.uint8)
    
    return img, skeleton

def create_graph(skeleton):
    # Create a graph from the skeleton
    G = nx.Graph()
    
    # Find all road pixels
    road_pixels = np.column_stack(np.where(skeleton > 0))
    
    # Add nodes to the graph
    for pixel in road_pixels:
        G.add_node(tuple(pixel))
    
    # Add edges to the graph
    for pixel in road_pixels:
        y, x = pixel
        neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1),
                     (y-1, x-1), (y-1, x+1), (y+1, x-1), (y+1, x+1)]
        for neighbor in neighbors:
            if neighbor in G.nodes():
                G.add_edge(tuple(pixel), neighbor, weight=1)
    
    return G

def find_shortest_path(G, start, end):
    try:
        path = nx.shortest_path(G, start, end, weight='weight')
        return path
    except nx.NetworkXNoPath:
        return None

def draw_path(img, path):
    if path is None:
        return img
    
    path_img = img.copy()
    for i in range(len(path) - 1):
        cv2.line(path_img, path[i][::-1], path[i+1][::-1], (0, 255, 0), 2)
    
    return path_img

def save_output(img, filename, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cv2.imwrite(os.path.join(output_folder, filename), img)

def main(image_path, start_point, end_point):
    # Process image
    original_img, skeleton = process_image(image_path)
    
    # Create graph
    G = create_graph(skeleton)
    
    # Find shortest path
    path = find_shortest_path(G, start_point, end_point)
    
    # Draw path on original image
    result_img = draw_path(original_img, path)
    
    # Save outputs
    output_folder = 'output'
    save_output(skeleton, 'skeleton.jpg', output_folder)
    save_output(result_img, 'shortest_path.jpg', output_folder)
    
    return result_img

if __name__ == "__main__":
    # This part will be called from the GUI script
    pass