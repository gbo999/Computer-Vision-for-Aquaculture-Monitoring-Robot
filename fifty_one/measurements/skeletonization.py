import cv2
import numpy as np
from skimage.morphology import skeletonize
import networkx as nx

def load_multiple_coords_from_txt(txt_file):
    """
    Loads multiple (x, y) coordinates sets from a .txt file.
    Each line in the file contains space-separated coordinates: x1 y1 x2 y2 ... xn yn.
    Each line represents a different segmentation.
    
    Returns:
        List of segmentations, each as a list of (y, x) tuples.
    """
    segmentations = []
    with open(txt_file, 'r') as file:
        for line in file:
            coords = list(map(float, line.strip().split()))  # Read all floats from the line
            # Convert into (y, x) pairs
            polygon = [(coords[i+1], coords[i]) for i in range(0, len(coords), 2)]
            segmentations.append(polygon)
    return segmentations

def create_filled_binary_mask(coords, img_height, img_width):
    """
    Creates a binary mask with a filled polygon from a single segmentation's (y, x) coordinates.
    """
    binary_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    polygon = np.array(coords, np.int32)  # Convert to NumPy array of int32
    polygon = polygon.reshape((-1, 1, 2))  # Reshape for OpenCV fillPoly
    cv2.fillPoly(binary_mask, [polygon], color=1) 

    #smooth the mask
    kernel = np.ones((5,5),np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    
    # Fill the polygon on the binary mask
    return binary_mask

def skeletonize_mask(binary_mask):
    """
    Skeletonizes the binary mask using skimage's skeletonize function.
    """
    skeleton = skeletonize(binary_mask)
    # cv2.imwrite(f'skeleton.png', skeleton.astype(np.uint8) * 255)

    return skeleton

def draw_skeleton(image, skeleton_coords, color=(0, 255, 0), thickness=1):
    """
    Draws the skeleton on the image.
    skeleton_coords: List of (y, x) points representing the skeleton.
    color: The color to draw the skeleton (in BGR format).
    thickness: The thickness of the lines representing the skeleton.
    """
    for i in range(len(skeleton_coords) - 1):
        start_point = (skeleton_coords[i][1], skeleton_coords[i][0])  # (x, y)
        end_point = (skeleton_coords[i+1][1], skeleton_coords[i+1][0])  # (x, y)
        cv2.line(image, start_point, end_point, color=color, thickness=thickness)
    return image

def draw_longest_path(image, longest_path, color=(0, 0, 255), thickness=2):
    """
    Draws the longest path on the image in a different color.
    longest_path: List of (y, x) points representing the longest path.
    color: The color to draw the longest path (in BGR format).
    thickness: The thickness of the lines representing the longest path.
    """
    for i in range(len(longest_path) - 1):
        start_point = (longest_path[i][1], longest_path[i][0])  # (x, y)
        end_point = (longest_path[i+1][1], longest_path[i+1][0])  # (x, y)
        cv2.line(image, start_point, end_point, color=color, thickness=thickness)
    return image

def ensure_integer_coords(coords):
    """
    Ensures that all coordinates are integers.
    """
    return [(int(round(y)), int(round(x))) for y, x in coords]

def scale_path(coords, original_size, new_size):
    """
    Scales a list of coordinates from the original image size to the new image size.
    
    coords: List of (y, x) coordinates.
    original_size: (height, width) of the original image.
    new_size: (height, width) of the new image.
    
    Returns:
        List of scaled (y, x) coordinates as floats.
    """
    orig_height, orig_width = original_size
    new_height, new_width = new_size
    y_scale = new_height / orig_height
    x_scale = new_width / orig_width

    scaled_coords = [(y * y_scale, x * x_scale) for y, x in coords]
    return scaled_coords

def normalize_coords(coords, img_size):
    """
    Normalizes the coordinates to be in the range [0, 1] relative to the image size.
    img_size: (height, width) of the image.
    
    Returns:
        List of normalized (y_norm, x_norm) coordinates.
    """
    img_height, img_width = img_size
    normalized_coords = [(y / img_height, x / img_width) for y, x in coords]
    return normalized_coords

def calculate_path_length(coords):
    """
    Calculates the Euclidean distance between consecutive points in a path.
    coords: List of (y, x) coordinates representing the path.
    """
    # length = 0
    # for i in range(len(coords) - 1):
    #     dy = coords[i+1][0] - coords[i][0]
    #     dx = coords[i+1][1] - coords[i][1]
    #     length += np.sqrt(dy ** 2 + dx ** 2)
    length = 0
    if len(coords) > 1:
        dy = coords[-1][0] - coords[0][0]
        dx = coords[-1][1] - coords[0][1]
        length = np.sqrt(dy ** 2 + dx ** 2)    
    return length

def find_longest_path(skeleton_coords, original_size, new_size):
    """
    Finds the longest path in the skeleton using graph analysis, and returns:
    - The longest path in normalized coordinates relative to the new image size.
    - The max length of the path in pixels in the new image size.
    
    skeleton_coords: List of (y, x) points representing the skeleton.
    original_size: (height, width) of the original image (e.g., 640x640).
    new_size: (height, width) of the new image size.
    """
    # Ensure coordinates are integers
    original_scaled_coords = ensure_integer_coords(skeleton_coords)
    
    # Remove duplicates
    original_scaled_coords = list(set(original_scaled_coords))
    
    # Create a graph using the original coordinates
    G = nx.Graph()
    for y, x in original_scaled_coords:
        G.add_node((y, x))  # Add node at each skeleton point

    # Add edges based on 8-connectivity
    for y, x in original_scaled_coords:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue  # Skip the current point itself
                neighbor = (y + dy, x + dx)
                if neighbor in G.nodes:
                    G.add_edge((y, x), neighbor)

    # Check if graph has any edges
    if G.number_of_edges() == 0:
        print("No edges found in the graph. Check skeleton coordinates.")
        return [], 0

    # Find the largest connected component
    connected_components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    largest_component = max(connected_components, key=len)
    endpoints = [node for node, degree in dict(largest_component.degree()).items() if degree == 1]

    if len(endpoints) < 2:
        print("Not enough endpoints to determine a path.")
        return [], 0

    max_length = 0
    longest_path = []

    # Find the longest path by checking endpoints
    for source in endpoints:
        lengths = nx.single_source_shortest_path_length(largest_component, source)
        farthest_node = max(lengths, key=lengths.get)
        length = lengths[farthest_node]
        if length > max_length:
            max_length = length
            longest_path = nx.shortest_path(largest_component, source, farthest_node)

    if not longest_path:
        print("No path found in the largest component.")
        return [], 0

    # Scale the longest path to the new image size
    scaled_longest_path = scale_path(longest_path, original_size, new_size)


        
    # Calculate the length of the longest path in pixel units for the new image size
    path_length_in_pixels = calculate_path_length(scaled_longest_path)

    # Normalize the longest path coordinates to [0, 1] based on the new image size
    normalized_longest_path = normalize_coords(scaled_longest_path, new_size)







    return normalized_longest_path, path_length_in_pixels

