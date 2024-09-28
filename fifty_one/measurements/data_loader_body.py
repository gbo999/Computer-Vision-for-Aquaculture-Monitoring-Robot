from MEC import welzl
from utils import calculate_euclidean_distance
from utils import calculate_real_width, calculate_bbox_area
import os
import ast
from tqdm import tqdm
import fiftyone as fo
import pandas as pd
from MEC import Point
import fiftyone.core.labels as fol
import cv2
from skeletonization import skeletonize_mask, draw_skeleton, draw_longest_path, find_longest_path,create_filled_binary_mask
import numpy as np
import math
from skimage.morphology import thin
class ObjectLengthMeasurer:
    def __init__(self, image_width, image_height, horizontal_fov, vertical_fov, distance_mm):
        self.image_width = image_width
        self.image_height = image_height
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov
        self.distance_mm = distance_mm
        self.scale_x, self.scale_y = self.calculate_scaling_factors()
        self.to_scale_x=image_width/640
        self.to_scale_y=image_height/360



    def calculate_scaling_factors(self):
        fov_x_rad = math.radians(self.horizontal_fov)
        fov_y_rad = math.radians(self.vertical_fov)
        scale_x = (2 * self.distance_mm * math.tan(fov_x_rad / 2)) / self.image_width
        scale_y = (2 * self.distance_mm * math.tan(fov_y_rad / 2)) / self.image_height
        # print(f"Scale X: {scale_x}, Scale Y: {scale_y}")  # Debugging
        return scale_x, scale_y

    def normalize_angle(self, angle):
        if angle < -45:
            angle += 90
        normalized = abs(angle)
        # print(f"Original Angle: {angle}, Normalized Angle: {normalized}")  # Debugging
        return normalized

    def compute_length(self, predicted_box_length, angle_deg):
        angle_rad = math.radians(angle_deg)
        combined_scale = math.sqrt((self.scale_x * math.cos(angle_rad)) ** 2 + 
                               (self.scale_y * math.sin(angle_rad)) ** 2)
        length_mm = predicted_box_length * combined_scale
        # print(f"Predicted Length (pixels): {predicted_box_length}, Length (mm): {length_mm}")  # Debugging
        return length_mm

    def measure_object_length(self, rect):
        angle = rect[-1]
        normalized_angle = self.normalize_angle(angle)
        predicted_box_length = max(rect[1])*8.3
    
        length_mm = self.compute_length(predicted_box_length, normalized_angle)

        # if length_mm > 130 and length_mm < 250:
        #     print(f'angle: {angle}, normalized_angle: {normalized_angle}')
        #     print(f'scaling factors: {self.scale_x}, {self.scale_y}') 
        #     print(f"Object Length: {length_mm:.2f}" )  # Debugging


        return length_mm, normalized_angle


def load_data(filtered_data_path, metadata_path):
    filtered_df = pd.read_excel(filtered_data_path)
    metadata_df = pd.read_excel(metadata_path)
    return filtered_df, metadata_df

def create_dataset():
    dataset = fo.Dataset("prawn_full_body", overwrite=True)
    #clear the dataset
    

    return dataset


def process_segmentations(segmentation_path):
    """
    Process the segmentations from the TXT file, calculate the minimum enclosing circle for each prawn.
    """
    segmentations = []
    # skeletons=[]
    # hulls=[]
    # # skeletons_straight=[]
    # # skeletons_straight_2=[]
    # seg_closeds=[]
    # skeletons_2=[]
    # box_diagonal=[] 
    boxes=[]
    # masks=[]
    # convexs=[]
    # cont_pair=[]






    
    # Open the segmentation file and process each line
    with open(segmentation_path, 'r') as file:
        for line in file:
            coords = [float(x) for x in line.strip().split()]
            coords_array = np.array(coords).reshape(-1, 2)
            

            
            binary_mask_no = create_filled_binary_mask(coords_array, 360, 640, gaussian_blur=False) 
            
            if np.sum(binary_mask_no) == 0:   
                continue





            # binary_dilated = cv2.dilate(binary_mask_no, np.ones((15, 15), np.uint8), iterations=1)     


            # #contour dilated
            # contures_dil, _ = cv2.findContours(binary_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    


            # prawn_conture_dil = max(contures_dil, key=cv2.contourArea)

            # coords_contour_dil = np.column_stack(prawn_conture_dil).flatten()

            # normalized_coords_bin=[(coords_contour_dil[i]/640, coords_contour_dil[i+1]/360) for i in range(0, len(coords_contour_dil), 2)]  # Extract points (x, y)


            # masks.append(fo.Polyline(
            #     points=[normalized_coords_bin],
            #     closed=True,
            #     filled=False,
            # ))


            
            # thinned_2=skeletonize_mask(binary_mask_no)

            # # skeleton = skeletonize_mask(binary_mask)
            # skeleton_2 = thinned_2
            # skeleton_coords_2 = np.column_stack(np.nonzero(skeleton_2))
            # normalized_coords_2,max_length_2 = find_longest_path(skeleton_coords_2,(360,640),(2988,5312))

            # normalized_coords_2 = [(x, y) for y, x in normalized_coords_2]  # Convert to (y, x) format

            # #only the first and last points of the skeleton
            # normalized_coords_straight_2 = [normalized_coords_2[0], normalized_coords_2[-1]]  





             
            #convex hull diameter
            contures, _ = cv2.findContours(binary_mask_no, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            
            prawn_conture = max(contures, key=cv2.contourArea) 
            
        

            # Compute the minimum area rectangle enclosing the shrimp
            rect = cv2.minAreaRect(prawn_conture)
            box_points = cv2.boxPoints(rect)
            # box_points = np.int0(box_points)

            original_size = (640, 360)
            new_size = (5312, 2988)

            # Scaling factors
            scale_x = new_size[0] / original_size[0]  # 5312 / 640
            scale_y = new_size[1] / original_size[1]  # 2988 / 640

            # print(f"Scale X: {scale_x}, Scale Y: {scale_y}")  # Should both be 8.3


            # pair, max_length_cont=find_furthest_points(prawn_conture)
            # if len(pair) < 2:
            #     raise ValueError(f"Expected pair to contain at least two points, but got {len(pair)}")
                
            # normalized_pair= [(pair[0][0]/640, pair[0][1]/360), (pair[1][0]/640, pair[1][1]/360)]

            # cont_pair.append(fo.Polyline(
            #     points=[normalized_pair],
            #     closed=False,
            #     filled=False,
            #     max_length=max_length_cont
            # ))



            # scaled_pair= [(pair[0][0] * scale_x, pair[0][1] * scale_y), (pair[1][0] * scale_x, pair[1][1] * scale_y)]
               
            # max_length_cont=np.linalg.norm(np.array(scaled_pair[0])-np.array(scaled_pair[1]))




            box_points_scaled = np.array([(point[0] * scale_x, point[1] * scale_y) for point in box_points])

            rotated_rect_width = rect[1][0] * scale_x
            rotated_rect_height = rect[1][1] * scale_y

# The max length of the rectangle is the longer side
            max_length_box = max(rotated_rect_width, rotated_rect_height)

            # print(f"Rotated Rect Width: {rotated_rect_width}, Height: {rotated_rect_height}, Max Length Box: {max_length_box}")

        
            # Convert theta from degrees to radians for FiftyOne
            theta_radians = np.deg2rad(rect[2])
            # normalized_bounding_box = [(box_points[i][0]/640, box_points[i][1]/640) for i in range(0, len(box_points))] 
            
            # image_center_x = 640 / 2

            # xc_adjusted = rect[0][0] - image_center_x

            # Extract points (x, y) 
            box=fo.Polyline.from_rotated_box(
                xc=rect[0][0] ,
                yc=rect[0][1],
                w=rect[1][0],
                h=rect[1][1],
                theta =theta_radians,
                frame_size=(640, 360)

            )


            boxes.append(box)


            # hull_points = cv2.convexHull(prawn_conture, returnPoints=True)

            
#             points,furthest_pair, max_length_cont = find_furthest_points(prawn_conture)

# # Scaling ffurthest_pair, max_length_cont = find_furthest_points(prawn_contour)

#             # If you need to scale the points and distance
#             original_size = (640, 360)
#             new_size = (5312, 2988)

#             scale_x = new_size[0] / original_size[0]
#             scale_y = new_size[1] / original_size[1]

#             scaled_pair = [(p[0] * scale_x, p[1] * scale_y) for p in furthest_pair]


#             #eucledian distance between scaled pair
#             max_distance = np.linalg.norm(np.array(scaled_pair[0])-np.array(scaled_pair[1]))


#             # The result is max_distance (in pixels) in the 5312x2988 image


#             normalzied_points_hull = [(point[0]/640, point[1]/360) for point in points]  # Extract points (x, y)

#             hull=fo.Polyline(
#                 points=[[(point[0]/640, point[1]/360) for point in furthest_pair]],
#                 closed=False,
#                 filled=False,
#                 max_length=max_distance
#             )

           
#             hull_convex=fo.Polyline(
#                 points=[normalzied_points_hull],
#                 closed=False,
#                 filled=False,
#                 max_length=max_distance
#             )
               
               
#             convexs.append(hull_convex)





                    

        # skeleton_straight=fo.Polyline(
        #     points=[normalized_coords_straight],
        #     closed=False,
        #     filled=False,
        #     max_length=max_length
        # )
        # skeletons_straight.append(skeleton_straight)

            # skeleton_straight_2=fo.Polyline(
            #     points=[normalized_coords_straight_2],
            #     closed=False,
            #     filled=False,
            #     max_length=max_length_2,
                
            # )
            # skeletons_straight_2.append(skeleton_straight_2)




            # hulls.append(hull)

            # skeleton=fo.Polyline(
            #     points=[normalized_coords],
            #     closed=False,
            #     filled=False,
            #     max_length=max_length
            # )

            # skeletons.append(skeleton)
            
            # skeleton_2=fo.Polyline( 
            #     points=[normalized_coords_2],
            #     closed=False,
            #     filled=False,
            #     max_length=max_length_2)
            # skeletons_2.append(skeleton_2)
              # Convert the line to a list of floats
            normalzied_points = [(coords[i]/640, coords[i + 1]/360) for i in range(0, len(coords), 2)]  # Extract points (x, y)
            # points = [Point(x*5312, y*2988) for x, y in normalzied_points] 
            

            scaled_contour = []
            for point in prawn_conture:
                scaled_x = int(point[0][0] * scale_x)
                scaled_y = int(point[0][1] * scale_y)
                scaled_contour.append([[scaled_x, scaled_y]])

            scaled_contour = np.array(scaled_contour, dtype=np.int32)

             # Convert to Point objects    
            # Calculate the minimum enclosing circle (center and radius)
            center, radius = cv2.minEnclosingCircle(scaled_contour)



            diameter = radius * 2


            #center in 640x360
            center_640 = (center[0] / scale_x, center[1] / scale_y)
        

            segmentation = fo.Polyline(
                points=[normalzied_points],
                closed=True,
                filled=False,
                diameter=diameter,
                center=center,
                max_length_box=max_length_box,
                rect=rect,
                # max_length_cont=max_length_cont
            )

            #smooth segmentation  wirh closing
            # seg_closed=fo.Polyline(
            #     points=[normalized_coords_bin],
            #     closed=True,
            #     filled=False,
            #     max_length=max_length
            # )

            # seg_closeds.append(seg_closed)                


            segmentations.append(segmentation)

            
            # # Calculate diagonal of the rotated rectangle
            # rect_diagonal = np.linalg.norm(np.array(box_points_scaled[0]) - np.array(box_points_scaled[2]))

            # # Compare with the convex hull furthest distance
            # print(f"Rect Diagonal: {rect_diagonal}")
            # print(f"Convex Hull Max Distance: {max_distance}")
            # if rect_diagonal < max_distance:
            #     print("Unexpected: Rect diagonal is smaller than hull's furthest distance.")




                     # Store the segmentation information (center, radius, and diameter)

    return segmentations,boxes

# def calculate_minimum_enclosing_circle(points):
#     """
#     Calculate the minimum enclosing circle for a set of points using Welzl's algorithm.
#     Returns the center and radius of the circle.
#     """
#     mec = welzl(points)  
#     center=[]
#     center.append(mec.C.X)
#     center.append(mec.C.Y)
#     return center, mec.R
def find_closest_circle_center(prawn_bbox, segmentations):
    """
    Find the closest segmentation circle center to the bounding box (bx, by) using Euclidean distance.
    """
    prawn_point = (prawn_bbox[0], prawn_bbox[1])  # Top-left corner of bounding box (bx, by)
    min_distance = float('inf')
    closest_center = None

    seg=None    
    # Iterate over each segmentation and calculate the distance to the bounding box corner
    for segmentation in segmentations:

        center = segmentation['center']  
        # print(f'center {str(center)}')   
        
        # Get the circle center (cx, cy)
        distance = calculate_euclidean_distance(prawn_point, center)

        if distance < min_distance:
            min_distance = distance
            closest_center = center
            seg=segmentation
    return seg, min_distance
def process_detection_by_circle(segmentation, sample, filename, prawn_id, filtered_df):
    """
    Process the prawn detection based on the enclosing circle's diameter.
    Update the filtered dataframe with the real-world size of the prawn.
    """

    

    # Fetch height in mm and other metadata
    height_mm = sample['heigtht(mm)'] 

    #focal length based on pond type
    if sample.tags[0] == 'circular-small' or sample.tags[0] == 'circular-big':
        focal_length = 23.64
    else:
        focal_length = 24.72


    # focal_length = 24.22  # Camera focal length
    pixel_size = 0.00716844  # Pixel size in mm

    poly=segmentation[0]

    # fov=75
    # FOV_width=2*height_mm*math.tan(math.radians(fov/2))


    # # Get the diameter of the circle in pixels
    # predicted_diameter_pixels = poly['diameter']


    # predicted_skeleton_length=poly['max_length_skeleton']  

    # predicted_hull_length=poly['max_length_hull']

    
    rect=poly['rect']

    predicted_box_length_pixels=poly['max_length_box']


    ObjectLengthMeasurer_obj = ObjectLengthMeasurer(5312, 2988, 75.2, 46, height_mm)

    predicted_box_length_fov, angle_deg = ObjectLengthMeasurer_obj.measure_object_length(rect)



    # Calculate the real-world prawn size using the box
    real_length_mm_box_focal = calculate_real_width(focal_length, height_mm, predicted_box_length_pixels, pixel_size) 


    # hull_length_cm = calculate_real_width(focal_length, height_mm, predicted_hull_length, pixel_size)    

    # # Calculate the real-world prawn size using the enclosing circle's diameter
    # real_length_cm = calculate_real_width(focal_length, height_mm, predicted_diameter_pixels, pixel_size)

    # ske_length_cm = calculate_real_width(focal_length, height_mm, predicted_skeleton_length, pixel_size)    

    #add height to dtaframe
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Height_mm'] = height_mm    



    # hull_length_fov=FOV_width*predicted_hull_length/5312
    # diameter_length_fov=FOV_width*predicted_diameter_pixels/5312
    # skeleton_length_fov=FOV_width*predicted_skeleton_length/5312

    # box_length_fov=FOV_width*predicted_box_length/5312

    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_box_focal'] = real_length_mm_box_focal

    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_FOV_box'] = predicted_box_length_fov

    # filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Hull_Length_FOV'] = hull_length_fov
    # filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Skeleton_Length_FOV'] = skeleton_length_fov
    # filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Diameter_FOV'] = diameter_length_fov

    # #add to filtered dataframe the number of pixels in the hull and skeleton and diameter
    # filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Hull_Length_pixels'] = predicted_hull_length
    # filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Skeleton_Length_pixels'] = predicted_skeleton_length
    # filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Diameter_pixels'] = predicted_diameter_pixels
    
    #box pixels
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Box_Length_pixels'] = predicted_box_length_pixels


    #error percentage min of Length_1, Length_2, Length_3
    min_true_length=min(abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1'].values[0]),abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2'].values[0]),abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3'].values[0]))
    max_true_length=max(abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1'].values[0]),abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2'].values[0]),abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3'].values[0]))
    median_true_length= (abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1'].values[0])+abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2'].values[0])+abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3'].values[0])-min_true_length-max_true_length)

    #True length in pixels is length_1*scale_1/10

    length_1=abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1'].values[0])
    length_2=abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2'].values[0])
    length_3=abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3'].values[0])

    scale_1=abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Scale_1'].values[0])
    scale_2=abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Scale_2'].values[0])
    scale_3=abs(filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Scale_3'].values[0])


    max_number_pixels=max(length_1*scale_1/10,length_2*scale_2/10,length_3*scale_3/10)
    min_number_pixels=min(length_1*scale_1/10,length_2*scale_2/10,length_3*scale_3/10)
    median_number_pixels=(length_1*scale_1/10+length_2*scale_2/10+length_3*scale_3/10)-max_number_pixels-min_number_pixels


    #min percentage error box in pixels
    min_error_percentage_box_pixels = abs(predicted_box_length_pixels - min_number_pixels) / min_number_pixels * 100
    max_error_percentage_box_pixels = abs(predicted_box_length_pixels - max_number_pixels) / max_number_pixels * 100
    median_error_percentage_box_pixels = abs(predicted_box_length_pixels - median_number_pixels) / median_number_pixels * 100

    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Error_percentage_min_box_pixels'] = min_error_percentage_box_pixels
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Error_percentage_max_box_pixels'] = max_error_percentage_box_pixels
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Error_percentage_median_box_pixels'] = median_error_percentage_box_pixels

    #abs error box in pixels
    min_abs_error_box_pixels = abs(predicted_box_length_pixels - min_number_pixels)
    max_abs_error_box_pixels = abs(predicted_box_length_pixels - max_number_pixels)
    median_abs_error_box_pixels = abs(predicted_box_length_pixels - median_number_pixels)

    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Abs_error_min_box_pixels'] = min_abs_error_box_pixels
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Abs_error_max_box_pixels'] = max_abs_error_box_pixels
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Abs_error_median_box_pixels'] = median_abs_error_box_pixels


    min_error_percentage_box_fov = abs(predicted_box_length_fov - min_true_length) / min_true_length * 100
    max_error_percentage_box_fov = abs(predicted_box_length_fov - max_true_length) / max_true_length * 100
    median_error_percentage_box_fov = abs(predicted_box_length_fov - median_true_length) / median_true_length * 100

    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Error_percentage_min_box_fov'] = min_error_percentage_box_fov    
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Error_percentage_max_box_fov'] = max_error_percentage_box_fov
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Error_percentage_median_box_fov'] = median_error_percentage_box_fov

    #abs error
    min_abs_error_box_fov = abs(predicted_box_length_fov - min_true_length)
    max_abs_error_box_fov = abs(predicted_box_length_fov - max_true_length)
    median_abs_error_box_fov = abs(predicted_box_length_fov - median_true_length)

    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Abs_error_min_box_fov'] = min_abs_error_box_fov
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Abs_error_max_box_fov'] = max_abs_error_box_fov
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Abs_error_median_box_fov'] = median_abs_error_box_fov

    #min percentage error focal
    min_error_percentage_box_focal = abs(real_length_mm_box_focal - min_true_length) / min_true_length * 100
    max_error_percentage_box_focal = abs(real_length_mm_box_focal - max_true_length) / max_true_length * 100
    median_error_percentage_box_focal = abs(real_length_mm_box_focal - median_true_length) / median_true_length * 100

    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Error_percentage_min_box_focal'] = min_error_percentage_box_focal
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Error_percentage_max_box_focal'] = max_error_percentage_box_focal
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Error_percentage_median_box_focal'] = median_error_percentage_box_focal

    #abs error focal
    min_abs_error_box_focal = abs(real_length_mm_box_focal - min_true_length)
    max_abs_error_box_focal = abs(real_length_mm_box_focal - max_true_length)
    median_abs_error_box_focal = abs(real_length_mm_box_focal - median_true_length)

    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Abs_error_min_box_focal'] = min_abs_error_box_focal
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Abs_error_max_box_focal'] = max_abs_error_box_focal
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Abs_error_median_box_focal'] = median_abs_error_box_focal






    # filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'RealLength_Hull_focal'] = hull_length_cm

    # # Update the filtered dataframe with the calculated real length
    # filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'RealLength_MEC_focal'] = real_length_cm
    # filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'RealLength_Skeleton_focal'] = ske_length_cm

    #add pond type to the filtered dataframe
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'PondType'] = sample.tags[0]        

    #put height in mm in the filtered dataframe
    filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Height_mm'] = height_mm

    # Fetch the true length from the dataframe
    # true_length = filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Avg_Length'].values[0]
    # true_length = (filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_1'].values[0],filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_2'].values[0],filtered_df.loc[(filtered_df['Label'] == f'full body:{filename}') & (filtered_df['PrawnID'] == prawn_id), 'Length_3'].values[0]).max()
    # Calculate the larges  length from Length_1, Length_2, Length_3

    # true_length = max(lengths)
    # Compute error and create label for the closest detection

    # error_percentage_MEC_fov = abs(diameter_length_fov - true_length) / true_length * 100

    # error_percentage_hull_fov = abs(hull_length_fov - true_length) / true_length * 100

    # error_percentage_skeleton_fov = abs(skeleton_length_fov - true_length) / true_length * 100  

    # error_percentage = abs(real_length_cm - true_length) / true_length * 100

    # error_percentage_skeleton = abs(ske_length_cm - true_length) / true_length * 100    

    # error_percentage_hull = abs(hull_length_cm - true_length) / true_length * 100

    error_max_percentage_box_fov = abs(predicted_box_length_fov - max_true_length) / max_true_length * 100

    error_min_percentage_box_fov = abs(predicted_box_length_fov - min_true_length) / min_true_length * 100

    error_median_percentage_box_fov = abs(predicted_box_length_fov - median_true_length) / median_true_length * 100


    min_min=min(abs(error_max_percentage_box_fov), abs(error_min_percentage_box_fov),abs(error_median_percentage_box_fov))

    closest_detection_label = f'MIN{min_min:.2f}% max true length:{max_true_length:.2f}mm, error max percentage box: {error_max_percentage_box_fov:.2f}%,error min {error_min_percentage_box_fov:.2f}%, error median {error_median_percentage_box_fov:.2f}% , pred length: {predicted_box_length_fov:.2f}cm, '
    poly.label = closest_detection_label
    poly.attributes["prawn_id"] = fo.Attribute(value=prawn_id)
    # Attach information to the sample

    # Tagging the sample based on error percentage
    if min_min > 25:
        if "MPE_box>25" not in sample.tags:
            sample.tags.append("MPE_box>25")
    elif min_min > 15 and min_min <= 25:
        if "MPE_box>15-25" not in sample.tags:
            sample.tags.append("MPE_box>15-25")
    elif min_min > 10 and min_min <= 15:
        if "MPE_box>10-15" not in sample.tags:
            sample.tags.append("MPE_box>10-15")
    elif min_min > 5 and min_min <= 10:
        if "MPE_box>5-10" not in sample.tags:
            sample.tags.append("MPE_box>5-10")
    else:
        if "MPE_box<5" not in sample.tags:
            sample.tags.append("MPE_box<5")


    
   
def process_images(image_paths, prediction_folder_path, filtered_df, metadata_df, dataset, pond_tag):
    """
    Processes images by matching segmentation with bounding boxes and calculating prawn sizes.
    """
    for image_path in tqdm(image_paths):
        filename = os.path.splitext(os.path.basename(image_path))[0]


        core_name=filename.split('.')[0]
        # Construct the path to the prediction (segmentation) file
        prediction_txt_path = os.path.join(prediction_folder_path, f"{core_name}_segmentations.txt")
        if not os.path.exists(prediction_txt_path):
            print(f"No segmentation file found for {filename}")
            continue


                 
        # Parse the segmentations to get the minimum enclosing circles
        segmentations,boxes = process_segmentations(prediction_txt_path)

    # # relevant_part = str('_'.join(compatible_file_name)[0],str('_'.join(compatible_file_name)[2]))
    
    #     metadata_row = metadata_df[metadata_df['file_name_new'] == relevant_part]

    #     if not metadata_row.empty:
    #         metadata = metadata_row.iloc[0].to_dict() 
    #     for key, value in metadata.items():
    #         if key != 'file name':
    #             sample[key] = value
    #     else:
    #         print(f"No metadata found for {relevant_part}")
        
        if 'undistorted' in filename:
            filename = filename.replace('undistorted_', '')


        compatible_file_name= filename.split('_')[0:3]

        comp=compatible_file_name[2].split('-')[0]

        compatible_file_name[2]=comp

        print(f'compatible {compatible_file_name}')

        matching_rows = filtered_df[filtered_df['Label'].str.contains('_'.join(compatible_file_name))]

        # Save the modified image (with circles drawn)
       
        # if pond_tag == 'circular-small' or pond_tag == 'pond_1\carapace\right' or pond_tag == 'pond_1\carapace\car':
        # Load bounding boxes from the filtered data
        # matching_rows = filtered_df[filtered_df['Label'] == f'full body:{filename}']
        
        if matching_rows.empty:
            continue


        # Create a new sample for FiftyOne
        sample = fo.Sample(filepath=image_path)

        # Iterate over each bounding box in the filtered data
        sample["segmentations"] = fol.Polylines(polylines=segmentations)

        # sample["skeletons"] = fol.Polylines(polylines=skeletons)

        # sample["hulls"] = fol.Polylines(polylines=hulls)    

        # sample["skeletons_straight"] = fol.Polylines(polylines=skeletons_straight)

        # sample['seg_closeds']=fol.Polylines(polylines=seg_closeds)
        sample.tags.append(f"{pond_tag}")

        # sample['skeletons_no_smooth']=fol.Polylines(polylines=skeletons_2)

        # sample["skeletons_straight_no_smooth"] = fol.Polylines(polylines=skeletons_straight_2)

        sample['boxes']=fol.Polylines(polylines=boxes)

        # sample['masks']=fol.Polylines(polylines=masks)
        # Add the processed sample to the FiftyOne dataset
        add_metadata(sample, filename, filtered_df, metadata_df)
        dataset.add_sample(sample)

    # Save the updated dataframe with the calculated real lengths
    output_file_path = r'Updated_full_Filtered_Data_with_real_length.xlsx'
    filtered_df.to_excel(output_file_path, index=False)
    print("Processed and saved the updated filtered data.")


def add_metadata(sample, filename, filtered_df, metadata_df, swimmingdf=None):
    """
    Add metadata to a sample in FiftyOne.
    """


    if 'undistorted' in filename:
        filename = filename.replace('undistorted_', '')


    compatible_file_name= filename.split('_')[0:3]

    comp=compatible_file_name[2].split('-')[0]

    compatible_file_name[2]=comp

    print(f'compatible {compatible_file_name}')

    matching_rows = filtered_df[filtered_df['Label'].str.contains('_'.join(compatible_file_name))]

    filename=matching_rows['Label'].values[0].split(':')[1] 

    joined_string ='_'.join([compatible_file_name[0],compatible_file_name[2]])
    


    relevant_part =joined_string 
    

    # Filter matching rows from the filtered data
    # matching_rows = filtered_df[filtered_df['Label'] == f'full body:{filename}']


    # compatible_file_name= filename.split('_')[0:3]

    # comp=compatible_file_name[2].split('-')[0]

    # compatible_file_name[2]=comp


    # print(f'compatible {compatible_file_name}')

    # joined_string ='_'.join([compatible_file_name[0],compatible_file_name[2]])
    # relevant_part =joined_string
        # Extract relevant parts from the filename
    # parts = filename.split('_')
    # relevant_part = f"{parts[1][-3:]}_{parts[3].split('.')[0]}"
    
    # metadata_df['file name'] = metadata_df['file name'].str.strip()

    # Look for metadata based on the relevant part of the filename
    metadata_row = metadata_df[metadata_df['file_name_new'] == relevant_part]

    # Add the metadata to the sample
    if not metadata_row.empty:
        metadata = metadata_row.iloc[0].to_dict()
        for key, value in metadata.items():
            if key != 'file name':  # Skip the 'file name' column
                sample[key] = value
    else:
        print(f"No metadata found for {relevant_part}")
    
    # Add prawn detections using the segmentation data
    add_prawn_detections(sample, matching_rows, filtered_df, filename)
def add_prawn_detections(sample, matching_rows, filtered_df, filename):
    """
    Add prawn detections based on bounding box information and segmentations (minimum enclosing circles).
    """
    # true_detections = []
    # max_diagonal_line_1=[]
    # max_diagonal_line_2=[]

    # min_diagonal_line_1=[]
    # min_diagonal_line_2=[]

    diagonal_line_1_1=[]
    diagonal_line_1_2=[]

    diagonal_line_2_1=[]
    diagonal_line_2_2=[]

    diagonal_line_3_1=[]
    diagonal_line_3_2=[]



    



    for _, row in matching_rows.iterrows():
        prawn_id = row['PrawnID']

        prawn_pixel_lenght_1=row['Length_1']*row['Scale_1']/10
        prawn_pixel_lenght_2=row['Length_2']*row['Scale_2']/10
        prawn_pixel_lenght_3=row['Length_3']*row['Scale_3']/10


        bbox_1=ast.literal_eval(row['BoundingBox_1'])
        bbox_1= tuple(float(coord) for coord in bbox_1)

        bbox_2=ast.literal_eval(row['BoundingBox_2'])
        bbox_2= tuple(float(coord) for coord in bbox_2)

        bbox_3=ast.literal_eval(row['BoundingBox_3'])
        bbox_3= tuple(float(coord) for coord in bbox_3)


        normalized_bbox_1 = [
            bbox_1[0] / 5312, bbox_1[1] / 2988,
            bbox_1[2] / 5312, bbox_1[3] / 2988
        ]

        normalized_bbox_2 = [
            bbox_2[0] / 5312, bbox_2[1] / 2988,
            bbox_2[2] / 5312, bbox_2[3] / 2988
        ]

        normalized_bbox_3 = [
            bbox_3[0] / 5312, bbox_3[1] / 2988,
            bbox_3[2] / 5312, bbox_3[3] / 2988
        ]

        # Corners in normalized coordinates
        top_left_1 = [normalized_bbox_1[0], normalized_bbox_1[1]]
        top_right_1 = [normalized_bbox_1[0] + normalized_bbox_1[2], normalized_bbox_1[1]]
        bottom_left_1 = [normalized_bbox_1[0], normalized_bbox_1[1] + normalized_bbox_1[3]]
        bottom_right_1 = [normalized_bbox_1[0] + normalized_bbox_1[2], normalized_bbox_1[1] + normalized_bbox_1[3]]

        # Diagonals
        diagonal1_1 = [top_left_1, bottom_right_1]
        diagonal1_2 = [top_right_1, bottom_left_1]

        # Create polylines for diagonals
        diagonal1_polyline_1 = fo.Polyline(
            label="Diagonal 1",
            points=[diagonal1_1],
            closed=False,
            filled=False,
            line_color="blue",
            thickness=2,
            pixel_length=prawn_pixel_lenght_1
        )

        diagonal1_polyline_2 = fo.Polyline(
            label="Diagonal 2",
            points=[diagonal1_2],
            closed=False,
            filled=False,
            line_color="green",
            thickness=2,
            pixel_length=prawn_pixel_lenght_1
        )


        diagonal_line_1_1.append(diagonal1_polyline_1)
        diagonal_line_1_2.append(diagonal1_polyline_2)


        # Corners in normalized coordinates
        top_left_2 = [normalized_bbox_2[0], normalized_bbox_2[1]]
        top_right_2 = [normalized_bbox_2[0] + normalized_bbox_2[2], normalized_bbox_2[1]]
        bottom_left_2 = [normalized_bbox_2[0], normalized_bbox_2[1] + normalized_bbox_2[3]]
        bottom_right_2 = [normalized_bbox_2[0] + normalized_bbox_2[2], normalized_bbox_2[1] + normalized_bbox_2[3]]

        # Diagonals
        diagonal2_1 = [top_left_2, bottom_right_2]
        diagonal2_2 = [top_right_2, bottom_left_2]

        # Create polylines for diagonals
        diagonal2_polyline_1 = fo.Polyline(
            label="Diagonal 1",
            points=[diagonal2_1],
            closed=False,
            filled=False,
            line_color="blue",
            thickness=2,
            pixel_length=prawn_pixel_lenght_2
        )

        diagonal2_polyline_2 = fo.Polyline(
            label="Diagonal 2",
            points=[diagonal2_2],
            closed=False,
            filled=False,
            line_color="green",
            thickness=2,
            pixel_length=prawn_pixel_lenght_2
        )


        diagonal_line_2_1.append(diagonal2_polyline_1)
        diagonal_line_2_2.append(diagonal2_polyline_2)


        # Corners in normalized coordinates
        top_left_3 = [normalized_bbox_3[0], normalized_bbox_3[1]]
        top_right_3 = [normalized_bbox_3[0] + normalized_bbox_3[2], normalized_bbox_3[1]]
        bottom_left_3 = [normalized_bbox_3[0], normalized_bbox_3[1] + normalized_bbox_3[3]]
        bottom_right_3 = [normalized_bbox_3[0] + normalized_bbox_3[2], normalized_bbox_3[1] + normalized_bbox_3[3]]

        # Diagonals
        diagonal3_1 = [top_left_3, bottom_right_3]
        diagonal3_2 = [top_right_3, bottom_left_3]

        # Create polylines for diagonals
        diagonal3_polyline_1 = fo.Polyline(
            label="Diagonal 1",
            points=[diagonal3_1],
            closed=False,
            filled=False,
            line_color="blue",
            thickness=2,
            pixel_length=prawn_pixel_lenght_3
        )

        diagonal3_polyline_2 = fo.Polyline(
            label="Diagonal 2",
            points=[diagonal3_2],
            closed=False,
            filled=False,
            line_color="green",
            thickness=2,
            pixel_length=prawn_pixel_lenght_3
        )


        diagonal_line_3_1.append(diagonal3_polyline_1)
        diagonal_line_3_2.append(diagonal3_polyline_2)






        # # # bounding_boxes = []
        # # for bbox_key in ['BoundingBox_1', 'BoundingBox_2', 'BoundingBox_3']:
        # #     if pd.notna(row[bbox_key]):
        # #         bbox = ast.literal_eval(row[bbox_key])
        # #         bbox = tuple(float(coord) for coord in bbox) 
                
                
                
        #          # Convert bounding box to tuple of floats
        #         # bounding_boxes.append(bbox)
        
        # # if not bounding_boxes:
        # #     print(f"No bounding boxes found for prawn ID {prawn_id} in {filename}.")
        # #     continue

        # # # Select the largest bounding box based on area
        # # largest_bbox = max(bounding_boxes, key=calculate_bbox_area)

        # # Normalize the largest bounding box coordinates
        # prawn_max_normalized_bbox = [
        #     largest_bbox[0] / 5312, largest_bbox[1] / 2988,
        #     largest_bbox[2] / 5312, largest_bbox[3] / 2988
        # ]

        # x_min = prawn_max_normalized_bbox[0]
        # y_min = prawn_max_normalized_bbox[1]
        # width = prawn_max_normalized_bbox[2]
        # height = prawn_max_normalized_bbox[3]

        # # Corners in normalized coordinates
        # top_left = [x_min, y_min]
        # top_right = [x_min + width, y_min]
        # bottom_left = [x_min, y_min + height]
        # bottom_right = [x_min + width, y_min + height]

        # # Diagonals
        # diagonal1 = [top_left, bottom_right]
        # diagonal2 = [top_right, bottom_left]

        # # Create polylines for diagonals
        # diagonal1_polyline = fo.Polyline(
        #     label="Diagonal 1",
        #     points=[diagonal1],
        #     closed=False,
        #     filled=False,
        #     line_color="blue",
        #     thickness=2,
        #     #pixel length in 5312x2988 
        
        # )

        # diagonal2_polyline = fo.Polyline(
        #     label="Diagonal 2",
        #     points=[diagonal2],
        #     closed=False,
        #     filled=False,
        #     line_color="green",
        #     thickness=2,

        # )


        # max_diagonal_line_1.append(diagonal1_polyline)
        # max_diagonal_line_2.append(diagonal2_polyline)


        # #diagonal line 1  of prawn_noramalized_bbox as fo polyline

        # smallest_bbox = min(bounding_boxes, key=calculate_bbox_area)

        # # Normalize the smallest bounding box coordinates
        # prawn_min_normalized_bbox = [
        #     smallest_bbox[0] / 5312, smallest_bbox[1] / 2988,
        #     smallest_bbox[2] / 5312, smallest_bbox[3] / 2988
        # ]

        # x_min = prawn_min_normalized_bbox[0]
        # y_min = prawn_min_normalized_bbox[1]
        # width = prawn_min_normalized_bbox[2]
        # height = prawn_min_normalized_bbox[3]

        # # Corners in normalized coordinates
        # top_left = [x_min, y_min]
        # top_right = [x_min + width, y_min]
        # bottom_left = [x_min, y_min + height]
        # bottom_right = [x_min + width, y_min + height]

        # # Diagonals
        # diagonal1 = [top_left, bottom_right]
        # diagonal2 = [top_right, bottom_left]

        # # Create polylines for diagonals
        # diagonal1_polyline = fo.Polyline(
        #     label="Diagonal 1",
        #     points=[diagonal1],
        #     closed=False,
        #     filled=False,
        #     line_color="blue",
        #     thickness=2,
        # )

        # diagonal2_polyline = fo.Polyline(
        #     label="Diagonal 2",
        #     points=[diagonal2],
        #     closed=False,
        #     filled=False,
        #     line_color="green",
        #     thickness=2,
        # )


        # min_diagonal_line_1.append(diagonal1_polyline)
        # min_diagonal_line_2.append(diagonal2_polyline)



        #draw    
        # Add the largest bounding box as a polyline    



        # Convert bounding box to normalized coordinates
        

        # Add true prawn detection based on the bounding box
        # true_detections.append(fo.Detection(label="prawn_true", bounding_box=prawn_normalized_bbox))

    

        # Find the closest segmentation circle to the bounding box
        segmentation = find_closest_circle_center(bbox_1, sample["segmentations"]['polylines'])  # segmentations should be part of the sample

        # Process the prawn detection using the circle's diameter
        if segmentation is not None:
            process_detection_by_circle(segmentation, sample, filename, prawn_id, filtered_df)

    # Store true detections in the sample
    # sample["true_detections"] = fo.Detections(detections=true_detections)
    sample["diagonal_line_1_1"] =   fo.Polylines(polylines=diagonal_line_1_1)
    sample["diagonal_line_1_2"] =   fo.Polylines(polylines=diagonal_line_1_2)

    sample["diagonal_line_2_1"] =   fo.Polylines(polylines=diagonal_line_2_1)
    sample["diagonal_line_2_2"] =   fo.Polylines(polylines=diagonal_line_2_2)

    sample["diagonal_line_3_1"] =   fo.Polylines(polylines=diagonal_line_3_1)
    sample["diagonal_line_3_2"] =   fo.Polylines(polylines=diagonal_line_3_2)


    

def calculate_mean_bbox(bounding_boxes):
    """
    Calculate the mean bounding box from a list of bounding boxes.
    Each bounding box is in the format (x_min, y_min, x_max, y_max).
    """
    # Unzip the list of bounding boxes into separate lists for x_min, y_min, x_max, and y_max
    x_mins, y_mins, x_maxs, y_maxs = zip(*bounding_boxes)
    
    # Calculate the mean for each coordinate
    mean_bbox = (
        sum(x_mins) / len(x_mins),
        sum(y_mins) / len(y_mins),
        sum(x_maxs) / len(x_maxs),
        sum(y_maxs) / len(y_maxs)
    )
    
    return mean_bbox

# Convert pixel length to real-world length
# Use camera parameters and scaling as described earlier
