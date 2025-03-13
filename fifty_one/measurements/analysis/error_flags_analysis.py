import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import fiftyone as fo
import os

def calculate_mape(estimated_lengths, true_lengths):
    """
Calculate Mean Absolute Percentage Error between estimated and true lengths.
Parameters:
-----------
estimated_lengths : array-like
    Estimated length measurements (Len_e)
true_lengths : array-like
    True length measurements (Len_t)
Returns:
--------
list
    List of individual absolute percentage errors for each measurement pair
"""
# Calculate individual absolute percentage errors
# Formula: |estimated - true| / estimated * 100
    absolute_percentage_errors = [abs(est - true) / est * 100 for est, true in zip(estimated_lengths, true_lengths)]

    return absolute_percentage_errors




def main():
    parser = argparse.ArgumentParser('flags_analysis')
    parser.add_argument('--type', type=str,choices=['carapace','body'], default='carapace')
    parser.add_argument('--weights_type', type=str,choices=['all','kalkar','car'], default='all')
    parser.add_argument('--error_size', type=str,choices=['min','median','mean','max'], default='mean')
    args = parser.parse_args()




    """
Error Flags Analysis Script
--------------------------
This script analyzes measurement errors in prawn length data and identifies potential error sources
through a series of flags. It visualizes relationships between error flags, error rates, and measurement 
characteristics using both Matplotlib/Seaborn and Plotly.
Key components:
1. Data loading and preprocessing
2. Error calculation and flagging
3. Statistical analysis of errors and flags
4. Visualization of findings through various plot types

"""

   

# ----- Data Loading and Preprocessing -----

# Load the dataset from Excel file
    data_path = f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/updated_filtered_data_with_lengths_{args.type}-{args.weights_type}.xlsx'




    df = pd.read_excel(data_path)

# Remove any rows with missing values to ensure clean analysis
    df = df.dropna()

# Standardize pond type names for consistency in analysis and visualization
    df['Pond_Type'] = df['Pond_Type'].replace({
    'car': 'square',          # Rename 'car' to 'square'
    'right': 'circle_female', # Rename 'right' to 'circle_female'
    'left': 'circle_male',    # Rename 'left' to 'circle_male'
})
    

    df['choice'] = f'{args.type}_{args.weights_type}_{args.error_size}'

    df['median_scale'] = df[['Scale_1', 'Scale_2', 'Scale_3']].median(axis=1)

    df['Length_fov(mm)'] = df['Length_fov(mm)']

# ----- Error Calculation -----
    df['annotation_length_1'] = df['Length_ground_truth_annotation_pixels'] / df['Scale_1'] * 10
    df['annotation_length_2'] = df['Length_ground_truth_annotation_pixels'] / df['Scale_2'] * 10
    df['annotation_length_3'] = df['Length_ground_truth_annotation_pixels'] / df['Scale_3'] * 10
    df['MPE_length1'] = abs(df['Length_1'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100
    df['MPE_length2'] = abs(df['Length_2'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100
    df['MPE_length3'] = abs(df['Length_3'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100
    df['MPE_annotation_length_1'] = abs(df['annotation_length_1'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100
    df['MPE_annotation_length_2'] = abs(df['annotation_length_2'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100
    df['MPE_annotation_length_3'] = abs(df['annotation_length_3'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100

    df['mae_length1'] = abs(df['Length_1']-df['Length_fov(mm)'])
    df['mae_length2'] = abs(df['Length_2']-df['Length_fov(mm)'])
    df['mae_length3'] = abs(df['Length_3']-df['Length_fov(mm)'])


    df['mae_with_sign_1'] =  df['Length_1']- df['Length_fov(mm)'] 
    df['mae_with_sign_2'] = df['Length_2']- df['Length_fov(mm)']
    df['mae_with_sign_3'] = df['Length_3']- df['Length_fov(mm)']   

    df['mpe_with_sign_1'] = (df['Length_1'] - df['Length_fov(mm)']) / df['Length_1'] * 100
    df['mpe_with_sign_2'] = (df['Length_2'] - df['Length_fov(mm)']) / df['Length_2'] * 100
    df['mpe_with_sign_3'] = (df['Length_3'] - df['Length_fov(mm)']) / df['Length_3'] * 100


    df['mae_with_sign'] = df[['mae_with_sign_1', 'mae_with_sign_2', 'mae_with_sign_3']].min(axis=1)
    df['mpe_with_sign'] = df[['mpe_with_sign_1', 'mpe_with_sign_2', 'mpe_with_sign_3']].min(axis=1)




    df['mae_annotation_length_1'] = abs(df['Length_fov(mm)'] - df['annotation_length_1'])
    df['mae_annotation_length_2'] = abs(df['Length_fov(mm)'] - df['annotation_length_2'])
    df['mae_annotation_length_3'] = abs(df['Length_fov(mm)'] - df['annotation_length_3'])
    
    df['mpe_with_annotation'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3', 'MPE_annotation_length_1', 'MPE_annotation_length_2', 'MPE_annotation_length_3']].min(axis=1)
    df['mae_with_annotation'] = df[['mae_length1', 'mae_length2', 'mae_length3', 'mae_annotation_length_1', 'mae_annotation_length_2', 'mae_annotation_length_3']].min(axis=1)
    
    
    if  args.error_size == 'min':    

   
    # Determine the minimum MPE across all three measurements for each row
    # This represents the best-case error for each measuremen
        #with annotation
        df[f'mpe_with_annotation'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3', 'MPE_annotation_length_1', 'MPE_annotation_length_2', 'MPE_annotation_length_3']].min(axis=1)
        df['mae_with_annotation'] = df[['mae_length1', 'mae_length2', 'mae_length3', 'mae_annotation_length_1', 'mae_annotation_length_2', 'mae_annotation_length_3']].min(axis=1)

        #without annotation
        df['mpe'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].min(axis=1)
        df['mae'] = df[['mae_length1', 'mae_length2', 'mae_length3']].min(axis=1)

        
        df['mpe_with_sign'] = df[['mpe_with_sign_1', 'mpe_with_sign_2', 'mpe_with_sign_3']].median(axis=1)
        df['mae_with_sign'] = df[['mae_with_sign_1', 'mae_with_sign_2', 'mae_with_sign_3']].median(axis=1)
        
        df['mpe_with_sign'] = df[['mpe_with_sign_1', 'mpe_with_sign_2', 'mpe_with_sign_3']].min(axis=1)
        df['mae_with_sign'] = df[['mae_with_sign_1', 'mae_with_sign_2', 'mae_with_sign_3']].min(axis=1)
        
        df['accounting_length'] = df.apply(
            lambda row: row['Length_1'] if row['mpe_with_sign_1'] == row['mpe'] else
                        (row['Length_2'] if row['mpe_with_sign_2'] == row['mpe'] else row['mpe_with_sign_3']),
            axis=1
)   
        df['mpe'] = df[['mpe_with_sign']]
        df['mae'] = df[['mae_with_sign']]


    elif args.error_size == 'median':

        df['median_annotation_length'] = df['Length_ground_truth_annotation_pixels'] / df['median_scale'] * 10

        df['MPE_median_annotation_length'] = abs(df['median_annotation_length'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100
        df['mae_median_annotation_length'] = abs(df['median_annotation_length'] - df['Length_fov(mm)'])
        df['mpe_with_annotation'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3', 'MPE_median_annotation_length']].median(axis=1)
        df['mae_with_annotation'] = df[['mae_length1', 'mae_length2', 'mae_length3', 'mae_median_annotation_length']].median(axis=1)

        #without annotation
        
        
        
        df['mpe_with_sign'] = df[['mpe_with_sign_1', 'mpe_with_sign_2', 'mpe_with_sign_3']].median(axis=1)
        df['mae_with_sign'] = df[['mae_with_sign_1', 'mae_with_sign_2', 'mae_with_sign_3']].median(axis=1)
        
        
        
        df['mpe'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].median(axis=1)
        df['mae'] = df[['mae_length1', 'mae_length2', 'mae_length3']].median(axis=1)

        df['mpe'] =df[['mpe_with_sign']]
        df['mae'] =df[['mae_with_sign']]
        
    
        df['accounting_length'] = df.apply(
            lambda row: row['Length_1'] if row['mpe_with_sign_1'] == row['mpe_with_sign'] else
                        (row['Length_2'] if row['mpe_with_sign_2'] == row['mpe_with_sign'] else row['Length_3']),
            axis=1
)

        df['accounting_length_pixels'] = df.apply(
            lambda row: row['Length_1_pixels'] if row['mpe_with_sign_1'] == row['mpe_with_sign'] else   
                        (row['Length_2_pixels'] if row['mpe_with_sign_2'] == row['mpe_with_sign'] else row['Length_3_pixels']),
            axis=1
)
        

        df['accounting_scale'] = df.apply(
            lambda row: row['Scale_1'] if row['mpe_with_sign_1'] == row['mpe_with_sign'] else
                        (row['Scale_2'] if row['mpe_with_sign_2'] == row['mpe_with_sign'] else row['Scale_3']),
            axis=1
)

    elif args.error_size == 'mean':
        df['mean_scale'] = df[['Scale_1', 'Scale_2', 'Scale_3']].mean(axis=1)
        df['mean_annotation_length'] = df['Length_ground_truth_annotation_pixels'] / df['mean_scale'] * 10
        df['MPE_mean_annotation_length'] = abs(df['mean_annotation_length'] - df['Length_fov(mm)']) / df['Length_fov(mm)'] * 100
        df['mae_mean_annotation_length'] = abs(df['mean_annotation_length'] - df['Length_fov(mm)'])
        df['mpe_with_annotation'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3', 'MPE_mean_annotation_length']].mean(axis=1)
        df['mae_with_annotation'] = df[['mae_length1', 'mae_length2', 'mae_length3', 'mae_mean_annotation_length']].mean(axis=1)

        #without annotation
        df['mpe'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].mean(axis=1)
        df['mae'] = df[['mae_length1', 'mae_length2', 'mae_length3']].mean(axis=1)


        
        df['mpe_with_sign'] = df[['mpe_with_sign_1', 'mpe_with_sign_2', 'mpe_with_sign_3']].mean(axis=1)
        df['mae_with_sign'] = df[['mae_with_sign_1', 'mae_with_sign_2', 'mae_with_sign_3']].mean(axis=1)
        
        df['mpe'] = df[['mpe_with_sign']]
        df['mae'] = df[['mae_with_sign']]   



    elif args.error_size == 'max':

        
        #without annotation
        df['mpe'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].max(axis=1)
        df['mae'] = df[['mae_length1', 'mae_length2', 'mae_length3']].max(axis=1)
        
        df['mpe_with_annotation'] = df[['MPE_length1', 'MPE_length2', 'MPE_length3', 'MPE_annotation_length_1', 'MPE_annotation_length_2', 'MPE_annotation_length_3']].max(axis=1)
        df['mae_with_annotation'] = df[['mae_length1', 'mae_length2', 'mae_length3', 'mae_annotation_length_1', 'mae_annotation_length_2', 'mae_annotation_length_3']].max(axis=1)

        df['mpe_with_sign'] = df[['mpe_with_sign_1', 'mpe_with_sign_2', 'mpe_with_sign_3']].max(axis=1)
        df['mae_with_sign'] = df[['mae_with_sign_1', 'mae_with_sign_2', 'mae_with_sign_3']].max(axis=1)

        df['accounting_length'] = df.apply(
            lambda row: row['Length_1'] if row['mpe_with_sign_1'] == row['mpe'] else
                        (row['Length_2'] if row['mpe_with_sign_2'] == row['mpe'] else row['mpe_with_sign_3']),
            axis=1
)
        
        df['mpe'] = df[['mpe_with_sign']]
        df['mae'] = df[['mae_with_sign']]




    #mean scale

    #make annotation_length




# ----- Best Length Determination -----

# Create a mask identifying which length measurement gave the minimum MPE
    min_mpe_mask = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].eq(df['mpe'], axis=0)

# Map column names to their corresponding indices for reference
    column_to_index = {
    'MPE_length1': 1,
    'MPE_length2': 2,
    'MPE_length3': 3
}

# Get the column name with minimum MPE for each row
    
    min_mpe_column = df[['MPE_length1', 'MPE_length2', 'MPE_length3']].idxmin(axis=1)

# Map to the correct index (1, 2, or 3) for each measurement
    min_mpe_index = min_mpe_column.map(column_to_index)

# Use the index to get the corresponding best length measurement for each row
# This selects the length with minimum error for further analysis
    df['best_length'] = df.apply(lambda row: row[f'Length_{min_mpe_index[row.name]}'], axis=1)

# Similarly, get the best length in pixels
    df['best_length_pixels'] = df.apply(lambda row: row[f'Length_{min_mpe_index[row.name]}_pixels'], axis=1)

# ----- Pixel Error Calculations -----

# Calculate prediction scale (pixels per mm)
    df['pred_scale'] = df['pred_Distance_pixels'] / df['Length_fov(mm)'] * 10
    print(f"pred_scale: {df['pred_scale']}")
# Normalize expert measurements to pixels for comparison with predictions
# This converts the best expert measurement to the same pixel scale as predictions
    df['expert_normalized_pixels'] = df.apply(
    lambda row: row['best_length_pixels'] * row['pred_scale'] / row[f'Scale_{min_mpe_index[row.name]}'],
    axis=1
)
    # df['flag_scale_error'] =(abs(df['pred_scale'] - df['Scale_1'])/df['Scale_1']*100 > 10) & (abs(df['pred_scale'] - df['Scale_2'])/df['Scale_2']*100 > 10 )& (abs(df['pred_scale'] - df['Scale_3'])/df['Scale_3']*100 > 10)
   
   
   
        #avg scale error
    df['scale_error_1'] = (df['pred_scale'] - df['Scale_1'])/df['Scale_1']*100
    df['scale_error_2'] = (df['pred_scale'] - df['Scale_2'])/df['Scale_2']*100
    df['scale_error_3'] = (df['pred_scale'] - df['Scale_3'])/df['Scale_3']*100

    df['avg_scale_error'] = df[['scale_error_1', 'scale_error_2', 'scale_error_3']].mean(axis=1)
    df['flag_avg_scale_error'] = abs(df['avg_scale_error'])  > 15
   
   
   
   
   
   
   
    # df['flag_scale_error'] = abs(df['pred_scale'] - df['accounting_scale'])/df['accounting_scale']*100 > 10
# Calculate minimum error in pixels (absolute difference between expert and prediction)
    df['min_error_pixels'] = abs(df['expert_normalized_pixels'] - df['pred_Distance_pixels'])

# Calculate minimum MAPE in pixels (percentage error in pixel space)
    df['min_mape_pixels'] = df['min_error_pixels'] / df['pred_Distance_pixels'] * 100

# ----- Flag High Errors -----

# Create a flag for measurements with errors > 10%
    df['high_error'] = abs(df['mpe']) >10  


# Calculate and display the percentage of measurements with high errors
    high_error_pct = df['high_error'].mean() * 100
    print(f"Percentage of measurements with errors > 10%: {high_error_pct:.1f}%")

# ----- Additional Pixel Differences -----

# Calculate absolute pixel differences between predicted and expert measurements
    df['pred_pixels_diff'] = abs(df['pred_Distance_pixels'] - df['expert_normalized_pixels'])

# Calculate absolute pixel differences between ground truth annotation and prediction
    df['pred_pixel_gt_diff'] = (df['Length_ground_truth_annotation_pixels'] - df['pred_Distance_pixels'])

# Calculate absolute pixel differences for each expert measurement
    df['pixel_diff_1'] = abs(df['pred_Distance_pixels'] - df['Length_1_pixels']) 
    df['pixel_diff_2'] = abs(df['pred_Distance_pixels'] - df['Length_2_pixels'])
    df['pixel_diff_3'] = abs(df['pred_Distance_pixels'] - df['Length_3_pixels'])

# Calculate percentage differences for each measurement
    df['pixel_diff_pct_1'] = df['pixel_diff_1'] / df['Length_1_pixels'] * 100
    df['pixel_diff_pct_2'] = df['pixel_diff_2'] / df['Length_2_pixels'] * 100
    df['pixel_diff_pct_3'] = df['pixel_diff_3'] / df['Length_3_pixels'] * 100

# ----- Flagging Pixel Errors -----

# Define threshold for high pixel percentage error
    pixel_pct_threshold = 15  # 10% threshold

    # df['flag_accounting_pixel_error'] = abs(df['pred_Distance_pixels'] - df['accounting_length_pixels'])/df['Length_ground_truth_annotation_pixels']*100 > pixel_pct_threshold
    # Flag when all three measurements exceed the threshold
# This indicates consistent high error across all expert measurements
    df['flag_all_high_pixel_error'] = (
    (df['pixel_diff_pct_1'] > pixel_pct_threshold) & 
    (df['pixel_diff_pct_2'] > pixel_pct_threshold) & 
    (df['pixel_diff_pct_3'] > pixel_pct_threshold)
)

# Flag when any measurement exceeds the threshold
# This captures cases where at least one measurement has high error
    df['flag_any_high_pixel_error'] = (
    (df['pixel_diff_pct_1'] > pixel_pct_threshold) | 
    (df['pixel_diff_pct_2'] > pixel_pct_threshold) | 
    (df['pixel_diff_pct_3'] > pixel_pct_threshold)
)

#flag all smaller than 10%
    df['flag_all_small_pixel_error'] = (
    (df['pixel_diff_pct_1'] < pixel_pct_threshold) & 
    (df['pixel_diff_pct_2'] < pixel_pct_threshold) & 
    (df['pixel_diff_pct_3'] < pixel_pct_threshold)
)
    df['pixel_pct'] = df[['pixel_diff_pct_1', 'pixel_diff_pct_2', 'pixel_diff_pct_3']].median(axis=1)
    if args.error_size == 'median':
        df['pixel_pct'] = df[['pixel_diff_pct_1', 'pixel_diff_pct_2', 'pixel_diff_pct_3']].median(axis=1)
    elif args.error_size == 'mean':
        df['pixel_pct'] = df[['pixel_diff_pct_1', 'pixel_diff_pct_2', 'pixel_diff_pct_3']].mean(axis=1)
    elif args.error_size == 'min':
        df['pixel_pct'] = df[['pixel_diff_pct_1', 'pixel_diff_pct_2', 'pixel_diff_pct_3']].min(axis=1)
    elif args.error_size == 'max':
        df['pixel_pct'] = df[['pixel_diff_pct_1', 'pixel_diff_pct_2', 'pixel_diff_pct_3']].max(axis=1)

# Calculate average pixel percentage error across all three measurements
    df['avg_pixel_error_pct'] = (df['pixel_diff_pct_1'] + df['pixel_diff_pct_2'] + df['pixel_diff_pct_3']) / 3

# Flag high average pixel error
    df['flag_high_avg_pixel_error'] = abs(df['avg_pixel_error_pct'])  > pixel_pct_threshold

# ----- GT-Expert Pixel Differences -----

# Calculate absolute pixel differences between ground truth and each expert measurement
    df['gt_expert_diff_1'] = (df['Length_ground_truth_annotation_pixels'] - df['Length_1_pixels'])
    df['gt_expert_diff_2'] = (df['Length_ground_truth_annotation_pixels'] - df['Length_2_pixels'])
    df['gt_expert_diff_3'] = (df['Length_ground_truth_annotation_pixels'] - df['Length_3_pixels'])

# Calculate percentage differences relative to expert measurements
    df['gt_expert_diff_pct_1'] = df['gt_expert_diff_1'] / df['Length_1_pixels'] * 100
    df['gt_expert_diff_pct_2'] = df['gt_expert_diff_2'] / df['Length_2_pixels'] * 100
    df['gt_expert_diff_pct_3'] = df['gt_expert_diff_3'] / df['Length_3_pixels'] * 100

    # df['flage_gt_expert_accounting_length_pixels'] = abs(df['Length_ground_truth_annotation_pixels'] - df['accounting_length_pixels'])/df['Length_ground_truth_annotation_pixels']*100 > 10



# ----- Flagging GT-Expert Errors -----

# Define threshold for high GT-Expert pixel percentage error
    gt_expert_pct_threshold = 15  # 10% threshold

# Flag when all three GT-Expert measurements exceed the threshold
    df['flag_all_high_gt_expert_error'] = (
    (df['gt_expert_diff_pct_1'] > gt_expert_pct_threshold) & 
    (df['gt_expert_diff_pct_2'] > gt_expert_pct_threshold) & 
    (df['gt_expert_diff_pct_3'] > gt_expert_pct_threshold)
)
    
    #all smaller than 10%
    df['flag_all_small_gt_expert_error'] = (
    (df['gt_expert_diff_pct_1'] < gt_expert_pct_threshold) & 
    (df['gt_expert_diff_pct_2'] < gt_expert_pct_threshold) & 
    (df['gt_expert_diff_pct_3'] < gt_expert_pct_threshold)
)

# Flag when any GT-Expert measurement exceeds the threshold
    df['flag_any_high_gt_expert_error'] = (
    (df['gt_expert_diff_pct_1'] > gt_expert_pct_threshold) | 
    (df['gt_expert_diff_pct_2'] > gt_expert_pct_threshold) | 
    (df['gt_expert_diff_pct_3'] > gt_expert_pct_threshold)
)

# Calculate average GT-Expert pixel percentage error
    df['avg_gt_expert_error_pct'] = (
    df['gt_expert_diff_pct_1'] + df['gt_expert_diff_pct_2'] + df['gt_expert_diff_pct_3']
) / 3

# Flag high average GT-Expert pixel error
    df['flag_high_avg_gt_expert_error'] = abs(df['avg_gt_expert_error_pct']) > gt_expert_pct_threshold

# ----- Potential Error Source Flags -----

# Define flags for potential error sources
# 1. High pixel difference between ground truth and expert (normalized)

# 2. High pixel difference between prediction and expert (percentage)

# 3. Low pose evaluation score (if available)
# Check which pose evaluation column exists in the dataset
    if 'pose_eval' in df.columns:
        df['flag_low_pose_eval'] = df['pose_eval'] < 0.75
    elif 'pose_eval_iou' in df.columns:
        df['flag_low_pose_eval'] = df['pose_eval_iou'] < 0.75
    else:
        df['flag_low_pose_eval'] = False
        print("Warning: No pose evaluation column found!")

# 4. High pixel difference between prediction and ground truth annotation
    df['flag_pred_gt_diff'] = abs(df['pred_pixel_gt_diff']/df['Length_ground_truth_annotation_pixels']*100) > 15

# ----- Flag Count and Multiple Error Images -----


# ----- Identify Images with All High Errors -----

# Get a complete list of unique image labels
    all_image_labels = df['Label'].unique()

# Count total measurements per image
    total_measurements_by_image = df.groupby('Label').size()

# Count high error measurements per image
    high_error_df = df[abs(df['mpe']) >15]  # Filter for high errors
    high_error_counts_by_image = high_error_df.groupby('Label').size()

# Reindex both Series to ensure they have the same index
# This is critical for correct comparison
    total_measurements_by_image = total_measurements_by_image.reindex(all_image_labels, fill_value=0)
    high_error_counts_by_image = high_error_counts_by_image.reindex(all_image_labels, fill_value=0)

# Find images where all measurements have high errors (and have more than 1 measurement)
    image_100_error_rate = high_error_counts_by_image[
    (high_error_counts_by_image == total_measurements_by_image) & 
    (total_measurements_by_image > 1)
].index.tolist()

# Flag measurements from these images
    df['flag_all_high_error_rate_image'] = df['Label'].isin(image_100_error_rate)

# Find images with multiple high errors
    images_with_multiple_high_errors = high_error_counts_by_image[high_error_counts_by_image > 1].index.tolist()

# Flag measurements from images with multiple high errors
    df['flag_image_multiple_errors'] = df['Label'].isin(images_with_multiple_high_errors)



# # ----- VISUALIZATION SECTION -----

# # ----- 1. Matplotlib/Seaborn Bar Chart -----

#     """
# This visualization creates a conditional bar chart comparing error rates
# when flags are present vs. absent. It helps understand the impact of
# each flag on error rates.
# """
# # Create visualization: Conditional bar chart showing percentage of high errors by flag
#     fig, ax = plt.subplots(figsize=(10, 6))  # Create figure and axes with specified size

# # For each flag, calculate percentage of high errors
#     flag_cols = ['flag_all_high_pixel_error', 'flag_all_high_gt_expert_error', 'flag_all_high_error_rate_image', 'flag_image_multiple_errors']
#     flag_labels = ['All High Pixel Error', 'All High GT-Expert Error', 'All High Error Rate Image', 'Multiple Errors in Image']
#     flag_true_error_pcts = []  # Will store error percentages when flag is True
#     flag_false_error_pcts = [] # Will store error percentages when flag is False

#     for flag in flag_cols:
#     # Calculate percentage of high errors when flag is True
#         if len(df[df[flag]]) > 0:
#             flag_true_error_pcts.append(df[df[flag]]['high_error'].mean() * 100)
#         else:
#             flag_true_error_pcts.append(0)
    
#     # Calculate percentage of high errors when flag is False
#         if len(df[~df[flag]]) > 0:
#             flag_false_error_pcts.append(df[~df[flag]]['high_error'].mean() * 100)
#         else:
#             flag_false_error_pcts.append(0)

# # Create bar chart with grouped bars
#     x = np.arange(len(flag_labels))  # Positions for the bars
#     width = 0.35  # Width of the bars

# # Create paired bars for each flag - one for when flag is present, one for when absent
#     ax.bar(x - width/2, flag_true_error_pcts, width, label='Flag Present', color='#e74c3c')
#     ax.bar(x + width/2, flag_false_error_pcts, width, label='Flag Absent', color='#2ecc71')

# # Configure the axes and labels
#     ax.set_xticks(x)  # Set tick positions
#     ax.set_xticklabels(flag_labels)  # Set tick labels
#     ax.set_ylabel('Percentage of Measurements with Error >10%')  # Y-axis label
#     ax.set_title('Impact of Error Flags on High Error Rate')  # Plot title
#     ax.legend()  # Add legend for bar colors

# # Add a horizontal line for the overall error rate as reference
#     ax.axhline(y=high_error_pct, color='gray', linestyle='--', 
#            label=f'Overall Error Rate ({high_error_pct:.1f}%)')
#     ax.set_ylim(0, 100)  # Set y-axis range from 0 to 100%

# # Adjust layout and save/display the figure
#     plt.tight_layout()  # Adjust spacing to prevent clipping of labels
#     plt.savefig('error_flags_impact.png', dpi=300)  # Save as high-resolution PNG
#     plt.show()  # Display the plot

# # ----- 2. Correlation Heatmap with Seaborn -----

#     """
# This visualization creates a correlation matrix heatmap to show relationships
# between different flags and MPE. It helps identify which flags are most
# strongly associated with error.
# """
# # Create a DataFrame for correlation analysis
#     corr_matrix = pd.DataFrame()

# # Define columns and labels for the enhanced correlation matrix
#     


# # Add flag columns (binary variables) to correlation matrix
#     for i, flag_col in enumerate(flag_columns):
#         corr_matrix[flag_labels[i]] = df[flag_col].astype(float)

# # Add MPE as a continuous variable
#     corr_matrix['MPE'] = df['mpe']

# # Calculate correlation matrix
#     correlation_matrix = corr_matrix.corr()

# # Plot the enhanced correlation heatmap
#     plt.figure(figsize=(10, 8))  # Create figure with specified size
# # Create heatmap with annotations, coolwarm colormap, and value range from -1 to 1
#     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
#     plt.title('Correlation Matrix: Error Flags and MPE')  # Add title
#     plt.tight_layout()  # Adjust spacing
#     plt.show()  # Display the plot

# # ----- 3. Flag Co-occurrence Heatmap with Seaborn -----

#     """
# This visualization creates a heatmap showing co-occurrence of flags.
# The diagonal shows the percentage of measurements with each flag,
# while off-diagonal cells show the percentage of measurements with one flag
# that also have another flag.
# """
# # Original heatmap showing co-occurrence of flags
#     heatmap_data = pd.DataFrame(index=flag_labels, columns=flag_labels)

# # Populate the heatmap data
#     for i, flag1 in enumerate(flag_columns):
#         for j, flag2 in enumerate(flag_columns):
#             if i == j:
#             # Diagonal: Percentage of measurements with this flag
#                 heatmap_data.iloc[i, j] = df[flag1].mean() * 100  
#             else:
#             # Off-diagonal: Percentage of measurements with flag1 that also have flag2
#                 if df[flag1].sum() > 0:
#                     heatmap_data.iloc[i, j] = df[df[flag1]][flag2].astype(float).mean() * 100
#                 else:
#                     heatmap_data.iloc[i, j] = 0.0

# # Convert to float to ensure compatibility with heatmap
#     heatmap_data = heatmap_data.astype(float)

# # Create the heatmap
#     plt.figure(figsize=(10, 8))  # Create figure with specified size
# # Create heatmap with annotations, YlOrRd colormap, and percentage range from 0 to 100
#     sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.1f', vmin=0, vmax=100)
#     plt.title('Flag Co-occurrence (%)')  # Add title
#     plt.tight_layout()  # Adjust spacing
#     plt.show()  # Display the plot

    flag_columns = ['flag_all_high_pixel_error', 'flag_all_high_gt_expert_error', 
                        'flag_all_high_error_rate_image', 'flag_image_multiple_errors','high_error']
    


# Create a new comprehensive box plot with mutually exclusive categories
    exclusive_flags_fig = go.Figure()

# Define priorities based on percentage thresholds
# New approach: Add columns with the actual percentage values for each flag

    df['pred_gt_diff_pct'] = (df['pred_pixel_gt_diff'] / df['pred_Distance_pixels'] * 100)

# Set the pose value to a comparable scale (0-100)
    if 'pose_eval' in df.columns:
        df['pose_pct'] = (1 - df['pose_eval']) * 100  # Convert to percent error (higher = worse)
    elif 'pose_eval_iou' in df.columns:
        df['pose_pct'] = (1 - df['pose_eval_iou']) * 100
    else:
        df['pose_pct'] = 0

# Now determine the primary flag based on the highest percentage
    def get_primary_flag_by_pct(row):
        # First check if we have any errors at all
        if not row['high_error']:
            return 'No Flags'
        elif row['flag_all_high_error_rate_image'] and row['flag_avg_scale_error']:
            return 'All High error rate image and Scale error >15%'
        elif row['flag_image_multiple_errors'] and row['flag_avg_scale_error']:
            return 'Multiple Errors in Same Image and Scale error >15%'
        
        # Now we know high_error is True, so no need to check it again
        elif row['flag_pred_gt_diff'] and row['flag_high_avg_gt_expert_error']:
            return 'Prediction-GT pixel diff over 15% and GT-Expert pixel diff over >15%'
        
        elif row['flag_pred_gt_diff'] and not row['flag_high_avg_pixel_error']:
            return 'Prediction-GT pixel diff over 10% and pred pixel error <15%'
        elif row['flag_avg_scale_error']:
            return 'Scale error >15%'
        
        
        elif row['flag_pred_gt_diff']:
            return 'Prediction-GT pixel diff over 15%'
        # elif row['flag_high_avg_gt_expert_error'] and row['flag_avg_scale_error']:
        #     return 'All GT-Expert error >10% and Scale error >10%'
        elif row['flag_high_avg_gt_expert_error'] and row['flag_high_avg_pixel_error']:
            return 'GT-Expert pixel diff over >15%'

       
        
        
    
        elif row['flag_high_avg_pixel_error']:
            return 'pred pixel error >15%'        
        

        elif row['pose_pct'] > 25:
            return 'Pose error >25%'
        
       
        else:   
            return 'Unclassified Error'  # More descriptive than 'strange'

# Assign each measurement to exactly one category based on highest percentage
    df['assigned_category'] = df.apply(get_primary_flag_by_pct, axis=1)

# Create a category for measurements with no flags
    # df.loc[df['assigned_category'].isna(), 'assigned_category'] = 'No Flags'

# The order for visualization still needs to be defined
    priority_names = [
    'Prediction-GT pixel diff over 15%',
    'Prediction-GT pixel diff over 15% and pred pixel error <15%',
        'GT-Expert pixel diff over >15%',
        'All GT-Expert error >15% and Scale error >15%',
        'Prediction-GT pixel diff over 15% and GT-Expert pixel diff over >15%',
        'pred pixel error >15%',
        'Scale error >15%',
        'Pose error >25%',
        'All High error rate image and Scale error >15%',
        'Multiple Errors in Same Image and Scale error >15%',
        'Unclassified Error'
]
    




# Print some statistics about our prioritized assignments
    print("\nAssignments based on percentage values:")
    for category in priority_names + ['No Flags']:
        count = len(df[df['assigned_category'] == category])
        print(f"{category}: {count} measurements")

# Store colors for visualization
    priority_colors = ['#9b59b6', '#2ecc71', '#f39c12', '#3498db', '#e74c3c', '#c0392b', '#8e44ad', '#27ae60', '#f1c40f', '#e67e22', '#34495e']
    df['flag_count'] = 0
    for flag in flag_columns:
        df['flag_count'] += df[flag].astype(int) 

    df['min_gt_diff']=0
    df['gt_diff_pct']=0
# Add a box plot for each exclusive category
    categories = [cat for cat in priority_names] + ['No Flags']
    for i, category in enumerate(categories):
        cat_df = df[df['assigned_category'] == category]
    
        if len(cat_df) > 0:  # Only add if there are measurements in this category
            color = priority_colors[i] if i < len(priority_colors) else 'gray'
            
            # Create separate traces for each pond type
            for pond_type in cat_df['Pond_Type'].unique():
                pond_df = cat_df[cat_df['Pond_Type'] == pond_type]
                
                # Define marker symbol based on pond type
                symbol = ('circle' if pond_type == 'square' else 
                         'square' if pond_type == 'circle_female' else 
                         'diamond' if pond_type == 'circle_male' else 
                         'cross')
                
                exclusive_flags_fig.add_trace(go.Box(
                    y=pond_df['mpe'],
                    name=category,
                    boxmean=True,
                    marker_color=color,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.5,
                    width=0.6,
                    quartilemethod="linear",
                    marker=dict(
                        symbol=symbol,
                        size=8
                    ),
                    showlegend=True if pond_type == cat_df['Pond_Type'].iloc[0] else False,  # Show legend only once per category
                    legendgroup=category,  # Group traces by category
                    hovertemplate=
                        f"<b>{category}</b><br>" +
                        "<b>ID:</b> %{customdata[0]}<br>" +
                        "<b>Image:</b> %{customdata[1]}<br>" +
                        "<b>Pond Type:</b> %{customdata[9]}<br>" +
                        "<b>Error:</b> %{y:.1f}%<br>" +
                        "<b>GT Diff:</b> %{customdata[2]:.1f}px (%{customdata[4]:.1f}%)<br>" +
                        "<b>Pred-GT Diff:</b> %{customdata[5]:.1f}px (%{customdata[6]:.1f}%)<br>" +
                        "<b>Pred Diff:</b> %{customdata[7]:.1f}px (%{customdata[8]:.1f}%)<br>" +
                        "<b>Flag Count:</b> %{customdata[3]}<br>" +
                        "<b>best pixels</b> %{customdata[10]:.1f}px<br>" +
                        "<b>pred pixels</b> %{customdata[11]:.1f}px<br>" +
                        "<b>pixel diff 1</b> %{customdata[12]:.1f}%<br>" +
                        "<b>pixel diff 2</b> %{customdata[13]:.1f}%<br>" +
                        "<b>pixel diff 3</b> %{customdata[14]:.1f}%<br>" +
                        "<b>gt expert diff 1</b> %{customdata[15]:.1f}%<br>" +
                        "<b>gt expert diff 2</b> %{customdata[16]:.1f}%<br>" +
                        "<b>gt expert diff 3</b> %{customdata[17]:.1f}%<br>" +
                        "<b>pose error</b> %{customdata[18]:.1f}%<br>" +
                        "<b>avg scale error</b> %{customdata[19]:.1f}%<br>" +
                        "<b>avg gt expert error</b> %{customdata[20]:.1f}%<br>",
                    customdata=pond_df[['PrawnID', 'Label', 'min_gt_diff', 'flag_count', 
                                    'gt_diff_pct', 'pred_pixel_gt_diff', 'pred_gt_diff_pct',
                                    'min_error_pixels', 'min_mape_pixels','Pond_Type','best_length_pixels',
                                    'pred_Distance_pixels','pixel_diff_pct_1','pixel_diff_pct_2',
                                    'pixel_diff_pct_3','gt_expert_diff_pct_1','gt_expert_diff_pct_2',
                                    'gt_expert_diff_pct_3','pose_pct','avg_scale_error','avg_gt_expert_error_pct']].values
                ))

# Add horizontal line at 10% error
    exclusive_flags_fig.add_shape(
    type='line',
    x0=-0.5, x1=len(categories) - 0.5,
    y0=10, y1=10,
    line=dict(color='red', dash='dash')
)


    exclusive_flags_fig.add_shape(
    type='line',
    x0=-0.5, x1=len(categories) - 0.5,
        y0=-10, y1=-10,
    line=dict(color='red', dash='dash')
)



# Update layout
    exclusive_flags_fig.update_layout(
    title='Error Distribution by Exclusive Flag Categories (Prioritizing Multiple Errors in Image)',
    yaxis_title='Min MPE (%)',
    height=800, width=2000,
    boxmode='group',
    yaxis=dict(
        range=[-50, max(50, df['mpe'].max() * 1.1)]
    ),
    margin=dict(l=50, r=50, t=80, b=120)
)

# Add counts to the box plot names
    for i, trace in enumerate(exclusive_flags_fig.data):
        category = trace.name
        count = len(df[df['assigned_category'] == category])
        trace.name = f"{category} (n={count})"


    os.makedirs('graphs', exist_ok=True)
    exclusive_flags_fig.write_html(f'/Users/gilbenor/Documents/code projects/msc/counting_research_algorithms/fifty_one/measurements/analysis/graphs/exclusive_flags_fig_{args.type}_{args.weights_type}_{args.error_size}.html')


    #create the same but for the 3 pond types in the same html file 
    
   









    """
    statistics:
    -----------
    mae: mean absolute error without flags
    mae by pond type: mean absolute error by pond type without flags
    mape: mean absolute percentage error without flags
    mape by pond type: mean absolute percentage error by pond type without flags
    mpe: mean percentage error without flags
    mpe by pond type: mean percentage error by pond type without flags
    rmse: root mean square error without flags
    rmse by pond type: root mean square error by pond type without flags
    r2: r-squared without flags

    mae_flags: mean absolute error with flags
    mape_flags: mean absolute percentage error with flags
    mpe_flags: mean percentage error with flags
    rmse_flags: root mean square error with flags
    r2_flags: r-squared with flags

    """

    mae_flags = df[df['assigned_category'] == 'No Flags']['mae'].mean()
    mape_flags = df[df['assigned_category'] == 'No Flags']['mpe'].mean()
    mae_by_pond_type = df[df['assigned_category'] == 'No Flags'].groupby('Pond_Type')['mae'].mean()
    mape_by_pond_type = df[df['assigned_category'] == 'No Flags'].groupby('Pond_Type')['mpe'].mean()
    

    std_mae_flags = df[df['assigned_category'] == 'No Flags']['mae'].std()
    std_mape_flags = df[df['assigned_category'] == 'No Flags']['mpe'].std()

    std_mae_by_pond_type = df[df['assigned_category'] == 'No Flags'].groupby('Pond_Type')['mae'].std()
    std_mape_by_pond_type = df[df['assigned_category'] == 'No Flags'].groupby('Pond_Type')['mpe'].std()



    mae_with_flags = df[df['assigned_category'] != 'No Flags']['mae'].mean()
    mape_with_flags = df[df['assigned_category'] != 'No Flags']['mpe'].mean()
    mae_by_pond_type_with_flags = df[df['assigned_category'] != 'No Flags'].groupby('Pond_Type')['mae'].mean()
    mape_by_pond_type_with_flags = df[df['assigned_category'] != 'No Flags'].groupby('Pond_Type')['mpe'].mean()

    std_mae_with_flags = df[df['assigned_category'] != 'No Flags']['mae'].std()
    std_mape_with_flags = df[df['assigned_category'] != 'No Flags']['mpe'].std()

    std_mae_by_pond_type_with_flags = df[df['assigned_category'] != 'No Flags'].groupby('Pond_Type')['mae'].std()
    std_mape_by_pond_type_with_flags = df[df['assigned_category'] != 'No Flags'].groupby('Pond_Type')['mpe'].std()



    # Print MAE and MPE by category
    print("\n=== MAE and MPE by Category ===")
    print(f"{'Category':<30} {'MAE Without Category':>25} {'MAE With Only Category':>25} {'MPE Without Category':>25} {'MPE With Only Category':>25}")
    print("-" * 130)


    
    for category in categories:
        # Calculate MAE and MPE without the current category
        mae_without_category = df[df['assigned_category'] != category]['mae'].mean()
        mape_without_category = df[df['assigned_category'] != category]['mpe'].mean()


        # Calculate MAE and MPE with only the current category
        mae_with_only_category = df[df['assigned_category'] == category]['mae'].mean()
        mape_with_only_category = df[df['assigned_category'] == category]['mpe'].mean()
        

        std_mae_without_category = df[df['assigned_category'] != category]['mae'].std()
        std_mape_without_category = df[df['assigned_category'] != category]['mpe'].std()

        std_mae_with_only_category = df[df['assigned_category'] == category]['mae'].std()
        std_mape_with_only_category = df[df['assigned_category'] == category]['mpe'].std()


        print(f"{category:<30} {mae_without_category:>25.2f}±{std_mae_without_category:.2f} {mae_with_only_category:>25.2f}±{std_mae_with_only_category:.2f} {mape_without_category:>25.2f}±{std_mape_without_category:.2f} {mape_with_only_category:>25.2f}±{std_mape_with_only_category:.2f} {std_mae_flags:>25.2f} {std_mape_flags:>25.2f} ")







    mae_high_error_rate_image_and_gt_expert_error_and_high_pixel_error = df[
        (df['flag_all_high_error_rate_image'] == 1) |
        (df['flag_all_high_gt_expert_error'] == 1) |
        (df['flag_all_high_pixel_error'] == 1) |
        (df['flag_image_multiple_errors'] == 1)
    ]['pixel_pct'].mean()

    mape_high_error_rate_image_and_gt_expert_error_and_high_pixel_error = df['pixel_pct'].mean()

    print(f"{'High error rate image and gt expert error and high pixel error':<30} {mae_high_error_rate_image_and_gt_expert_error_and_high_pixel_error:>25.2f} {mape_high_error_rate_image_and_gt_expert_error_and_high_pixel_error:>25.2f}")
    
    
    

    for category in categories:
        for pond_type in df['Pond_Type'].unique():
            print(f"\n=== {category} for {pond_type} ===")
            print(f"{'Metric':<30} {'Without Category':>20} {'With Only Category':>20}")
            print("-" * 70)
            
            # Calculate statistics for the current category and pond type
            mae_without_category = df[(df['assigned_category'] != category) & (df['Pond_Type'] == pond_type)]['mae'].mean()
            mape_without_category = df[(df['assigned_category'] != category) & (df['Pond_Type'] == pond_type)]['mpe'].mean()
            
            mae_with_only_category = df[(df['assigned_category'] == category) & (df['Pond_Type'] == pond_type)]['mae'].mean()
            mape_with_only_category = df[(df['assigned_category'] == category) & (df['Pond_Type'] == pond_type)]['mpe'].mean()

            std_mae_without_category = df[(df['assigned_category'] != category) & (df['Pond_Type'] == pond_type)]['mae'].std()
            std_mape_without_category = df[(df['assigned_category'] != category) & (df['Pond_Type'] == pond_type)]['mpe'].std()

            std_mae_with_only_category = df[(df['assigned_category'] == category) & (df['Pond_Type'] == pond_type)]['mae'].std()
            std_mape_with_only_category = df[(df['assigned_category'] == category) & (df['Pond_Type'] == pond_type)]['mpe'].std()   

            print(f"{category:<30} {mae_without_category:>25.2f} + {std_mae_without_category:>25.2f} {mae_with_only_category:>25.2f} + {std_mae_with_only_category:>25.2f} {mape_without_category:>25.2f} + {std_mape_without_category:>25.2f} {mape_with_only_category:>25.2f} + {std_mape_with_only_category:>25.2f}")

            
        





    # Existing overall statistics table
    print("\n=== Overall Statistics ===")
    print(f"{'Metric':<30} {'Without Flags':>15} {'With Flags':>15}")
    print("-" * 60)
    print(f"{'MAE (Mean Absolute Error)':<30} {mae_flags:>15.2f}±{std_mae_flags:>25.2f} {mae_with_flags:>15.2f}±{std_mae_with_flags:>25.2f}")
    print(f"{'MAPE (Mean Absolute % Error)':<30} {mape_flags:>15.2f}±{std_mape_flags:>25.2f} {mape_with_flags:>15.2f}±{std_mape_with_flags:>25.2f}")




    # Print MAE by pond type table
    print("\n=== MAE by Pond Type ===")
    print(f"{'Pond Type':<20} {'Without Flags':>15} {'With Flags':>15}")
    print("-" * 50)
    for pond_type in mae_by_pond_type.index:
        print(f"{pond_type:<20} {mae_by_pond_type[pond_type]:>15.2f}±{std_mae_by_pond_type[pond_type]:>25.2f} {mae_by_pond_type_with_flags[pond_type]:>15.2f}±{std_mae_by_pond_type_with_flags[pond_type]:>25.2f}")

    # Print MAPE by pond type table
    print("\n=== MAPE by Pond Type ===")
    print(f"{'Pond Type':<20} {'Without Flags':>15} {'With Flags':>15}")
    print("-" * 50)
    for pond_type in mape_by_pond_type.index:
        print(f"{pond_type:<20} {mape_by_pond_type[pond_type]:>15.2f}±{std_mape_by_pond_type[pond_type]:>25.2f} {mape_by_pond_type_with_flags[pond_type]:>15.2f}±{std_mape_by_pond_type_with_flags[pond_type]:>25.2f}")

    # Print sample counts
    print("\n=== Sample Counts ===")
    print(f"{'Pond Type':<20} {'Without Flags':>15} {'With Flags':>15} {'Total':>15}")
    print("-" * 65)
    for pond_type in df['Pond_Type'].unique():
        without_flags = len(df[(df['Pond_Type'] == pond_type) & (df['assigned_category'] == 'No Flags')])
        std_without_flags = df[(df['Pond_Type'] == pond_type) & (df['assigned_category'] == 'No Flags')]['mae'].std()
        with_flags = len(df[(df['Pond_Type'] == pond_type) & (df['assigned_category'] != 'No Flags')])
        std_with_flags = df[(df['Pond_Type'] == pond_type) & (df['assigned_category'] != 'No Flags')]['mae'].std()
        total = without_flags + with_flags
        std_total = df[(df['Pond_Type'] == pond_type)]['mae'].std()
        print(f"{pond_type:<20} {without_flags:>15} + {std_without_flags:>25.2f} {with_flags:>15} + {std_with_flags:>25.2f} {total:>15} + {std_total:>25.2f}")

    total_without_flags = len(df[df['assigned_category'] == 'No Flags'])
    total_with_flags = len(df[df['assigned_category'] != 'No Flags'])
    print("-" * 65)
    print(f"{'Total':<20} {total_without_flags:>15} {total_with_flags:>15} {len(df):>15}")


    #count statistics for each flag and pond type using primary flag using the assigned category
    for category in categories:
        print(f"\n=== {category} Statistics ===")
        print(f"{'Pond Type':<20} {'Count':>15}")
        print("-" * 30)
        
        for pond_type in df['Pond_Type'].unique():
            count = len(df[(df['Pond_Type'] == pond_type) & (df['assigned_category'] == category)])
            print(f"{pond_type:<20} {count:>15}")

        
    






    return __name__



if __name__ == "__main__":
    main()