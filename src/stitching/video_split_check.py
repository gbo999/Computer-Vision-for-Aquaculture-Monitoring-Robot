import cv2
import os

def split_video_to_images(video_path, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Read frames from the video and save them as images
    frame_count = 0
    while True:
        # Read the next frame
        ret, frame = video.read()

        # If the frame was not read successfully, we have reached the end of the video
        if not ret:
            break

        # Save the frame as an image
        image_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(image_path, frame)

        frame_count += 1

    # Release the video file
    video.release()


# Get the current working directory
current_dir = os.getcwd()
print(f"Current Directory: {current_dir}")
relative_path='src\\stitching\\video.mp4'

video_path = os.path.join(current_dir, relative_path)

output_folder = os.path.join(current_dir, "src\stitching\output")
split_video_to_images(video_path, output_folder)
