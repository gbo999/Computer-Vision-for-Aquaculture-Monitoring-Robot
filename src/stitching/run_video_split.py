import os

from src.video.parameters import Parameters
from src.stitching.video2dataset_mine import Video2Dataset
import glob
work_dir = os.getcwd()

args= { 

    "input": f"C:/Users/gbo10/Dropbox/research videos/31.12/65-31.12/GX010065.MP4",
    "output": f"C:/Users/gbo10/OneDrive/pictures/to_contrast/2.1",
    "start": 0,
    "end": None,
    "output_resolution": None,
    "blur_threshold":230,
    "distance_threshold":40,
    "black_ratio_threshold": None,
    "pixel_black_threshold": None,
    "use-srt": None,
    "limit": None,
    "frame_format": "jpg",
    "stats_file":f"{work_dir}/src/stitching/output/stats.csv"}


video_paths = []  # Initialize an empty list to store the video paths

for video_path in glob.glob("C:/Users/gbo10/Dropbox/research videos/2.1.2024/65/*.mp4"):
    video_paths.append(video_path)


for video_path in video_paths:
    args["input"] = video_path
    print(f'Processing video {video_path}')
    Video2Dataset2process = Video2Dataset(Parameters(args))
    Video2Dataset2process.ProcessVideo()

