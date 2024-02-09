import os

from src.video.parameters import Parameters
from src.stitching.video2dataset_mine import Video2Dataset
work_dir = os.getcwd()

args= { 

    "input": f"C:/Users/gbo10/Dropbox/research videos/31.12/65-31.12/GX010065.MP4",
    "output": f"C:/Users/gbo10/OneDrive/pictures/to_contrast/GX010065",
    "start": 0,
    "end": None,
    "output_resolution": None,
    "blur_threshold":200,
    "distance_threshold":20,
    "black_ratio_threshold": None,
    "pixel_black_threshold": None,
    "use-srt": None,
    "limit": None,
    "frame_format": "jpg",
    "stats_file":f"{work_dir}/src/stitching/output/stats.csv"}




Video2Dataset2process=Video2Dataset(Parameters(args))
Video2Dataset2process.ProcessVideo()

