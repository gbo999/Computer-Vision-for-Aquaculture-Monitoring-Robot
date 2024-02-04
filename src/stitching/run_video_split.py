import os

from src.video.parameters import Parameters
from src.stitching.video2dataset_mine import Video2Dataset
work_dir = os.getcwd()

args= { 

    "input": f'{work_dir}/src/stitching/video.mp4',
    "output": f'{work_dir}/src/stitching/output',
    "start": 0,
    "end": None,
    "output_resolution": 1024,
    "blur_threshold":150,
    "distance_threshold":15,
    "black_ratio_threshold": None,
    "pixel_black_threshold": None,
    "use-srt": None,
    "limit": None,
    "frame_format": "jpg",
    "stats_file":f"{work_dir}/src/stitching/output/stats.csv"}




Video2Dataset2process=Video2Dataset(Parameters(args))
Video2Dataset2process.ProcessVideo()

