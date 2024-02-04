import os

from src.video.parameters import Parameters
from src.stitching.video2dataset_mine import Video2Dataset
work_dir = os.getcwd()

args= { 

    "input": f'{work_dir}/src/stitching/video.mp4',
    "output": f'{work_dir}/src/stitching/output',
    "start": 0,
    "end": None,
    "output-resolution": 1024,
    "blur-threshold": 300,
    "distance-threshold": None,
    "black-ratio-threshold": 0.98,
    "pixel-black-threshold": 0.30,
    "use-srt": None,
    "limit": None,
    "frame-format": "jpg",
    "stats-file": f'{work_dir}/src/stitching/output/stats.csv',
}


video2dataset_mine = Video2Dataset(Parameters(args))