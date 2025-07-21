class PrawnDataset:
    """Handles FiftyOne dataset creation and visualization"""
    def __init__(self, name="prawn_dataset"):
        self.dataset = fo.Dataset(name, overwrite=True)
        self._setup_skeleton()
    
    def add_sample(self, image_path, predictions, ground_truth):
        """Add a sample with predictions and ground truth""" 
    
    def _setup_skeleton(self):
        """Setup keypoint skeleton with 4 points"""
        self.dataset.default_skeleton = fo.KeypointSkeleton(
            labels=["tail", "start_carapace", "eyes", "rostrum"],
            edges=[
                [0, 1],  # tail to start_carapace
                [1, 2],  # start_carapace to eyes
                [2, 3]   # eyes to rostrum
            ]
        ) 