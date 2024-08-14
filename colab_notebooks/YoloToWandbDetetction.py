from typing import Any, Callable, Dict, List, Optional

from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import get_flops, get_num_params


import wandb
from wandb.sdk.lib import telemetry
import os

import cv2
import numpy as np

def read_bboxes_from_txt(txt_path: str) -> list:
    """
    Read bounding boxes from a .txt file.
    """
    bboxes = []
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Split each line and convert to appropriate data type
            values = line.strip().split()
            label, x_center, y_center, width, height = int(values[0]), float(values[1]), float(values[2]), float(values[3]), float(values[4])
            bboxes.append((label, x_center, y_center, width, height))
    return bboxes

def draw_bounding_boxes(
    image: np.ndarray,image_file_name, ground_truth: list, predictions: list, class_names: list
) -> np.ndarray:
    """
    Draw bounding boxes on the image using ground truth and predictions.

    Args:
        image (np.ndarray): The image to draw bounding boxes on.
        ground_truth (list): A list of ground truth bounding boxes.
        predictions (list): A list of predicted bounding boxes.
        class_names (list): A list of class names.

    Returns:
        np.ndarray: The image with bounding boxes drawn on it.
    """
    output_image = image.copy()

    # Convert center, width, height to xmin, ymin, xmax, ymax
    def cwh_to_corners(x_center, y_center, width, height):
        x_min = int((x_center - width / 2) * image.shape[1])
        y_min = int((y_center - height / 2) * image.shape[0])
        x_max = int((x_center + width / 2) * image.shape[1])
        y_max = int((y_center + height / 2) * image.shape[0])
        return x_min, y_min, x_max, y_max

    # Draw ground truth bounding boxes with green color
    for label, x_center, y_center, width, height in ground_truth:
        x0, y0, x1, y1 = cwh_to_corners(x_center, y_center, width, height)
        cv2.rectangle(output_image, (x0, y0), (x1, y1), (0, 0, 255), 2)
        

    # Draw predicted bounding boxes with red color
    for label, x_center, y_center, width, height in predictions:
        x0, y0, x1, y1 = cwh_to_corners(x_center, y_center, width, height)
        cv2.rectangle(output_image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        

    cv2.imwrite(f'{image_file_name}.jpg', output_image) # Save the image with bounding boxes
    return output_image

def log_to_wandb(images_folder, ground_truth_folder, predictions_folder, class_names):
    # Create a wandb.Table() for logging
    table = wandb.Table(columns=["Image", "Ground Truth", "Predictions",'Ground Truth and Predictions'])

    # Iterate through the images
    for image_file in os.listdir(images_folder):
        image_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(images_folder, image_file)

        gt_txt_path = os.path.join(ground_truth_folder, f"{image_name}.txt")
        predictions_txt_path = os.path.join(predictions_folder, f"{image_name}.txt")

        image = cv2.imread(image_path)
        ground_truth = read_bboxes_from_txt(gt_txt_path)
        predictions = read_bboxes_from_txt(predictions_txt_path)

        # Draw ground truth and predictions on separate images
        image_gt = draw_bounding_boxes(image.copy(),image_name, ground_truth, [], class_names)
        image_pred = draw_bounding_boxes(image.copy(),image_name, [], predictions, class_names)
         
        combined_image = draw_bounding_boxes(image, ground_truth, predictions, class_names)
        # Log images
        wandb_combined_image = wandb.Image(combined_image, caption="Ground Truth and Predictions")    

        # Log images
        wandb_gt_image = wandb.Image(image_gt, caption="Ground Truth")
        wandb_pred_image = wandb.Image(image_pred, caption="Predictions")
        
        table.add_data(wandb.Image(image, caption="Original"), wandb_gt_image, wandb_pred_image, wandb_combined_image)

    # Log table
    wandb.log({"results": table})


def compute_errors_and_log(ground_truth_path, predicted_path):
    total_squared_error = 0
    total_absolute_error = 0
    num_files = 0

    for filename in os.listdir(ground_truth_path):
        ground_truth_file = os.path.join(ground_truth_path, filename)
        predicted_file = os.path.join(predicted_path, filename)

        if not os.path.exists(predicted_file):
            continue

        with open(ground_truth_file, 'r') as gt, open(predicted_file, 'r') as pred:
            gt_lines = gt.readlines()
            pred_lines = pred.readlines()

            # Calculate the difference in the number of objects
            num_gt_objects = len(gt_lines)
            num_pred_objects = len(pred_lines)

            squared_error = (num_gt_objects - num_pred_objects) ** 2
            absolute_error = abs(num_gt_objects - num_pred_objects)

            total_squared_error += squared_error
            total_absolute_error += absolute_error
            num_files += 1

    # Calculate the Root Mean Squared Error (RMSE)
    if num_files > 0:
        rmse = math.sqrt(total_squared_error / num_files)
        mae = total_absolute_error / num_files
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Error (MAE): {mae}")

        # Log to wandb
        wandb.log({"prediction_RMSE": rmse, "prediction_MAE": mae})
        wandb.run.summary["prediction_RMSE"] = rmse
        wandb.run.summary["prediction_MAE"] = mae
    else:
        print("No matching files found.")

class WandbCallback:
    """An internal YOLO model wrapper that tracks metrics, and logs models to Weights & Biases.

    Usage:
    ```python
    from wandb.integration.yolov8.yolov8 import WandbCallback

    model = YOLO("yolov8n.pt")
    wandb_logger = WandbCallback(
        model,
    )
    for event, callback_fn in wandb_logger.callbacks.items():
        model.add_callback(event, callback_fn)
    ```
    """

    def __init__(
        self,
        yolo: YOLO,
        run_name: Optional[str] = None,
        project: Optional[str] = None,
        tags: Optional[List[str]] = None,
        resume: Optional[str] = None,
        **kwargs: Optional[Any],
    ) -> None:
        """A utility class to manage wandb run and various callbacks for the ultralytics YOLOv8 framework.

        Args:
            yolo: A YOLOv8 model that's inherited from `:class:ultralytics.yolo.engine.model.YOLO`
            run_name, str: The name of the Weights & Biases run, defaults to an auto generated run_name if `trainer.args.name` is not defined.
            project, str: The name of the Weights & Biases project, defaults to `"YOLOv8"` if `trainer.args.project` is not defined.
            tags, List[str]: A list of tags to be added to the Weights & Biases run, defaults to `["YOLOv8"]`.
            resume, str: Whether to resume a previous run on Weights & Biases, defaults to `None`.
            **kwargs: Additional arguments to be passed to `wandb.init()`.
        """
        self.yolo = yolo
        self.run_name = run_name
        self.project = project
        self.tags = tags
        self.resume = resume
        self.kwargs = kwargs

    def on_pretrain_routine_start(self, trainer: BaseTrainer) -> None:
        """Starts a new wandb run to track the training process and log to Weights & Biases.

        Args:
            trainer: A task trainer that's inherited from `:class:ultralytics.yolo.engine.trainer.BaseTrainer`
                    that contains the model training and optimization routine.
        """
        if wandb.run is None:
            self.run = wandb.init(
                name=self.run_name if self.run_name else trainer.args.name,
                project=self.project
                if self.project
                else trainer.args.project or "YOLOv8",
                tags=self.tags if self.tags else ["YOLOv8"],
                config=vars(trainer.args),
                resume=self.resume if self.resume else None,
                **self.kwargs,
            )
        else:
            self.run = wandb.run
        self.run.define_metric("epoch", hidden=True)
        self.run.define_metric(
            "train/*", step_metric="epoch", step_sync=True, summary="min"
        )

        self.run.define_metric(
            "val/*", step_metric="epoch", step_sync=True, summary="min"
        )

        self.run.define_metric(
            "metrics/*", step_metric="epoch", step_sync=True, summary="max"
        )
        self.run.define_metric(
            "lr/*", step_metric="epoch", step_sync=True, summary="last"
        )

        with telemetry.context(run=wandb.run) as tel:
            tel.feature.ultralytics_yolov8 = True

    def on_pretrain_routine_end(self, trainer: BaseTrainer) -> None:
        self.run.summary.update(
            {
                "model/parameters": get_num_params(trainer.model),
                "model/GFLOPs": round(get_flops(trainer.model), 3),
            }
        )

    def on_train_epoch_start(self, trainer: BaseTrainer) -> None:
        """On train epoch start we only log epoch number to the Weights & Biases run."""
        # We log the epoch number here to commit the previous step,
        self.run.log({"epoch": trainer.epoch + 1})

    def on_train_epoch_end(self, trainer: BaseTrainer) -> None:
        """On train epoch end we log all the metrics to the Weights & Biases run."""
        self.run.log(
            {
                **trainer.metrics,
                **trainer.label_loss_items(trainer.tloss, prefix="train"),
                **trainer.lr,
            },
        )
        # Currently only the detection and segmentation trainers save images to the save_dir
        if not isinstance(trainer, ClassificationTrainer):
            self.run.log(
                {
                    "train_batch_images": [
                        wandb.Image(str(image_path), caption=image_path.stem)
                        for image_path in trainer.save_dir.glob("train_batch*.jpg")
                    ]
                }
            )

    def on_fit_epoch_end(self, trainer: BaseTrainer) -> None:
        """On fit epoch end we log all the best metrics and model detail to Weights & Biases run summary."""
        if trainer.epoch == 0:
            speeds = [
                trainer.validator.speed.get(
                    key,
                )
                for key in (1, "inference")
            ]
            speed = speeds[0] if speeds[0] else speeds[1]
            if speed:
                self.run.summary.update(
                    {
                        "model/speed(ms/img)": round(speed, 3),
                    }
                )
        if trainer.best_fitness == trainer.fitness:
            self.run.summary.update(
                {
                    "best/epoch": trainer.epoch + 1,
                    **{f"best/{key}": val for key, val in trainer.metrics.items()},
                }
            )

    def on_train_end(self, trainer: BaseTrainer) -> None:
        """On train end we log all the media, including plots, images and best model artifact to Weights & Biases."""
        # Currently only the detection and segmentation trainers save images to the save_dir
        if not isinstance(trainer, ClassificationTrainer):
            self.run.log(
                {
                    "plots": [
                        wandb.Image(str(image_path), caption=image_path.stem)
                        for image_path in trainer.save_dir.glob("*.png")
                    ],
                    "val_images": [
                        wandb.Image(str(image_path), caption=image_path.stem)
                        for image_path in trainer.validator.save_dir.glob("val*.jpg")
                    ],
                },
            )

        if trainer.best.exists():
            self.run.log_artifact(
                str(trainer.best),
                type="model",
                name=f"{self.run.name}_{trainer.args.task}.pt",
                aliases=["best", f"epoch_{trainer.epoch + 1}"],
            )

    def on_model_save(self, trainer: BaseTrainer) -> None:
        """On model save we log the model as an artifact to Weights & Biases."""
        self.run.log_artifact(
            str(trainer.last),
            type="model",
            name=f"{self.run.name}_{trainer.args.task}.pt",
            aliases=["last", f"epoch_{trainer.epoch + 1}"],
        )

    def on_val_end(self,validator):
        if validator.training==False:
            self.run.summary.update(
                    {
                        
                        **{f"prediction/{key}": val for key, val in validator.metrics.items()},
                    })
            self.run.log({
            "prediction_plots": [
                            wandb.Image(str(image_path), caption=image_path.stem)
                            for image_path in validator.save_dir.glob("*.png")
                        ]
            })
    def on_pred_end(self,predictor):
        images_folder = "some_folder/test/images"
        ground_truth_folder = "some_folder/test/labels"
        predictions_folder = f'{predictor.save_dir}/labels'  # Adjust this based on your exact folder structure
        class_names = ['prawn']  # Adjust based on your classes
        log_to_wandb(images_folder, ground_truth_folder, predictions_folder, class_names)
        compute_errors_and_log(ground_truth_folder, predictions_folder)
        artifact = wandb.Artifact('predictions_folder', type='dataset')
        artifact.add_dir(predictor.save_dir)
        self.run.log_artifact(artifact)


    def on_train_start(self,trainer):
    #     dataset_artifact = wandb.Artifact(
    #     'first_photos',
    #     type='dataset',
    #     description='first photos taken at the ponds',
    #     metadata={'version': '1.0'}
    # )
    #     dataset_artifact.add_dir('some_path')
    #     self.run.log_artifact(dataset_artifact)  
        wandb.run.use_artifact('first_photos:latest')

    # def teardown(self, _trainer: BaseTrainer) -> None:
    #     """On teardown, we finish the Weights & Biases run and set it to None."""
    #     self.run.finish()
    #     self.run = None

    @property
    def callbacks(
        self,
    ) -> Dict[str, Callable]:
        """Property contains all the relevant callbacks to add to the YOLO model for the Weights & Biases logging."""
        return {
            "on_pretrain_routine_start": self.on_pretrain_routine_start,
            "on_pretrain_routine_end": self.on_pretrain_routine_end,
            "on_train_epoch_start": self.on_train_epoch_start,
            "on_train_epoch_end": self.on_train_epoch_end,
            "on_fit_epoch_end": self.on_fit_epoch_end,
            "on_train_end": self.on_train_end,
            "on_model_save": self.on_model_save,
            "on_val_end": self.on_val_end,
            "on_pred_end": self.on_pred_end,
            "on_train_start": self.on_train_start,
            # "teardown": self.teardown,
        }


def add_callbacks(
    yolo: YOLO,
    run_name: Optional[str] = None,
    project: Optional[str] = None,
    tags: Optional[List[str]] = None,
    resume: Optional[str] = None,
    **kwargs: Optional[Any],
) -> YOLO:
    """A YOLO model wrapper that tracks metrics, and logs models to Weights & Biases.

    Args:
        yolo: A YOLOv8 model that's inherited from `:class:ultralytics.yolo.engine.model.YOLO`
        run_name, str: The name of the Weights & Biases run, defaults to an auto generated name if `trainer.args.name` is not defined.
        project, str: The name of the Weights & Biases project, defaults to `"YOLOv8"` if `trainer.args.project` is not defined.
        tags, List[str]: A list of tags to be added to the Weights & Biases run, defaults to `["YOLOv8"]`.
        resume, str: Whether to resume a previous run on Weights & Biases, defaults to `None`.
        **kwargs: Additional arguments to be passed to `wandb.init()`.

    Usage:
    ```python
    from wandb.integration.yolov8 import add_callbacks as add_wandb_callbacks

    model = YOLO("yolov8n.pt")
    add_wandb_callbacks(
        model,
    )
    model.train(
        data="coco128.yaml",
        epochs=3,
        imgsz=640,
    )
    ```
    """
    wandb.termwarn(
        """The wandb callback is currently in beta and is subject to change based on updates to `ultralytics yolov8`.
        The callback is tested and supported for ultralytics v8.0.43 and above.
        Please report any issues to https://github.com/wandb/wandb/issues with the tag `yolov8`.
        """,
        repeat=False,
    )

    if RANK in [-1, 0]:
        wandb_logger = WandbCallback(
            yolo, run_name=run_name, project=project, tags=tags, resume=resume, **kwargs
        )
        for event, callback_fn in wandb_logger.callbacks.items():
            yolo.add_callback(event, callback_fn)
        return yolo
    else:
        wandb.termerror(
            "The RANK of the process to add the callbacks was neither 0 or -1."
            "No Weights & Biases callbacks were added to this instance of the YOLO model."
        )
    return yolo
