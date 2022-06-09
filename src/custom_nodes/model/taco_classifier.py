"""
YOLOv5n Model for detecting litter (TACO Dataset)
"""
import torch
import torch.backends.cudnn as cudnn

from typing import Any, Dict

from peekingduck.pipeline.nodes.node import AbstractNode

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


class Node(AbstractNode):
    """Initializes YOLOv5n using trained weights to detect and predict 
    bounding boxes for objects in an image frame.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """
    class_labels = {0: "Aluminium Foil",
                    1: "Bottle",
                    2: "Bottle Cap",
                    3: "Can",
                    4: "Carton",
                    5: "Cup",
                    6: "Food Waste",
                    7: "Other Plastic",
                    8: "Paper Trash",
                    9: "Plastic Trash",
                    10: "Unlabeled Litter",
                    11: "Cigarette"}

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        # initialize/load any configs and models here
        # configs can be called by self.<config_name> e.g. self.filepath
        # self.logger.info(f"model loaded with configs: config")
        super().__init__(config, node_path=__name__, **kwargs)
        self.device = select_device('0')
        self.model = DetectMultiBackend("models/yolov5n_taco_best.pt", device=self.device)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt

        self.imgsz = check_img_size(640, s=self.stride)
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = False
        self.max_det = 1000
        self.line_thickness = 3
        self.hide_labels = False
        self.hide_conf = False

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """

        # result = do_something(inputs["in1"], inputs["in2"])
        # outputs = {"out1": result}
        # return outputs
