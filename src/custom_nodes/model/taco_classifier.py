"""
YOLOv5n Model for detecting litter (TACO Dataset)
"""
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from typing import Any, Dict

from peekingduck.pipeline.nodes.node import AbstractNode

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
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

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        # initialize/load any configs and models here
        # configs can be called by self.<config_name> e.g. self.filepath
        # self.logger.info(f"model loaded with configs: config")
        super().__init__(config, node_path=__name__, **kwargs)
        self.device = select_device('0')
        self.model = DetectMultiBackend(self.weights_path, device=self.device)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt

        self.imgsz = check_img_size(self.img_size, s=self.stride)
        self.classes = None
        self.agnostic_nms = False
        self.auto = True

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        bboxes = []
        labels = []
        scores = []

        # Read image
        im0 = inputs["img"]

        # Padded resize
        im = letterbox(im0, self.imgsz, stride=self.stride, auto=self.auto)[0]

        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred,
                                   self.conf_thres,
                                   self.iou_thres,
                                   self.classes,
                                   self.agnostic_nms,
                                   max_det=self.max_det)

        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                det = det.cpu().numpy()
                # Normalize bboxes coordinates
                det[:,[1, 3]] /= im0.shape[0]
                det[:,[0, 2]] /= im0.shape[1]
                bboxes = det[:, :4]
                labels = [*map(self.class_label_map.get, det[:,5])]
                scores = det[:,4]

        outputs = {"bboxes": bboxes, "bbox_labels": labels, "bbox_scores": scores}
        return outputs
