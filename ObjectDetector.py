import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import norfair
from norfair import Tracker
from norfair import Detection, Tracker, draw_tracked_boxes, draw_tracked_objects
from typing import List
from numpy import random as nprand
import threading
import traceback

# These imports are from your custom modules
import my_classes.utils as ut
import my_classes.norfair_distances as norfair_distances
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from utils.plots import Annotator, colors

class ObjectDetector:
    def __init__(self, args):
        self.args = args
        self.confidence = float(args.confidence)

        self.device = select_device("cpu" if args.cpu else 0)
        self.model = DetectMultiBackend(args.model, device=self.device, dnn=False, data='dataset.yaml', fp16=False)
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[nprand.randint(0, 255) for _ in range(3)] for _ in self.names]

        cudnn.benchmark = True
        self.tracked_objects = []
        self._stop_detection = False

    def stop_detection(self):
        """Sets the stop_detection flag to gracefully stop the detection loop."""
        self._stop_detection = True

    def preprocess_image(self, img_np):
        """Preprocesses an image and converts it to a format suitable for the object detection model."""
        img = np.array(img_np)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = ut.letterbox(img)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        return img

    def detect_objects(self, img):
        """Detects objects in the preprocessed image using the object detection model."""
        pred = self.model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.confidence)
        return pred

    def draw_bounding_boxes(self, pred, manipulated_image, img, lock):
        """Draws bounding boxes on the original image based on the object detection predictions."""
        boxes = []
        im0 = np.array(img)
        for i, det in enumerate(pred):
            annotator = Annotator(im0, line_width=1, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(manipulated_image.shape[2:], det[:, :4], img.shape).round()

                for *x, conf, cls in reversed(det): #xyxy
                    c = int(cls)
                    label = f'{self.names[c]} {conf:.2f}'
                    annotator.box_label(x, label, color=self.colors[int(cls)])
                    lock.acquire()
                    boxes.append((self.names[int(cls)], (int(x[0]), int(x[1]), int(x[2]), int(x[3]))))
                    lock.release()
        return im0, boxes

    def norfair_track_bb(self, pred, tracker, frame, boxes):
        """Performs object tracking using Norfair and updates the tracked_objects attribute."""
        norfair_detections = []
        for box in boxes:
            bbox = np.array([
                [box[1][0], box[1][1]],
                [box[1][2], box[1][3]],
            ])
            norfair_detections.append(Detection(bbox, label=box[0]))

        self.tracked_objects = tracker.update(norfair_detections)
        norfair.draw_tracked_boxes(frame=frame, objects=self.tracked_objects, border_width=1, id_thickness=1, color_by_label=True, draw_labels=True, label_size=1, label_width=1)
        return frame

    def detect_screen(self):
        """The main loop for the object detector, which captures the screen, preprocesses the image, detects objects, draws bounding boxes, and performs object tracking."""
        tracker = Tracker(distance_function=norfair_distances.iou_opt, distance_threshold=1.33)
        lock = threading.Lock()
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        self.model.warmup()  # check image size
        while not self._stop_detection:
            try:
                frame = ut.get_frame()
                img = self.preprocess_image(frame)
                predictions = self.detect_objects(img)
                detected_frame_detection, boxes = self.draw_bounding_boxes(predictions, img, frame, lock)
                detected_frame_track = self.norfair_track_bb(predictions, tracker, frame, boxes)
                detected_frame = detected_frame_track if not self.args.no_track else detected_frame_detection
                cv2.imshow("screen", detected_frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    cv2.destroyAllWindows()
                    self.stop_detection()
            except Exception as e:
                print(f"Error in object detection: {e}")
                traceback.print_exc()
                break
