import os
import cv2
import time
import argparse
import numpy as np
import random
from YOLOv3 import YOLOv3
from deep_sort import DeepSort
from util import COLORS_10, draw_bboxes, act_pred
from act_recog import predict_single_action


class Detector(object):
    def __init__(self, args):
        self.args = args
        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture(0)
        self.yolo3 = YOLOv3(args.yolo_cfg,
                            args.yolo_weights,
                            args.yolo_names,
                            is_xywh=True,
                            conf_thresh=args.conf_thresh,
                            nms_thresh=args.nms_thresh)
        self.deepsort = DeepSort(args.deepsort_checkpoint)
        self.class_names = self.yolo3.class_names

    def __enter__(self):
        assert os.path.isfile(self.args.VIDEO_PATH), "Error: path error"
        self.vdo.open(self.args.VIDEO_PATH)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 20,
                                          (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):
        # data structure 
        offset = (0, 0)
        ids_list = np.array([0],dtype=int)
        objects_frame_list = np.array([[
            np.zeros((64, 64, 3), dtype=float),
            np.zeros((64, 64, 3), dtype=float),
            np.zeros((64, 64, 3), dtype=float),
            np.zeros((64, 64, 3), dtype=float),
            np.zeros((64, 64, 3), dtype=float),
            np.zeros((64, 64, 3), dtype=float),
            np.zeros((64, 64, 3), dtype=float),
            np.zeros((64, 64, 3), dtype=float),
            np.zeros((64, 64, 3), dtype=float),
            np.zeros((64, 64, 3), dtype=float),
            np.zeros((64, 64, 3), dtype=float),
            np.zeros((64, 64, 3), dtype=float),
            np.zeros((64, 64, 3), dtype=float),
            np.zeros((64, 64, 3), dtype=float),
            np.zeros((64, 64, 3), dtype=float),
            np.zeros((64, 64, 3), dtype=float),
            np.zeros((64, 64, 3), dtype=float),
            np.zeros((64, 64, 3), dtype=float),
            np.zeros((64, 64, 3), dtype=float),
            np.zeros((64, 64, 3), dtype=float),]])
        actions = np.array(['walk'])
        # looping over frames of a video
        while self.vdo.grab():
            _, ori_im = self.vdo.retrieve()
            start = time.time()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            im = ori_im
            bbox_xcycwh, cls_conf, cls_ids = self.yolo3(im)
            if bbox_xcycwh is not None:
                # select classes(person) to detect
                mask = cls_ids == 0
                bbox_xcycwh = bbox_xcycwh[mask]
                bbox_xcycwh[:, 3:] *= 1.2
                cls_conf = cls_conf[mask]
                outputs = self.deepsort.update(bbox_xcycwh, cls_conf, im)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    lost_ids_indices = np.where((identities != ids_list))[0]
                    print("identities = ", identities)
                    print("ids_list = ", ids_list)
                    print("lost_ids_indices = ", lost_ids_indices)
                    print("lost_ids_indices length = ", len(lost_ids_indices))
                    if len(lost_ids_indices) > 0:
                        for i in range(len(lost_ids_indices)):
                            print("ID to be  deleted = ",lost_ids_indices[i])
                            if lost_ids_indices[i] != 0:
                                ids_list = np.delete(
                                    ids_list, lost_ids_indices[i])
                                objects_frame_list = np.delete(
                                    objects_frame_list, lost_ids_indices[i])
                                actions = np.delete(
                                    actions, lost_ids_indices[i])
                    # /////////////////////////////
                    print("bbox_xyxy = ", bbox_xyxy)

                    # processing further over each detections in a frame
                    for i, box in enumerate(bbox_xyxy):
                        x1, y1, x2, y2 = [int(i) for i in box]
                        x1 += offset[0]
                        x2 += offset[0]
                        y1 += offset[1]
                        y2 += offset[1]
                        # box text and bar
                        id = int(identities[i]
                                 ) if identities is not None else 0
                        # print("id-1 = ",id)
                        # id = '{}{:d}'.format("", id)
                        # print("id-2 = ",id)
                        # xmin, ymin, xmax, ymax = bbox
                        cropped_object = im[y1:y2, x1:x2]
                        resized_obj_img = cv2.resize(cropped_object, (64, 64))

                        normalized_frame = resized_obj_img / 255
                        # adding data structure if new person with detected
                        if id not in ids_list:
                            ids_list = np.append(ids_list, id)
                            actions = np.append(actions, 'walk')
                            temp_arr = [np.zeros((64, 64, 3), dtype=float),
                                        np.zeros((64, 64, 3), dtype=float),
                                        np.zeros((64, 64, 3), dtype=float),
                                        np.zeros((64, 64, 3), dtype=float),
                                        np.zeros((64, 64, 3), dtype=float),
                                        np.zeros((64, 64, 3), dtype=float),
                                        np.zeros((64, 64, 3), dtype=float),
                                        np.zeros((64, 64, 3), dtype=float),
                                        np.zeros((64, 64, 3), dtype=float),
                                        np.zeros((64, 64, 3), dtype=float),
                                        np.zeros((64, 64, 3), dtype=float),
                                        np.zeros((64, 64, 3), dtype=float),
                                        np.zeros((64, 64, 3), dtype=float),
                                        np.zeros((64, 64, 3), dtype=float),
                                        np.zeros((64, 64, 3), dtype=float),
                                        np.zeros((64, 64, 3), dtype=float),
                                        np.zeros((64, 64, 3), dtype=float),
                                        np.zeros((64, 64, 3), dtype=float),
                                        np.zeros((64, 64, 3), dtype=float),
                                        np.zeros((64, 64, 3), dtype=float),
                                        ]
                            objects_frame_list = np.append(
                                objects_frame_list, [temp_arr], axis=0)
                        id_index = np.where(ids_list == id)[0][0]
                        print("id_index = ", id_index)
                        for i, j in enumerate(objects_frame_list[id_index]):
                            # print(np.shape(objects_frame_list[id_index,i]))
                            # print(np.shape(np.zeros((64, 64, 3), dtype=float)))
                            if i == 19 and np.all(objects_frame_list[id_index][i] == 0, axis=None):
                                for i in range(20):
                                    if(i):
                                        print("i===",i)
                                        objects_frame_list[id_index][i] = np.zeros((64, 64, 3), dtype=float)
                            if np.all(objects_frame_list[id_index][i] == 0, axis=None):
                                objects_frame_list[id_index,
                                                   i] = normalized_frame
                                action, confidence = predict_single_action(
                                    objects_frame_list[id_index])
                                actions[id_index] = action
                                # print("action = ",action)
                                if (action == "shoot_gun"):
                                    print(
                                        f"======================ALERT({action} = {confidence})======================")
                                break
                        print(
                            "---------------data updating,loop 1 end---------------")
                    # /////////////////////////////
                    print("actions list = ", actions)
                    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                    ori_im = draw_bboxes(ori_im, bbox_xyxy, actions)
                    # print("bbox_xyxy:", bbox_xyxy,
                    #       "bbox_shape = ", bbox_xyxy.shape)
                    # for i, cropped_object in enumerate(cropped_objects):
                    #     cv2.imshow(f"Cropped Object {i+random.randint(0,90)}", cropped_object)
                    #     # Save the crop ped object as an image file
                    #     cv2.imwrite(f"cropped_object_{i+random.randint(0,90)}.jpg", cropped_object)
                    # print("shape:",ori_im.shape
                    # ori_im = draw_bboxes(ori_im, bbox_xyxy, identities)

            end = time.time()
            # print("time: {}s, fps: {}".format(end - start, 1 / (end - start)))

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.output.write(ori_im)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--yolo_cfg",
                        type=str,
                        default="YOLOv3/cfg/yolo_v3.cfg")
    parser.add_argument("--yolo_weights",
                        type=str,
                        default="YOLOv3/yolov3.weights")
    parser.add_argument("--yolo_names",
                        type=str,
                        default="YOLOv3/cfg/coco.names")
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--nms_thresh", type=float, default=0.4)
    parser.add_argument("--deepsort_checkpoint",
                        type=str,
                        default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.2)
    parser.add_argument("--ignore_display",
                        dest="display",
                        action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="demo.avi")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with Detector(args) as det:
        det.detect()
