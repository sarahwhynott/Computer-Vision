# Filename: motfinalproj.py
# Author: Sarah Whynott
# Date Created: 11/28/2022

import numpy as np
import cv2
import math
import sys
import os
import random
import csv

class ObjectDetection:
    def __init__(self, weights_path="yolo4_model/yolov4.weights", cfg_path="yolo4_model/yolov4.cfg"):
        self.nmsThreshold = 0.4
        self.confThreshold = 0.5
        self.image_size = 608

        # Load Network
        net = cv2.dnn.readNet(weights_path, cfg_path)

        # Enable GPU CUDA
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model = cv2.dnn_DetectionModel(net)

        self.classes = []
        self.load_class_names()
        self.colors = np.random.uniform(0, 255, size=(80, 3))

        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1/255)

    def load_class_names(self, classes_path="yolo4_model/classes.txt"):

        with open(classes_path, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.colors = np.random.uniform(0, 255, size=(80, 3))
        return self.classes

    def detect(self, frame):
        return self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)




def main():

    obj_detection = ObjectDetection()

    if len(sys.argv) < 1:
        print(f'{sys.argv[0]} [groundtruth] [img dir]')
        sys.exit()
    

    groundtruth = sys.argv[1]

    dir = sys.argv[2]

    file = open(groundtruth, 'w', newline='') 
    writer = csv.writer(file)

    frame_num = 1

    prev_boxes = []

    tracked_boxes = {}

    box_id = 0

    color = {}

    with os.scandir(dir) as it:

        # sort the files in alphabetical order so the video
        # appears in sequential order
        it = list(it)
        it.sort(key=lambda x: x.name)

        for entry in it:
            if not entry.name.startswith('.') and entry.is_file():

                # Read in the image
                frame = cv2.imread(dir + "/" + entry.name)

            curr_boxes = []

            class_ids, scores, boxes = obj_detection.detect(frame)

            for box in boxes:
            
                x, y, width, height = box

                newbox = x, y, width, height, int((x + x + width) / 2), int((y + y + height) / 2)

                curr_boxes.append(newbox)

            win1 = "Frame"
            cv2.namedWindow(win1, cv2.WINDOW_NORMAL)

            threshold = 20

            if frame_num <= 3:

                for curr_box in curr_boxes:
                    
                    for prev_box in prev_boxes:

                        distance = math.hypot(prev_box[4] - curr_box[4], prev_box[5] - curr_box[5])

                        if distance < threshold:
                            
                            tracked_boxes[box_id] = curr_box
                            color[box_id] = (random.randint(64,255),random.randint(64 ,255),random.randint(64,255))

                            box_id = box_id + 1

                frame_num = frame_num + 1
            else:

                curr_boxes_copy = curr_boxes.copy()

                for obj_id, prev_box in tracked_boxes.copy().items():

                    obj_exists = False 

                    for curr_box in curr_boxes:
                        
                        distance = math.hypot(prev_box[4] - curr_box[4], prev_box[5] - curr_box[5])

                        if distance < threshold:
                            
                            tracked_boxes[obj_id] = curr_box

                            obj_exists = True

                            if curr_box in curr_boxes:
                            
                                curr_boxes.remove(curr_box)

                            continue

                    if not obj_exists:

                        tracked_boxes.pop(obj_id)


                for curr_box in curr_boxes:

                    tracked_boxes[box_id] = curr_box
                    color[box_id] = (random.randint(64,255),random.randint(64,255),random.randint(64,255))
                    

                    box_id = box_id + 1
            
            for obj_id, box in tracked_boxes.items():
                cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color[obj_id], 2)
                cv2.putText(frame, str(obj_id), (box[4], box[1]), 0, 1, (0, 255, 0), 2)
                writer.writerow([frame_num, obj_id, box[0], box[1], box[2], box[3], -1, -1, -1, -1])
        

            cv2.imshow(win1,frame)

            prev_boxes = curr_boxes.copy()
            
            key = cv2.waitKey(1)

            frame_num = frame_num + 1

            if key == ord('q'):
                break
            
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()