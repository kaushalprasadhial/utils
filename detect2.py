from PIL import Image
import numpy as np
import os, cv2
import sys
import time
import torch
from sort import Sort, track

from models.experimental import attempt_load
from utils.datasets import LoadImagesFromNumpy #, LoadImagesFromFrameQueue
from utils.general import (
    check_img_size, non_max_suppression, scale_coords)
# from utils.torch_utils import select_device, time_synchronized
from utils import torch_utils
import logging

counts = None
analyser = []
classes = ['person']

total_count = {cls:0 for cls in classes}
current_count = {cls:0 for cls in classes}
colors = [(255, 0, 0), (0, 255, 255), (0, 255)]

class DetectYOLOPyTorch:
    """
    Detects the vehicles from YOLOv5 pytorch model
    """
    def __init__(self, config, imgsz = 640, device='cpu'):
        self.model_file = config['MODELS_FILE']
        self.num_classes = int(config['NUM_CLASSES'])

        # Initialize
        self.device = torch_utils.select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.model_file, map_location=self.device)  # load FP32 model

        self.imgsz = check_img_size(imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16


    
    def detect(self, image_src, conf_thresh=0.4, iou_thresh=0.5, agnostic=False):
        # try:
        #rows, cols, _ = image_src.shape
        dataset = LoadImagesFromNumpy(image_src, img_size=self.imgsz)

        # Run inference
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap in dataset:
            img_input_height = img.shape[1]
            img_input_width = img.shape[2]
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # # Inference
            pred = self.model(img, augment=False)[0]

            
            # Apply NMS
            pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=None, agnostic=agnostic)
        

            output = []
            if pred[0] != None:
                for i, det in enumerate(pred):
                    p, s, im0 = path, '', im0s
                    
                    if len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        det=det[:,:6].tolist()
                    for i in range(0,len(det)):
                        # print(det)
                        x1, y1, x2, y2 = det[i][:4]
                        class_id = round(det[i][5])
                        x_start = int(x1)
                        y_start = int(y1 )
                        x_end = int(x2) 
                        y_end = int(y2)
                        conf = det[i][4]

                        output.append([class_id, x_start,y_start,x_end,y_end, conf])
                        
        return output
            
        # except Exception as e:
        #     exc_type, _, exc_tb = sys.exc_info()
        #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        #     logging.error(f"{e}, {exc_type}, {fname}, {exc_tb.tb_lineno}")
        #     print(f"{e}, {exc_type}, {fname}, {exc_tb.tb_lineno}")
        #     logging.info('Model initiation issue in YOLO_PyTorch : {}'.format(e))

def Diff(li1, li2):
    return list(set(li2) - set(li1))

def draw_bounding_box(img, class_id, conf, x, y, x_plus_w, y_plus_h, id=None):
    if id is not None:
        color = (255, 0, 0)
        label = classes[class_id]+"id:"+str(id)
    else:
        color = (0,255,0)
        label = classes[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    # cv2.putText(img, label+' : '+str(round(conf, 2)), (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



    return img

if __name__=="__main__":
    video_file = '/media/kaushal/1db7515a-52b7-4566-bb04-4243044fca34/kaushal/Documents/train_yolo/data/mumbai_port/input/mumbai_port_1.avi'
    model_file = 'mumbai_port_yolov5s.pt' #'/media/kaushal/1db7515a-52b7-4566-bb04-4243044fca34/kaushal/Documents/train_yolo/yolov5/runs/train/exp13/weights/best.pt'
    model = DetectYOLOPyTorch({'MODELS_FILE':model_file,
                                'NUM_CLASSES':3})
    
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    detection_out = cv2.VideoWriter('detection_out.avi',cv2.VideoWriter_fourcc(*'DIVX'), cap.get(cv2.CAP_PROP_FPS), size)
    tracking_out = cv2.VideoWriter('tracking_out.avi',cv2.VideoWriter_fourcc(*'DIVX'), cap.get(cv2.CAP_PROP_FPS), size)
    # Check if camera opened successfully
    if (cap.isOpened()== False): print("Error opening video  file")
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    # video = cv2.VideoWriter('video.avi', fourcc, 1, (frame_width, frame_height))
    # Read until video is completed
    max_age_val = 3
    min_hits_val = 2
    sort = Sort(max_age=max_age_val, min_hits=min_hits_val, use_dlib = False)

    missing_person = {}
    previous_ids = []
    total_people = 0
    while ret:
        current_ids = []
        s = ''
        counts = ''
        current_count = {}
        if ret == True:
            result = model.detect(frame)
            # for c in result[:, 0].unique():
            #     n = (det[:, 0] == c).sum()  # detections per class
            #     s += f"{n} {classes[int(c)]}{'s' * (n > 1)}, "  # add to string
            #     counts += f"{n} {classes[int(c)]}{'s' * (n > 1)}, "
            detections = []
            for det in result:
                if det:
                    detections.append(det[1:])
            tracking_img = frame.copy()
            # Display the resulting frame
            for detection in result:
                # print("normal detection", detection)
                frame = draw_bounding_box(frame, 
                                            detection[0], 
                                            int(detection[5]),
                                            int(detection[1]),
                                            int(detection[2]), 
                                            int(detection[3]),
                                            int(detection[4])
                                            )
                
                if classes[detection[0]] in current_count.keys():
                    current_count[classes[detection[0]]]+=1
                else:
                    current_count[classes[detection[0]]]=1
            
            tracks = track(sort, frame, detections)
            for i,detection in enumerate(tracks):
                # print("track detection", detection)
                id = int(detection[4])
                tracking_img = draw_bounding_box(tracking_img, 
                                            result[i][0], 
                                            result[i][-1], 
                                            int(detection[0]), 
                                            int(detection[1]), 
                                            int(detection[2]),
                                            int(detection[3]),
                                            id)
                current_ids.append(id)
            #     if id in track_person:
            #         track_person[id]+=1
            #     else:
            #         track_person[id]+=0

            # for key in track_person:
            #     if track_person[key]>3:
            #         track_person.pop(key)
            potential_lost_ids = Diff(previous_ids, current_ids)
            print(f"potential_lost_ids {potential_lost_ids}")
            print(f"missing_person {missing_person}")
            to_remove_ids = []
            for id in missing_person:
                if id in current_ids:
                    to_remove_ids.append(id)
                    # else:
                    #     if id in missing_person:
                    #         if missing_person[id]>=3:
                    #             missing_person.pop(id)
                    #         else:
                    #             missing_person[id]+=1
                    #     else:
                    #         missing_person[id]=1
                else:
                    if missing_person[id]>=3:
                        to_remove_ids.append(id)
                    else:
                        missing_person[id]+=1
            for id in to_remove_ids:
                missing_person.pop(id)
            for id in potential_lost_ids:
                missing_person[id]=1

            potential_new_ids = Diff(current_ids, previous_ids)
            for id in potential_new_ids:
                if not id in missing_person:
                    total_people+=1
                    
            frame = cv2.putText(frame, str(current_count), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                   3, (255, 0, 0), 5, cv2.LINE_AA)
            tracking_img = cv2.putText(tracking_img, str(total_people), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                   3, (255, 0, 0), 5, cv2.LINE_AA)
            
            cv2.imshow('detection', frame)
            cv2.imshow('tracking', tracking_img)
            detection_out.write(frame)
            tracking_out.write(tracking_img)
        
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                detection_out.release()
                tracking_out.release()
                break
            previous_ids = current_ids
            time.sleep(1)
            ret, frame = cap.read()
        # Break the loop
        else: 
            break
    
    # When everything done, release 
    # the video capture object
    cap.release()
    detection_out.release()
    tracking_out.release()
    # Closes all the frames
    cv2.destroyAllWindows()