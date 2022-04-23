import cv2
import os

input_path = "videos"
output_path = "dataset"
if not os.path.exists(output_path):
    os.makedirs(output_path)
dir_list = os.listdir(input_path)
print(dir_list)
for file in dir_list:
    temp = os.path.basename(file).split(".")
    video_name, ext = ".".join(temp[:-1]), temp[-1]
    print(video_name, ext)
    vidcap = cv2.VideoCapture(os.path.join(input_path, file))
    success,image = vidcap.read()
    count = 0
    while success:
        if count%200==0: cv2.imwrite(os.path.join(output_path, video_name+"_%d.jpg" % count), image)     # save frame as JPEG file      
        success,image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1