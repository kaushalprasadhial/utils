from audioop import lin2adpcm
import cv2
import glob, os
import random

image_foler_path = "/media/kaushal/1db7515a-52b7-4566-bb04-4243044fca34/kaushal/Documents/mahindra_logistics/dataset/all/images"
label_folder_path = "/media/kaushal/1db7515a-52b7-4566-bb04-4243044fca34/kaushal/Documents/mahindra_logistics/dataset/all/labels"
names_file_path = "./obj.names"

images_paths = glob.glob(os.path.join(image_foler_path, "*.jpeg"))

name_file = open(names_file_path, 'r')
names = []
for line in name_file.readlines():
    names.append(line.replace("\n", ""))
name_file.close()

colours = []

while len(names)!=len(colours):
    colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    if not colour in colours:
        colours.append(colour)

print(colours)

for image_path in images_paths:
    img = cv2.imread(image_path)
    dh, dw, _ = img.shape

    label_path = os.path.join(label_folder_path, os.path.basename(image_path).replace("jpeg", "txt"))
    if os.path.exists:
        fl = open(label_path, 'r')
    else:
        fl = open(label_path, 'w')
        fl.close()
        fl = open(label_path, 'r')
    data = fl.readlines()
    fl.close()


    for dt in data:
        # Split string to float
        c, x, y, w, h = map(float, dt.split(' '))

        # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
        # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)
        
        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1
        print(f"names[int(c)] {names[int(c)]} colours[int(c)] {colours[int(c)]}")
        cv2.putText(img, names[int(c)], (l-10,t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colours[int(c)], 2)
        cv2.rectangle(img, (l, t), (r, b), colours[int(c)], 1)

    cv2.imshow("img",img)
    cv2.waitKey()