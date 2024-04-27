import os
import json
from PIL import Image
import shutil

keypointsp = ["0 : Nose","1 : Left-eye","2 : Right-eye","3 : Left-ear","4 : Right-ear","5 : Left-shoulder","6 : Right-shoulder",
             "7 : Left-elbow","8 : Right-elbow","9 : Left-wrist","10 : Right-wrist","11 : Left-hip","12 : Right-hip",
             "13 : Left-knee","14 : Right-knee","15 : Left-ankle","16 : Right-ankle","17 : C7"]
# IMAGE_FOLDER = "./import/data/images/"
# YOLO_TEXT_FOLDER = "./import/data/labels/"
# OUTPUT_FOLDER = "./import/output/"

IMAGE_FOLDER = "./import/data/image/"
YOLO_TEXT_FOLDER = "./import/data/label/"
OUTPUT_FOLDER = "./import/output/"

def yolo_to_labelme(class_id, x_center, y_center, width, height, keypoints, imageWidth, imageHeight):
    x1 = (x_center - width/2) * imageWidth
    y1 = (y_center - height/2) * imageHeight
    x2 = (x_center + width/2) * imageWidth
    y2 = (y_center + height/2) * imageHeight

    bbox = {
        "label": class_id,
        "points": [[x1, y1], [x2, y2]],
        "shape_type": "rectangle",
        "flags": {}
    }

    keypoint_shapes = []
    z = 0
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i:i+3]
        kp_x = x * imageWidth
        kp_y = y * imageHeight
        
        keypoint_shapes.append({
            "label": keypointsp[z],
            "points": [[kp_x, kp_y]],
            "group_id": int(v),
            "shape_type": "point",
            "flags": {}
            })
        z += 1

    return [bbox] + keypoint_shapes

def convert_yolo_txt_to_labelme_json(image_folder, yolo_txt_folder, output_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    
    for image_file in image_files:
        
        image_path = os.path.join(image_folder, image_file)
        print(image_path)
        shutil.copy(image_path,output_folder)
        # Retrieve image dimensions
        with Image.open(image_path) as img:
            imageWidth, imageHeight = img.size
        
        yolo_txt_path = os.path.join(yolo_txt_folder, os.path.splitext(image_file)[0] + '.txt')
        json_output_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + '.json')
        
        with open(yolo_txt_path, 'r') as yolo_file:
            lines = yolo_file.readlines()
            shapes = []

            for line in lines:
                
                parts = line.strip().split()
                class_id = parts[0]
                x_center, y_center, width, height = map(float, parts[1:5])
                keypoints = list(map(float, parts[5:]))
                
                shapes.extend(yolo_to_labelme(class_id, x_center, y_center, width, height, keypoints, imageWidth, imageHeight))
            
            
            labelme_format = {
                "version": "0.3.3",
                "flags": {},
                "shapes": shapes,
                "imagePath": image_file,
                "imageData": None,
                "imageHeight": imageHeight,
                "imageWidth": imageWidth
            }

            with open(json_output_path, 'w') as json_file:
                json.dump(labelme_format, json_file, indent=4)
            

# Parameters (adjust these)
convert_yolo_txt_to_labelme_json(IMAGE_FOLDER, YOLO_TEXT_FOLDER, OUTPUT_FOLDER)


