import os
import cv2
import albumentations as A
from tqdm import tqdm

transform = A.Compose([
    A.HorizontalFlip(p=0.5), 
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8), 
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5), 
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3), 
    A.OneOf([
        A.Blur(blur_limit=3, p=1.0),
        A.Sharpen(alpha=(0.2, 0.5), p=1.0),
    ], p=0.3),
])

input = "../raw"
output = "../raw_aug"
augX = 5  

if not os.path.exists(output):
    os.makedirs(output)
for folder in os.listdir(input):
    input_folder_path = os.path.join(input, folder)
    output_folder_path = os.path.join(output, folder)
    
    if not os.path.isdir(input_folder_path):
        continue
        
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    
    for img_name in tqdm(os.listdir(input_folder_path)):
        img_path = os.path.join(input_folder_path, img_name)
        image = cv2.imread(img_path)
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cv2.imwrite(os.path.join(output_folder_path, f"og_{img_name}"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        for i in range(augX):
            augmented = transform(image=image)["image"]
            aug_name = f"aug{i}_{img_name}"
            cv2.imwrite(os.path.join(output_folder_path, aug_name), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

print("stop")