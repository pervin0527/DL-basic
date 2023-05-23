import cv2
import torch
import numpy as np
from tqdm import tqdm
from glob import glob

def get_mean_rgb(img_path):
    total_pixels = 0
    sum_red, sum_green, sum_blue = 0, 0, 0

    images = glob(f"{img_path}/*/*")
    for idx in tqdm(range(len(images))):
        image = cv2.imread(images[idx])
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0

        sum_red += np.sum(image[:, :, 2])
        sum_green += np.sum(image[:, :, 1])
        sum_blue += np.sum(image[:, :, 0])

        total_pixels += image.shape[0] * image.shape[1]
    
    mean_red = sum_red / total_pixels
    mean_green = sum_green / total_pixels
    mean_blue = sum_blue / total_pixels

    return mean_red, mean_green, mean_blue

def get_std_rgb(img_path, mean_rgb):
    total_pixels = 0
    sum_squared_diff_red = 0
    sum_squared_diff_green = 0
    sum_squared_diff_blue = 0

    images = glob(f"{img_path}/*/*")
    for idx in tqdm(range(len(images))):
        image = cv2.imread(images[idx])
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0

        sum_squared_diff_red += np.sum((image[:, :, 2] - mean_rgb[0]) ** 2)
        sum_squared_diff_green += np.sum((image[:, :, 1] - mean_rgb[1]) ** 2)
        sum_squared_diff_blue += np.sum((image[:, :, 0] - mean_rgb[2]) ** 2)

        total_pixels += image.shape[0] * image.shape[1]

    std_red = np.sqrt(sum_squared_diff_red / total_pixels)
    std_green = np.sqrt(sum_squared_diff_green / total_pixels)
    std_blue = np.sqrt(sum_squared_diff_blue / total_pixels)

    return std_red, std_green, std_blue

class Scale_Jitter:
    def __init__(self, min_size=256, max_size=512):
        self.scale_factor = np.random.uniform(0.5, 2.0)
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image):
        image = image.numpy()
        ## Rescale the image while keeping the aspect ratio
        height, width = image.shape[:2]
        new_height = int(self.scale_factor * height) ## scale_factor 만큼 이미지가 커지거나 작아짐.
        new_width = int(self.scale_factor * width)
        if height < width: ## aspect ratio 유지.
            new_width = int(new_height * (width / height))
        else:
            new_height = int(new_width * (height / width))
        image = cv2.resize(image, (new_width, new_height))

        ## Ensure that the resulting image size is within the desired range of 256 to 512 pixels
        if new_height < 256 or new_width < 256:
            image = cv2.resize(image, (256, 256))
        elif new_height > 512 or new_width > 512:
            image = cv2.resize(image, (512, 512))

        ## Randomly crop the image to 224 x 224
        crop_size = (224, 224)
        y = np.random.randint(0, image.shape[0] - crop_size[0] + 1) ## 좌상단 x, y 지정.
        x = np.random.randint(0, image.shape[1] - crop_size[1] + 1)
        image = image[y:y+crop_size[0], x:x+crop_size[1]] 

        return torch.tensor(image)