import cv2
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from glob import glob
from random import random


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

class ScaleJitterTransform(object):
    def __init__(self, min_size=256, max_size=512, crop_size=(224, 224), p=0.5):
        self.min_size = min_size
        self.max_size = max_size
        self.crop_size = crop_size
        self.p = p

    def __call__(self, image):
        if self.p < random():
            return scale_jitter(image, self.min_size, self.max_size, self.crop_size)
        else:
            return image

def scale_jitter(image, min_size=256, max_size=512, crop_size=(224, 224)):
    scale_factor = np.random.uniform(0.5, 2.0)

    # Rescale the image while keeping the aspect ratio
    height, width = image.shape[:2]
    new_height = int(scale_factor * height)
    new_width = int(scale_factor * width)
    if height < width:
        new_width = int(new_height * (width / height))
    else:
        new_height = int(new_width * (height / width))
    image = cv2.resize(image, (new_width, new_height))

    # Ensure that the resulting image size is within the desired range
    if new_height < min_size or new_width < min_size:
        image = cv2.resize(image, (min_size, min_size))
    elif new_height > max_size or new_width > max_size:
        image = cv2.resize(image, (max_size, max_size))

    # Randomly crop the image
    y = np.random.randint(0, image.shape[0] - crop_size[0] + 1)
    x = np.random.randint(0, image.shape[1] - crop_size[1] + 1)
    image = image[y:y+crop_size[0], x:x+crop_size[1]]

    return image

def data_visualize(dataloader, mean_rgb, std_rgb, img_size):
    reverse_transform = transforms.Compose([
        transforms.Normalize(mean=[-m/s for m, s in zip(mean_rgb, std_rgb)],
                             std=[1/s for s in std_rgb]),
        transforms.Resize(size=img_size, antialias=False),
    ])

    for batch in dataloader:
        preprocessed_data = batch[0]

        restored_data = reverse_transform(preprocessed_data) ## (batch_size, 3, 224, 224)
        for idx, img in enumerate(restored_data):
            img = np.transpose(img.numpy(), (1, 2, 0))
            img = img[:, :, ::-1]

            cv2.imshow(str(idx), img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()