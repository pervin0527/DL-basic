import cv2
import numpy as np

def compute_mean_std(files):
    images = np.zeros((len(files), 224, 224, 3), dtype=np.uint8)
    for idx, file in enumerate(files):
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        images[idx, :, : ,:] = image
    
    mean_rgb = np.mean(images, axis=(0, 1, 2))
    std_rgb = np.std(images, axis=(0, 1, 2))

    return mean_rgb, std_rgb