import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
# from tensorflow.keras.utils import to_categorical
# from PIL import Image
import os
import sys
import csv
from PIL import Image

new_size = 28

np.set_printoptions(threshold=sys.maxsize)

def resize_images(dir, label, new_size=28):
    """"Resize images in a dir, give them a class label and convert them into a matrix."""
    new_images = np.empty(shape=(len(os.listdir(dir)), new_size * new_size * 3 + 1)) # np.array([[]])
    index = 0

    print("Resizing in directory: " + dir)

    for filename in os.listdir(dir):
        print(".", end='')
        # Grab all images from the dir
        img = Image.open(dir + filename)

        # Check aspect ratio and crop to a square image accordingly
        width, height = img.size
        if width > height:
            new_width = height
            new_height = height
        else:
            new_width = width
            new_height = width
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        new_img = img.crop((left, top, right, bottom))

        # Resize the image and add it to the list
        new_img2 = new_img.resize((new_size, new_size))
        img_flat = np.array(new_img2)
        img_flat = img_flat.ravel()
        img_flat = np.append(img_flat, label)
        new_images[index] = img_flat
        index = index + 1

    print()
    return new_images

# Directories
dir_isic= r'd:\Users\Renzo\Desktop\Skin cancer ISIC The International Skin Imaging Collaboration/Train/'  # "C:/Data/Skin cancer ISIC The International Skin Imaging Collaboration/Test/actinic keratosis/"

# Sub-directories
dir_isic_akiec = dir_isic + r'actinic keratosis/'
dir_isic_bcc = dir_isic + r'basal cell carcinoma/'
dir_isic_bkl = dir_isic + r'pigmented benign keratosis/'
dir_isic_df = dir_isic + r'dermatofibroma/'
# dir_isic_nv = dir_isic + ONTBREEKT IN DATA
dir_isic_vasc = dir_isic + r'vascular lesion/'
dir_isic_mel = dir_isic + r'melanoma/'

# Flat image arrays
flat_isic_akiec = resize_images(dir_isic_akiec, 0, new_size)
flat_isic_bcc = resize_images(dir_isic_bcc, 1, new_size)
flat_isic_bkl = resize_images(dir_isic_bkl, 2, new_size)
flat_isic_df = resize_images(dir_isic_df, 3, new_size)
# 4 ONTBREEKT IN DATA
flat_isic_vasc = resize_images(dir_isic_vasc, 5, new_size)
flat_isic_mel = resize_images(dir_isic_mel, 6, new_size)


# Create a 2D array for all pixel values and their label
col_names = np.array([])

for i in range(0, new_size * new_size * 3):
    col_names = np.append(col_names, "pixel" + str(i).zfill(4))

col_names = np.append(col_names, 'label')
flat_imgs = np.concatenate((flat_isic_akiec, flat_isic_bcc, flat_isic_bkl, flat_isic_df, flat_isic_vasc, flat_isic_mel), axis=0)

# Convert to a CSV file
with open('ISIC_' + str(new_size) + '_' + str(new_size) + '_RGB.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerow(col_names)
    writer.writerows(flat_imgs)
