import cv2 as cv
import numpy as np
import math
import os

def read(image_path,labels):
    image = cv.imread(image_path)
    image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    height,width = image[:2]
    
    image = cv.resize(image,(448,448))
    image = image/255

    label_matrix = np.zeros([7,7,30],dtype=float)

    for label in labels:
        label = label.rstrip("\n").split(" ")
        label = np.array(label)

        cls = int(label[0])
        b_x = float(label[1])
        b_y = float(label[2])
        b_h = float(label[3])
        b_w = float(label[4])

        loc = [math.floor(7 * b_x),math.floor(7 * b_y)]
        if label_matrix[loc[0],loc[1],0] == 0:
            label_matrix[loc[0],loc[1],0] = 1
            label_matrix[loc[0],loc[1],1:5] = [b_x, b_y, b_h, b_w]
            label_matrix[loc[0],loc[1],5+cls] = 1
    
    return image, label_matrix


def load(image_dir_path,label_dir_path):
    images_name = os.listdir(image_dir_path)
    labels_name = os.listdir(label_dir_path)
    images_mat= []
    labels_mat = []
    
    for index,image_name in enumerate(images_name):
        image_path = os.path.join(image_dir_path,image_name)

        with open(os.path.join(label_dir_path,labels_name[index]),"r") as file:
            labels = file.readlines()

        i , j = read(image_path,labels)
        images_mat.append(i)
        labels_mat.append(j)

    images_mat = np.array(images_mat)
    labels_mat = np.array(labels_mat)

    print("images_mat : ",images_mat.shape)
    print("labels_mat : ",labels_mat.shape)

load("./image","./label")




 