import os
import csv
import random
import numpy as np

from general_utils import bb_util, img_util


def read_annot(file_path):
    '''
    This function reads the annotation file
    file_path : path to the annotation file
    return : list of obj class and annotations
    '''
    annot_list = []
    with open(file_path) as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            row = [float(x) for x in row]
            annot_list.append((row[0], row[1:]))
    f.close()
    return annot_list

def read_annot_list(path_list):
    '''
    This function reads a list of annotation files
    path_list : list of images and annotation files path
    return : tuple of image path and the corresponding annotation list
    '''
    img_path_annot_list = []
    for item in path_list:
        (img_path, annot_path) = item
        img_path_annot_list.append((img_path, read_annot(annot_path)))
    return img_path_annot_list


def make_list(dir_path):
    '''
    This function makes a list of images and the corresponding annotation files path
    dir_path : path to the directory containing images(*.png) and annotations(*.txt)
    return : list of images and the corresponding annotation files path
    '''
    try:
        list_annot = []
        list_img = []
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for name in files:
                if name.endswith(".txt"):
                    annot_file_path = os.path.join(root, name)
                    img_file_path = annot_file_path.replace("annot_", "").replace(".txt", ".png")
                    if os.path.exists(img_file_path) and os.path.exists(annot_file_path):
                        list_img.append(img_file_path)
                        list_annot.append(annot_file_path)

        if len(list_img) != len(list_annot):
            print("Error : Length of annotation and images list must be equal!!!")
            return []
        else:
            final_list = [(img, annot) for img, annot in zip(list_img, list_annot)]
            return final_list
    except:
        print("Error : The path {} doesn't exist!!!".format(dir_path))

