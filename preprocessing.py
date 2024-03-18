import numpy as np
import random
import math
from general_utils import bb_util, img_util, file_util
from PIL import Image

def generate_datalist(input_path):
    '''
    This function generates the list of images and the corresponding annotations
    img_file: path to the image file
    return : the image
    '''
    output_list = file_util.make_list(input_path)
    output_list = file_util.read_annot_list(output_list)
    data_list = []
    for item in output_list:
        img = img_util.read_image((item[0]))
        (img_height, img_width, _) = np.shape(img)
        bb_list = bb_util.convert_yolo_bb_to_abs([item[1][0][1]], [img_width, img_height])
        # crop the img with double the size of the bb
        img = img_util.crop_obj_double_bb(img, bb_list[0])
        # calculate the diagonal size of the bb
        diag_size = int(math.sqrt(bb_list[0][2] ** 2 + bb_list[0][3] ** 2)) + 20 #added a padding of 20 pixels
        # if the bb is too small, skip the sample
        if (np.shape(img)[0] == 0 or np.shape(img)[1] == 0):
            print("Error !!! problematic sample : ", item[0])
        else:
            print("The sample is imported ... ", item[0], "... OK.")
            data_list.append([img, diag_size])

    print("Number of generated samples : ", np.shape(data_list))
    return data_list


def generate_dataset(opt, data_list, desired_size=144):
    '''
    This function generates the rotated images and the corresponding labels
    data_list : list of images and the corresponding annotations
    nc : number of classes (angles) in degrees
    desired_size : the size of the output image
    return : the rotated images and the corresponding labels
    '''
    dataset_img = []
    dataset_label = []
    for item in data_list:
        img = np.array(item[0])
        bb_diag_size = item[1]

        # generate all the angles
        sample_angles = range(0, opt.n_class, 1)
        for sample_angle in sample_angles:
            # generate a valid rotated image based on sample_angle and
            new_img = img_util.generate_rotated_image(img, sample_angle, crop_center=False, crop_largest_rect=False)
            # resize the image to the diagonal size of the bb in a square shape
            new_img = img_util.resize_with_border(new_img, desired_size=bb_diag_size, resize=False)
            # crop the image to the diagonal size of the bb
            new_img = img_util.crop_image_from_center(new_img, (bb_diag_size, bb_diag_size))
            # resize the image to a squared shape with the desired size
            new_img = img_util.resize_with_border(new_img, desired_size=desired_size, resize=True)
            dataset_img.append(new_img)
            dataset_label.append(sample_angle)

    # visualizing the generated samples in RGB format
    N = len(dataset_img)
    random.seed(42)
    sampled_image_indecies = random.sample(range(N), 100)
    # sampled_image_indecies = range(0,101)
    sampled_images = [Image.fromarray(dataset_img[i]) for i in sampled_image_indecies]
    sampled_labels = [dataset_label[i] for i in sampled_image_indecies]
    img_util.display_image_grid(sampled_images, 10, sampled_labels, output_path=opt.model_path)

    # converting all images to a np array
    data = np.stack(dataset_img, axis=0)

    return data, dataset_label


if __name__ == '__main__':

    l = generate_datalist()
    generate_dataset(l)
