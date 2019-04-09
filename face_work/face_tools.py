import random

import face_recognition
from PIL import Image
from keras.preprocessing import image
import cv2

def rescale_sub_face(sub_img, desired_size):
    im = image.img_to_array(sub_img)
    old_size = im.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)


def find_face(img):
    '''Find faces via pre-trained CNN'''
    __img = img
    _image = face_recognition.load_image_file(img)

    face_locations = face_recognition.face_locations(_image, number_of_times_to_upsample=0, model="cnn")
    found_faces = []

    for face_location in face_locations:
        top, right, bottom, left = face_location

        # increase box size to catch more face features
        top = int(top / 1.1)
        right = int(right * 1.1)
        bottom = int(bottom * 1.1)
        left = int(left / 1.1)
        _face = _image[top:bottom, left:right]

        # rescale to 350,350,3
        _face = rescale_sub_face(_face,350)

        new_fname = __img.replace('user_photo', 'user_photo_' + str(random.randint(1, 10)))
        cv2.imwrite('../pics/' + new_fname, _face)
        found_faces.append(new_fname)

    return found_faces
