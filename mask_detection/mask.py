import os
import sys
import random
import numpy as np
import PIL

Image, ImageFile

import face_recognition

BULE_IMAGE_PATH = 'data/blue_mask.jpg'


def create_mask(image_path):
    pic_path = image_path
    mask_path = BULE_IMAGE_PATH
    show = False
    model = 'hog'
    FaceMasker(pic_path, mask_path, show, model).mask()


class FaceMasker:
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')

    def __init__(self, face_path, mask_path, show=False, model='hog'):
        self.face_path = face_path
        self.mask_path = mask_path
        self.show = show
        self.model = model

    def mask(self):
        face_image_np = face_recognition.load_image_file(self.face_path)
        face_locations = face_recognition.face_locations(face_image_np, model=self.model)
        face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations, model=self.model)

        self._face_img = Image.fromarray(self.face_path)
        self._mask_img = Image.open(self.mask_path)

        found_face = False

        for face_landmark in face_landmarks:
            skip = False
            for facial_feature in self.KEY_FACIAL_FEATURES:
                if facial_feature not in face_landmark:
                    skip = True
                    break

            if skip:
                continue

        if found_face:
            if self.show():
                self._face_img.show()

            self._face_img.save('masked.jpg')


if __name__ == '__main__':
    create_mask()
