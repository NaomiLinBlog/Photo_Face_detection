# -*- coding: utf-8 -*-        
import os
import time
import random

import numpy
from cv2 import cv2 as cv2
from PIL import Image
from PIL import ImageDraw
import face_recognition

path = "LCC_FASD/evaluation"
#path = "LCC_FASD/development/spoof"

def image_show(pil_image):
    #img = cv2.cvtColor(numpy.asarray(pil_image),cv2.COLOR_RGB2BGR)
    img = numpy.asarray(pil_image)

    cv2.imshow('tmp', img)
    c = cv2.waitKeyEx(0) % 256

    if c == ord('q') or c ==  27:
        exit()
    elif cv2.getWindowProperty('tmp', 1) == -1:
        exit()
    else:
        print('you press ' + str(c))

def main():
    files = os.listdir(path)
    #random.shuffle(files)
    
    for file_name in files:
        full_path = path + "/" + file_name
        print(full_path)

        #image = face_recognition.load_image_file(full_path)
        image = cv2.imread(full_path)
        #print(img.shape)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings):
            pil_image = Image.fromarray(image).convert('RGB')
            
            # Create a Pillow ImageDraw Draw instance to draw with
            draw = ImageDraw.Draw(pil_image)
            
            # Draw a box around the face using the Pillow module
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

            image_show(pil_image)

if __name__ == '__main__':
    main()

    