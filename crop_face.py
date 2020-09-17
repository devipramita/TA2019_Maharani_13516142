import os
import cv2
import glob
import numpy
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN

face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

def get_filenames(path):
    paths_result = []
    for filename in glob.glob(path):
        # print(filename)
        paths_result.append(filename)
    return paths_result

def get_faces(filenames, target_path):
    for filename in filenames:
        print(filename)
        img = cv2.imread(filename)
        # equ = cv2.equalizeHist(img)
        R, G, B = cv2.split(img)
        output1_R = cv2.equalizeHist(R)
        output1_G = cv2.equalizeHist(G)
        output1_B = cv2.equalizeHist(B)
        equ = cv2.merge((output1_R, output1_G, output1_B))
        faces = face_cascade.detectMultiScale(equ)
        for x,y,w,h in faces:
            x1 = x
            x2 = x+w+10
            y1 = y
            y2 = y+h+10
            crop_face = img[y1:y2, x1:x2]
            crop_face = cv2.resize(crop_face, (300,300))
            folder = filename.split('/')
            name = folder[2].split('\\')
            cv2.imwrite(target_path+name[1], crop_face)
    print("[INFO] Done get faces on", target_path)

# def get_faces(filenames, target_path):
#     for filename in filenames:
#         print(filename)
#         im = cv2.imread(filename)
#         detector = MTCNN()
#         faces = detector.detect_faces(im)
#         for face in faces:
#             x,y,w,h = face["box"]
#             crop_face = im[y:y+h, x:x+w]
#             crop_face = cv2.resize(crop_face, (300,300))
#             folder = filename.split('/')
#             name = folder[2].split('\\')
#             cv2.imwrite(target_path+name[1], crop_face)
#     print("[INFO] Done get faces on", target_path)

def crop_faces():
    for x in ["angelman","apert","cdl","down","fragilex","williams","normal"]:
        # get filenames
        print("[INFO] Get filenames")
        paths_syndrome = get_filenames("../0. dataset/he_"+x+"/*.jpg")
        print("%s : %d" % (x, len(paths_syndrome)))
        # crop face on each file
        if not os.path.exists("crop_faces/cascade/"+x+"/"):
            os.makedirs("crop_faces/cascade/"+x+"/")
        print("[INFO] Get faces %s" % x)
        get_faces(paths_syndrome, "crop_faces/cascade/"+x+"/")
        # if not os.path.exists("crop_faces/mtcnn/normal/"):
        #     os.makedirs("crop_faces/mtcnn/normal/")
        # print("[INFO] Get faces: normal")
        # get_faces(paths_normal, "crop_faces/mtcnn/normal/")
 
if __name__ == "__main__":
    crop_faces()
    