import cv2
import joblib
import argparse
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
# from sklearn.externals import joblib

import face_alignment
# from mtcnn.mtcnn import MTCNN

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from crop_face import get_faces
from generate_dataset import calculate_distances

face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

def show_image(img, bbox, label):
    # img = cv2.imread(img_path)
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 4)
    print(label)
    if label == 0:
        print("%d : No Syndrome" % (label))
        cv2.putText(img, "Label 0", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
    elif label == 1:
        print("%d : Has Angelman Syndrome" % (label))
        cv2.putText(img, "Label 1", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
    elif label == 2:
        print("%d : Has Apert Syndrome" % (label))
        cv2.putText(img, "Label 2", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
    elif label == 3:
        print("%d : Has Cornelia De Lange Syndrome" % (label))
        cv2.putText(img, "Label 3", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
    elif label == 4:
        print("%d : Has Down Syndrome" % (label))
        cv2.putText(img, "Label 4", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
    elif label == 5:
        print("%d : Has FragileX Syndrome" % (label))
        cv2.putText(img, "Label 5", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
    elif label == 6:
        print("%d : Has Williams Syndrome" % (label))
        cv2.putText(img, "Label 6", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
    
    return img
    # if label==0:
    #     cv2.putText(img, "Label 0", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
    # elif label==1:
    #     cv2.putText(img, "Label 1", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
    
    
    # cv2.imwrite("")
    # cv2.imshow("PREDICTION RESULT", img)
    # cv2.waitKey(0)

def predict_image(img_path, n_pca, pca_path, model_path):
    # read image & pca & model
    print("[INFO] Loading image & models")
    img = cv2.imread(img_path)
    # img = cv2.resize(img, (300,300))
    # print(img)
    pca = joblib.load(pca_path)
    model = joblib.load(model_path)
    # # get bbox
    # print("[INFO] Preprocess image")
    # faces = face_cascade.detectMultiScale(img)
    # print(faces)
    # for x,y,w,h in faces:
    #     bbox = (x,y,w,h)
    #     crop_face = img[y:y+h, x:x+w]
    # print(crop_face)
    fa_2d = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, face_detector='sfd')
    preds = fa_2d.get_landmarks(img)
    # print(preds)
    dist = calculate_distances(preds[0])
    # print(dist)
    infer = []
    infer.append(dist)
    initial_feature_names = ["%s-%s" % (str(i), str(j)) for i in range(1,69) for j in range(i+1,69)]
    # initial_feature_names.append("label")
    infer_df = pd.DataFrame(data=infer, columns=initial_feature_names)
    # print(len(infer_df))
    # transform image using PCA model
    print("[INFO] Transform image using PCA")
    new_infer = pca.transform(infer_df)
    new_infer = pd.DataFrame(new_infer, columns=["%s" % i for i in range(n_pca)])
    # predict
    print("[INFO] Start predicting image")
    label = model.predict(new_infer)
    # show
    print("[INFO] Done!")
    return label
    # img_result = show_image(img, bbox, label)
    # return img_result
    # return bbox, label

if __name__ == "__main__":
    # get img path
    parser = argparse.ArgumentParser(description="Predict 1 image")
    parser.add_argument("--n-pca", type=int, help="n_components value for PCA.")
    parser.add_argument("--pca", type=str, help="path to PCA model.")
    parser.add_argument("--model", type=str, help="path to classifier model.")
    parser.add_argument("--img-path", type=str, help="path to source image.")
    args = parser.parse_args()
    n_pca = args.n_pca
    pca_path = args.pca
    model_path = args.model
    img_path = args.img_path
    bbox, label = predict_image(img_path, n_pca, pca_path, model_path)
    show_image(img_path, bbox, label)