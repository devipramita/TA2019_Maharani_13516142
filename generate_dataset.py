import cv2
import glob
import math
import numpy as np
import pandas as pd
import face_alignment
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def get_faces(source_path):
    result = []
    for path in glob.glob(source_path):
        print(path)
        img = cv2.imread(path)
        result.append(img)
    return result

def extract_feature(model, list_image):
    preds_result = []
    idx = 0
    for img in list_image:
        print("feature img:", idx)
        preds_result.append(model.get_landmarks(img))
        idx += 1
    return preds_result

def euclidean_distance(p1, p2):
    distance = math.sqrt(math.pow((p1[0] - p2[0]),2) + math.pow((p1[1] - p2[1]),2))
    return distance

def calculate_distances(pred):
    list_distance = []
    for i in range(len(pred)):
        for j in range(i+1, len(pred), 1):
            distance = euclidean_distance(pred[i], pred[j])
            list_distance.append(distance)
    return list_distance

def get_all_distances(list_preds, label):
    df = []
    for i in range(len(list_preds)):
        print("preds:", i)
        distances = calculate_distances(list_preds[i][0])
        distances.append(label)
        df.append(distances)
    return df

def generate_dataset():
    # get images
    print("[INFO] Get images")
    images_syndrome = get_faces("crop_faces/mtcnn/syndrome/*.jpg")
    images_normal = get_faces("crop_faces/mtcnn/normal/*.jpg")
    # images_syndrome = get_faces("../0. dataset/0.syndrome/*.jpg")
    # images_normal = get_faces("../0. dataset/0.normal/*.jpg")
    # extract facial features
    print("[INFO] Load alignment model")
    fa_2d = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, face_detector='sfd')
    print("[INFO] Extract features: syndrome")
    preds_syndrome = extract_feature(fa_2d, images_syndrome)
    print(len(preds_syndrome))
    print("[INFO] Extract features: normal")
    preds_normal = extract_feature(fa_2d, images_normal)
    print(len(preds_normal))
    # get all feature distances
    print("[INFO] Get distances: syndrome")
    dist_syndrome = get_all_distances(preds_syndrome, 1)
    print("[INFO] Get distances: normal")
    dist_normal = get_all_distances(preds_normal, 0)
    dist = dist_syndrome + dist_normal
    # into DataFrame
    print("[INFO] Turn into DataFrame")
    initial_feature_names = ["%s-%s" % (i, j) for i in range(1,69) for j in range(i+1,69)]
    initial_feature_names.append("label")
    df = pd.DataFrame(data=dist, columns=initial_feature_names)
    df = shuffle(df)
    train_df, test_df = train_test_split(df, test_size=0.3)
    # save DataFrame to csv
    df.to_csv("dataset/mtcnn_df.csv", encoding="utf-8", index=False)
    train_df.to_csv("dataset/mtcnn_train_df.csv", encoding="utf-8", index=False)
    test_df.to_csv("dataset/mtcnn_test_df.csv", encoding="utf-8", index=False)
    print("[INFO] Done! New csv files on 'dataset' folder: mtcnn_df.csv, mtcnn_train_df.csv, mtcnn_test_df.csv")

if __name__ == "__main__":
    generate_dataset()