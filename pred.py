from __future__ import print_function
import argparse
import os
import face_recognition
import numpy as np
import sklearn
import pickle
from face_recognition import face_locations
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2
import pandas as pd
from skimage import io
import urllib.request

# we are only going to use 4 attributes
COLS = ['Male', 'Asian', 'White', 'Black']
N_UPSCLAE = 1
def extract_features(img_path, url=False):
    """Exctract 128 dimensional features
    """
    X_img = io.imread(img_path) if url else face_recognition.load_image_file(img_path)
    locs = face_locations(X_img, number_of_times_to_upsample = N_UPSCLAE)
    if len(locs) == 0:
        return None, None
    face_encodings = face_recognition.face_encodings(X_img, known_face_locations=locs)
    return face_encodings, locs

def predict_one_image(img_path, clf, labels, url=False):
    """Predict face attributes for all detected faces in one image
    """
    face_encodings, locs = extract_features(img_path, url)
    if not face_encodings:
        return None, None
    pred = pd.DataFrame(clf.predict_proba(face_encodings),
                        columns = labels)
    pred = pred.loc[:, COLS]
    return pred, locs
def draw_attributes(img_path, df):
    """Write bounding boxes and predicted face attributes on the image
    """
    resp = urllib.request.urlopen(img_path)
    img = np.asarray(bytearray(resp.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    # img  = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    race_map = { 0: 'Asian', 1: 'White', 2: 'Black' }
    for row in df.iterrows():
        top, right, bottom, left = row[1][4:].astype(int)
        if row[1]['Male'] >= 0.5:
            gender = 'Male'
        else:
            gender = 'Female'

        race = race_map[np.argmax(row[1][1:4])]
        text_showed = "{} {}".format(race, gender)

        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        img_width = img.shape[1]
        cv2.putText(img, text_showed, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    return img



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_url', type=str,
                         required = True,
                        help='input url')
    parser.add_argument('--output_dir', type=str,
                        default='results/',
                        help='output directory to save the results (default: results/')
    parser.add_argument('--model', type=str,
                        default='face_model.pkl',
                        help='path to trained model (default: face_model.pkl)')

    args = parser.parse_args()
    output_dir = args.output_dir
    image_url = args.img_url
    model_path = args.model

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    # load the model
    with open(model_path, 'rb') as f:
        clf, labels = pickle.load(f, encoding="latin1")
    pred, locs = predict_one_image(image_url, clf, labels, True)
    if not locs:
        print("No people found!")
        return
    locs = pd.DataFrame(locs, columns = ['top', 'right', 'bottom', 'left'])
    df = pd.concat([pred, locs], axis=1)
    img = draw_attributes(image_url, df)
    cv2.imwrite(os.path.join(output_dir, "image.jpg"), img)
    output_csvpath = os.path.join(output_dir,'image.csv')
    df.to_csv(output_csvpath, index = False)

if __name__ == "__main__":
    main()
