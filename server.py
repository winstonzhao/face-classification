import uuid
from flask import Flask, request
import pickle
from pred import predict_one_image, write_photo_with_attributes
import sys
import numpy as np
import json

app = Flask(__name__)

MODEL_PATH='face_model.pkl'
clf = None
labels = None

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        global clf
        global labels
        clf, labels = pickle.load(f, encoding='latin1')
    return

@app.route('/')
def hello_world():
    url = request.args.get("url")
    if not url:
        return json.dumps([])

    try:
        race_map = {0: 'Asian', 1: 'White', 2: 'Black'}
        pred, locs = predict_one_image(url, clf, labels, True)
        print(pred.shape[0], file=sys.stdout)
        output = []
        for row in pred.iterrows():
            if row[1]['Male'] >= 0.5:
                gender = 'Male'
            else:
                gender = 'Female'

            race = race_map[np.argmax(row[1][1:4])]
            text_showed = "{} {}".format(race, gender)
            if race == 'Asian':
                write_photo_with_attributes(url, pred, locs, "results/{}.jpg".format(str(uuid.uuid1())))
            output.append(text_showed)


        return json.dumps(output)
    except Exception as e:
        print(e, file=sys.stderr)
        return json.dumps([])
if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0')
