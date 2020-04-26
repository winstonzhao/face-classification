from flask import Flask
import pickle
from pred import predict_one_image
import sys

app = Flask(__name__)

MODEL_PATH='face_model.pkl'

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        clf, labels = pickle.load(f, encoding='latin1')
    pred, locs = predict_one_image("https://imagesvc.meredithcorp.io/v3/mm/image?url=https%3A%2F%2Fstatic.onecms.io%2Fwp-content%2Fuploads%2Fsites%2F20%2F2020%2F01%2Fkobe-bryant-family-1.jpg", clf, labels, True)
    print(pred, file=sys.stdout)
    print(locs, file=sys.stdout)
    return

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0')
