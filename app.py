import json

from flask import Flask, request
import pandas as pd
import pickle
from datetime import datetime
from waitress import serve

from utils.preprocessing.generic_preprocessing import convert_object_to_categorical

model = pickle.load(
    open(
        f'C:/Users/lallij/PycharmProjects/housePrices/model/model_{datetime.today().strftime("%Y-%m-%d")}.pkl',
        "rb",
    )
)


app = Flask(__name__)


@app.route("/housePrices", methods=["POST"])
def get_predictions():
    data = request.data
    data = json.loads(data)
    data = pd.json_normalize(data)
    data = convert_object_to_categorical(data)
    prediction = pd.DataFrame(model.predict(data), columns=["Prediction"])
    prediction["id"] = prediction.index
    prediction = prediction[["id", "Prediction"]]
    prediction = prediction.to_json(orient="records")
    return prediction


@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return "Ready"


if __name__ == "__main__":

    serve(app, port=8099)
