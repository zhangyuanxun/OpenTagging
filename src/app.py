from flask import Flask,request,jsonify
from inference import get_model
from config import *
from flask import Flask, request,jsonify, render_template

app = Flask(__name__)
model = get_model()



@app.route('/')
def index():
	return render_template("index.html")



@app.route("/predict", methods=['POST'])
def predict():
    context = request.json["context"]
    attribute = request.json["attribute"]
    try:
        out = model.predict(context, attribute)
        return jsonify({"result": out})
    except Exception as e:
        return jsonify({"result": "Model Failed"})


@app.route("/process", methods=['POST'])
def process():
    context = request.form["context"]
    attribute = request.form["attribute"]
    try:
        out = model.predict(context, attribute)
        return jsonify({"errcode": 1, "result": out})
    except Exception as e:
        return jsonify({"errcode": 0})


if __name__ == "__main__":
    app.run('0.0.0.0', port=PORT_NUMBER)

