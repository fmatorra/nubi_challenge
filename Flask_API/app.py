import torch
import flask
import time
from flask import Flask
from flask import request
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
import pandas as pd
import numpy as np



app = Flask(__name__)  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def is_smart(text):
    PATH = '../transformers_classification/models/NubiTvSmartOther'
    tokenizer = AutoTokenizer.from_pretrained(PATH)
    model = AutoModelForSequenceClassification.from_pretrained(PATH, num_labels=3).to(device)
    categories = ['Televisión', 'Smart', 'Otro']
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    del tokenizer
    del model
    torch.cuda.empty_cache()
    return categories[probs.argmax()]

def product(text):
    PATH = '../transformers_classification/models/NubiTvCel'
    tokenizer = AutoTokenizer.from_pretrained(PATH)
    model = AutoModelForSequenceClassification.from_pretrained(PATH, num_labels=3).to(device)
    categories = ['Celular', 'Televisión', 'Otro']
    # prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    # perform inference to our model
    outputs = model(**inputs)
    # get output probabilities by doing softmax
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label
    del tokenizer
    del model
    torch.cuda.empty_cache()
    return categories[probs.argmax()]

def spacy_ent(text, product):
    if product == 'tv':
        nlp = spacy.load('../spacy_ner/TVs/models/NUBI-TV-NER-1077/model-best')
    else:
        nlp = spacy.load('../spacy_ner/cel/models/NUBI-CEL-NER-1187/model-best')

    doc = nlp(text)
    ner = ''
    for ent in doc.ents:
        ner = ent.text
        break
    if not ner:
        ner = 'Marca no encontrada'
    del nlp
    del doc
    return ner


@app.route("/predict")
def predict():
    title = request.args.get("title")
    start_time = time.time()
    producto = product(title)
    if producto == 'Televisión':
        caracteristica = is_smart(title)
        marca = spacy_ent(title, 'tv')
        response = {}
        response["response"] = {
            "producto": str(producto),
            "marca": str(marca),
            "caracteristica": str(caracteristica),
            "time_taken": str(time.time() - start_time),
        }
    else:
        marca = spacy_ent(title, 'cel')
        response = {}
        response["response"] = {
            "producto": str(producto),
            "marca": str(marca),
            "time_taken": str(time.time() - start_time),
        }
    return flask.jsonify(response)

@app.route("/report")
def report():
    marca = request.args.get("marca")
    producto = request.args.get("producto")
    start_time = time.time()
    final_db = pd.read_csv("../final_database_cel_tv.csv")

    if marca:
        tipo = "Marca"
        search = marca
        cantidad = final_db.marcas_corregidas.str.count(marca).sum()
        porcentaje = (cantidad * 100) / len(final_db)
        sellers = final_db[final_db['marcas_corregidas'] == marca]['seller_id'].nunique()

    elif producto:
        tipo = "Producto"
        search = producto
        cantidad = final_db.Producto.str.count(str(producto)).sum()
        porcentaje = (cantidad * 100) / len(final_db)
        sellers = final_db[final_db['Producto'] == producto]['seller_id'].nunique()

    response = {}
    response["response"] = {
            "tipo": str(tipo),
            "busqueda": str(search),
            "cantidad": int(cantidad),
            "porcentaje": int(porcentaje),
            "vendedores": int(sellers),
            "time_taken": str(time.time() - start_time),
    }
    del final_db
    return flask.jsonify(response)




if __name__ == "__main__":
    app.run(debug=True)