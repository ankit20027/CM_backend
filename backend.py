import pandas as pd
import pickle
import spacy
import requests
from nlp import NLP
from fastapi import FastAPI

app = FastAPI()
app.my_nlp = NLP(pd.read_csv("Training.csv"), spacy.load("en_core_web_sm"))
app.model = pickle.load(open('model.pickle', 'rb'))

def gptSearch(gpt_query: str):
    gpt_url = "https://chatgpt-42.p.rapidapi.com/gpt4"
    payload = {
        "messages": [
            {
                "role": "user",
                "content": gpt_query
            }
        ],
        "tone": "Balanced"
    }
    headers = {
        "content-type": "application/json",
        "X-RapidAPI-Key": "52aff55dd2msh19213cf6e447630p1d05ffjsn793803be2081",
        "X-RapidAPI-Host": "chatgpt-42.p.rapidapi.com"
    }
    return requests.post(gpt_url, json=payload, headers=headers)


@app.get('/getDignostics')
def getDignostics(query: str):
    x_pred = app.my_nlp.get_encodings(query)
    if (x_pred == -1):
        return "No Dignosis"
    disease = app.model.predict([x_pred])[0]
    
    early_symptoms = gptSearch(f"Early-stage symptoms of {disease}").content
    precautions = gptSearch(f"Precautions for {disease}").content
    medications = gptSearch(f"Medications for {disease}").content

    return { "Disease": disease, 
             "GPTResponse": {
                 "early_symptoms": early_symptoms,
                 "precautions": precautions,
                 'medications': medications
             } }