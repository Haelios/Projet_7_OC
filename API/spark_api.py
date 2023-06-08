import io
import string
import time
import gc
import os
import numpy as np
import pandas as pd
from contextlib import contextmanager
from flask import Flask, jsonify, request
import lightgbm as lgb

app = Flask(__name__)

# Importer le modèle optimisé
model = lgb.Booster(model_file='models/lgbm_opti.txt')

# Importer les données pré-traitées
data = pd.read_csv('df_train.csv')

@app.route('/')
def main_page() :
    iddg = str(data['SK_ID_CURR'][0])
    return iddg

@app.route('/loan_prediction', methods=['POST'])
def predict_loan() :
    # Récupérer l'id via la requête API
    id_loan = request.json['loan_id']

    # Transmet les données du client
    client_data = data[data['SK_ID_CURR']==id_loan].to_dict(orient='list')
    if client_data==np.NaN:
        return jsonify(id_loan, {'error': "La requête a échoué, votre saisie."})
    else:
        prediction_client = client_data['TARGET']

        # Renvoie la prédiction pour le client
        response = {'prediction' : prediction_client,
                    'data' : client_data
                    }

        return jsonify(response)


if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0')
