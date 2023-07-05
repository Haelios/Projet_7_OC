import pandas as pd
from flask import Flask, jsonify, request
import joblib
import warnings
import shap

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

app = Flask(__name__)


class API:

    def __init__(self):
        self.data_train = pd.read_csv('Data/df_train_sample.csv')
        self.data_test = pd.read_csv('Data/df_test_sample.csv')
        self.model = joblib.load('Data/lgb_opti.pkl')

    def score(self):
        """
        Calcule le score client, qui correspond à la probabilité que le client ne rembourse pas le prêt,
        calculée grâce à notre modèle.
        :param index: Index de la ligne correspondant à l'identifiant entré par l'utilisateur.
        """

        score_client = self.model.predict_proba(self.data_test.drop(['TARGET', 'SK_ID_CURR'], axis=1))[:, 1].tolist()
        return score_client

    def prediction(self):
        """
        Retourne le résultat de la demande de tous les prêts. Ce résultat se trouve dans la colonne 'TARGET' du
        dataset. 0 = client accepté, 1 = client refusé
        """
        predict_client = self.model.predict(self.data_test.drop(['TARGET','SK_ID_CURR'], axis=1)).tolist()
        return predict_client

    def feature_importance(self, index):
        """
        Afficher les features importances locales du client afin d'analyser au mieux les éléments qui peuvent causer un
        refus.
        :param index: Position du client dans le dataset
        :return: Retourne les infos nécessaires à la création du graph.
        """
        explainer = shap.TreeExplainer(self.model, self.data_train.drop(['TARGET', 'SK_ID_CURR'], axis=1),
                                       model_output='probability')
        expected_val = explainer.expected_value
        shap_val = explainer.shap_values(self.data_train.drop(['TARGET', 'SK_ID_CURR'], axis=1).iloc[index],
                                         check_additivity=False).tolist()
        return expected_val, shap_val

    def clients_comparison(self, iden):
        """
        Retourne les données pour la comparaison entre le client et les prêts acceptés.
        Les variables étudiées sont les variables les plus importantes selon les features importances du modèle.
        :return: Les valeurs du client pour chaque var. La valeur moyenne pour les prêts acceptés par var.
        """
        ext_source1_client = self.data_test[self.data_test['SK_ID_CURR'] == iden]['EXT_SOURCE_1'][0]
        ext_source2_client = self.data_test[self.data_test['SK_ID_CURR'] == iden]['EXT_SOURCE_2'][0]
        ext_source3_client = self.data_test[self.data_test['SK_ID_CURR'] == iden]['EXT_SOURCE_3'][0]
        days_emp_client = -(self.data_test[self.data_test['SK_ID_CURR'] == iden]['DAYS_EMPLOYED'][0])

        ext_source1_mean = self.data_train[self.data_train['TARGET'] == 0]['EXT_SOURCE_1'].mean()
        ext_source2_mean = self.data_train[self.data_train['TARGET'] == 0]['EXT_SOURCE_2'].mean()
        ext_source3_mean = self.data_train[self.data_train['TARGET'] == 0]['EXT_SOURCE_3'].mean()
        days_emp_mean = -(self.data_train[self.data_train['TARGET'] == 0]['DAYS_EMPLOYED'].mean())

        ext_sources = [ext_source1_client, ext_source2_client, ext_source3_client,
                       ext_source1_mean, ext_source2_mean, ext_source3_mean]
        emp_days = [days_emp_client, days_emp_mean]
        return ext_sources, emp_days

    def bivar_graphs(self):
        """
        :return:
        """
        ext1 = self.data_test['EXT_SOURCE_1'].tolist()
        ext2 = self.data_test['EXT_SOURCE_2'].tolist()
        ext3 = self.data_test['EXT_SOURCE_3'].tolist()
        return [ext1, ext2, ext3]

    def get_client_data(self, input_id):
        """
        Fonction qui crée la réponse à la requête envoyée par le dashboard. Fait appel à toutes les fonctions
        définies précédemment pour extraire les données nécessaires grâce au dataset et au modèle.
        :param input_id: Identifiant entré par l'utilisateur.
        :return: Dictionnaire comprenant les données.
        """

        if input_id not in self.data_test['SK_ID_CURR'].tolist():
            return jsonify(input_id, {'error': "La requête a échoué, votre saisie est incorrecte."})
        else:
            # Prendre l'index du dataset correspondant à l'id entré
            ind = self.data_test[self.data_test['SK_ID_CURR'] == input_id].index

            # Données nécessaires pour le Force Plot de feature importance
            all_data_columns = self.data_test.columns.tolist()
            all_data_columns.remove('TARGET')
            all_data_columns.remove('SK_ID_CURR')
            all_data_values = self.data_test.iloc[ind][all_data_columns].values.tolist()
            expected_value, shap_values = self.feature_importance(ind)

            # Calculer le résultat de la demande de prêt
            predict = self.prediction()
            # Calculer le score (probabilité) du client
            predict_score = self.score()

            # Données pour l'affichage des graphs de comparaison entre clients
            ext_sources, days_emp = self.clients_comparison(input_id)

            # Données pour l'affichage des graphs d'analyse bivariée
            bivars = self.bivar_graphs()

        return {'index' : ind[0].tolist(),
                'all_client': [all_data_values, all_data_columns],
                'prediction_client': predict, 'score_client': predict_score,
                'expected_val': expected_value, 'shap_val': shap_values,
                'comp_graphs': [ext_sources, days_emp],
                'bivar_graphs': bivars }


# Chemin d'accès à la page utilisée pour la requête
@app.route('/prediction', methods=['POST'])
def predict_loan():
    # Générer la classe
    api = API()

    # Récupérer l'id via la requête API
    id_loan = request.json['loan_id']

    # Appel aux fonctions définies plus haut pour extraire les informations
    client_data = api.get_client_data(id_loan)

    # Renvoie la réponse contenant toutes infos nécessaires au dashboard
    response = {'index': client_data['index'],
                'prediction': client_data['prediction_client'],
                'score': client_data['score_client'],
                'shap_val': client_data['shap_val'],
                'expected_val': client_data['expected_val'],
                'all_client_val':  client_data['all_client'][0],
                'all_client_col': client_data['all_client'][1],
                'comp_graphs': client_data['comp_graphs'],
                'bivar_graphs': client_data['bivar_graphs']
                }

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
