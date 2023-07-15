from API.spark_api import API
import pytest
from app import create_app

@ pytest.fixture # On crée un fixture pour conserver la génération de l'objet
def api():
    return API()    # Crée une instance d'API


@pytest.fixture
def client():
    app = create_app({"TESTING": True})
    with app.test_client() as client:
        yield client

# Tests de fonctions définies dans le script
def test_score(api):
    # On va tester que la fonction renvoie bien une valeur entre 0 et 1.
    score = api.score()
    assert(x >= 0 | x <= 1 for x in score)


def test_prediction(api):
    # On teste que la fonction renvoie un 0 ou un 1
    prediction = api.prediction()
    assert( x == 0| x == 1 for x in prediction)


def test_bivar_graphs(api):
    # On teste que nos 3 listes sont bien complètes
    bivars = api.bivar_graphs()
    assert(len(bivars[0]) == len(bivars[1]) == len(bivars[2]))


def test_get_client_data(api, input_id=350290):
    # On teste la fonction de génération de toutes les données avec un id existant
    client_data = api.get_client_data(input_id)
    # On vérifie que le dictionnaire contient toutes les valeurs nécessaires
    assert(len(client_data) == 8)


# Test de fonctionnement de l'appli Flask 

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200


def test_should_return_hello_world(client):
    response = client.get('/')
    data = response.data.decode()
    assert data == 'Hello World'
