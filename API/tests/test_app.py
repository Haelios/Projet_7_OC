import pytest
from app import create_app

@pytest.fixture
def client():
    app = create_app({"TESTING": True})
    with app.test_client() as client:
        yield client

# Test de fonctionnement de l'appli Flask 

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200


def test_should_return_hello_world(client):
    response = client.get('/')
    data = response.data.decode()
    assert data == 'Hello World'
