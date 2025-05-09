from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_read_hello_name():
    response = client.get("/hello/Test")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello Test"}