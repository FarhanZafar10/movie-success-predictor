import requests

sample_movie = {
    "budget": 150000000,
    "runtime": 148,
    "release_year": 2024,
    "release_month": 7,
    "genres": ["Action", "Science Fiction", "Adventure"],
    "num_production_companies": 3,
    "num_production_countries": 2,
    "num_spoken_languages": 2
}

response = requests.post('http://127.0.0.1:5001/predict', json=sample_movie)
print(response.json())