import requests

def get_soil_ph(lat, lon):
    """
    Fetch soil pH (0–5 cm depth) from SoilGrids API for given lat/lon.
    """
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = {
        "lon": 25.543123,
        "lat":  83.980642,
        "property": "phh2o",
        "depth": "0-5cm",
        "value": "mean"
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"SoilGrids API error: {response.status_code}")

    soil_data = response.json()
    
    # Extract soil pH at 0–5 cm
    ph_layer = soil_data["properties"]["layers"][0]
    ph_value = ph_layer["depths"][0]["values"]["mean"]

    return ph_value

if __name__ == "__main__":
    # Example: Cuttack, Odisha (lat=20.4625, lon=85.8828)
    lat, lon = 20.4625, 85.8828
    ph = get_soil_ph(lat, lon)
    print(f"Soil pH (0–5 cm) at Buxar: {ph}")
