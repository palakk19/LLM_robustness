import requests

try:
    r = requests.get("http://ollama.warhol.informatik.rwth-aachen.de")
    print("Server reachable:", r.status_code)
except:
    print("Server NOT reachable")