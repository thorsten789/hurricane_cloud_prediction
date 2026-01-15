import json
from datetime import datetime

def save_json_with_timestamp(model_name, param1, param2, param3):
    """
    Speichert eine JSON-Datei mit Zeitstempel im Dateinamen.
    
    Args:
        model_name: Name des Modells
        param1: Erster Parameter
        param2: Zweiter Parameter
        param3: Dritter Parameter
    
    Returns:
        str: Pfad zur gespeicherten Datei
    """
    # Aktuelles Datum und Uhrzeit im Format: YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Dateiname zusammensetzen
    filename = f"{model_name}_{timestamp}.json"
    
    # Daten als Dictionary vorbereiten
    data = {
        "Param1": param1,
        "Param2": param2,
        "Param3": param3
    }
    
    # JSON-Datei speichern
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Datei gespeichert: {filename}")
    return filename

# Beispielaufruf
if __name__ == "__main__":
    save_json_with_timestamp(
        model_name="MeinModell",
        param1="Wert1",
        param2=42,
        param3=[1, 2, 3]
    )
