import json
from datetime import datetime

def save_json_with_timestamp(model_name, params_dict):
    """
    Saves a JSON file with a timestamp in the filename.
    
    Args:
        model_name: Name of the model
        params_dict: Dictionary containing hyperparameters and performance statistics
    
    Returns:
        str: Path to the saved file
    """
    # Current date and time in format: YYYY-MM-DD_HH-MM-SS
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Compose filename
    filename = f"{model_name}_{timestamp}.json"
    
    # Use the provided dictionary directly
    data = params_dict
    
    # Save JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    
    print(f"Hyperparameters and statistics saved: {filename}")
    return filename

# Example usage
if __name__ == "__main__":
    params = {
        "Param1": "Value1",
        "Param2": 42,
        "Param3": [1, 2, 3]
    }
    
    save_json_with_timestamp(
        model_name="MyModel",
        params_dict=params
    )
