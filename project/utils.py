import json

def read_json(file_path):
    """Reading json data as dict.

    Args:
      file_path (str): File path.

    Return:
      json_dict (dict): Dictionary format of json data.

    """
    with open(file_path, 'r') as f:
        json_dict = json.load(f)
    return json_dict