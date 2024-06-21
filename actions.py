import os
import pandas as pd
import glob

def get_datasets():
    datasets_folder = os.path.join("static", "datasets")
    file_list = os.listdir(datasets_folder)
    dataset_items = [{"filename": filename, "id": idx} for idx, filename in enumerate(file_list, start=1)]
    return dataset_items

def get_sensor(filename):
    file_path = f"static/datasets/{filename}"
    try:
        df = pd.read_excel(file_path)
        columns = df.columns.tolist()
        return {"filename": filename, "columns": columns}
    except Exception as e:
        print("Error:", e)
        return {"error": str(e)}

def get_models(filename):
    base_directory = "static/models"
    base_name = os.path.splitext(os.path.basename(filename))[0]
    search_pattern = os.path.join(base_directory, f"{base_name}*")
    matching_files = glob.glob(search_pattern)
    results = []
    for file_path in matching_files:
        normalized_path = os.path.normpath(file_path).replace(os.path.sep, '/')
        file_name = os.path.basename(file_path)
        results.append({"file": file_name, "path": normalized_path})
  
    return results