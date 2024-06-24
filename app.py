from flask import Flask,render_template,request,jsonify
import actions
import core
import pandas as pd
from config import Config

app = Flask(__name__, static_url_path='/static', static_folder='static')

app.config.from_object(Config)
def convert_keys_to_str(d):
    """ Recursively convert dictionary keys to strings. """
    if isinstance(d, dict):
        return {str(k): convert_keys_to_str(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_keys_to_str(i) for i in d]
    else:
        return d
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    selected_model = request.form.get('selectedModel') 

    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    try:
        df = pd.read_excel(file)
        predictions =  core.predict_code(df,selected_model)
        columns_list = df.columns.astype(str).tolist()
        predictions = {str(k): float(v) for k, v in predictions.items()}

        response_data = {
            'filename': file.filename,
            'columns': columns_list,
            'predictions': predictions
        }
        print(response_data)
        return response_data
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        selected_filename = request.form.get('datasetSelect')
        training = request.form.get('training')
        if training :
            filename = request.form.get('datasetSelect')
            taget = request.form.get('sensorSelect')
            json_return = core.main (filename,taget)
            return json_return
        if selected_filename:
            sensors = actions.get_sensor(selected_filename)
            models = actions.get_models(selected_filename)
            response = {
                "sensors": sensors,
                "models": models
            }
            return jsonify(response)
    else:
        x_datasets = actions.get_datasets()
        return render_template("index.html", x_datasets=x_datasets)


@app.route("/train")
def train():
    return render_template("home.html")

# @app.route("/upload")
# def upload():

#     return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)