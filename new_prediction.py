from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Function to process data and make predictions
def prediction_newdata(df, features_col_name, target_col_name, x_columns):
    # Drop columns not needed for prediction
    df.drop(x_columns, axis=1, inplace=True)
    
    # Normalize features
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(df.drop(columns=target_col_name))
    
    # Prepare sequences
    window_size = 10
    sequences = []
    labels = df['Stable_Flag'].values
    for i in range(len(features_normalized) - window_size):
        sequence = features_normalized[i:i + window_size]
        sequences.append(sequence)
    X = np.array(sequences)
    X = np.reshape(X, (X.shape[0], window_size, X.shape[2]))
    
    # Load the pre-trained LSTM model
    specif_target = 'Stable_Flag'  # Replace with your specific target
    model = load_model(f"lstm_model_{specif_target}.keras")
    
    # Predict on new data
    y_pred_prob = model.predict(X)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels[window_size:], y_pred)
    
    return y_pred, accuracy, labels[window_size:]

# Route for uploading file and making predictions
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Read Excel file
            df = pd.read_excel(file)
            
            # Define column names
            features_col_name = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"]
            target_col_name = ["Cooler_Condition", "Valve_Condition", "Internal_Pump_Leakage", "Hydraulic_Accumulator", "Stable_Flag"]
            x_columns = ["Cooler_Condition", "Valve_Condition", "Internal_Pump_Leakage", "Hydraulic_Accumulator"]
            
            # Get predictions and accuracy
            predictions, accuracy, actual_labels = prediction_newdata(df, features_col_name, target_col_name, x_columns)
            
            # Plotting (optional)
            plt.figure(figsize=(12, 6))
            plt.plot(np.arange(len(actual_labels)), actual_labels, label='Actual Data', color='green')
            plt.plot(np.arange(len(actual_labels)), predictions, label='Predicted Data', color='red')
            plt.title('Comparison of Actual and Predicted Data')
            plt.xlabel('Time')
            plt.ylabel('Stable Flag')
            plt.legend()
            plt.grid(True)
            plt.savefig('/path/to/static/plot.png')  # Save plot to static directory
            plt.close()
            
            # Render results page with accuracy and plot
            return render_template('results.html', accuracy=accuracy)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
