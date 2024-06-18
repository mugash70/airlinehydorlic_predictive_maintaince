import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.models import load_model

from flask import  jsonify
plt.style.use('ggplot')
import os
# %matplotlib inline

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def handle_missing_values(data):
    if isinstance(data, np.ndarray):
        missing_values = np.isnan(data)
    elif isinstance(data, pd.DataFrame):
        missing_values = data.isnull()
    else:
        raise ValueError("Unsupported data type. Please provide either a NumPy array or a pandas DataFrame.")
    if np.any(missing_values):
        if isinstance(data, np.ndarray):
            data[missing_values] = np.nanmean(data)
        elif isinstance(data, pd.DataFrame):
            data.fillna(data.mean(), inplace=True)

    return data

def load_and_process_datasets(file_prefixes, directory, axis=1):
    data = {}
    for prefix in file_prefixes:
        file_path = f"{directory}/{prefix}.txt"
        data[prefix] = handle_missing_values(np.genfromtxt(file_path).mean(axis=axis))
    return data

def process_and_append_datasets(data, directory, df,target_col,target_file=None, axis=1,):
    df = df.assign(**data)
    if target_file:
        target = np.genfromtxt(f"{directory}/{target_file}.txt")
        print(target_col)
        target_col = [col for col in target_col]
        df_temp = pd.DataFrame(target,columns=target_col)
        df = pd.concat([df, df_temp], axis=1)
    return df

def generate_sequences(df, seq_length, seq_cols):
    sequences = []
    for id_val in df['id'].unique():
        id_df = df[df['id'] == id_val]
        if len(id_df) < seq_length:
            padding = pd.DataFrame(np.zeros((seq_length - len(id_df), len(seq_cols))), columns=seq_cols)
            id_df = pd.concat([padding, id_df], ignore_index=True)
        data_array = id_df[seq_cols].values
        for start in range(len(id_df) - seq_length + 1):
            sequences.append(data_array[start:start + seq_length])
    return np.array(sequences)

def gen_label(id_df, seq_length, seq_cols, label):
    df_zeros = pd.DataFrame(np.zeros((seq_length-1, id_df.shape[1])), columns=id_df.columns)
    id_df = pd.concat([df_zeros, id_df], ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    y_label = []
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        y_label.append(id_df[label].iloc[stop])
    return np.array(y_label)



def main (filename,target):
    base_directory = r"C:\Users\Administrator\Desktop\frontend\static\datasets"
    directory = os.path.join(base_directory, filename)
    # print(directory)
    # df = pd.DataFrame(directory)
    df = pd.read_excel(directory)
    # features_col_name = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6","EPS1","FS1", "FS2","TS1", "TS2", "TS3", "TS4","VS1","CE", "CP","SE"]
    target_col_name=["Cooler_Condition", "Valve_Condition", "Internal_Pump_Leakage", "Hydraulic_Accumulator", "Stable_Flag"]
   
    # data = load_and_process_datasets(features_col_name, directory, axis=1)
    # df = process_and_append_datasets(data, directory, df,target_col_name,target_file ="profile")

    window_size = 96
    scaler = MinMaxScaler()
    features_normalized =df.drop(columns=target_col_name)
    labels = df[target].values
    sequences = []
    next_labels = []
    for i in range(len(features_normalized) - window_size):
        sequence = features_normalized[i:i + window_size]
        label = labels[i + window_size]
        sequences.append(sequence)
        next_labels.append(label)
    X = np.array(sequences)
    y = np.array(next_labels)
    X = np.reshape(X, (X.shape[0], window_size, X.shape[2]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(input_shape=(X_train.shape[1], X_train.shape[2]),units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(  units=50,return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=100, validation_split=0.05, verbose=1,callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])
    scores = model.evaluate(X_train, y_train, verbose=1, batch_size=200)
    base_directory = r"C:\Users\Administrator\Desktop\frontend\static\models"

    name, _ = os.path.splitext(filename)
    model.save(os.path.join(base_directory, f"{name}_lstm_model_{target}.keras"))

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # conf_matrix = confusion_matrix(y_test, y_pred)
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    # plt.title('Confusion Matrix')
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.show()
    return jsonify({
            'df': {
                'y_test': y_test.tolist(),  # Convert tolist() if y_test is a numpy array
                'y_pred': y_pred.tolist(),  # Convert tolist() if y_pred is a numpy array
                'X_train': X_train.tolist(),  # Convert tolist() if X_train is a numpy array
                'y_train': y_train.tolist()  # Convert tolist() if y_train is a numpy array
            },
            'accuracy_train': format(scores[1], '.4f'),  # Format accuracy_train to 4 decimal places
            'accuracy_test': accuracy_score(y_test, y_pred)
        })
    # return jsonify({'df':{'y_test':y_test,'y_pred':y_pred,'X_train':X_train,'y_train':y_train}accuracy_train':format(scores[1]),'accuracy_test': accuracy_score(y_test, y_pred)})

# x = main ('original.xlsx','Stable_Flag')
# print(x)

def predict_code(df,modelfile):
    pattern = r'^.*_lstm_model_|\.keras$'
    specif_target = re.sub(pattern, '', filename)
    target_col_name=["Cooler_Condition", "Valve_Condition", "Internal_Pump_Leakage", "Hydraulic_Accumulator", "Stable_Flag"]
    window_size = 10
    scaler = MinMaxScaler()
    features_normalized =df.drop(columns=target_col_name)
    labels = df[specif_target].values
    sequences = []
    next_labels = []
    for i in range(len(features_normalized) - window_size):
        sequence = features_normalized[i:i + window_size]
        label = labels[i + window_size]
        sequences.append(sequence)
        next_labels.append(label)
    X = np.array(sequences)
    y_test= np.array(next_labels)
    X_test = np.reshape(X, (X.shape[0], window_size, X.shape[2]))
    model=load_model(f"lstm_model_{specif_target}.keras")
    y_pred = model.predict(X_test)
    #y_pred = (y_pred_prob > 0.5).astype(int)
    accuracy =accuracy_score(y_test, y_pred)
    return y_pred,accuracy
