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
import re
from flask import  jsonify,current_app
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

def gen_sequence(id_df, seq_length, seq_cols):
    df_zeros = pd.DataFrame(np.zeros((seq_length - 1, len(id_df.columns))), columns=id_df.columns)
    id_df = pd.concat([df_zeros, id_df], ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array = []
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        lstm_array.append(data_array[start:stop, :])
    return np.array(lstm_array)

def gen_label(id_df, seq_length, seq_cols, label):
    df_zeros = pd.DataFrame(np.zeros((seq_length - 1, len(id_df.columns))), columns=id_df.columns)
    id_df = pd.concat([df_zeros, id_df], ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    y_label = []
    for stop in range(seq_length, num_elements):
        y_label.append(id_df[label].iloc[stop])
    return np.array(y_label)



def main (filename,target):
    base_directory =current_app.config['DATASET_BASE_DIRECTORY']
    # r"C:\Users\Administrator\Desktop\frontend\static\datasets"
    
    directory = os.path.join(base_directory, filename)
    df = pd.read_excel(directory)
    window_size = 96
    scaler = MinMaxScaler()
    features_col_name =features_col_name = df.columns.difference([target,'id'])  
    target_col_name= target
    df[features_col_name]=scaler.fit_transform(df[features_col_name])
    if 'id' not in df.columns:
        df['id'] = range(1, len(df) + 1)
    df_x = [gen_sequence(df[df['id'] == id], window_size, features_col_name) for id in df['id'].unique()]
    X = np.concatenate(df_x, axis=0)
    df_y = [gen_label(df[df['id'] == id], window_size, features_col_name, target_col_name) for id in df['id'].unique()]
    y = np.concatenate(df_y, axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Sequential()
    model.add(LSTM(input_shape=(X_train.shape[1], X_train.shape[2]),units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(  units=50,return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','mse','mae'])
    model.fit(X_train, y_train, epochs=10, batch_size=200, validation_split=0.05, verbose=1,
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])
    
    scores = model.evaluate(X_train, y_train, verbose=1, batch_size=200)
    base_directory = current_app.config['MODELS_BASE_DIRECTORY']
    # r"C:\Users\Administrator\Desktop\frontend\static\models"
    # 

    name, _ = os.path.splitext(filename)
    model.save(os.path.join(base_directory, f"{name}_lstm_model_{target}.keras"))

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    accuracy_train=format(scores[1], '.4f')
    accuracy_test =accuracy_score(y_test, y_pred)

    return X_train,y_train,y_test,y_pred,accuracy_train,accuracy_test

def gen_sequence_pred(id_df, seq_length, seq_cols):
    num_elements = len(id_df)
    if num_elements >= seq_length:
        lstm_array = []
        for start in range(0, num_elements - seq_length + 1):
            sequence = id_df.iloc[start:start + seq_length][seq_cols].values
            lstm_array.append(sequence)
    else:
        df_zeros = pd.DataFrame(np.zeros((seq_length - num_elements, len(seq_cols))), columns=seq_cols)
        padded_df = pd.concat([df_zeros, id_df], ignore_index=True)
        lstm_array = [padded_df[seq_cols].values]
    return np.array(lstm_array)

def predict_code(df_test, modelfile):
    normalized_path = os.path.normpath(modelfile)
    model = load_model(normalized_path)
    pattern = r'^.*_lstm_model_|\.keras$'
    target = re.sub(pattern, '', normalized_path)
    pattern2 = r'^.*?(?=_lstm_model)'
    original_name = re.search(pattern2, os.path.basename(normalized_path)).group(0)
    base_directory =current_app.config['DATASET_BASE_DIRECTORY']
    # r"C:\Users\Administrator\Desktop\frontend\static\datasets"  
    directory = os.path.join(base_directory, f"{original_name}.xlsx")
    df = pd.read_excel(directory)
    scaler = MinMaxScaler()
    features_col_name = df_test.columns.difference([target,'id'])
    scaler.fit(df_test[features_col_name])
    df_test[features_col_name]=scaler.transform(df_test[features_col_name])
    window_size = 96
    machine_test = [gen_sequence_pred(df_test[df_test['id'] == id], window_size, features_col_name) for id in df_test['id'].unique()]
    predictions = {}
    for idx, machine_id in enumerate(df_test['id'].unique()):
        machine_seq = machine_test[idx]
        m_pred = model.predict(machine_seq)
        failure_prob = list(m_pred[-1] * 100)[0]
        predictions[machine_id] = failure_prob
    return predictions
