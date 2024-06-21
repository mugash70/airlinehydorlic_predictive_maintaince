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
    base_directory = r"C:\Users\Administrator\Desktop\frontend\static\datasets"
    directory = os.path.join(base_directory, filename)
    df = pd.read_excel(directory)
    window_size = 96
    scaler = MinMaxScaler()
    features_col_name =features_col_name = df.columns.difference([target])  
    target_col_name= target
    df[features_col_name]=scaler.fit_transform(df[features_col_name])
    if 'id' not in df.columns:
        df['id'] = range(1, len(df) + 1)

    df_x = [gen_sequence(df[df['id'] == id], window_size, features_col_name) for id in df['id'].unique()]
    X = np.concatenate(df_x, axis=0)

    df_y = [gen_label(df[df['id'] == id], window_size, features_col_name, target_col_name) for id in df['id'].unique()]
    y = np.concatenate(df_y, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Define the LSTM model
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
    base_directory = r"C:\Users\Administrator\Desktop\frontend\static\models"

    name, _ = os.path.splitext(filename)
    model.save(os.path.join(base_directory, f"{name}_lstm_model_{target}.keras"))

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # return jsonify({
    #         'df': {
    #             'y_test': y_test.tolist(),  # Convert tolist() if y_test is a numpy array
    #             'y_pred': y_pred.tolist(),  # Convert tolist() if y_pred is a numpy array
    #             'X_train': X_train.tolist(),  # Convert tolist() if X_train is a numpy array
    #             'y_train': y_train.tolist()  # Convert tolist() if y_train is a numpy array
    #         },
    #         'accuracy_train': format(scores[1], '.4f'),  # Format accuracy_train to 4 decimal places
    #         'accuracy_test': accuracy_score(y_test, y_pred)
    #     })
    return y_test,y_pred, format(scores[1], '.4f'),accuracy_score(y_test, y_pred)

def predict_code(df_test, modelfile):
    normalized_path = os.path.normpath(modelfile)
    model = load_model(normalized_path)
    pattern = r'^.*_lstm_model_|\.keras$'
    target = re.sub(pattern, '', normalized_path)
    features_col_name = df_test.columns.difference([target])
    window_size = 96
    predictions = {}

    for machine_id in df_test['id'].unique():
        machine_df = df_test[df_test['id'] == machine_id]
        machine_test = gen_sequence(machine_df, window_size, features_col_name)
        m_pred = model.predict(machine_test)
        failure_prob = list(m_pred[-1] * 100)[0]
        predictions[machine_id] = failure_prob
    return predictions
 
# dataset ='./static/datasets/original.xlsx'
modelfile ='./static/models/sensors_lstm_model_label_bc.keras'
# df = pd.read_excel(dataset)
# print(df)
y_test,y_pred,accuracy_train,accuracy_test = main('sensors.xlsx','label_bc')
df_test = './static/datasets/sensors1.xlsx'
df_test = pd.read_excel(df_test)
# print(df_test)
print(predict_code(df_test,modelfile))