import os
import re
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras import layers, models, mixed_precision
from tensorflow import keras
import pandas as pd
import sqlite3
from glob import glob
import pickle

print("[DONE] Imports")

"""
def rmsd(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))
"""

def rmsd(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    if tf.reduce_any(tf.math.is_nan(y_true)):
        print("Warning: NaN detected in y_true")
    if tf.reduce_any(tf.math.is_nan(y_pred)):
        print("Warning: NaN detected in y_pred")
    
    if tf.reduce_any(tf.math.is_inf(y_true)):
        print("Warning: Infinite values detected in y_true")
    if tf.reduce_any(tf.math.is_inf(y_pred)):
        print("Warning: Infinite values detected in y_pred")

    diff_squared = tf.square(y_pred - y_true)

    if tf.reduce_any(tf.math.is_nan(diff_squared)):
        print("Warning: NaN detected in the squared differences")
    if tf.reduce_any(tf.math.is_inf(diff_squared)):
        print("Warning: Infinite values detected in the squared differences")

    return tf.sqrt(tf.reduce_mean(diff_squared))

class FlightData:
    def __init__(self, flight_id, timestamps, latitudes, longitudes, altitudes, groundspeeds, tracks, vertical_rates, u_component_of_winds, v_component_of_winds, specific_humidities, icao24, aircraft_type, flight_duration, taxiout_time, flown_distance):
        self.flight_id = flight_id
        self.timestamps = timestamps
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.altitudes = altitudes
        self.groundspeeds = groundspeeds
        self.tracks = tracks
        self.vertical_rates = vertical_rates
        self.u_component_of_winds = u_component_of_winds
        self.v_component_of_winds = v_component_of_winds
        self.specific_humidities = specific_humidities
        self.icao24 = icao24
        self.aircraft_type = aircraft_type
        self.flight_duration = flight_duration
        self.taxiout_time = taxiout_time
        self.flown_distance = flown_distance

def load_aircraft_types_from_db():
    conn = sqlite3.connect('flights_data.db')
    aircraft_types = pd.read_sql_query("SELECT DISTINCT aircraft_type FROM flights", conn)
    conn.close()
    return aircraft_types['aircraft_type'].tolist()

def sort_files_by_day_then_month(files):
    def extract_date(file_name):
        match = re.search(r"(\d{4})-(\d{2})-(\d{2})", file_name)
        if match:
            year, month, day = map(int, match.groups())
            return year, month, day
        return None
    
    files_sorted = sorted(files, key=lambda f: (extract_date(f)[2], extract_date(f)[1], extract_date(f)[0]))
    return files_sorted

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

mixed_precision.set_global_policy('mixed_float16')

def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=input_shape))
    model.add(layers.GRU(128, return_sequences=True))
    model.add(layers.GRU(64))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, dtype='float32'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[rmsd])
    return model

def process_flight_data(data, tow, max_length=36871):
    X = []
    y = []
    
    for flight_data, tow_value in zip(data, tow):
        if tow_value is None or flight_data.aircraft_type is None:
            continue
        
        flight_length = len(flight_data.altitudes)
        latitudes = flight_data.latitudes
        longitudes = flight_data.longitudes
        altitudes = flight_data.altitudes
        groundspeeds = flight_data.groundspeeds
        tracks = flight_data.tracks
        vertical_rates = flight_data.vertical_rates
        
        aircraft_type_encoded = aircraft_encoder.transform([flight_data.aircraft_type])[0]
        
        combined_features = np.array([
            latitudes, longitudes, altitudes, groundspeeds, tracks, vertical_rates,
            [aircraft_type_encoded] * flight_length,
            [flight_data.flight_duration] * flight_length,
            [flight_data.taxiout_time] * flight_length
        ]).T
        
        X.append(combined_features)
        y.append(tow_value)
    
    X_padded = keras.utils.pad_sequences(X, maxlen=max_length, padding='post', dtype='float32')
    
    return np.array(X_padded), np.array(y)

scaler_X, scaler_y = None, None

def generate_scaler_y_from_database():
    global scaler_y
    conn = sqlite3.connect('flights_data.db')
    tow_values = pd.read_sql_query("SELECT tow FROM flights WHERE tow IS NOT NULL", conn)
    conn.close()

    tow_values = tow_values['tow'].values.reshape(-1, 1)  # Reshape for scaling
    scaler_y = StandardScaler()
    scaler_y.fit(tow_values)  # Fit scaler on all TOW values in the database

    with open('scaler_y.pkl', 'wb') as f:
        pickle.dump(scaler_y, f)
    print(f"Scaler y fitted on {len(tow_values)} TOW values.")
    print(f"Scaler y params: {scaler_y.__dict__}")

if os.path.exists('scaler_y.pkl'):
    with open('scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)
else:
    generate_scaler_y_from_database()

def normalize_data(X, y):
    global scaler_X, scaler_y
    if scaler_X is None:
        scaler_X = StandardScaler()

        X_all = X.reshape(-1, X.shape[-1])  # Reshape to 2D array

        X_normalized = scaler_X.fit_transform(X_all).reshape(X.shape)

        with open('scaler_X.pkl', 'wb') as f:
            pickle.dump(scaler_X, f)
    else:
        X_normalized = scaler_X.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    y_normalized = scaler_y.transform(y.reshape(-1, 1)).flatten()
    return X_normalized, y_normalized

print("[DONE] Definitions")

aircraft_types_db = load_aircraft_types_from_db()

aircraft_encoder = LabelEncoder()
aircraft_encoder.fit(aircraft_types_db)

parquet_files = glob('./competition-data/*.parquet')
sorted_files = sort_files_by_day_then_month(parquet_files)

model = None
print("BEGIN!")

model_path = './models/model_epoch_latest.keras'
if os.path.exists(model_path):
    model = keras.models.load_model(model_path, custom_objects={'rmsd': rmsd})
else:
    input_shape = (36871, 9)  # 9 features (lat, lon, alt, etc.)
    model = create_model(input_shape)

for idx, file_path in enumerate(sorted_files):
    df = pd.read_parquet(file_path)

    conn = sqlite3.connect('flights_data.db')
    df_sqlite = pd.read_sql_query("SELECT * FROM flights WHERE tow IS NOT NULL", conn)  # Only select flights with TOW
    conn.close()

    df['flight_id'] = df['flight_id'].astype(str)
    df_sqlite['flight_id'] = df_sqlite['flight_id'].astype(str)

    df_merged = pd.merge(df, df_sqlite, on='flight_id', how='inner')  # Inner join to only keep flights with TOW

    grouped = df_merged.groupby('flight_id').agg({
        'timestamp': list,
        'latitude': list,
        'longitude': list,
        'altitude': list,
        'groundspeed': list,
        'track': list,
        'vertical_rate': list,
        'u_component_of_wind': list,
        'v_component_of_wind': list,
        'specific_humidity': list,
        'icao24': 'first',
        'aircraft_type': 'first',
        'flight_duration': 'first',
        'taxiout_time': 'first',
        'flown_distance': 'first',
        'tow': 'first'
    })

    flights_data = []
    tow_values = []
    
    for row in grouped.itertuples():
        flight_data = FlightData(
            flight_id=row.Index,
            timestamps=row.timestamp,
            latitudes=row.latitude,
            longitudes=row.longitude,
            altitudes=row.altitude,
            groundspeeds=row.groundspeed,
            tracks=row.track,
            vertical_rates=row.vertical_rate,
            u_component_of_winds=row.u_component_of_wind,
            v_component_of_winds=row.v_component_of_wind,
            specific_humidities=row.specific_humidity,
            icao24=row.icao24,
            aircraft_type=row.aircraft_type,
            flight_duration=row.flight_duration,
            taxiout_time=row.taxiout_time,
            flown_distance=row.flown_distance
        )
        flights_data.append(flight_data)
        tow_values.append(row.tow)
    
    X, y = process_flight_data(flights_data, tow_values)
    X_normalized, y_normalized = normalize_data(X, y)

    model.fit(X_normalized, y_normalized, epochs=10, batch_size=32)
    
    model.save(f'./models/model_epoch_{idx}.keras')
    print(f"Trained on day {idx+1}/{len(sorted_files)}")

model.save('./models/model_final.keras')