import os
import re
import numpy as np
import tensorflow as tf
import pandas as pd
import sqlite3
from glob import glob
import keras
from keras import Model
from sklearn.preprocessing import LabelEncoder
import pickle  # For loading the scaler

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

def load_most_recent_model(model_directory='./models'):
    model_files = glob(f'{model_directory}/model_epoch_*.keras')
    if not model_files:
        raise ValueError("No models found in the directory.")
    
    def extract_epoch_number(filename):
        match = re.search(r'epoch_(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0
    
    model_files_sorted = sorted(model_files, key=extract_epoch_number, reverse=True)
    latest_model_path = model_files_sorted[0]
    model = tf.keras.models.load_model(latest_model_path, custom_objects={'rmsd': rmsd})
    
    print(f"Loaded model from {latest_model_path}")
    return model

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

def process_flight_data_submission(data, aircraft_encoder):
    X = []
    for flight_data in data:
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

    return np.array(X)

def find_last_processed_flight(output_csv):
    if not os.path.exists(output_csv):
        return None
    results_df = pd.read_csv(output_csv)
    if results_df.empty:
        return None
    return results_df['flight_id'].iloc[-1]  # Return the last processed flight ID

def predict_tow_for_submission(db_path='final_submission_data.db', parquet_dir='./competition-data', output_csv='team_dependable_gorilla_v0_results.csv'):
    model = load_most_recent_model()

    with open('./scalers/scaler_y.pkl', 'rb') as f:
        scaler_y = pickle.load(f)

    conn = sqlite3.connect(db_path)
    submission_df = pd.read_sql_query("SELECT * FROM flights", conn)

    all_aircraft_types = submission_df['aircraft_type'].unique()

    aircraft_encoder = LabelEncoder()
    aircraft_encoder.fit(all_aircraft_types)

    conn.close()

    last_processed_flight_id = find_last_processed_flight(output_csv)
    print(f"Last processed flight ID: {last_processed_flight_id}")

    if last_processed_flight_id is not None:
        submission_df = submission_df[submission_df['flight_id'] > last_processed_flight_id]

    predictions = []

    for _, row in submission_df.iterrows():
        flight_id = row['flight_id']
        date_str = row['date']  # Get the date to find the corresponding parquet file
        aircraft_type = row['aircraft_type']

        print(f"Processing flight: {flight_id}, Date: {date_str}")
        parquet_file = glob(f'{parquet_dir}/{date_str}.parquet')[0]
        if not parquet_file:
            print(f"No parquet file found for date {date_str}")
            continue

        df = pd.read_parquet(parquet_file)

        flight_data_df = df[df['flight_id'].astype(str) == str(flight_id)]

        if flight_data_df.empty:
            print(f"No data found for flight ID {flight_id}")
            continue

        if flight_data_df.isnull().values.any():
            print(f"Flight data for flight ID {flight_id} contains NaN values, skipping.")
            continue

        flight_data = FlightData(
            flight_id=flight_id,
            timestamps=flight_data_df['timestamp'].tolist(),
            latitudes=flight_data_df['latitude'].tolist(),
            longitudes=flight_data_df['longitude'].tolist(),
            altitudes=flight_data_df['altitude'].tolist(),
            groundspeeds=flight_data_df['groundspeed'].tolist(),
            tracks=flight_data_df['track'].tolist(),
            vertical_rates=flight_data_df['vertical_rate'].tolist(),
            u_component_of_winds=flight_data_df['u_component_of_wind'].tolist(),
            v_component_of_winds=flight_data_df['v_component_of_wind'].tolist(),
            specific_humidities=flight_data_df['specific_humidity'].tolist(),
            icao24=flight_data_df['icao24'].iloc[0],
            aircraft_type=aircraft_type,
            flight_duration=row['flight_duration'],
            taxiout_time=row['taxiout_time'],
            flown_distance=row['flown_distance']
        )

        X = process_flight_data_submission([flight_data], aircraft_encoder=aircraft_encoder)

        if np.isnan(X).any():
            print(f"Flight {flight_id} contains NaN values in feature array X, skipping.")
            continue

        tow_prediction_scaled = model.predict(X)[0][0]  # Assuming model returns 1D output

        if np.isnan(tow_prediction_scaled):
            print(f"Predicted TOW for flight {flight_id} is NaN, skipping.")
            continue

        tow_prediction = scaler_y.inverse_transform([[tow_prediction_scaled]])[0][0]

        print(f"Predicted TOW for flight {flight_id}: {tow_prediction}")

        predictions.append((flight_id, tow_prediction))

        results_df = pd.DataFrame(predictions, columns=['flight_id', 'tow'])
        results_df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
        predictions.clear()  # Clear the predictions list after appending

    print(f"All predictions processed and saved to {output_csv}")

predict_tow_for_submission()