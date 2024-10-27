import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import sqlite3
from collections import defaultdict

class FlightData:
    def __init__(self, flight_id, timestamps, latitudes, longitudes, altitudes, groundspeeds, tracks, vertical_rates, track_unwrappeds, u_component_of_winds, v_component_of_winds, specific_humidities, icao24, aircraft_type, flight_duration, taxiout_time, flown_distance):
        self.flight_id = flight_id
        self.timestamps = timestamps
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.altitudes = altitudes
        self.groundspeeds = groundspeeds
        self.tracks = tracks
        self.vertical_rates = vertical_rates
        self.track_unwrappeds = track_unwrappeds
        self.u_component_of_winds = u_component_of_winds
        self.v_component_of_winds = v_component_of_winds
        self.specific_humidities = specific_humidities
        self.icao24 = icao24
        self.aircraft_type = aircraft_type
        self.flight_duration = flight_duration
        self.taxiout_time = taxiout_time
        self.flown_distance = flown_distance

df_parquet = pd.read_parquet("./competition-data/2022-01-01.parquet")
print(df_parquet.head())

conn = sqlite3.connect('flights_data.db')
df_sqlite = pd.read_sql_query("SELECT * FROM flights", conn)
conn.close()

df_parquet['flight_id'] = df_parquet['flight_id'].astype(str)
df_sqlite['flight_id'] = df_sqlite['flight_id'].astype(str)

df_merged = pd.merge(df_parquet, df_sqlite, on='flight_id', how='left')

grouped = df_merged.groupby('flight_id').agg({
    'timestamp': list,
    'latitude': list,
    'longitude': list,
    'altitude': list,
    'groundspeed': list,
    'track': list,
    'vertical_rate': list,
    'track_unwrapped': list,
    'u_component_of_wind': list,
    'v_component_of_wind': list,
    'specific_humidity': list,
    'icao24': 'first',  # Same for all rows
    'aircraft_type': 'first',  # Same for all rows
    'flight_duration': 'first',  # Same for all rows
    'taxiout_time': 'first',  # Same for all rows
    'flown_distance': 'first'  # Same for all rows
}).reset_index()

data = [
    FlightData(
        row['flight_id'], row['timestamp'], row['latitude'], row['longitude'],
        row['altitude'], row['groundspeed'], row['track'], row['vertical_rate'],
        row['track_unwrapped'], row['u_component_of_wind'], row['v_component_of_wind'],
        row['specific_humidity'], row['icao24'], row['aircraft_type'],
        row['flight_duration'], row['taxiout_time'], row['flown_distance']
    )
    for _, row in grouped.iterrows()
]

for record in data:
    print(f"Flight ID: {record.flight_id} - Altitudes: {record.altitudes}")