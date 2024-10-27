import pyarrow.parquet as pq
import os
import json

def find_max_entries(directory):
    max_length = 0
    for parquet_file in os.listdir(directory):
        if parquet_file.endswith('.parquet'):
            filepath = os.path.join(directory, parquet_file)
            print(f"Processing file {filepath}")
            table = pq.read_table(filepath)
            df = table.to_pandas()
            
            flight_lengths = df.groupby('flight_id').size()
            max_flight_length = flight_lengths.max()
            
            if max_flight_length > max_length:
                max_length = max_flight_length
                print(f"New max: {max_length}")

    max_length = int(max_length)
    
    with open('max_flight_length.json', 'w') as f:
        json.dump({'max_flight_length': max_length}, f)
    
    print(f"Maximum number of entries for a flight_id: {max_length}")

if __name__ == "__main__":
    find_max_entries('./competition-data/')