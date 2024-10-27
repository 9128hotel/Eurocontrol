import sqlite3
import pandas as pd

csv_file = "./competition-data/final_submission_set.csv"
df = pd.read_csv(csv_file)

conn = sqlite3.connect('final_submission_data.db')

df.to_sql('flights', conn, if_exists='replace', index=False)

conn.close()

print("Data has been written to the SQLite database.")
