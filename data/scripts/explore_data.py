import pandas as pd
import json

df = pd.read_csv('data/Data_MLB_2025_StatcastPostseason_PitchByPitch_20251102a.csv')
with open('schema.txt', 'w') as f:
    f.write("Columns:\n")
    for col in df.columns:
        f.write(f"- {col}\n")
    
    f.write("\nSample Data:\n")
    f.write(df.head(2).to_string())
    
    f.write("\n\nValue counts for pitch_name:\n")
    f.write(str(df['pitch_name'].value_counts()))

print("Data explored. Output saved to schema.txt")
