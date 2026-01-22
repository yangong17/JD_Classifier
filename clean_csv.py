"""
Script to clean boilerplate text from job descriptions CSV.
"""
import pandas as pd
import re

# Read the CSV
print("Loading CSV...")
df = pd.read_csv('job_descriptions_clean.csv')
print(f"Loaded {len(df)} rows")

# Text to remove (the disaster service worker boilerplate)
boilerplate_texts = [
    r'DISASTER SERVICE WORKERÂ\s*Â\s*In accordance with Government Code Section 3100.*?respond accordingly\.?',
    r'DISASTER SERVICE WORKER\s*In accordance with Government Code Section 3100.*?respond accordingly\.?',
    r'DISASTER SERVICE WORKER.*$',
]

# Find the description column
desc_col = None
for col in df.columns:
    if 'description' in col.lower():
        desc_col = col
        break

if desc_col:
    print(f"Cleaning column: {desc_col}")
    
    # Show sample before
    sample_before = str(df[desc_col].iloc[0])[-300:]
    print(f"\nSample BEFORE (last 300 chars):\n{sample_before}\n")
    
    # Clean the descriptions
    for pattern in boilerplate_texts:
        df[desc_col] = df[desc_col].astype(str).str.replace(pattern, '', regex=True, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove special characters and extra whitespace
    df[desc_col] = df[desc_col].str.replace(r'[ÂÂ]+', '', regex=True)
    df[desc_col] = df[desc_col].str.strip()
    
    # Show sample after
    sample_after = str(df[desc_col].iloc[0])[-300:]
    print(f"Sample AFTER (last 300 chars):\n{sample_after}\n")
    
    # Save
    df.to_csv('job_descriptions_cleaned.csv', index=False)
    print("Saved to job_descriptions_cleaned.csv!")
else:
    print("Could not find description column!")
    print("Available columns:", df.columns.tolist())
