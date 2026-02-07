import gdown
import zipfile
import pandas as pd
import os
import re
from pathlib import Path

# Temporary folder for downloads & extraction
tmp_folder = "./energy_dataset"
os.makedirs(tmp_folder, exist_ok=True)

# ---------------------------
# Step 1: Download Core + Bonus + Advanced Core ZIPs
# ---------------------------
zip_files = {
    "core": "https://drive.google.com/uc?id=13o_2ojFRCCqwmYMN3w3qu5fQxieXATTd",
    "bonus": "https://drive.google.com/uc?id=1Hvqi5nv66m3b1aEN23NnUOBkVKQrfP5z",
    "advanced_core": "PASTE_ADVANCED_CORE_URL_HERE" # Replace with actual URL
}

extracted_csv_paths = []

for name, url in zip_files.items():
    zip_path = os.path.join(tmp_folder, f"{name}_dataset.zip")
    
    if os.path.exists(zip_path):
        print(f'{name} data already downloaded, skipping download')
    elif "PASTE" in url:
        print(f"Skipping {name}: No URL provided.")
        continue
    else:
        print(f"\nDownloading {name} ZIP...")
        gdown.download(url, zip_path, quiet=False)
    
    if os.path.exists(zip_path):
        print(f"Extracting CSVs from {name} ZIP...")
        with zipfile.ZipFile(zip_path, "r") as z:
            for member in z.namelist():
                # Filter out MacOS metadata and ignore weather files
                if member.endswith(".csv") and "__MACOSX" not in member and "weather" not in member.lower():
                    print(f"  Extracting {member}")
                    z.extract(member, tmp_folder)
                    extracted_csv_paths.append(os.path.join(tmp_folder, member))

# ---------------------------
# Step 2: Load CSVs into Pandas
# ---------------------------
pdf_dict = {}
for csv_path in list(set(extracted_csv_paths)):
    csv_name = os.path.basename(csv_path)
    print(f"\nLoading {csv_name}...")
    pdf_dict[csv_name] = pd.read_csv(csv_path, encoding="latin1")

# ---------------------------
# Step 3: Advanced Merge Logic (Padded SIMS Code)
# ---------------------------
print("\nProcessing Master Dataset...")

# 1. Combine Meter Readings
reading_dfs = [df for name, df in pdf_dict.items() if "meter-readings" in name.lower()]

if reading_dfs:
    df_readings = pd.concat(reading_dfs, ignore_index=True)
    
    # PAD SIMSCODE: Convert 241.0 -> '0241'
    # We use fillna(0) to handle missing codes and astype(int) to remove decimals
    df_readings['sims_key'] = df_readings['simscode'].fillna(0).astype(int).astype(str).str.zfill(4)
    
    # 2. Process Metadata (Extract from Building Name)
    meta_key = next((name for name in pdf_dict.keys() if "metadata" in name.lower()), None)
    
    if meta_key:
        df_meta = pdf_dict[meta_key]
        
        # EXTRACT CODE: Look for 4 digits inside brackets in "sitename"
        # Example: "Ackerman Rd, 650 (0241)" -> "0241"
        df_meta['sims_key_meta'] = df_meta['buildingname'].str.extract(r'\((\d{4})\)')
        
        # 3. Perform Merge using the specific keys
        df_master = pd.merge(
            df_readings, 
            df_meta, 
            left_on='sims_key', 
            right_on='sims_key_meta', 
            how='left'
        )
        
        # Clean up temporary columns used for joining
        df_master = df_master.drop(columns=['sims_key', 'sims_key_meta'])
        
        # Save output
        pdf_dict["master_dataset_combined.csv"] = df_master
        
        df_master.to_csv("master_dataset_combined.csv", index=False)
        
        print(f"Success! Master dataset created.")
        print(f"Total Rows: {len(df_master)}")
        print(f"Sample of Building Names joined: {df_master['sitename'].unique()[:5]}")
    else:
        print("Error: Metadata file not found. Check zip extraction.")
else:
    print("Error: No meter readings found.")