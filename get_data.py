import gdown
import zipfile
import pandas as pd
import os
import re
from pathlib import Path

def calculate_monthly_eui(df_meter, df_building):
    # --- STEP 1: PREPARE DATA & DATES ---
    # Ensure date column is datetime objects
    df_meter['readingtime'] = pd.to_datetime(df_meter['readingtime'])
    
    # Extract Month and Year for grouping
    df_meter['month'] = df_meter['readingtime'].dt.month
    df_meter['year'] = df_meter['readingtime'].dt.year
    
    # Clean IDs for joining
    df_meter['simscode_clean'] = df_meter['simscode'].astype(str).str.replace(r'\.0$', '', regex=True)
    df_building['buildingnumber_clean'] = df_building['buildingnumber'].astype(str).str.replace(r'\.0$', '', regex=True)
    df_building['grossarea'] = pd.to_numeric(df_building['grossarea'], errors='coerce')

    # --- STEP 2: HANDLE DOUBLE COUNTING (Same logic as before) ---
    # Identify buildings with STEAM and remove their HEAT readings
    steam_buildings = set(df_meter[df_meter['utility'] == 'STEAM']['simscode_clean'])
    mask_keep = (df_meter['utility'] != 'HEAT') | (~df_meter['simscode_clean'].isin(steam_buildings))
    df_clean_meter = df_meter[mask_keep].copy()

    # --- STEP 3: CONVERT TO KBTU ---
    conversion_factors = {
        'ELECTRICITY': 3.412,
        'HEAT': 3.412,
        'GAS': 3.412,
        'COOLING': 3.412,
        'OIL28SEC': 3.412,
        'STEAM': 2.632 
    }
    
    df_energy = df_clean_meter[df_clean_meter['utility'].isin(conversion_factors.keys())].copy()
    df_energy['factor'] = df_energy['utility'].map(conversion_factors)
    df_energy['kbtu'] = df_energy['readingvalue'] * df_energy['factor']

    # --- STEP 4: AGGREGATE BY MONTH ---
    # Group by Building AND Month/Year
    df_monthly = df_energy.groupby(['simscode_clean', 'year', 'month'])['kbtu'].sum().reset_index()

    # --- STEP 5: JOIN & CALCULATE MONTHLY EUI ---
    df_final = pd.merge(
        df_monthly,
        df_building[['buildingnumber_clean', 'buildingname', 'grossarea']],
        left_on='simscode_clean',
        right_on='buildingnumber_clean',
        how='inner'
    )

    # Calculate Monthly EUI
    df_final = df_final[df_final['grossarea'] > 0]
    df_final['Monthly_EUI'] = df_final['kbtu'] / df_final['grossarea']

    # Sort by time for plotting
    return df_final.sort_values(['buildingname', 'year', 'month'])

# Make temp folder
tmp_folder = "./energy_dataset"
os.makedirs(tmp_folder, exist_ok=True)

# ---------------------------
# Step 1: Download Core + Bonus ZIPs
# ---------------------------
zip_files = {
    "core": "https://drive.google.com/uc?id=13o_2ojFRCCqwmYMN3w3qu5fQxieXATTd",
    "bonus": "https://drive.google.com/uc?id=1Hvqi5nv66m3b1aEN23NnUOBkVKQrfP5z"
}

extracted_csv_paths = []

for name, url in zip_files.items():
    zip_path = os.path.join(tmp_folder, f"{name}_dataset.zip")
    if os.path.exists(zip_path):
        print('Data already downloaded, skipping download')
        continue
    print(f"\nDownloading {name} ZIP...")
    gdown.download(url, zip_path, quiet=False)
    
    print(f"Extracting CSVs from {name} ZIP...")
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            if member.endswith(".csv") and "__MACOSX" not in member:
                print(f"  Extracting {member}")
                z.extract(member, tmp_folder)
                extracted_csv_paths.append(os.path.join(tmp_folder, member))

# ---------------------------
# Step 2: Print list of CSV files
# ---------------------------
print("\nAll extracted CSV files:")
if not extracted_csv_paths:
    extracted_csv_paths = list(Path(tmp_folder).glob('**/*.csv'))
for csv_path in extracted_csv_paths:
    print(f" - {os.path.basename(csv_path)}")

# ---------------------------
# Step 3: Load CSVs into Pandas
# ---------------------------
pdf_dict = {}
for csv_path in extracted_csv_paths:
    csv_name = os.path.basename(csv_path)
    print(f"\nLoading {csv_name} into Pandas...")
    pdf_dict[csv_name] = pd.read_csv(csv_path, encoding="latin1")
    print(f"  {csv_name} loaded, shape: {pdf_dict[csv_name].shape}")

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
        
        # Remove NaN
        
        df_master = df_master
        df_master.to_csv("master_dataset_combined.csv", index=False)
        
        print(f"Success! Master dataset created.")
        print(f"Total Rows: {len(df_master)}")
        print(f"Sample of Building Names joined: {df_master['sitename'].unique()[:5]}")
    else:
        print("Error: Metadata file not found. Check zip extraction.")
else:
    print("Error: No meter readings found.")
    
print("\nCalculate EUI Monthly...")
building_data = pdf_dict['building_metadata.csv']

power_usage = calculate_monthly_eui(df_master, building_data)

power_usage = power_usage
power_usage = power_usage.reset_index(drop=True)
power_usage.to_csv('energy_EUI_monthly.csv', index=False)

cols_to_use_from_df2 = [
    'simscode_clean', 'year', 'month', 
    'kbtu', 'Monthly_EUI', 'buildingnumber_clean'
]
df2_subset = power_usage[cols_to_use_from_df2]

df_master['simscode'] = df_master['simscode'].astype(str).str.replace(r'\.0$', '', regex=True)
df2_subset['simscode_clean'] = df2_subset['simscode_clean'].astype(str).str.strip()

# 2. Perform the Merge
mega_table = pd.merge(
    left=df_master,
    right=df2_subset,
    # Map the columns from Table 1 to Table 2
    left_on=['simscode', 'year', 'month'], 
    right_on=['simscode_clean', 'year', 'month'],
    how='left'
)

# 3. Cleanup (Optional)
# If you don't want the extra "clean" columns hanging around after the join:
mega_table = mega_table.drop(columns=['simscode_clean_x', 'simscode_clean_y', 'buildingnumber_clean'])
mega_table = mega_table
mega_table = mega_table.reset_index(drop=True)

print('Saving big mega dataset')
mega_table.to_csv('big_mega_dataset.csv', index=False)
