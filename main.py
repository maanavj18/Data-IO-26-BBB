# ---------------------------
# Install gdown
# ---------------------------


import gdown
import zipfile
import pandas as pd
import os

# Temporary folder for downloads & extraction
tmp_folder = "/tmp/energy_dataset"
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

# Example usage
train = pdf_dict.get("meter-readings-march-2025.csv")
train.head()