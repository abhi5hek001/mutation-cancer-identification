import pandas as pd
import requests
from tqdm import tqdm

# SETTINGS
INPUT_FILE = "dataset/blca_tcga_pan_can_atlas_2018.tar/blca_tcga_pan_can_atlas_2018/data_mutations.txt"
OUTPUT_FILE = "BLCA_silent.csv"
WINDOW_SIZE = 20
MAX_SILENT = 4000
SKIP_ROWS = 10000

# Get cancer type from output file
CANCER_TYPE = OUTPUT_FILE.split("/")[-1][:4]

# Iterator over file, skipping first 10,000 rows
print(f"Reading from {INPUT_FILE}, skipping first {SKIP_ROWS} rows...")
silent_mutations = []
reader = pd.read_csv(INPUT_FILE, sep="\t", comment='#', low_memory=False, chunksize=1000, skiprows=range(1, SKIP_ROWS + 1))

for chunk in reader:
    chunk = chunk.dropna(subset=["Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2"])
    silent_chunk = chunk[chunk["Variant_Classification"] == "Silent"]
    silent_mutations.append(silent_chunk)

    total = sum(len(c) for c in silent_mutations)
    if total >= MAX_SILENT:
        break

# Concatenate and trim to 4000
df = pd.concat(silent_mutations).head(MAX_SILENT)
print(f"✅ Found {len(df)} 'Silent' mutations.")

# Function to get genomic sequence around mutation
def fetch_sequence(chrom, pos, strand="+", window=20):
    start = int(pos) - window
    end = int(pos) + window
    strand_code = 1 if strand == "+" else -1
    url = f"https://rest.ensembl.org/sequence/region/human/{chrom}:{start}..{end}:{strand_code}"
    headers = {"Content-Type": "text/plain"}

    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.ok:
            return r.text.strip()
        else:
            return None
    except Exception as e:
        print(f"Error fetching {chrom}:{pos} - {e}")
        return None

# Fetch sequences from Ensembl
sequences = []
print("Fetching sequences from Ensembl...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    chrom = str(row["Chromosome"]).replace("chr", "")
    pos = int(row["Start_Position"])
    strand = "+" if row.get("Strand", 1) == 1 else "-"
    seq = fetch_sequence(chrom, pos, strand, window=WINDOW_SIZE)
    sequences.append(seq)

# Add sequences and cancer type to DataFrame
df["Genomic_Context_Sequence"] = sequences
df["Cancer_Type"] = CANCER_TYPE

# Select key columns to save
columns_to_keep = [
    "Hugo_Symbol", "Chromosome", "Start_Position", "End_Position",
    "Reference_Allele", "Tumor_Seq_Allele2", "Variant_Classification",
    "Variant_Type", "Strand", "Genomic_Context_Sequence", "Cancer_Type"
]

# Save output
df[columns_to_keep].to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved {len(df)} rows to: {OUTPUT_FILE}")
