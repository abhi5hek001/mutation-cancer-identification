import pandas as pd
import requests
from tqdm import tqdm

# SETTINGS
INPUT_FILE = "dataset/blca_tcga_pan_can_atlas_2018.tar/blca_tcga_pan_can_atlas_2018/data_mutations.txt"
OUTPUT_FILE = "BLCA_silent.csv"
WINDOW_SIZE = 20 

# Load MAF or mutation file (tab-delimited)
df = pd.read_csv(INPUT_FILE, sep="\t", comment='#', low_memory=False, nrows=100000)

# Filter out rows missing key fields
df = df.dropna(subset=["Chromosome", "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2"])

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

# Apply sequence fetching row-by-row
sequences = []
print("Fetching sequences from Ensembl...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    chrom = str(row["Chromosome"]).replace("chr", "")
    pos = int(row["Start_Position"])
    strand = "+" if row.get("Strand", 1) == 1 else "-"
    seq = fetch_sequence(chrom, pos, strand, window=WINDOW_SIZE)
    sequences.append(seq)

# Add the result to the dataframe
df["Genomic_Context_Sequence"] = sequences

# Select key columns (you can modify this list as needed)
columns_to_keep = [
    "Hugo_Symbol", "Chromosome", "Start_Position", "End_Position",
    "Reference_Allele", "Tumor_Seq_Allele2", "Variant_Classification",
    "Variant_Type", "Strand", "Genomic_Context_Sequence"
]

# Save output
df[columns_to_keep].to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved results to: {OUTPUT_FILE}")
