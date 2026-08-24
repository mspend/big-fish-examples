import pandas as pd
from pathlib import Path

root_path = Path("/data/smfish/20260311_bartelle_smFISH_cryo_48hr_male/big_fish/results/all_tiles_2D/")

n_bits = 16

for bit in num_bits(range(1, n_bits+1)):
    if bit == 1:
        # df_name = f"bit{str(bit).zfill(3)}_df"
        filepath = root_path / f"spots_bit_{bit}.csv"
        df = pd.read_csv(filepath, header = "infer", index_col = 0)
    else:
        # df_name = f"bit{str(bit).zfill(3)}_df"
        filepath = root_path / f"spots_bit_{bit}.csv"
        df = pd.read_csv(filepath, header=None, index_col = 0)        