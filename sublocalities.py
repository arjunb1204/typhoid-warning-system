import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_csv("typhoid_with_time_locality.csv")

# Define Delhi sublocalities
delhi_sublocalities = ["Delhi-North", "Delhi-South", "Delhi-East", "Delhi-West"]

# Treat "Urban" rows as Delhi → assign Delhi sublocalities
mask = df["Location"] == "Urban"
np.random.seed(42)  # reproducibility
df.loc[mask, "SubLocality"] = np.random.choice(delhi_sublocalities, size=mask.sum())

# Save updated dataset
df.to_csv("typhoid_with_time_locality_v2.csv", index=False)

print("✅ Updated file saved as typhoid_with_time_locality_v2.csv")
print(df[mask].head())
