import pandas as pd
import numpy as np

# Load your original dataset
df = pd.read_csv("typhoid.csv")

# -----------------------
# 1. Add Time Dimension
# -----------------------
np.random.seed(42)  # reproducibility
df["Week"] = np.random.randint(1, 53, df.shape[0])

# -----------------------
# 2. Add Locality Dimension
# -----------------------
# Example sublocality map
sublocalities_map = {
    "Delhi": ["Delhi-North", "Delhi-South", "Delhi-East", "Delhi-West"],
    "Mumbai": ["Mumbai-North", "Mumbai-South", "Mumbai-East", "Mumbai-West"],
    "Bangalore": ["Bangalore-Central", "Bangalore-North", "Bangalore-South"],
    "Chennai": ["Chennai-North", "Chennai-South", "Chennai-East", "Chennai-West"]
}

def assign_sublocality(location):
    if location in sublocalities_map:
        return np.random.choice(sublocalities_map[location])
    else:
        return f"{location}-General"

df["SubLocality"] = df["Location"].apply(assign_sublocality)

# Save the updated dataset
df.to_csv("typhoid_with_time_locality.csv", index=False)

print("✅ Added Week & SubLocality without altering anything else.")
print(df.head())
