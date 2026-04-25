# eda_typhoid.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("typhoid.csv")

# -------------------------------
# 1. Basic Info
# -------------------------------
print("----- Basic Info -----")
print(df.head())
print(df.info())
print(df.describe(include="all"))

# -------------------------------
# 2. Distribution of Typhoid Status
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Typhoid Status", palette="Set2")
plt.title("Distribution of Typhoid Cases")
plt.show()

# -------------------------------
# 3. Age vs Typhoid Status
# -------------------------------
plt.figure(figsize=(8,5))
sns.histplot(data=df, x="Age", hue="Typhoid Status", bins=20, kde=True)
plt.title("Age Distribution by Typhoid Status")
plt.show()

# -------------------------------
# 4. Gender vs Typhoid Status
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Gender", hue="Typhoid Status", palette="coolwarm")
plt.title("Gender vs Typhoid Status")
plt.show()

# -------------------------------
# 5. Water Source vs Typhoid
# -------------------------------
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Water Source Type", hue="Typhoid Status", palette="Set1")
plt.xticks(rotation=45)
plt.title("Water Source vs Typhoid Cases")
plt.show()

# -------------------------------
# 6. Sanitation vs Typhoid
# -------------------------------
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Sanitation Facilities", hue="Typhoid Status", palette="Set3")
plt.xticks(rotation=45)
plt.title("Sanitation Facilities vs Typhoid Cases")
plt.show()

# -------------------------------
# 7. Hand Hygiene vs Typhoid
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(data=df, x="Hand Hygiene", hue="Typhoid Status", palette="pastel")
plt.title("Hand Hygiene vs Typhoid Cases")
plt.show()

# -------------------------------
# 8. Correlation (Numeric Features)
# -------------------------------
plt.figure(figsize=(10,8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Numeric Features)")
plt.show()

# -------------------------------
# 8. Age vs Fever duration (Days)
# -------------------------------
plt.figure(figsize=(10,7))
sns.barplot(data=df, x="Age", y="Fever Duration (Days)", hue="Typhoid Status", palette="Set1")
plt.title("Age vs Fever Duration")
plt.xlabel("Age")
plt.ylabel("Fever Duration (Days)")
plt.show()
