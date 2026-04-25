import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("typhoid.csv")

# Define age bins and labels
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # adjust max if needed
labels = ["1-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-80", "81-90", "91-100"]

# Create a new column for age brackets
df["Age Bracket"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)

# Group by Age Bracket and calculate average fever duration
age_bracket_fever = df.groupby("Age Bracket")["Fever Duration (Days)"].mean().reset_index()

plt.figure(figsize=(12,6))
sns.barplot(data=df, x="Age Bracket", y="Fever Duration (Days)", hue="Typhoid Status", palette="Set1")
plt.title("Average Fever Duration by Age Bracket and Typhoid Status")
plt.xlabel("Age Bracket (Years)")
plt.ylabel("Average Fever Duration (Days)")
plt.show()
