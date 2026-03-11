import pandas as pd

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true])

# Keep only required columns
data = data[["text", "label"]]

# Save new dataset
data.to_csv("dataset/news_dataset.csv", index=False)

print("Dataset prepared successfully!")
print("Total rows:", len(data))