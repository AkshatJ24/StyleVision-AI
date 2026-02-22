import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# --- DEFINE YOUR FOLDER PATH ---
# This tells the script where to look for everything
DATASET_FOLDER = "StyleVision AI Dataset"

# 1. Load the metadata (The spreadsheet inside your folder)
csv_path = os.path.join(DATASET_FOLDER, 'styles.csv')
df = pd.read_csv(csv_path, on_bad_lines='skip')

print("📊 Dataset Overview:")
print(f"Total items: {len(df)}")
print("\nTop 5 Categories (articleType):")
print(df['articleType'].value_counts().head(5))

# 2. Visualize a few random images
# We will pick 4 random items from the dataset
sample_items = df.sample(4)

plt.figure(figsize=(12, 4))

for i, (index, row) in enumerate(sample_items.iterrows()):
    # The image path is now: StyleVision AI Dataset/images/12345.jpg
    img_path = os.path.join(DATASET_FOLDER, 'images', f"{row['id']}.jpg")
    
    if os.path.exists(img_path):
        img = mpimg.imread(img_path)
        plt.subplot(1, 4, i + 1)
        plt.imshow(img)
        # We show the category and the specific product name
        plt.title(f"{row['articleType']}\n{row['productDisplayName'][:20]}...", fontsize=9)
        plt.axis('off')
    else:
        print(f"⚠️ Could not find image: {img_path}")

plt.tight_layout()
plt.show()