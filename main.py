#Progress report (5%) – due 11:59pm on November 11th

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = "T_ONTIME_MARKETING.csv"  # Replace with your dataset path
data = pd.read_csv(file_path)

# Display the first few rows to understand the structure
print(data.head())

# Data Cleaning: Removing duplicates
data = data.drop_duplicates()

# Add a synthetic "Delay" column for demonstration purposes (binary classification)
np.random.seed(42)
data['Delay'] = np.random.choice([0, 1], size=len(data), p=[0.8, 0.2])  # 80% on-time, 20% delayed

# Selecting features and target
X = data[['ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'ORIGIN_CITY_MARKET_ID', 'DEST_CITY_MARKET_ID']]
y = data['Delay']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(random_state=42, n_estimators=100)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the results
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", conf_matrix)

# Calculate the counts for origin and destination airports
origin_counts = data['ORIGIN_AIRPORT_ID'].value_counts()
dest_counts = data['DEST_AIRPORT_ID'].value_counts()

# Top 10 most frequent origin and destination airports
top_origin_counts = origin_counts.head(10)
top_dest_counts = dest_counts.head(10)

# Plotting the graph with thicker bars and better visibility
plt.figure(figsize=(16, 8))

# Adjusting bar positions to avoid overlap
bar_width = 0.4
x_origin = [i - bar_width / 2 for i in range(len(top_origin_counts))]
x_dest = [i + bar_width / 2 for i in range(len(top_dest_counts))]

# Plot origin and destination bars with thicker widths
plt.bar(x_origin, top_origin_counts.values, width=bar_width, label='Origin Airports', alpha=0.7, linewidth=2, edgecolor='black')
plt.bar(x_dest, top_dest_counts.values, width=bar_width, label='Destination Airports', alpha=0.5, linewidth=2, edgecolor='black')

# Adding frequency values above the bars
for i, value in enumerate(top_origin_counts.values):
    plt.text(x_origin[i], value + 2, str(value), ha='center', va='bottom', fontsize=10)
for i, value in enumerate(top_dest_counts.values):
    plt.text(x_dest[i], value + 2, str(value), ha='center', va='bottom', fontsize=10)

# Titles and labels
plt.title('Top 10 Most Frequent Airports (Origin and Destination)', fontsize=14)
plt.xlabel('Airport ID', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()

# Replace numeric indices with airport IDs
all_airport_ids = top_origin_counts.index.tolist()
plt.xticks(range(len(all_airport_ids)), all_airport_ids, rotation=45, fontsize=10)

# Ensure proper scaling of axes
plt.tight_layout()

# Show the plot
plt.show()