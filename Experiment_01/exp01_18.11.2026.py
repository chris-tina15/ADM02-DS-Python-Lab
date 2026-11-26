# Experiment 1 - 18.11.2026
import csv
import json
import os
import pandas as pd

os.makedirs("Experiment_01/data", exist_ok=True)

# Step 1: Create CSV file
pos_rows = [
    ["order_id", "customer", "product", "qty", "unit_price", "branch"],
    [101, "Anitha", "Laptop", 1, 54000, "Chennai"],
    [102, "Rahul", "Mouse", 3, 450, "Chennai"],
    [103, "Priya", "Keyboard", 2, 1200, "Chennai"],
    [104, "Karthik", "Monitor", 1, 9500, "Chennai"]
]

with open("Experiment_01/data/pos_orders.dxccx", "w", newline="") as f:
    csv.writer(f).writerows(pos_rows)

# Step 2: Create JSON file
web_orders = [
    {"order_id": 201, "customer": "Divya", "product": "Laptop",
     "qty": 2, "unit_price": 52500, "branch": "Online"},
    {"order_id": 202, "customer": "Arjun", "product": "Headset",
     "qty": 1, "unit_price": 2300, "branch": "Online"},
    {"order_id": 203, "customer": "Meena", "product": "Mouse",
     "qty": 5, "unit_price": 430, "branch": "Online"}
]

with open("Experiment_01/data/web_orders.json", "w") as f:
    json.dump(web_orders, f, indent=2)

# Step 3: Create TXT file
log_lines = [
    "301|Suresh|Monitor|2|9300|Mobile",
    "302|Lakshmi|Keyboard|1|1150|Mobile",
    "303|Vignesh|Laptop|1|55000|Mobile"
]

with open("Experiment_01/data/mobile_orders.txt", "w") as f:
    f.write("\n".join(log_lines))

print("RETAIL ORDER DATA COLLECTION")
print("=" * 55)

# Step 4: Read CSV
df_pos = pd.read_csv("Experiment_01/data/pos_orders.csv")
print("\nCSV DATA:")
print(df_pos.to_string(index=False))

# Step 5: Read JSON
with open("Experiment_01/data/web_orders.json") as f:
    df_web = pd.DataFrame(json.load(f))

print("\nJSON DATA:")
print(df_web.to_string(index=False))

# Step 6: Read TXT
cols = ["order_id", "customer", "product", "qty", "unit_price", "branch"]

df_mob = pd.read_csv(
    "Experiment_01/data/mobile_orders.txt",
    sep="|",
    header=None,
    names=cols
)

print("\nMOBILE DATA:")
print(df_mob.to_string(index=False))

# Step 7: Integrate datasets
orders = pd.concat(
    [df_pos, df_web, df_mob],
    ignore_index=True
)

orders["amount"] = orders["qty"] * orders["unit_price"]

print("\nINTEGRATED DATASET:")
print(orders.to_string(index=False))

print("\nDataset Shape:", orders.shape)

# Step 8: Store in different formats
orders.to_csv("Experiment_01/data/all_orders.csv", index=False)

orders.to_json(
    "Experiment_01/data/all_orders.json",
    orient="records",
    indent=2
)

orders.to_excel(
    "Experiment_01/data/all_orders.xlsx",
    index=False
)

print("\nSTORED FILE SIZES:")

for name in ["all_orders.csv", "all_orders.json", "all_orders.xlsx"]:
    path = "Experiment_01/data/" + name
    print(name, ":", os.path.getsize(path), "bytes")

# Step 9: Retrieve stored Excel data
back = pd.read_excel("Experiment_01/data/all_orders.xlsx")

print("\nRETRIEVED ROWS:", len(back))

# Step 10: Revenue by branch
print("\nREVENUE BY BRANCH:")

print(
    back.groupby("branch")["amount"]
    .sum()
    .sort_values(ascending=False)
)

# Step 11: High-value orders
print("\nORDERS ABOVE Rs. 10,000:")

print(
    back.loc[
        back["amount"] > 10000,
        ["order_id", "customer", "product", "amount"]
    ].to_string(index=False)
)
