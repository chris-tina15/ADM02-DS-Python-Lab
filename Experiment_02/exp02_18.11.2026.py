#Experiment-18.11.202
import sqlite3
import pandas as pd
import json

# Step 1: Create database
conn = sqlite3.connect("Experiment_02/hospital.db")
cur = conn.cursor()

# Step 2: Create tables
cur.executescript("""
CREATE TABLE doctors (
    doctor_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department TEXT NOT NULL
);

CREATE TABLE patients (
    patient_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER,
    gender TEXT,
    city TEXT
);

CREATE TABLE visits (
    visit_id INTEGER PRIMARY KEY,
    patient_id INTEGER REFERENCES patients(patient_id),
    doctor_id INTEGER REFERENCES doctors(doctor_id),
    visit_date TEXT,
    fee REAL
);
""")

# Step 3: Insert doctors
cur.executemany(
    "INSERT INTO doctors VALUES (?,?,?)",
    [
        (1, "Dr. Ramesh", "Cardiology"),
        (2, "Dr. Kavitha", "Orthopaedics"),
        (3, "Dr. Imran", "General Medicine")
    ]
)

# Step 4: Insert patients
cur.executemany(
    "INSERT INTO patients VALUES (?,?,?,?,?)",
    [
        (1, "Ajay", 45, "M", "Chennai"),
        (2, "Bhavana", 32, "F", "Madurai"),
        (3, "Charles", 67, "M", "Chennai"),
        (4, "Deepa", 29, "F", "Coimbatore"),
        (5, "Elango", 54, "M", "Madurai"),
        (6, "Fathima", 38, "F", "Chennai")
    ]
)

# Step 5: Insert visits
cur.executemany(
    "INSERT INTO visits VALUES (?,?,?,?,?)",
    [
        (1, 1, 1, "2026-01-05", 800),
        (2, 2, 2, "2026-01-06", 600),
        (3, 3, 1, "2026-01-08", 800),
        (4, 4, 3, "2026-01-09", 300),
        (5, 5, 1, "2026-01-12", 800),
        (6, 1, 3, "2026-01-15", 300),
        (7, 6, 2, "2026-01-18", 600),
        (8, 3, 3, "2026-01-20", 300),
        (9, 2, 2, "2026-01-22", 600),
        (10, 5, 3, "2026-01-25", 300)
    ]
)

conn.commit()

# Function to execute SQL queries
def run(title, sql):
    print(f"\n--- {title} ---")
    print("SQL:", " ".join(sql.split()))
    print(pd.read_sql_query(sql, conn).to_string(index=False))


print("HOSPITAL PATIENT RECORDS")
print("=" * 60)

# Q1: SELECT query
run(
    "Q1. Patients older than 40",
    """
    SELECT name, age, city
    FROM patients
    WHERE age > 40
    ORDER BY age DESC
    """
)

# Q2: JOIN query
run(
    "Q2. Visits with patient and doctor names",
    """
    SELECT v.visit_id,
           p.name AS patient,
           d.name AS doctor,
           d.department,
           v.fee
    FROM visits v
    JOIN patients p ON v.patient_id = p.patient_id
    JOIN doctors d ON v.doctor_id = d.doctor_id
    ORDER BY v.visit_id
    """
)

# Q3: GROUP BY query
run(
    "Q3. Revenue per department",
    """
    SELECT d.department,
           COUNT(*) AS visits,
           SUM(v.fee) AS revenue
    FROM visits v
    JOIN doctors d ON v.doctor_id = d.doctor_id
    GROUP BY d.department
    ORDER BY revenue DESC
    """
)

# Q4: HAVING query
run(
    "Q4. Patients with more than one visit",
    """
    SELECT p.name, COUNT(*) AS visits
    FROM visits v
    JOIN patients p ON v.patient_id = p.patient_id
    GROUP BY p.name
    HAVING COUNT(*) > 1
    """
)

# Step 6: UPDATE and DELETE operations
cur.execute(
    "UPDATE doctors SET department = 'Internal Medicine' WHERE doctor_id = 3"
)

cur.execute("DELETE FROM visits WHERE visit_id = 10")
conn.commit()

print("\nUpdated doctor 3 department; deleted visit 10.")

run(
    "Q5. Doctors table after update",
    "SELECT * FROM doctors"
)

print(
    "\nTotal visits after delete:",
    cur.execute("SELECT COUNT(*) FROM visits").fetchone()[0]
)

# Step 7: Load SQL data into Pandas
df = pd.read_sql_query(
    """
    SELECT p.city, p.gender, v.fee
    FROM visits v
    JOIN patients p ON v.patient_id = p.patient_id
    """,
    conn
)

print("\nCITY-WISE AVERAGE FEE USING PANDAS")

print(
    df.groupby("city")["fee"]
    .agg(["count", "mean"])
    .round(2)
    .to_string()
)

# Step 8: Convert relational data into NoSQL documents
rows = pd.read_sql_query(
    """
    SELECT p.patient_id, p.name, p.age, p.city,
           d.department, v.visit_date, v.fee
    FROM patients p
    JOIN visits v ON p.patient_id = v.patient_id
    JOIN doctors d ON v.doctor_id = d.doctor_id
    ORDER BY p.patient_id, v.visit_date
    """,
    conn
)

conn.close()

collection = []

for pid, grp in rows.groupby("patient_id"):
    first = grp.iloc[0]

    collection.append({
        "_id": int(pid),
        "name": first["name"],
        "age": int(first["age"]),
        "city": first["city"],
        "visits": [
            {
                "dept": r.department,
                "date": r.visit_date,
                "fee": float(r.fee)
            }
            for r in grp.itertuples()
        ]
    })

with open("Experiment_02/patients_nosql.json", "w") as f:
    json.dump(collection, f, indent=2)

print("\nNOSQL DOCUMENT MODEL")

print(json.dumps(collection[0], indent=2))

# Step 9: NoSQL-style query
def find(coll, condition):
    return [doc for doc in coll if condition(doc)]

print("\nMongoDB-style query:")
print("Find Chennai patients who visited Cardiology")

result = find(
    collection,
    lambda d: d["city"] == "Chennai"
    and any(v["dept"] == "Cardiology" for v in d["visits"])
)

for doc in result:
    print(f"{doc['_id']} {doc['name']} visits = {len(doc['visits'])}")

print("\nEXPERIMENT 2 COMPLETED SUCCESSFULLY!")
