
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Generate customer dataset
np.random.seed(5)

centres = [
    (25, 80),
    (55, 20),
    (85, 85),
    (55, 50),
    (25, 20)
]

rows = []

for i, (inc, score) in enumerate(centres):
    for _ in range(40):
        rows.append([
            np.random.normal(inc, 7),
            np.random.normal(score, 7)
        ])

customers = pd.DataFrame(
    np.round(rows, 1),
    columns=["annual_income_k", "spending_score"]
)

customers.insert(
    0, "customer_id",
    range(1, len(customers) + 1)
)

print("Mall Customer Segmentation using K-Means Clustering")
print("=" * 52)

print(customers.head().to_string(index=False))
print("Total customers:", len(customers))

# Standardise features
X = StandardScaler().fit_transform(
    customers[["annual_income_k", "spending_score"]]
)

# Elbow method and silhouette analysis
print("\n k     WCSS     Silhouette")

wcss = []
sil = []

for k in range(2, 9):

    km = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=0
    ).fit(X)

    wcss.append(km.inertia_)

    sil.append(
        silhouette_score(X, km.labels_)
    )

    print(
        f"{k:>2} {km.inertia_:>8.2f}    {sil[-1]:.4f}"
    )

best_k = range(2, 9)[int(np.argmax(sil))]

print("\nBest k by silhouette score:", best_k)

# Final K-Means model
km = KMeans(
    n_clusters=best_k,
    n_init=10,
    random_state=0
).fit(X)

customers["cluster"] = km.labels_

# Cluster profiling
profile = (
    customers.groupby("cluster")[
        ["annual_income_k", "spending_score"]
    ]
    .mean()
    .round(1)
)

profile["size"] = (
    customers["cluster"]
    .value_counts()
    .sort_index()
)

def describe(row):

    inc = (
        "High income"
        if row.annual_income_k > 70
        else "Low income"
        if row.annual_income_k < 40
        else "Mid income"
    )

    sp = (
        "high spender"
        if row.spending_score > 65
        else "low spender"
        if row.spending_score < 35
        else "average spender"
    )

    return f"{inc}, {sp}"

profile["segment"] = profile.apply(
    describe,
    axis=1
)

print("\nCluster profiles:")
print(profile.to_string())

# Convert centroids to original scale
scaler = StandardScaler()
scaler.fit(customers[["annual_income_k", "spending_score"]])

# Recreate the standardised data for plotting
X = scaler.transform(
    customers[["annual_income_k", "spending_score"]]
)

# Plot elbow curve and silhouette scores
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

ax[0].plot(
    range(2, 9),
    wcss,
    marker="o"
)

ax[0].set_title("Elbow Method")
ax[0].set_xlabel("Number of Clusters (k)")
ax[0].set_ylabel("WCSS")

ax[1].plot(
    range(2, 9),
    sil,
    marker="o"
)

ax[1].set_title("Silhouette Analysis")
ax[1].set_xlabel("Number of Clusters (k)")
ax[1].set_ylabel("Silhouette Score")

plt.tight_layout()

plt.savefig(
    "Experiment_08/exp08_elbow.png",
    dpi=150
)

plt.show()

# Cluster visualisation
plt.figure(figsize=(8, 5))

for cluster in sorted(customers["cluster"].unique()):

    part = customers[customers["cluster"] == cluster]

    plt.scatter(
        part["annual_income_k"],
        part["spending_score"],
        label=f"Cluster {cluster}"
    )

centres_original = scaler.inverse_transform(km.cluster_centers_)

plt.scatter(
    centres_original[:, 0],
    centres_original[:, 1],
    marker="X",
    s=200,
    edgecolors="black",
    label="Centroids"
)

plt.xlabel("Annual Income (Thousands)")
plt.ylabel("Spending Score")
plt.title("K-Means Customer Segmentation")
plt.legend()
plt.tight_layout()

plt.savefig(
    "Experiment_08/exp08_clusters.png",
    dpi=150
)

plt.show()

print("\nGraphs saved successfully!")
print("\nEXPERIMENT 8 COMPLETED SUCCESSFULLY!")
