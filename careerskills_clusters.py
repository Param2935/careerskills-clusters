import os
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


DATA_DIR = "data"
FIG_DIR = "figures"

K_RANGE = range(2, 16) # candidate cluster counts to sweep
RANDOM_STATE = 42
FINAL_K = 5           # chosen on interpretability
WEIGHT_ESSENTIAL = 1.0  

SEED_OCCUPATIONS = [
    "15-2051.00",   # Data scientists
    "15-1212.00",   # Information security analysts
    "25-4022.00",   # Librarians and Media collections specialists
    "13-1161.00",   # Market research analysts and marketing specialists
]

FILES = {
    "essential_skills.csv":    f"https://www.onetcenter.org/dl_files/database/db_31_0_csv/essential_skills.csv",
    "transferable_skills.csv": f"https://www.onetcenter.org/dl_files/database/db_31_0_csv/transferable_skills.csv",
    "job_zones.csv":           f"https://www.onetcenter.org/dl_files/database/db_31_0_csv/job_zones.csv",
    "occupation_data.csv":     f"https://www.onetcenter.org/dl_files/database/db_31_0_csv/occupation_data.csv",
}

HEADERS = {"User-Agent": "INST414-student-project/1.0 (educational use)"}


#Get Data 
def download_file(url, dest):
    """Download url to dest unless it is already cached."""
    if os.path.exists(dest):
        return
    import requests
    print(f"downloading {url}")
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)
    time.sleep(1)


def collect_data():
    """Download (if needed) and load the four source CSVs into DataFrames."""
    os.makedirs(DATA_DIR, exist_ok=True)
    frames = {}
    for filename, url in FILES.items():
        dest = os.path.join(DATA_DIR, filename)
        download_file(url, dest)
        df = pd.read_csv(dest)
        df.columns = df.columns.str.strip()
        frames[filename.replace(".csv", "")] = df
    return frames




# Feature matrix
# Each occupation is a 35-dimensional vector of skill importance ratings (1-5).
# Skills are standardized and similarity is Euclidean distance,which is what K-Means minimizes. The 10 essential skills are
# near-universal; WEIGHT_ESSENTIAL can down-weight them relative to the 25 transferable skills.

def build_matrix(skills, weight_essential=None):
    im = skills[skills["Scale ID"] == "IM"].copy()
 
    X_raw = im.pivot_table(
        index="O*NET-SOC Code",
        columns="Element Name",
        values="Data Value",
        aggfunc="mean",
    )
    X_raw = X_raw.dropna(axis=0, how="any")
 
    group = (skills.drop_duplicates("Element Name")
                   .set_index("Element Name")["Element ID"].str[:3])
    essential_cols = [c for c in X_raw.columns if group.get(c) == "2.A"]
 
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_raw),
        index=X_raw.index,
        columns=X_raw.columns,
    )
 
    w = WEIGHT_ESSENTIAL if weight_essential is None else weight_essential
    if w != 1.0 and essential_cols:
        X_scaled[essential_cols] *= w
 
    titles = (skills.drop_duplicates("O*NET-SOC Code")
                    .set_index("O*NET-SOC Code")["Title"]
                    .reindex(X_raw.index))
 
    return X_raw, X_scaled, titles




# Choosing k
def kmeans_sweep(X, k_range=K_RANGE):
    """Fit KMeans for each k; return a DataFrame of inertia by k."""
    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        km.fit(X)
        records.append({"k": k, "inertia": km.inertia_})
    return pd.DataFrame(records).set_index("k")
 
 
def plot_elbow(metrics, path, title_suffix=""):
    """Save the elbow plot (inertia vs k) used to choose the number of clusters."""
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(metrics.index, metrics["inertia"], "o-")
    ax.set_xlabel("k")
    ax.set_ylabel("inertia")
    ax.set_title("Elbow" + title_suffix)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
 
 
def pick_k():
    """Return the configured FINAL_K (chosen by reading the elbow plot)."""
    if FINAL_K is None:
        raise ValueError(
            "FINAL_K is not set. Inspect figures/k_selection.png and set FINAL_K.")
    print(f"Using FINAL_K = {FINAL_K}")
    return FINAL_K
 
 
def choose_k(X_scaled):
    """Sweep k and save the elbow figure used to choose k."""
    metrics = kmeans_sweep(X_scaled)
    plot_elbow(metrics, os.path.join(FIG_DIR, "k_selection.png"))
    return metrics
 
 
def fit_final(X_scaled, k):
    """Fit the final KMeans and return (model, labels Series)."""
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = pd.Series(km.fit_predict(X_scaled), index=X_scaled.index, name="cluster")
    print("cluster sizes:", labels.value_counts().sort_index().to_dict())
    return km, labels




# Analysis and description of clusters
def cluster_centroids(X_scaled, labels):
    """Mean standardized skill vector per cluster."""
    return X_scaled.groupby(labels).mean()
 
 
def _examples_near_centroid(X_scaled, labels, titles, cluster, n=5):
    """The n occupations closest to a cluster centroid (most typical members)."""
    members = X_scaled[labels == cluster]
    centroid = members.mean()
    dist = ((members - centroid) ** 2).sum(axis=1) ** 0.5
    return [titles.get(code, code) for code in dist.nsmallest(n).index]
 
 
def describe_clusters(X_raw, X_scaled, labels, titles, job_zones, n_skills=5, n_ex=5):
    """Summarize each cluster by top skills, example occupations, and Job Zone."""
    centroids = cluster_centroids(X_scaled, labels)
    jz = (job_zones.drop_duplicates("O*NET-SOC Code")
                   .set_index("O*NET-SOC Code")["Job Zone"]
                   .reindex(X_scaled.index))
 
    rows = []
    for c in sorted(centroids.index):
        top_skills = centroids.loc[c].nlargest(n_skills).index.tolist()
        examples = _examples_near_centroid(X_scaled, labels, titles, c, n_ex)
        jz_c = jz[labels == c].dropna()
        typical_zone = int(jz_c.mode().iloc[0]) if not jz_c.empty else None
        mean_zone = round(jz_c.mean(), 2) if not jz_c.empty else None
        size = int((labels == c).sum())
        suggested = " / ".join(top_skills[:2])   # placeholder; rename by hand
 
        print(f"\nCluster {c}  (n={size}, typical Job Zone {typical_zone})")
        print(f"  top skills: {', '.join(top_skills)}")
        print(f"  examples:   {'; '.join(examples)}")
 
        rows.append({
            "cluster": c,
            "size": size,
            "typical_job_zone": typical_zone,
            "mean_job_zone": mean_zone,
            "suggested_name": suggested,
            "top_skills": ", ".join(top_skills),
            "examples": "; ".join(examples),
        })
 
    return pd.DataFrame(rows).set_index("cluster")
 
 
def adjacent_careers(X_scaled, labels, titles, seeds=SEED_OCCUPATIONS, n=5):
    """For each seed occupation, list its n nearest neighbors in skill space."""
    nn = NearestNeighbors(n_neighbors=n + 1)    
    nn.fit(X_scaled.values)
 
    out = []
    for code in seeds:
        if code not in X_scaled.index:
            continue
        i = X_scaled.index.get_loc(code)
        dists, idxs = nn.kneighbors(X_scaled.values[i:i + 1])
        print(f"\nIf you like {titles.get(code, code)}:")
        for d, j in zip(dists[0], idxs[0]):
            ncode = X_scaled.index[j]
            if ncode == code:
                continue
            print(f"   - {titles.get(ncode, ncode)}  "
                  f"(cluster {labels.loc[ncode]}, dist {d:.2f})")
            out.append({"seed": titles.get(code, code),
                        "neighbor": titles.get(ncode, ncode),
                        "cluster": int(labels.loc[ncode]),
                        "distance": round(float(d), 3)})
    return pd.DataFrame(out)

# Plotting and saving figures
def make_figures(X_scaled, labels, titles, summary):
    """Save the PCA scatter, the summary table image, and the cluster CSVs."""
    os.makedirs(FIG_DIR, exist_ok=True)
    cmap = plt.get_cmap("tab10")
 
    coords = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)
    coords = pd.DataFrame(coords, index=X_scaled.index, columns=["PC1", "PC2"])
    fig, ax = plt.subplots(figsize=(10, 8))
    for c in sorted(labels.unique()):
        pts = coords[labels == c]
        ax.scatter(pts["PC1"], pts["PC2"], s=12, color=cmap(c), alpha=0.45,
                   label=f"cluster {c}", zorder=1)
    for code in SEED_OCCUPATIONS:
        if code in coords.index:
            x, y = coords.loc[code, "PC1"], coords.loc[code, "PC2"]
            ax.scatter([x], [y], s=80, facecolor="none", edgecolor="black",
                       linewidth=1.5, zorder=3)
            ax.annotate(
                titles.get(code, code),
                xy=(x, y), xytext=(12, 12), textcoords="offset points",
                fontsize=8, weight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.85),
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
            )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Occupations by skill profile (PCA, colored by cluster)")
    ax.legend(markerscale=1.5, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pca_scatter.png"), dpi=150)
    plt.close(fig)
 
    show = summary.reset_index()[["cluster", "size", "typical_job_zone",
                                  "suggested_name", "top_skills"]]
    fig, ax = plt.subplots(figsize=(13, 0.6 * len(show) + 1))
    ax.axis("off")
    tbl = ax.table(cellText=show.values, colLabels=show.columns, loc="center",
                   cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)
    fig.savefig(os.path.join(FIG_DIR, "cluster_summary_table.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
 
    summary.to_csv("cluster_summary.csv")
    pd.DataFrame({"title": titles, "cluster": labels}).to_csv("occupation_clusters.csv")
    print(f"\nsaved figures to ./{FIG_DIR}/ and cluster CSVs to project root")



def main():
    frames = collect_data()
    skills = pd.concat([frames["essential_skills"], frames["transferable_skills"]],
                       ignore_index=True)
 
    X_raw, X_scaled, titles = build_matrix(skills)
    choose_k(X_scaled)        
    k = pick_k()
    _, labels = fit_final(X_scaled, k)
 
    summary = describe_clusters(X_raw, X_scaled, labels, titles, frames["job_zones"])
    adjacent_careers(X_scaled, labels, titles)
    make_figures(X_scaled, labels, titles, summary)
 
 
if __name__ == "__main__":
    main()