# transaction_monitoring_system.py

# ---------------- XGBoost Fraud Detector ----------------
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# ---------------- XGBoost Fraud Detector ----------------
def xgboost_fraud_detector(df):
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("\n--- XGBoost Fraud Detection ---")
    print(classification_report(y_test, preds))
    print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))


# ---------------- KMeans Clustering ----------------
def run_kmeans_clustering(df):
    features = df.drop(columns=["is_fraud"], errors='ignore')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    sns.pairplot(df, hue="cluster")
    plt.savefig("cluster_plot.png")
    plt.close()


# ---------------- Autoencoder for Anomaly Detection ----------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))

def run_autoencoder(df):
    X = df.drop(columns=["is_fraud"], errors='ignore')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(X_tensor, batch_size=32, shuffle=True)

    model = Autoencoder(X.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(10):
        for batch in train_loader:
            output = model(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Autoencoder Epoch {epoch+1}, Loss: {loss.item():.4f}")


# ---------------- Isolation Forest ----------------
def detect_anomalies(df):
    features = df.drop(columns=["is_fraud"], errors='ignore')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    model = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly'] = model.fit_predict(X_scaled)
    df.to_csv("transactions_with_anomalies.csv", index=False)


# ---------------- Graph-Based Detection ----------------
def build_graph(graph_df):
    G = nx.from_pandas_edgelist(graph_df, source='sender', target='receiver', edge_attr=True)
    communities = nx.algorithms.community.greedy_modularity_communities(G)
    print(f"\n--- Graph Network Detection ---\nDetected {len(communities)} communities.")
    nx.draw(G, with_labels=True, node_size=50)
    plt.savefig("graph_network.png")
    plt.close()


# ---------------- NLP Keyword Scanner ----------------
SUSPICIOUS_KEYWORDS = ["drugs", "weapon", "cash", "bitcoin", "fraud", "illegal"]

def flag_descriptions(df):
    stop_words = set(stopwords.words('english'))
    def scan(text):
        words = word_tokenize(str(text).lower())
        words = [w for w in words if w not in stop_words]
        return int(any(k in words for k in SUSPICIOUS_KEYWORDS))
    df['suspicious_flag'] = df['description'].apply(scan)
    df.to_csv("transactions_with_flags.csv", index=False)


# ---------------- Main ----------------
def main():
    df = pd.read_csv("data/transactions.csv")
    graph_df = pd.read_csv("data/transactions_graph.csv")

    xgboost_fraud_detector(df)
    run_kmeans_clustering(df)
    run_autoencoder(df)
    detect_anomalies(df)
    build_graph(graph_df)
    flag_descriptions(df)


if __name__ == '__main__':
    main()
