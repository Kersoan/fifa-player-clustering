import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Page Config
st.set_page_config(page_title="FIFA Player Clustering", layout="wide")

# Title
st.title("⚽ FIFA Player Clustering App")
st.markdown("Analyze and group football players based on their skills using Machine Learning")

# Upload Dataset
uploaded_file = st.file_uploader("📂 Upload FIFA Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Sidebar options
    st.sidebar.header("⚙️ Settings")
    k = st.sidebar.slider("Select Number of Clusters", 2, 10, 4)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Feature selection
    features = [
        'pace','shooting','passing','dribbling',
        'defending','physic',
        'attacking_finishing',
        'movement_acceleration',
        'movement_sprint_speed'
    ]

    X = df[features].fillna(0)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    st.success("✅ Clustering Completed!")

    # Layout columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔍 Filter by Cluster")
        cluster_id = st.selectbox("Select Cluster", sorted(df['Cluster'].unique()))
        st.dataframe(df[df['Cluster'] == cluster_id][['short_name','Cluster']].head(20))

    with col2:
        st.subheader("🔎 Search Player")
        player = st.selectbox("Select Player", df['short_name'])
        st.dataframe(df[df['short_name'] == player])

    # PCA Visualization
    st.subheader("📈 Cluster Visualization")

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots()
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=df['Cluster'])
    plt.title("Player Clusters (PCA)")

    st.pyplot(fig)

    # Cluster Meaning
    st.subheader("🧠 Cluster Insights")
    st.markdown("""
    - **Cluster 0** → Likely Attackers (High Shooting)  
    - **Cluster 1** → Midfielders (Passing + Stamina)  
    - **Cluster 2** → Defenders (Defending + Strength)  
    - **Cluster 3** → Wingers (Speed + Dribbling)  
    """)

else:
    st.info("👆 Please upload a dataset to begin")