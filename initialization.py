from cycler import K
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from kneed import KneeLocator

def K_means(X):
    def WCSS(X):
        wcss = []
        K_range = range(1, 11)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X) 
            wcss.append(kmeans.inertia_)  # WCSS

        # Find the elbow (knee) point
        kl = KneeLocator(list(K_range), wcss, curve='convex', direction='decreasing')
        optimal_k = kl.elbow

        plt.figure(figsize=(10, 6))
        plt.plot(K_range, wcss, marker='o')
        if optimal_k is not None:
            plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal K = {optimal_k}')
        plt.title("Elbow Method for Optimal K")
        plt.xlabel("Number of Clusters (K)")
        plt.ylabel("WCSS")
        plt.legend()
        plt.show()
        print(f"Optimal number of clusters (K): {optimal_k}")
        return optimal_k


    K = WCSS(X)  # Get optimal K using WCSS

    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(X)  

    means = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # Elements in each cluster
    clusters = [X[labels == i] for i in range(K)]

    
    
    return means, clusters


