import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Given a Pandas Dataframe computes a kmeans clustering
def kMeans(df, clusters=5):
    km = KMeans(n_clusters=clusters)
    labels = km.fit_predict(df.values)
    return (labels, km.inertia_)

# Finds the best number of clusters to use based on 'elbow method'
def findBestK(df):
    results = [kMeans(df, i) for i in range(1, 11)]
    distortions = [x[1] for x in results]
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.show()
    return results


#Performs SVD on the data and returns singular values/variance of ith pc

def pca(df, i):
    vals = df.values
    u,e,vt = np.linalg.svd(vals, full_matrices=True)
    #Captures variance of the ith pc if we want it
    #ith principal component's variance is (ith singular value)^2/N
    var_i = e[i]**2 / len(vals)
    return u,e,vt, var_i



#Seaborn Heatmap
general = "https://raw.githubusercontent.com/saurav-c/NBAClassification/master/Data/More%20Player%20Data%20-%20general.csv"
df1 = pd.read_csv(general);
Boxout = "https://raw.githubusercontent.com/saurav-c/NBAClassification/master/Data/Playtype%20Data_Box%20Outs%20-%20PR%20Ball%20Handler.csv"
df2 = pd.read_csv(Boxout);
tracking_data2 = "https://raw.githubusercontent.com/saurav-c/NBAClassification/master/Data/tracking_data2.csv"
df3 = pd.read_csv(tracking_data2);

plt.figure(figsize=(16, 16))
sns.heatmap(df1.corr(), annot=True);
