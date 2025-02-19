import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats

#LOADING DATA
df=pd.read_csv("C:\\Users\\user\\Desktop\\Mall_Customers.csv")
print(df)

#REMOVING CATEGORIC COLUMNS
df1=df.drop(columns=["CustomerID","Gender","Age"])

#FILLING MISSING VALUES
df1=df1.fillna(df1.mean())

#REMOVING OUTLIERS
mask=(np.abs(stats.zscore(df1))<=3).any(axis=1)
df1=df1[mask]

#SCALING
scaler=StandardScaler()
scaled=scaler.fit_transform(df1)

#CATEGORIZING USING KMEANS ALGORITHM
kmeans=KMeans(n_clusters=3,random_state=42)
df["Categorie"]=kmeans.fit_predict(scaled)

#LABELING THE CATEGORIES
label_mapping = {0: "CUSTOMERS WITH LOWER WILLING TO SPEND", 1: "CUSTOMERS WITH MODERATE WILLING", 2: "CUSTOMERS WITH HIGH WILLING TO SPEND"}
df["cluster_labels"] = df["Categorie"].map(label_mapping)

#PLOTTING
plt.figure(figsize=(15,15))
sns.scatterplot(data=df,x="Annual Income (k$)",y="Spending Score (1-100)",hue="cluster_labels",style="Gender",alpha=1)
plt.title("Our Customers")
plt.show()