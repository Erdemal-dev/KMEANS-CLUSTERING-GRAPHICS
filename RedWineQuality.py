import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#LOAD THE DATA and display it
df=pd.read_csv("C:\\Users\\user\\Desktop\\winequality-red.csv")


#EXRACT THE OUTLIERS
mask=(np.abs(stats.zscore(df))<2).any(axis=1)
df=df[mask]

fig,ax=plt.subplots(1,2,figsize=(10,10))

df1=df[["fixed acidity","volatile acidity"]]
scaler=StandardScaler()
df1=scaler.fit_transform(df1)


kmeans=KMeans(n_clusters=4,random_state=42)
df["Absolute Acidity"]=kmeans.fit_predict(df1)
dic1={0:"Low Acidity",1:"Moderate Acidity",2:"Hıgh Acidity",3:"EXTREME ACIDITY!"}
df["Absolute Acidity"]=df["Absolute Acidity"].map(dic1)


df2=df[["residual sugar","alcohol"]]
df2=scaler.fit_transform(df2)
kmeans=KMeans(n_clusters=3,random_state=42)
df["Harmfulness"] =kmeans.fit_predict(df2)
dic2={0:"Harmless",1:"Little Harmful",2:"Harmfull"}
df["Harmfulness"]=df["Harmfulness"].map(dic2)

sns.scatterplot(data=df,x="fixed acidity",y="volatile acidity",hue="Absolute Acidity",ax=ax[0],alpha=1)
sns.kdeplot(data=df,x="fixed acidity",y="volatile acidity",hue="Absolute Acidity",ax=ax[0],alpha=1,fill=True)
plt.title("Absolute Acidity")
sns.scatterplot(data=df,x="residual sugar",y="alcohol",hue="Harmfulness",ax=ax[1],alpha=1)
sns.kdeplot(data=df,x="residual sugar",y="alcohol",hue="Harmfulness",ax=ax[1],alpha=1,fill=True)
plt.title("Harmfullnes")

plt.show()
