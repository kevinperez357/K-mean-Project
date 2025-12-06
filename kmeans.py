import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

plt.rcParams['figure.figsize'] = (7,4)
df = pd.read_csv(r"C:\Users\kevin\OneDrive\Documents\Programacion\Python\Credit Card GENERAL.csv")
df.head()

df.info()
df.isna().mean().sort_values(ascending=False).head(8)
data = df.copy()
if 'CUST_ID' in data.columns:
	data = data.drop(columns=['CUST_ID'])
data = data.fillna(data.median(numeric_only=True))

df.select_dtypes(include='object').columns
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.replace(',', '', regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')

data = df.copy()
if 'CUST_ID' in data.columns:
    data = data.drop(columns=['CUST_ID'])
data = data.fillna(data.median(numeric_only=True))

scaler = StandardScaler()
X = scaler.fit_transform(data)
X[:3]

inertias = []
K_range = range(2, 10)
for k in K_range:
	km = KMeans(n_clusters=k, n_init=20, random_state=42)
	km.fit(X)
	inertias.append(km.inertia_)

plt.plot(list(K_range), inertias, marker='o')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inertia (SSE)')
plt.title('Método del codo')
plt.show()

k = 5
kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
labels = kmeans.fit_predict(X)

pca = PCA(n_components=2, random_state=42)
X2 = pca.fit_transform(X)

plt.scatter(X2[:,0], X2[:,1], c=labels, alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'Clusters formados con k = {k}')
plt.show()

seg = df.copy()
seg['cluster'] = labels
profile_mean = seg.groupby('cluster').mean(numeric_only=True)
profile_mean.round(2)
