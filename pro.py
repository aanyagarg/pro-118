# within clusters sum of squares

import pandas as pd 
import plotly.express as pe
from sklearn.cluster import KMeans
import matplotlib.pyplot as mp 
import seaborn as sb 


data = pd.read_csv("data.csv")
 
graph = pe.scatter(data , x="size" , y = "light")
# graph.show()

# --------------------------------------------------------

X = data.iloc[: , [0,1]].values

# print(X)

# --------------------------- using wcss algo we found out the correct "k"----------------------------------------------
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i , init="k-means++" , random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

mp.figure(figsize=(10,5))

sb.lineplot(range(1,11) , wcss , marker='o' , color = 'black')
mp.title('The Elbow Graph')

mp.xlabel("Number of clusters")
mp.ylabel('WCSS')

mp.show()


# --------------------------------------------------------------------------

kmeans = KMeans( n_clusters = 3 , init="k-means++" , random_state=42)
yMeans = kmeans.fit_predict(X)


mp.figure(figsize=(15,7))

sb.scatterplot(X[yMeans == 0 , 0 ] , X[yMeans == 0 , 1 ] , color = 'red' , label='Cluster 1' )

sb.scatterplot(X[yMeans == 1 , 0 ] , X[yMeans == 1 , 1 ] , color = 'green' , label='Cluster 2' )

sb.scatterplot(X[yMeans == 2 , 0 ] , X[yMeans == 2 , 1 ] , color = 'blue' , label='Cluster 3' )

sb.scatterplot( kmeans.cluster_centers_[:,0] , kmeans.cluster_centers_[:,1]  , color = 'violet' , label='centroid' , s=100  )

mp.grid(False)

mp.title('Clusters Of Intestellar Objects')
mp.legend()
mp.xlabel("Size")
mp.ylabel('Light')

mp.show()



