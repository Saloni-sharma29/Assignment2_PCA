import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

iris= load_iris()
print(iris.keys())
#print(iris['DESCR']) #discription of dataset

df=pd.DataFrame(iris['data'], columns=iris['feature_names'])
print(df.head(150))
#plotting scatter plot for original data
x_index=0
y_index=1
formatter=plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target, s=10 )
plt.colorbar(ticks=[0,1,2], format=formatter)
plt.xlabel(iris.feature_names[x_index])
plt.ylabel(iris.feature_names[y_index])
#plt.tight_layout()
plt.text(5.841,4.439, 'Before PCA Scatter Plot')
plt.show()

scaler=StandardScaler()  #scale down all value 
scaler.fit(df)

scaled_data= scaler.transform(df)  #sandared deviation=1, mean =0
#print(scaled_data)
#perfoming pca over scaled data

pca=PCA(n_components=2, svd_solver='full') #reduced dimension will be 2

pca.fit(scaled_data)
x_pca=pca.transform(scaled_data)
#print(x_pca)

print("***********************PCA******************************")
print('Before performing PCA Dimension:{} '.format(scaled_data.shape))

print('After performing PCA Dimension:{}'.format(x_pca.shape) )
print("\n***********************KMeansCluster********************")

#plotting reduced data based on their target value
plt.scatter(x_pca[:,0],x_pca[:,1], c=iris['target'], s=10)
plt.text(-1.34, 2.77,'After PCA Scatter Plot')
formatter=plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])
plt.colorbar(ticks=[0,1,2], format=formatter)
plt.show()

#cluster alanysis using KMeans
kmeans=KMeans(n_clusters=4) #specifying the number of clusters prior
print("Number of clusters will be for K-Means {}".format(kmeans.n_clusters))

kModel=kmeans.fit(x_pca)#fits kmeans object to reduced dataset
#print(kModel)

#print(kModel.labels_)
#kModel.cluster_centers_
print('Data points belong to which cluster ')
print(pd.crosstab(iris.target, kModel.labels_ ))
#plt.scatter(x_pca[:,0],x_pca[:,1], c=iris['target'])
#plt.show()

#identifying clusters centers
clusters= kmeans.cluster_centers_
print('Cluster Centers are: \n{}'.format(clusters))
#print("\nCluster1: Blue\nCluster2: Red\nCluster3: Yellow\nCluster4: Green ")
y_km= kmeans.fit_predict(x_pca)
#print(y_km)
plt.scatter(x_pca[y_km==0,0], x_pca[y_km==0,1],s=20, color='red')
plt.scatter(x_pca[y_km==1,0], x_pca[y_km==1,1],s=20, color='blue')
plt.scatter(x_pca[y_km==2,0], x_pca[y_km==2,1],s=20, color='yellow')
plt.scatter(x_pca[y_km==3,0], x_pca[y_km==3,1],s=20, color='green')
#Marking centre of clusters
plt.scatter(clusters[0,0], clusters[0,1], marker='*', s=30, color='black')
plt.scatter(clusters[1,0], clusters[1,1], marker='*', s=30, color='black')
plt.scatter(clusters[2,0], clusters[2,1], marker='*', s=30, color='black')
plt.scatter(clusters[3,0], clusters[3,1], marker='*', s=30, color='black')
plt.text(-0.30, 2.69,'Making Clusters')
plt.show()

#performing linear regressions & #preprocessing the  data before performing linear regression
linreg= LinearRegression()
data1= x_pca[:,0].reshape(-1,1)  #listing values of 1st variable 
#print(data1)
data2= x_pca[:,1]  #listing values of 2nd variable 
#print('\n',data2)
linreg.fit(data1,data2)
#consider data2 as dependent variable and data1 as  independent variable
data2_predict= linreg.predict(data1)
#plotting linear regression line
print(linreg.coef_)
print(linreg.intercept_)
plt.scatter(data1,data2)
plt.plot(data1, data2_predict, color='red')
plt.text(-0.60, 2.71,'Linear Regression')
plt.show()





