"""
Cluster CCLE cell lines based on expression of
top 5000 most variable genes.

Michael Heskett, Oregon Health and Science University

INPUT: CCLE_Expression_Entrez_2012-09-29.gct after you fix awful formatting.

OUTPUT: Text file with KMeans clustered labels for each cell line.
"""

from sys import argv
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load gene expression table with Pandas.
expression = pd.read_csv(argv[1],sep='\t',header=0)\
             .transpose()\
             .drop(['Description'], axis=0)


# Reduce the dimensionality from 5000 to 50.
pca = PCA(n_components=50)
model = pca.fit_transform(expression.as_matrix())


# Cluster cell lines into 30 clusters. 
clusters = KMeans(n_clusters=30)
labels_array = clusters.fit_predict(model)


# Write text file with cluster labels.
file = open(argv[2],'w')
for i in range(len(expression.index)):
    file.write(expression.index[i] + '\t' + str(labels_array[i]) + '\n')
file.close()

"""
NOTE: This is a bare bones script for dimensionality reduction and clustering
for the purpose of testing BioMedical Evidence Graph
"""
