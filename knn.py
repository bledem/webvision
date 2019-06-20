# coding: utf-8
import os
import numpy as np
import falconn
import sys


"""Locality-Sensitive Hashing (LSH) is a class of methods for the nearest neighbor search problem, 
which is defined as follows: given a dataset of points in a metric space (e.g., Rd with the Euclidean distance), 
our goal is to preprocess the data set so that we can quickly answer nearest neighbor queries: given a previously unseen query point, 
we want to find one or several points in our dataset that are closest to the query point. LSH is one of the main techniques
 for nearest neighbor search in high dimensions"""


args = sys.argv
#Load data from a text file. Each row in the text file must have the same number of values.
points = np.loadtxt(args[1])   #two dimensionnal array (array of the features) of training data
querys = np.loadtxt(args[2])   #two dim array of val data
k = int(args[3]) #k nearest neighbor

#A function that sets default parameters based on the following dataset properties
parm = falconn.get_default_parameters(points.shape[0], points.shape[1], distance=falconn.DistanceFunction.NegativeInnerProduct) #dataset_size, dimension, distance function
#takes a dataset and builds an LSH data structure
lsh = falconn.LSHIndex(parm)
#Build the LSH data structure from a given dataset.
lsh.setup(points)
# construct a query object by calling construct_query_object() method of the LSHIndex object;
q = lsh.construct_query_object()

for p in querys:
#Find the keys of the k closest candidates in the probing sequence for q. The keys are returned in order of increasing distance to q.
    r = q.find_k_nearest_neighbors(p, k) #find the k closest training data from validation query p
    print(' '.join(map(str,r)))

