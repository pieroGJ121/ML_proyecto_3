#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from random import sample


def distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def Init_Centroide(data, k):
    return np.array(sample(data.tolist(), k))


def return_new_centroide(grupos, data, k):
    kCentroids = [[] for i in range(k)]

    for i in range(len(data)):
        kCentroids[grupos[i]].append(data[i])

    return np.array([np.mean(L) for L in kCentroids])


def get_cluster(data, centroides):
    clusters = []
    for p in data:
        idx = 0
        dist = distance(p, centroides[0])
        for i in range(len(centroides)):
            dist2C = distance(p, centroides[i])
            if dist > dist2C:
                dist = dist2C
                idx = i
        clusters.append(idx)

    return np.array(clusters)


def distancia_promedio_centroides(old_centroides, new_centroides):
    # Initialize array of distances
    promedios = []
    # Iterate each
    for i in range(len(old_centroides)):
        # Get distance between them
        dist = distance(old_centroides[i], new_centroides[i])
        # Append to mean
        promedios.append(dist)
    # Return mean value
    return np.mean(promedios)


def kmeans(data, k, umbral):
    centroides = Init_Centroide(data, k)
    clusters = get_cluster(data, centroides)
    new_centroides = return_new_centroide(clusters, data, k)
    while distancia_promedio_centroides(centroides, new_centroides) > umbral:
        centroides = new_centroides
        clusters = get_cluster(data, centroides)
        new_centroides = return_new_centroide(clusters, data, k)

    return new_centroides, clusters
