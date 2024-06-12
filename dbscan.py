#!/usr/bin/env python3

from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt


def DBSCAN(data, radio, vecinos_min):
    tree = KDTree(data)

    labels = [None for i in range(len(data))]
    clusters = 0

    for i in range(len(data)):
        if labels[i] is not None:
            continue

        # Use kdtree
        pI = data[i]
        neighbors = tree.query_ball_point(pI, radio)  # Implement

        if len(neighbors) < vecinos_min:
            labels[i] = -1  # Noise
            continue

        clusters += 1
        new_cluster = clusters
        labels[i] = new_cluster

        S = set(neighbors)  # Use kdtree to get close.
        # print(S)

        while S:
            q = S.pop()

            if labels[q] == -1:
                labels[q] = new_cluster

            if labels[q] is not None:
                continue

            pJ = data[q]
            neighbors = tree.query_ball_point(pJ, radio)  # Implement

            labels[q] = new_cluster

            if len(neighbors) < vecinos_min:
                continue

            S.update(neighbors)

    return labels
