# Standard Library Imports
import os
import json
import sys
import math
from random import shuffle
from collections import Counter

# Third Party Imports
import pandas as pd
import numpy as np


NUMBER_OF_USERS = 943
GENRES = ['Action', 'Adventure', 'Animation',
          'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
          'Thriller', 'War', 'Western']




def file_exists(file_name):
    """Checks if the specified file exists"""

    # return os.path.isfile(os.path.abspath(file_name))
    return False


def euclidian_distance(vector_a, vector_b):
    """Calculates the euclidian distance between two vectors"""

    return math.sqrt(sum([(vector_a - vector_b) ** 2 for vector_a, vector_b in zip(vector_a, vector_b)]))


def binary_search(array, item):
    """A binary search implementation"""

    first = 0
    last = len(array) - 1

    while first <= last:
        mid = (first + last) // 2

        if array[mid] == item:
            return True  # Found at mid
        elif array[mid] > item:
            last = mid - 1
        elif array[mid] < item:
            first = mid + 1
        else:
            return False


def vector_to_group(vector, groups):
    """
    Locates the group(Cm) that the given vector is placed

    Args:
        vector: The vector to find it's group.
        groups: The list of the generated groups(Cm).

    Returns:
       index: The index of the group that the given vector was found.
    """

    for index, group in enumerate(groups):
        group_index = binary_search(group, vector)

        # If vector is not found in the first group check the next
        if group_index == False:
            continue
        else:
            return index


def group_with_min_distance(vector, groups):
    """
    Returns the minimum euclidian distance between the giver vector and the remainings.

    Args:
        vector_index: This is the vector's index that we want compare.
        vectors: This is the list of all vectors.

    Returns:
       vectors[min_index]: the vector with the minimum distance for the given vector_index
       min_distance: the minimum distance
    """

    min_distance = 10000000
    min_index = 10000000

    for index, group in enumerate(groups):
        for target_vector in group:
            if euclidian_distance(vector, target_vector) < min_distance:
                min_distance = euclidian_distance(vector, target_vector)
                min_group = index

    return min_group, min_distance


def min_max_between_all(vectors):
    """
    Calculates all the possible distances between all the vector combinations

    Args:
        vectors: This is the list of all vectors.

    Returns:
       min(distances): the minimum distance
       max(distances): the maximum distance
    """

    distances = []

    for i in range(0, len(vectors)):
        for j in range(i + 1, len(vectors)):
            distance = euclidian_distance(vectors[i], vectors[j])
            distances.append(distance)

    return min(distances), max(distances)


def BSAS(threshold, q, vectors):
    """
    An implementation of BSAS algorithm. BSAS calculates the number of clusters in a dataset.

    Args:
        threshold: This is the list of all vectors.
        q: maximum number of clusters allowed.
        vectors: the vectors generated for the given dataset

    Returns:
        len(Cm): the count of the found clusters
    """

    m = 1
    Cm = [[vectors[0]]]  # List of groups

    vectors_count = len(vectors)

    for i in range(1, vectors_count):

        min_group, min_distance = group_with_min_distance(vectors[i], Cm)

        if min_distance > threshold and m < q:

            # Create a new group
            m = m + 1
            new_group = []
            # Adds the used vector to the new group
            new_group.append(vectors[i])

            # Adds the new group to Cm list of groups
            Cm.append(new_group)
        else:

            Cm[min_group].append(vectors[i])

    return len(Cm)


def clusters_count(a, b, c, s, vectors, q):
    """
    An implementation of BSAS algorithm. BSAS calculates the number of clusters in a dataset.

    Args:
        a: This is the list of all vectors.
        b: maximum number of clusters allowed.
        c: the vectors generated for the given dataset
        vectors:
        q:

    Returns:
        len(Cm): the count of the found clusters
    """

    groups = []
    theta = a

    while theta <= b:
        print("a = {}, b = {}, theta = {}".format(a, b, theta))
        temp = []
        for i in range(0, s):
        

            np.random.shuffle(vectors)
            
            count = BSAS(theta, q, vectors)
            temp.append(count)
        print(temp)
        temp = Counter(temp).most_common(1)[0][0]
        groups.append(temp)
        theta = theta + c

    return groups

def get_clusters_count(vectors, c, s):
    """Returns the most frequently occurring number of clusters"""

    a, b = min_max_between_all(vectors)

    clusters = clusters_count(a, b, c, s, vectors, 1000)

    clusters = Counter(clusters).most_common(1)[0][0]

    return clusters


