from random import sample
from PIL import Image
import numpy as np
from keras import backend as K
import math
import tensorflow as tf

def load_image(path, size = (224, 224)):
    """
    Loads and resizes image on given path
    Args:
        path: path to the image
    """
    img = Image.open(path)
    img = img.resize(size)
    vector = np.asarray(img)
    return vector

def paths2names(paths):
    """
    Extracts filenames from paths
    Args:
        paths: array of paths to slides
    """
    vector_names = []
    for index, path in enumerate(list(paths)):
        path_find = str(path).split('/')[-1].split('.')[0]
        vector_names.append(path_find)
    return vector_names

def sort_patients(patients):
    """
    Sorts patients according to the order they were taken in
    Args:
        patients: array of arrays, containing slides sorted to patients
    """
    sorted_patients = []
    for patient in patients:
        patient_values = []
        for slide in patient:
            suffix = slide.split('_')[1]
            suffix = ''.join(c for c in suffix if (c.isdigit() or c == '-'))
            suffix = suffix.replace('-', '.')
            if suffix[0] == '.':
                suffix = suffix[1:]
            patient_values.append(float(suffix))
        sorted_indeces = np.argsort(patient_values)
        sorted_patients.append(np.array(patient)[sorted_indeces].tolist())
    return sorted_patients

def split_dataset(patients, min_distance):
    """
    Splits patients into background knowledge and probes
    Args:
        patients: array of arrays, containing slides sorted to patients
        min_distance: min allowed distance between background knowledge and probes
    """
    neighbor_distance = 3
    if min_distance == 0:
        references = []
        new_patients = []
        for patient in patients:
            refence_index = sample(range(len(patient)), 1)[0]
            # refence_index = 0
            references.append(patient[refence_index])
            del patient[refence_index]
            new_patients.append(patient)
        # print(references)
        return new_patients, references

    gap_size = math.floor(min_distance/neighbor_distance)
    patients = sort_patients(patients)

    references = []
    new_patients = []
    for patient in patients:
        refence_index = sample(range(len(patient)), 1)[0]
        references.append(patient[refence_index])
        
        to_delete = sorted([i for i in range(max(refence_index - gap_size, 0), min(refence_index + gap_size, len(patient) - 1) + 1)], reverse=True)
        # print('index is {}, min_distance is {}, gap is {}, to_delete is {}'.format(refence_index, min_distance, gap_size, to_delete))
        for index in to_delete:
            del patient[index]
        # print(patient)
        new_patients.append(patient)
    return new_patients, references

def references2paths(references, paths):
    """
    Translates references names to paths
    Args:
        references: array of names of slides
        paths: array of paths to slides
    """
    i_indexes = [paths2names(paths).index(reference) for reference in references ]
    return [paths[i] for i in i_indexes]

def references2vectors(references, vectors, vector_names):
    """
    Translates references names to vectors
    Args:
        references: array of names of slides
        vectors: array of slide vectors
        vector_names: array of names in the same order as vectors
    """
    i_indexes = [vector_names.index(reference) for reference in references ]
    return [vectors[i] for i in i_indexes]

def load_data(paths, feature_extractor, size=(224, 224)):
    """
    Translates paths to vectors
    Args:
        paths: array of paths of slides
        feature_extractor: feature_extractor
    """
    vectors = []
    vector_names = []
    for path in list(paths):
        img = load_image(path, size)
        feature = feature_extractor.extract_feature(img)
        vectors.append(feature)
        path_find = str(path).split('/')[-1].split('.')[0]
        vector_names.append(path_find)

    if (isinstance(vectors[0], tf.python.framework.ops.Tensor)):
        new_vectors = []
        with tf.compat.v1.Session() as sess:
            tf.compat.v1.global_variables_initializer().run()
            for vector in vectors:
                o = sess.run(vector)
                new_vectors.append(o)
        return new_vectors, vector_names
    return vectors, vector_names