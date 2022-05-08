import numpy as np
from pathlib import Path
import pickle
import argparse
import pandas as pd
from utils.util import split_dataset, references2vectors, load_data
from utils.get_extractor import get_extractor
from attacker.get_probability_vectors import get_probability


def process_patient(patient, vector_names, references, vectors, metric, index):
    """
    Measures probability that one of the probes of the patient is mathed with him
    Args:
        patient: array of probes of the patient
        vector_names: array of the names of all images
        references: vectors of the background knowledge
        vectors: array of all image vectors
        metric: selected vector similarity metric
        index: index of the patient
    """
    i_indexes = [vector_names.index(slide) for slide in patient ]
    i_list = [vectors[i] for i in i_indexes]
    
    probabilities = []
    df = pd.DataFrame(columns=['representation', 'probe', 'cosine_distance', 'probability'])
    if len(i_list) > 0:
        for a, i in enumerate(i_list):
            probability_distribution, scores = get_probability(i, references, metric)
            probability_h = probability_distribution[index]
            
            probabilities.append(probability_h)
            
            df = df.append({"patient": index, 'probe': patient[a], 'cosine_distance': scores[index], 'probability': probability_h}, ignore_index = True)
        vpi = np.max(probabilities)
    else:
        vpi = 0
    return vpi, df

def main(map_path, metric, vectors, vector_names, min_distance = 0):
    """
    Measures V_{pi-max} and basic experiment statistics
    Args:
        map_path: path to the map
        metric: selected vector similarity metric
        vectors: array of vectors representing images
        vector_names: array of image filenames
        min_distance: minimum distance between background knoledge and probes
    """
    Vpis = []
    df = pd.DataFrame(columns=['patient', 'representation', 'probe', 'cosine_distance', 'probability'])

    with open(map_path, "rb") as fp:
        patients = pickle.load(fp)
    patients, references_names = split_dataset(patients, min_distance)
    references = references2vectors(references_names, vectors, vector_names)
    for index, patient in enumerate(patients):
        vpi, patient_df = process_patient(patient, vector_names, references, vectors, metric, index)
        patient_number = index

        patient_df['representation'] = references_names[patient_number]
        df = df.append(patient_df, ignore_index = True)
        Vpis.append(vpi/len(list(references)))
    Vpi_max = np.sum(np.nan_to_num(Vpis))
    
    # print(df)
    return Vpi_max, df


if __name__=='__main__':

    parser = argparse.ArgumentParser()

    # Data params
    parser.add_argument('--slide_dir', dest='slide_dir', type=Path, help='dataset path')
    parser.add_argument('--map_path', dest='map_path', type=Path, help='map path')
    parser.add_argument('--extractor', dest='extractor', type=str, default='resnet', help='extractor')
    parser.add_argument('--metric', dest='metric', type=str, default='cosine', help='metric')

    args = parser.parse_args()

    paths = np.array([path_.relative_to('/') for path_ in (args.slide_dir).glob('*.png')], dtype=object)

    vectors, vector_names = load_data(paths, get_extractor(args.extractor))

    Vpi_max, df = main(args.map_path, args.metric, vectors, vector_names)
    print('Vpi_max is : {}'.format(Vpi_max))
