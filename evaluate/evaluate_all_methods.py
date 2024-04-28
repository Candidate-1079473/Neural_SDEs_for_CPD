"""Evaluate the latent-SDE CP method or benchmark methods""" 
ensemble_count = 3
num_of_iterations_with_new_training_end_point = 0

latent_or_benchmarks = 'benchmarks'

if latent_or_benchmarks == 'latent':
    METHODS = ["SDE"]
    
else:
    METHODS = ["claspy",
        "AutoRegressive",
        "KernelizedMeanChange",
        "RBeast"]

DATASET_NAMES = {
    "Cricket": 300,
    "DiatomSizeReduction": 345,
    "EOGHorizontalSignal": 750,
    "EOGVerticalSignal": 750,
    "GestureMidAir": 360,
    "InsectEPGRegularTrain": 601,
    "InsectEPGSmallTrain": 601,
    "Mallat": 750,
    "NonInvasiveFetalECGThorax": 750,
     "Rock": 750,
     "SwedishLeaf": 128,
     "Trace": 275
     }

#how many datasets are created for each element of DATASET_NAMES 
BATCH_SIZE = 10

import json
import pandas as pd
import sys
from tqdm import tqdm
import os
import numpy as np
from collections import defaultdict

# module responsible for data generation:
from make_data import make_data

# the SDE change-point detector
sys.path.append("/mnt/sdd/MSc_projects/knowles/projects/sde-cpd")
from change_points_finder import SDEChangePointFinder

# import other CPD methods
import ruptures as rpt
from claspy.segmentation import BinaryClaSPSegmentation

def block_print():
    sys.stdout = open(os.devnull, "w")

def enable_print():
    sys.stdout = sys.__stdout__

block_print()
import Rbeast as rb

enable_print()


def _predict_ensemble(array, method, true_change_point, filename, result_folder):
    """Predicts change-point (CP) for SDE method using an ensemble"""
    time_series = SDEChangePointFinder(
        None,
        None,
        array,
        directory=result_folder + f"SDE_plots/" + filename[:-4],
        name=filename[:-4],
    )
    
    #change_point_lst will contain change points as defined using the mean change function 
    change_point_lst = np.array([])
    
    #change_point_lst_2 will contain change points as defined using the argmax of the difference of distances 
    change_point_lst_2 = np.array([])
    
    #train a total of ensemble_count SDE models, each with a different random seed initialisation
    for i in range(0,ensemble_count):
        time_series.set_seeds(2023 + i)
        time_series.find_change_point()
        change_point_lst = np.append(change_point_lst, time_series.predicted_change_point)
        change_point_lst_2 = np.append(change_point_lst_2, time_series.predicted_change_point_2)
    
    #optionally plot the results with uncertainty estimates (the mean of the ensemble plus or minus 1 standard deviation)
    time_series.plot_with_uncertainty_estimates(true_change_point, change_point_lst) 
    
    return [change_point_lst, change_point_lst_2]

def _predict_ensemble_SDE_with_iterations(array, method, true_change_point, filename, result_folder):
    """Predicts change-point (CP) using the latent-SDE method with iterating the hyperparameter T_0.
       only calculates the CP predictions using the mean change function"""

    mean_predictions_training_with_iterations = np.array([])
    
    time_series = SDEChangePointFinder(
        None,
        array,
        directory=result_folder + f"SDE_plots/" + filename[:-4],
        name=filename[:-4]
        )

    change_point_lst = np.array([])
    for i in range(0,ensemble_count):
        time_series.set_seeds(2023 + i)
        time_series.find_change_point()
        change_point_lst = np.append(change_point_lst, time_series.predicted_change_point)
    
    mean_pred = np.mean(change_point_lst)
    mean_predictions_training_with_iterations = np.append(mean_predictions_training_with_iterations, mean_pred)
        
    for j in range(0, num_of_iterations_with_new_training_end_point):
        time_series = SDEChangePointFinder(
            mean_pred,
            array,
            directory=result_folder + f"SDE_plots/" + filename[:-4],
            name=filename[:-4]
            )
        
        if j == num_of_iterations_with_new_training_end_point:
            max_i = ensemble_count
        else:
            max_i = ensemble_count
        
        for i in range(0,ensemble_count):
            time_series.set_seeds(2023 + i)
            time_series.find_change_point()
            change_point_lst = np.append(change_point_lst, time_series.predicted_change_point)
            
        mean_pred = np.mean(change_point_lst)
        mean_predictions_training_with_iterations = np.append(mean_predictions_training_with_iterations, mean_pred)
        
    np.save("./ITERATION_RESULTS/" + filename[:-4] + ".npy", np.abs(mean_predictions_training_with_iterations - true_change_point))
    
    return change_point_lst

def _predict(array, method, true_change_point, filename, result_folder):
    """Predicts change-point (CP) using benchmark methods"""
    
    if method == "claspy":
        print('this is being called')
        try:
            return BinaryClaSPSegmentation(n_segments=2).fit_predict(array[..., -1])[0]
        except IndexError:
            return -1

    elif method == "KernelizedMeanChange":
        return rpt.Dynp(model="l2").fit(array).predict(n_bkps=1)[0]

    elif method == "AutoRegressive":
        return rpt.Dynp(model="ar").fit(array).predict(n_bkps=1)[0]

    elif method == "RBeast":
        block_print()
        array = array[..., -1]
        result = rb.beast(array, scp_minmax=[1, 1], tcp_minmax=[1, 1], freq=750)
        result_season = result.season.cp[0]
        result_trend = result.trend.cp[0]
        result = min([result_season, result_trend], key=lambda r: abs(true_change_point - r))
        enable_print()
        return result
    

def _get_predictions(method, result_folder):
    """Retrieves CP predictions of given method"""

    predictions_path = result_folder + f"predictions_dict_{method}.json"

    if os.path.exists(predictions_path):
        with open(predictions_path, "r") as in_file:
            predictions_dict = json.load(in_file)
    else:
        predictions_dict = {}
        
    #again predictions 2 refers to the CP as determined by the argmax of the differences of distances
    predictions_path_2 = result_folder + f"predictions_dict_2_{method}.json"
    
    if os.path.exists(predictions_path_2):
        with open(predictions_path_2, "r") as in_file:
            predictions_dict_2 = json.load(in_file)
    else:
        predictions_dict_2 = {}

    root = "./data/"
    for filename in os.listdir(root):
        if filename.endswith(".npy") and filename not in predictions_dict:
            with open(root + filename, "rb") as time_series_file:
                array = np.load(time_series_file)
                true_change_point = int(filename[-10:-4])
                
                predicted_change_point_both_lsts = _predict_ensemble(
                        array, method, true_change_point, filename, result_folder
                    )
                
                predicted_change_point_lst = predicted_change_point_both_lsts[0]
                predicted_change_point_lst_2 = predicted_change_point_both_lsts[1]
                
                predictions_dict[filename] = (
                    true_change_point,
                    predicted_change_point_lst.tolist(),
                )
                
                predictions_dict_2[filename] = (
                    true_change_point,
                    predicted_change_point_lst_2.tolist(),
                )

    with open(predictions_path, "w") as out_file:
        json.dump(predictions_dict, out_file)
        
    with open(predictions_path_2, "w") as out_file:
        json.dump(predictions_dict_2, out_file)

    return [predictions_dict, predictions_dict_2]


def compute_performances_with_one_metric(predictions):
    """caclulate mean CPD accuracy, mean Euclidean distance and SD of Euclidean distance"""
    accuracies_epsilon_insensitive = []
    avg_euclidean_distances = []
    sd_euclidean_distances = []
    
    scores = defaultdict(list)
    euclidean_distances = defaultdict(list)

    for filename in predictions:
        true_change_point, predicted_change_point_lst = predictions[filename]
        dataset_name = filename[: filename.index("_")]
        predicted_change_point = np.mean(predicted_change_point_lst) 
        error_margin = DATASET_NAMES[dataset_name]

        if abs(true_change_point - predicted_change_point) * 0.5 <= error_margin:
            scores[dataset_name].append(1)
        else:
            scores[dataset_name].append(0)
            
        euclidean_distances[dataset_name].append(abs(true_change_point - predicted_change_point)) 
        
    accuracies_epsilon_insensitive = {
        dataset_name: np.sum(scores[dataset_name]) / len(scores[dataset_name])
        for dataset_name in DATASET_NAMES
    }
    avg_euclidean_distances = {
        dataset_name: np.sum(euclidean_distances[dataset_name]) / len(euclidean_distances[dataset_name])
        for dataset_name in DATASET_NAMES
    }
    sd_euclidean_distances = {
        dataset_name: np.std(euclidean_distances[dataset_name])
        for dataset_name in DATASET_NAMES
    }
    
    return [accuracies_epsilon_insensitive, avg_euclidean_distances, sd_euclidean_distances]


def compute_performance_mean(method, result_folder):
    """Returns performance metrics for each method"""
    predictions = _get_predictions(method, result_folder) #if the method is SDE then predictions is a list of size two
    
    final_results = []
    
    if latent_or_benchmarks == 'latent':
        for i in range(0,2):
            prediction_i = predictions_[i]
            result_i = compute_performances_with_one_metric(prediction_i)    
            final_results = final_results + [result_i]
    
    else:
        result = compute_performances_with_one_metric(predictions)
        final_results = final_results + result

    return final_results


def save_dataframes(accuracies_df, euclidean_distances_df, euclidean_SDs_df, paths):
    result_path_epsilon, result_path_euclidean, result_path_sd = paths
    accuracies_df = pd.DataFrame.from_dict(accuracies_df)
    accuracies_df["DATASET_NAMES"] = DATASET_NAMES.keys()
    accuracies_df = accuracies_df.set_index("DATASET_NAMES")
    accuracies_df.to_csv(result_path_epsilon, sep="\t")
    
    euclidean_distances_df = pd.DataFrame.from_dict(euclidean_distances_df)
    euclidean_distances_df["DATASET_NAMES"] = DATASET_NAMES.keys()
    euclidean_distances_df = euclidean_distances_df.set_index("DATASET_NAMES")
    euclidean_distances_df.to_csv(result_path_euclidean, sep="\t")
    
    euclidean_SDs_df = pd.DataFrame.from_dict(euclidean_SDs_df)
    euclidean_SDs_df["DATASET_NAMES"] = DATASET_NAMES.keys()
    euclidean_SDs_df = euclidean_SDs_df.set_index("DATASET_NAMES")
    euclidean_SDs_df.to_csv(result_path_sd, sep="\t")


if __name__ == "__main__" and latent_or_benchmarks == 'latent':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5, 6"

    # generate time series containing one change-point each
    if not os.path.exists("./data"):
        os.mkdir("./data")
        make_data("./data", DATASET_NAMES, batch_size=BATCH_SIZE)

    # where results are saved
    for i in [0,1]:
        result_folder = "./results/"
        result_path_epsilon = result_folder + f"all_accuracies_{i}.csv"
        result_path_euclidean = result_folder + f"all_euclidean_{i}.csv"
        result_path_sd = result_folder + f"all_sd_{i}.csv"
        
        if not os.path.exists(result_path_epsilon):
            os.mkdir(result_folder) if not os.path.exists(result_folder) else None
            os.mkdir(result_folder + "SDE_plots") if not os.path.exists(
                result_folder + "SDE_plots"
            ) else None

            # compute CPD accuracy of each method
            accuracies = {}
            accuracies_df = defaultdict(list)
            euclidean_distances = {}
            euclidean_distances_df = defaultdict(list)
            euclidean_SDs = {}
            euclidean_SDs_df = defaultdict(list)
            
            methods = tqdm(METHODS)

            for method in methods:
                methods.set_description(f"CPD Method: {method}")
                methods.refresh()
                
                performance_metrics = compute_performance_mean(method, result_folder)[i]
                
                accuracies[method] = performance_metrics[0]
                euclidean_distances[method] = performance_metrics[1]
                euclidean_SDs[method] = performance_metrics[2]
                
                for dataset_name in DATASET_NAMES:
                    accuracies_df[method].append(accuracies[method][dataset_name])
                    euclidean_distances_df[method].append(euclidean_distances[method][dataset_name])
                    euclidean_SDs_df[method].append(euclidean_SDs[method][dataset_name])
             
            paths = [result_path_epsilon, result_path_euclidean, result_path_sd]
            save_dataframes(accuracies_df, euclidean_distances_df, euclidean_SDs_df, paths)
            
if __name__ == "__main__" and latent_or_benchmarks == 'benchmarks':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5, 6"

    # generate time series containing one change-point each
    if not os.path.exists("./data"):
        os.mkdir("./data")
        make_data("./data", DATASET_NAMES, batch_size=BATCH_SIZE)

    # where results are saved
    result_folder = "./results/"
    result_path_epsilon = result_folder + "all_accuracies_other_methods.csv"
    result_path_euclidean = result_folder + "all_other_methods_euclidean.csv"
    result_path_sd = result_folder + "all_other_methods_sd.csv"
    
    if not os.path.exists(result_path_epsilon):
        os.mkdir(result_folder) if not os.path.exists(result_folder) else None
        os.mkdir(result_folder + "SDE_plots") if not os.path.exists(
            result_folder + "SDE_plots"
        ) else None
        
        # compute CPD accuracy of each method
        accuracies = {}
        accuracies_df = defaultdict(list)
        
        euclidean_distances = {}
        euclidean_distances_df = defaultdict(list)
        
        euclidean_SDs = {}
        euclidean_SDs_df = defaultdict(list)
        
        methods = tqdm(METHODS)
        for method in methods:
            methods.set_description(f"CPD Method: {method}")
            methods.refresh()
            
            performance_metrics = compute_performance_mean(method, result_folder)
            accuracies[method] = performance_metrics[0]
            euclidean_distances[method] = performance_metrics[1]
            euclidean_SDs[method] = performance_metrics[2]
            
            for dataset_name in DATASET_NAMES:
                accuracies_df[method].append(accuracies[method][dataset_name])
                euclidean_distances_df[method].append(euclidean_distances[method][dataset_name])
                euclidean_SDs_df[method].append(euclidean_SDs[method][dataset_name])

            paths = [result_path_epsilon, result_path_euclidean, result_path_sd]
            save_dataframes(accuracies_df, euclidean_distances_df, euclidean_SDs_df, paths)
    
    