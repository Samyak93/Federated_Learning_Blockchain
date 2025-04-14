"""
Edge Node Update Script

This script performs the following:
  1. Fetch the current global model (average signals) from the blockchain.
     (The model is stored as a flat vector of 240 parameters for 3 clusters (3x80).)
  2. Convert and reshape the global model into a (3 x 80) matrix.
  3. Load new local ECG training data and refine each centroid using the dtw_average update procedure.
  4. Test the updated model on 5 random new samples and print the predicted labels.
  5. Flatten and scale the updated model and push it back to the blockchain along with the sample counts
     (i.e. the number of new samples used for training for each cluster).

Usage: python edge_node_update.py [Account Number (1/2/3)] 
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, math
from copy import deepcopy
from web3 import Web3
from configparser import ConfigParser
import json

# =============================================================================
# === Global: Initializing Configurations======================================
# =============================================================================
config = ConfigParser()
config.read('../config/config.ini')

# =============================================================================
# === Helper Functions ========================================================
# =============================================================================

def print_usage():
    print("Usage: python edge_node_update.py [Account Number (1/2/3)]")
    print("Example: python edge_node_update.py 1")
    sys.exit(-1)

def resample_signal(signal, target_length):
    original_length = len(signal)
    original_indices = np.linspace(0, 1, original_length)
    target_indices = np.linspace(0, 1, target_length)
    return np.interp(target_indices, original_indices, signal)

def adaptive_scaling(signal1, new_length):
    if len(signal1) < new_length:
        return resample_signal(signal1, new_length)
    differences = np.diff(signal1)
    absolute_diffs = np.abs(differences)
    while len(signal1) > new_length:
        index = np.argmin(absolute_diffs)
        new_val = np.mean([signal1[index], signal1[index+1]])
        signal1[index] = new_val
        signal1 = np.delete(signal1, index+1)
        differences = np.delete(differences, index)
        absolute_diffs = np.delete(absolute_diffs, index)
        last_index = len(signal1) - 1
        if index != last_index:
            diff = signal1[index+1] - signal1[index]
            differences[index] = diff
            absolute_diffs[index] = abs(diff)
        if index != 0:
            diff = signal1[index] - signal1[index-1]
            differences[index-1] = diff
            absolute_diffs[index-1] = abs(diff)
    return signal1

def scale_data(data, new_length):
    nsignals = len(data)
    scaled_data = np.zeros([nsignals, new_length])
    for i in range(nsignals):
        scaled_data[i] = adaptive_scaling(data[i], new_length)
    return scaled_data

def split_data(data, data_labels, n, classes):
    # Splits data based on specified class labels.
    test_index = data.index
    i = 0
    for _ in classes:
        index = data_labels.loc[data_labels == classes[i]].index
        nclass = len(index)
        nclass = np.min([n, nclass])
        train_choices = np.random.choice(index, nclass, replace=False)
        test_index = test_index.drop(train_choices)
        if 'train_index' not in locals():
            train_index = train_choices
        else:
            train_index = np.concatenate([train_index, train_choices])
        i += 1
    return train_index, test_index

def upper_bound_partials(signal1, signal2):
    signal_list = [signal1, signal2]
    coordinate_list = [[], []]
    signal_lengths = [len(x) for x in signal_list]
    
    if signal_lengths[0] != signal_lengths[1]:
        index_long = np.argmax(signal_lengths)
        index_short = np.argmin(signal_lengths)
        long_len = len(signal_list[index_long])
        short_len = len(signal_list[index_short])
        fraction = short_len / long_len
        long_index_list = [i for i in range(long_len)]
        short_index_list = [int(np.floor(x * fraction)) for x in range(long_len)]
        warped_short = np.take(signal_list[index_short], short_index_list)
        ub_partials = np.abs(signal_list[index_long] - warped_short)[::-1]
        coordinate_list[index_long] = long_index_list
        coordinate_list[index_short] = short_index_list
    else:
        ub_partials = np.abs(signal_list[0] - signal_list[1])[::-1]
        fraction = 1
        index_list1 = [i for i in range(signal_lengths[0])]
        index_list2 = [i for i in range(signal_lengths[1])]
        coordinate_list[0] = index_list1
        coordinate_list[1] = index_list2
        
    ub_partials = np.cumsum(ub_partials)[::-1]
    coordinate_list = list(zip(coordinate_list[0], coordinate_list[1]))
    return coordinate_list, fraction, ub_partials

def pruned_dtw(matched, warped, window_size):
    N = len(matched)
    M = len(warped)
    ub_coordinate_list, fraction, ub_partials = upper_bound_partials(matched, warped)
    start_column = 1 
    end_column = 1 
    window_size = np.max([window_size, N - M])
    cost_matrix = np.ndarray((N+1, M+1))
    cost_matrix[:] = np.inf
    cost_matrix[0, 0] = 0
    UB = ub_partials[1]
    traceback_matrix = np.ones((N, M)) * np.inf
    
    for i in range(1, N+1):
        if N >= M:
            ub_col_index = int(np.floor(i * fraction))
        else:
            ub_col_index = int(np.floor(i / fraction))
        beg = np.max([start_column, ub_col_index - window_size])
        end = np.min([M+1, ub_col_index + window_size + 1])
        smaller_found = False
        end_column_next = ub_col_index
      
        for j in range(beg, end):
            cost = (matched[i-1] - warped[j-1])**2
            penalty = [cost_matrix[i-1, j-1], cost_matrix[i-1, j], cost_matrix[i, j-1]]
            penalty_index = np.argmin(penalty)
            traceback_matrix[i-1, j-1] = penalty_index
            cost_matrix[i, j] = cost + penalty[penalty_index]
            if (i-1, j-1) in ub_coordinate_list:
                ub_index = ub_coordinate_list.index((i-1, j-1))
                UB = cost_matrix[i, j] + ub_partials[ub_index]
            if cost_matrix[i, j] > UB:
                if not smaller_found:
                    start_column = j+1
                if j >= end_column:
                    break
            else:
                smaller_found = True
                end_column_next = j+1
        end_column = end_column_next
        
    i = N-1
    j = M-1
    path = [(i, j)]
    
    while (i > 0 or j > 0):
        tb_type = traceback_matrix[i, j]
        if tb_type == 0: 
            i -= 1
            j -= 1
        elif tb_type == 1:
            i -= 1
        else:
            j -= 1
        path.append((i, j))
    
    cost_matrix = cost_matrix[1:, 1:]
    distance = np.sqrt(cost_matrix[-1, -1])
    path = path[::-1]
    
    return cost_matrix, path, distance

def dtw_average(signals, average_signal, iterations, window_size):
    nsignals = len(signals)
    paths = []
    match_indices = []
    signal_indices = []
    
    for j in range(iterations):
        for n in range(nsignals):
            paths.append(pruned_dtw(average_signal, signals[n], window_size)[1])
            match_indices.append(np.array(paths[n])[:, 0])
            signal_indices.append(np.array(paths[n])[:, 1])
            
        n_average = len(average_signal)
        average_signal = []
        time_index = []
        
        for i in range(n_average):
            for n in range(nsignals):
                if i in match_indices[n]:
                    indices = np.where(match_indices[n] == i)
                    if 'signal_values' in locals():
                        signal_values = np.concatenate((signal_values, signals[n][signal_indices[n][tuple(indices)]]))
                    else:
                        signal_values = signals[n][signal_indices[n][tuple(indices)]]
            if 'signal_values' in locals():
                average_signal.append(np.mean(signal_values))
                del signal_values
                time_index.append(i)
        average_signal = np.array(average_signal)
        time_index = np.array(time_index)
    return average_signal

def get_boundary_signals(window, signal1):
    sig_length = len(signal1)
    ub_signal = np.zeros(sig_length)
    lb_signal = np.zeros(sig_length)
    for i in range(sig_length):
        low_index = np.max([0, i-window])
        up_index = np.min([i+window+1, sig_length])
        ub_signal[i] = np.max(signal1[low_index:up_index])
        lb_signal[i] = np.min(signal1[low_index:up_index])
    return ub_signal, lb_signal

def lb_keogh(ub_signal, lb_signal, signal2):
    sig_length = len(signal2)
    max_dists = np.zeros(sig_length)
    for i in range(sig_length):
        if signal2[i] > ub_signal[i]:
            max_dists[i] = (signal2[i] - ub_signal[i])**2
        elif signal2[i] < lb_signal[i]:
            max_dists[i] = (signal2[i] - lb_signal[i])**2
        else:
            max_dists[i] = 0
    lb_keogh_val = np.sqrt(np.sum(max_dists))
    return lb_keogh_val

def kmeans_dtw(data, n_clusters, km_iterations, av_iterations, dtw_window, data_labels):
    n = len(data)
    l_signal = len(data[0])
    clusters = np.random.choice(np.array(range(n_clusters)), size=n)
    centroids = np.zeros((n_clusters, data.shape[1]))
    distances = np.zeros(n)
    purities = []
    
    best_purity = 0

    for i in range(km_iterations):
        ub_matrix = np.zeros((n_clusters, l_signal))
        lb_matrix = np.zeros((n_clusters, l_signal))
        
        for k in range(n_clusters):
            indices = np.where(clusters == k)[0]
            if np.any(indices):
                if i == 0:
                    centroids[k] = data[np.random.choice(indices, 1)]
                else:
                    length_c = len(distances[indices])
                    centroid_index = indices[np.argsort(distances[indices])[length_c // 2]]
                    centroids[k] = data[centroid_index]
                centroids[k] = dtw_average(data[indices], centroids[k], av_iterations, dtw_window)
                ub_signal, lb_signal = get_boundary_signals(dtw_window, centroids[k])
                ub_matrix[k] = ub_signal
                lb_matrix[k] = lb_signal

        for j in range(n):
            best_so_far = np.inf
            best_cluster_index = 0
            signal1 = data[j]
            for k in range(n_clusters):
                centroid = centroids[k]
                ub_signal = ub_matrix[k]
                lb_signal = lb_matrix[k]
                lb_k = lb_keogh(ub_signal, lb_signal, signal1)
                if lb_k < best_so_far:
                    _, _, distance = pruned_dtw(centroid, signal1, dtw_window)
                    if distance < best_so_far:
                        best_so_far = distance
                        best_cluster_index = k
            clusters[j] = best_cluster_index
            distances[j] = best_so_far
        
        correspondence, purity = purity_score(data_labels, clusters)
        if purity > best_purity:
            best_purity = purity
            best_clusters = deepcopy(clusters)
            best_centroids = deepcopy(centroids)
            best_correspondence = deepcopy(correspondence)
        purities.append(purity)
        if purity == 1:
            break
    return best_clusters, best_centroids, purities, best_purity, best_correspondence

def purity_score(data_labels, clusters):
    in_cluster_count = {}
    correspondence = {}
    cluster_labels = np.unique(clusters)
    N = len(data_labels)
    for k in range(len(cluster_labels)):
        cluster_label = cluster_labels[k]
        cluster = np.where(clusters == cluster_label)
        if np.any(cluster):
            data_labels_arr = np.array(data_labels, dtype='int')
            in_cluster_count[cluster_label] = np.bincount(data_labels_arr[cluster]).max()
            correspondence[cluster_label] = np.bincount(data_labels_arr[cluster]).argmax()
    purity = sum(in_cluster_count.values()) / N
    return correspondence, purity

def get_best_cluster_index(purities, ncluster_list, threshold=0.005):
    best_index = 0
    best_number = ncluster_list[best_index]
    best_so_far = purities[best_index]
    for i in range(1, len(purities)):
        increase_from_best_so_far = (purities[i] - best_so_far) / (ncluster_list[i] - best_number)
        if increase_from_best_so_far > threshold:
            best_index = i
            best_number = ncluster_list[best_index]
            best_so_far = purities[i]
    return best_index

def predict_test_data_lb_k(test_filt_data, average_signal_list, window_size, correspondence):    
    nsignals = len(test_filt_data)
    nclasses = len(average_signal_list)
    l_signal = len(test_filt_data[0])
    
    up_bound_matrix = np.zeros((nclasses, l_signal))
    low_bound_matrix = np.zeros((nclasses, l_signal))
    
    for j in range(nclasses):
        ub_signal, lb_signal = get_boundary_signals(window_size, average_signal_list[j])
        up_bound_matrix[j] = ub_signal
        low_bound_matrix[j] = lb_signal

    predicted_labels = np.zeros(nsignals)
    
    for i in range(nsignals):
        best_so_far = np.inf
        best_class_index = 0
        signal1 = test_filt_data[i]
        for j in range(nclasses):
            if j in correspondence.keys():
                ub_signal = up_bound_matrix[j]
                lb_signal = low_bound_matrix[j]
                lb_k = lb_keogh(ub_signal, lb_signal, test_filt_data[i])
                if lb_k < best_so_far:
                    signal2 = average_signal_list[j]
                    cost_matrix, path, distance = pruned_dtw(signal1, signal2, window_size)
                    dist = cost_matrix[-1, -1]
                    if dist < best_so_far:
                        best_so_far = dist
                        best_class_index = j
        predicted_label = correspondence[best_class_index]
        predicted_labels[i] = predicted_label
    return predicted_labels

def scale_model_for_blockchain(model, scaling_factor=1e6):
    flattened = model.flatten()
    return [int(x * scaling_factor) for x in flattened]

# =============================================================================
# === PART 2: EDGE NODE UPDATE SCRIPT (Second Script) =========================
# =============================================================================

def edge_node_update():
    # Define scaling and model dimensions.
    scaling_factor = 1e6  # Same scaling factor used during training.
    n_clusters = 3
    model_length = 80  # Each centroid is 80 samples.
    
    # Connect to Ganache.
    ganache_url = config["URLS"]["ganache_url"]
    w3 = Web3(Web3.HTTPProvider(ganache_url))
    if not w3.is_connected():
        raise Exception("Failed to connect to Ganache. Check your RPC URL and Ganache status.")
    
    # Set your deployed contract address (for a 240-parameter model).
    contract_address_pull = config["CONTRACT_ADDRESS"]["global_model"]
    contract_address_push = config["CONTRACT_ADDRESS"]["account"+account_number]

    abi = json.loads(config["ABI"]["abi"])
    
    contract_push = w3.eth.contract(address=contract_address_push, abi=abi)
    contract_pull = w3.eth.contract(address=contract_address_pull, abi=abi)
    
    # Retrieve the current global model and sample counts from blockchain.
    stored_weights, stored_counts = contract_pull.functions.getGlobalModel().call()
    global_model = np.array(stored_weights, dtype=np.float64) / scaling_factor
    try:
        global_model = global_model.reshape((n_clusters, model_length))
    except Exception as e:
        raise ValueError("Error reshaping the retrieved model. Expected {} parameters but got {}: {}"
                         .format(n_clusters * model_length, len(global_model), e))
    print("Edge Node: Retrieved Global Model (reshaped to 3 x 80)")
    
    # Load new local ECG training data.
    print(f"Reading file: ../data/ecg{account_number}.csv")
    data = pd.read_csv(f'../data/ecg{account_number}.csv', header=None)
    data_labels = data[140].apply(lambda x: int(x))
    del data[140]
    _, new_train_index = split_data(data, data_labels, 20, [0, 1])
    new_train_data = data.reindex(new_train_index).to_numpy()
    new_train_labels = data_labels.reindex(new_train_index).to_numpy()
    new_train_data = scale_data(new_train_data, model_length)
    
    # Update each centroid using new local data.
    # Also compute the number of samples used for the update for each cluster.
    updated_model = np.zeros(global_model.shape)
    sample_counts = []  # Will store counts for each centroid.
    for i in range(n_clusters):
        distances = []
        for sample in new_train_data:
            _, _, d = pruned_dtw(global_model[i], sample, window_size=10)
            distances.append(d)
        distances = np.array(distances)
        num_samples = len(new_train_data)
        selected_indices = np.argsort(distances)[:max(1, num_samples // 2)]
        sample_counts.append(len(selected_indices))
        local_data = new_train_data[selected_indices]
        updated_centroid = dtw_average(local_data, global_model[i], iterations=5, window_size=10)
        updated_model[i] = updated_centroid

    print("\nEdge Node: Updated Model after local refinement\n")
    
    # ------------------- TESTING PART: Evaluate Updated Model ---------------------
    rand_indices = np.random.choice(len(new_train_data), size=5, replace=False)
    test_samples = new_train_data[rand_indices]
    test_true_labels = new_train_labels[rand_indices]
    default_correspondence = {0: 1, 1: 0, 2: 1}
    predicted_labels = predict_test_data_lb_k(test_samples, updated_model, window_size=10, correspondence=default_correspondence)
    print("Edge Node: Predictions on 5 random new samples:\n")
    for idx, pred, true_label in zip(rand_indices, predicted_labels, test_true_labels):
        correctness = "CORRECT" if pred == true_label else "WRONG"
        print("Sample index {}: True label: {}, Predicted label: {} ({})".format(idx, true_label, pred, correctness))
    # ------------------- END OF TESTING PART --------------------------------------
    
    # Prepare the updated model to push to blockchain.
    scaled_params = scale_model_for_blockchain(updated_model, scaling_factor)
    print("\nEdge Node: Scaled parameters length:", len(scaled_params))
    
    account = w3.eth.accounts[int(account_number)-1]
    print("\nEdge Node: Using blockchain account:", account)
    
    # Build the transaction now providing both the new model and new sample counts.
    tx = contract_push.functions.updateGlobalModel(scaled_params, sample_counts).build_transaction({
        'from': account,
        'nonce': w3.eth.get_transaction_count(account),
        'gas': 20000000,
        'gasPrice': w3.to_wei('20', 'gwei')
    })
    
    private_key = config["PRIVATE_KEYS"]["account"+account_number]
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print("\nEdge Node: Blockchain transaction receipt:")
    print(receipt)
    
    stored_new_weights, stored_new_counts = contract_push.functions.getGlobalModel().call()
    new_global_model = np.array(stored_new_weights, dtype=np.float64) / scaling_factor
    print("Edge Node: New Global Model on Blockchain after update:")
    print(new_global_model)
    print("\nEdge Node: New Sample Counts on Blockchain after update:")
    print(stored_new_counts)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print_usage()
        
    global account_number 
    account_number = sys.argv[1]

    edge_node_update()
