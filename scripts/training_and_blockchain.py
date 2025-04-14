"""
Initial Training and Blockchain Update Script

This script:
  1. Loads and preprocesses ECG data.
  2. Performs DTW-based k-means clustering to compute average signals.
     (The number of clusters is fixed to 3.)
  3. Visualizes clustering results.
  4. Scales the average signals (centroids) and sends them to a blockchain smart contract.
  5. Contains a function for testing the trained model on a single new ECG entry.

Make sure to update the contract_address and private_key before running.
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import time, math
from copy import deepcopy
from sklearn.metrics import accuracy_score, confusion_matrix
from web3 import Web3
from configparser import ConfigParser
import json

# =============================================================================
# === Global: Initializing Configurations======================================
# =============================================================================
config = ConfigParser()
config.read('../config/config.ini')


# =============================================================================
# === Helper: Resample Signal to a Fixed Length (if needed) ===================
# =============================================================================

def resample_signal(signal, target_length):
    """
    Resamples a 1D signal to a fixed target length using linear interpolation.
    """
    original_length = len(signal)
    original_indices = np.linspace(0, 1, original_length)
    target_indices = np.linspace(0, 1, target_length)
    resampled_signal = np.interp(target_indices, original_indices, signal)
    return resampled_signal

# =============================================================================
# === Adaptive Scaling with Upsampling if Needed =================================
# =============================================================================

def adaptive_scaling(signal1, new_length):
    """
    If the signal is shorter than new_length, upsample it using linear interpolation.
    Otherwise, reduce the length by merging the closest two points.
    """
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

# =============================================================================
# === DTW-based Clustering Functions ===========================================
# =============================================================================

def split_data(data, data_labels, n, classes):
    n_classes = len(np.unique(data_labels))
    test_index = data.index
    i = 0
    for x_class in classes:
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
# === PART 1: INITIAL TRAINING AND BLOCKCHAIN UPDATE =========================
# =============================================================================

def initial_training_and_update():
    # For initial training, use scale_number = 80.
    scale_number = 80
    
    # Load ECG data
    data = pd.read_csv('../data/ecg.csv', header=None)
    data_labels = data[140].apply(lambda x: int(x))
    del data[140]
    
    # Scale the data
    train_index, test_index = split_data(data, data_labels, 40, [0, 1])
    train_data = data.reindex(train_index).to_numpy()
    train_labels = data_labels.reindex(train_index).to_numpy()
    train_data = scale_data(train_data, scale_number)
    test_data = data.reindex(test_index).to_numpy()
    test_data = scale_data(test_data, scale_number)
    test_labels = data_labels.reindex(test_index).to_numpy()
    
    km_iterations = 10
    av_iterations = 1
    dtw_window = 10
    
    # FIX the number of clusters to 3
    n_clusters_fixed = 3
    
    # Instead of looping over multiple cluster counts, directly use 3.
    clusters, centroids, purities, best_purity, correspondence = kmeans_dtw(
        train_data, n_clusters_fixed, km_iterations, av_iterations, dtw_window, train_labels)
    
    plt.plot([n_clusters_fixed], [best_purity], 'o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Purity Score')
    plt.title('Chosen Purity Score at 3 Clusters')
    plt.show()
    
    print('Best purity: {}'.format(best_purity))
    print('Number of clusters fixed at: {}'.format(n_clusters_fixed))
    
    # Compute average signals (centroids refined using dtw_average)
    average_signals = np.zeros(centroids.shape)
    av_iterations = 10
    for i in correspondence.keys():
        corr = correspondence[i]
        average_signal = centroids[i]
        index1 = set(np.where(clusters == i)[0])
        index2 = set(np.where(train_labels == corr)[0])
        intersection = list(index1.intersection(index2))
        signals1 = train_data[intersection]
        average_signal = dtw_average(signals1, average_signal, av_iterations, dtw_window)
        average_signals[i] = average_signal
    
    # Visualize clusters and average signals
    k = 0
    colors = ['blue', 'red', 'purple', 'orange']
    for i in range(n_clusters_fixed):
        if i in correspondence.keys():
            label = correspondence[i]
            label_str = 'Normal' if label == 1 else 'Abnormal'
            c = colors[correspondence[i] - 1]
            indices = np.where(clusters == i)[0]
            for index in indices:
                plt.plot(train_data[index] + 5 * k, alpha=0.5)
            plt.plot(average_signals[i] + 5 * k, c=c, label=label_str)
        k += 1
    plt.title('Cluster Centroid Signals')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()
    
    # Test the prediction
    start_time = time.time()
    predicted_labels = predict_test_data_lb_k(test_data, average_signals, dtw_window, correspondence)
    end_time = time.time()
    print('Prediction runtime: {} s'.format(end_time - start_time))
    
    score = accuracy_score(test_labels, predicted_labels, normalize=True)
    c_mat = confusion_matrix(test_labels, predicted_labels)
    print('Total Accuracy Score: {}'.format(score))
    print('Confusion Matrix:\n', c_mat)
    
    # -------------------------------------------------------------------------
    # Blockchain Integration: Updating Global Model on Ganache
    # -------------------------------------------------------------------------
    # Flatten and scale the average_signals (each signal is 80 samples and 3 clusters yields 240 parameters)
    flattened_avg = average_signals.flatten()
    scaling_factor = 1e6  # adjust as needed
    scaled_params = [int(x * scaling_factor) for x in flattened_avg]
    print("Scaled weights length (to update on blockchain):", len(scaled_params))
    
    # Compute sample counts for each cluster – using a dummy example:
    total_train_samples = len(train_data)
    sample_counts = [int(total_train_samples // n_clusters_fixed)] * n_clusters_fixed
    print("Sample counts to update on blockchain:", sample_counts)
    
    # Connect to Ganache using Web3.py
    ganache_url = config["URLS"]["ganache_url"]
    w3 = Web3(Web3.HTTPProvider(ganache_url))
    if not w3.is_connected():
        raise Exception("Failed to connect to Ganache. Check your RPC URL and Ganache status.")
    
    # For 3 clusters at 80 samples each, we expect 240 parameters.
    contract_address = config["CONTRACT_ADDRESS"]["global_model"]
    
    abi = json.loads(config["ABI"]["abi"])
    
    contract = w3.eth.contract(address=contract_address, abi=abi)
    account = w3.eth.accounts[0]  # ACCOUNT 1
    print("Using blockchain account:", account)
    
    tx = contract.functions.updateGlobalModel(scaled_params, sample_counts).build_transaction({
        'from': account,
        'nonce': w3.eth.get_transaction_count(account),
        'gas': 20000000,
        'gasPrice': w3.to_wei('20', 'gwei')
    })
    
    private_key = config["PRIVATE_KEYS"]["global_model"] # ACCOUNT 1
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print("Initial model blockchain transaction receipt:")
    print(receipt)
    
    # Return the contract instance and scaling_factor for future edge updates.
    return contract, scaling_factor

# =============================================================================
# === PART 2: EDGE NODE UPDATE (Testing New Data on the Trained Model) =========
# =============================================================================

def edge_node_update(contract, scaling_factor):
    """
    This function simulates an edge node update.
    It retrieves the latest global model from the blockchain, uses it as initialization
    for local training on new data, and then sends the updated model back to the blockchain.
    """
    ganache_url = config["URLS"]["ganache_url"]
    w3 = Web3(Web3.HTTPProvider(ganache_url))
    if not w3.is_connected():
        raise Exception("Failed to connect to Ganache. Check your RPC URL and Ganache status.")
    
    # Retrieve the current global model (now also obtaining sampleCounts but we focus on weights)
    stored_weights, stored_counts = contract.functions.getGlobalModel().call()
    global_model = np.array(stored_weights, dtype=np.float64) / scaling_factor
    print("Edge Node: Retrieved Global Model from Blockchain\n")
    
    # Simulate new local training update.
    # Dummy update using a random gradient (replace with your actual training procedure).
    def local_training_update(model, learning_rate=0.001):
        grad = np.random.randn(*model.shape) * 0.01  # dummy gradient
        return model - learning_rate * grad
    
    updated_model = local_training_update(global_model, learning_rate=0.001)
    print("Edge Node: Updated Model after local training update\n")
    
    # (Optionally, you could recalc new sample_counts here based on new data.)
    # Here we use a dummy new sample count (for example purposes).
    new_sample_counts = [42, 42, 42]  # Update these values as needed.
    print("New sample counts to update on blockchain:", new_sample_counts)
    
    # Scale and flatten the updated model.
    flattened = updated_model.flatten()
    scaled_params = [int(x * scaling_factor) for x in flattened]
    print("Edge Node: Scaled parameters to update on blockchain (length={})\n".format(len(scaled_params)))
    
    account = w3.eth.accounts[0] #Default global account
    print("Edge Node: Using blockchain account:", account)
    
    tx = contract.functions.updateGlobalModel(scaled_params, new_sample_counts).build_transaction({
        'from': account,
        'nonce': w3.eth.get_transaction_count(account),
        'gas': 20000000,
        'gasPrice': w3.to_wei('20', 'gwei')
    })
    
    private_key = config["PRIVATE_KEYS"]["global_model"]
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    print("\nEdge Node: Blockchain transaction receipt:")
    print(receipt)
    
    stored_new_weights, stored_new_counts = contract.functions.getGlobalModel().call()
    new_global_model = np.array(stored_new_weights, dtype=np.float64) / scaling_factor
    print("Edge Node: New Global Model on Blockchain after update:")
    print(new_global_model)

# =============================================================================
# === PART 3: MODEL TESTING ON A SINGLE ENTRY (External Testing) ===============
# =============================================================================

def test_single_entry(contract, scaling_factor, test_sample, dtw_window=10, correspondence=None, average_signals=None):
    """
    Tests a single ECG test sample with the trained model.
    
    Parameters:
      contract: The blockchain contract instance.
      scaling_factor: The scaling factor used during model storage.
      test_sample: A 1D numpy array containing the ECG signal (should be of length 80).
      dtw_window: Window size for DTW (default: 10).
      correspondence, average_signals: Optionally, if already computed during training.
    
    The function prints the predicted label.
    """
    if average_signals is None:
        stored_weights, stored_counts = contract.functions.getGlobalModel().call()
        global_model = np.array(stored_weights, dtype=np.float64) / scaling_factor
        average_signals = global_model.reshape((-1, 80))
        print("Retrieved sample counts:", stored_counts)
    
    if correspondence is None:
        correspondence = {0: 1, 1: 0, 2: 1}  # adjust as needed
    
    predicted_label = predict_test_data_lb_k([test_sample], average_signals, dtw_window, correspondence)
    print("Prediction for the test sample:", predicted_label[0])
    return predicted_label[0]

# =============================================================================
# === MAIN =======================================================
# =============================================================================

def main():
    contract, scaling_factor = initial_training_and_update()
    
    print("\n----- Running Edge Node Dummy Update -----\n")
    edge_node_update(contract, scaling_factor)
    
    print("\n----- Running Single Entry Testing -----\n")
    data = pd.read_csv('../data/ecg.csv', header=None)
    data_labels = data[140].apply(lambda x: int(x))
    del data[140]
    scale_number = 80
    _, test_index = split_data(data, data_labels, 40, [0, 1])
    test_data = data.reindex(test_index).to_numpy()
    test_data = scale_data(test_data, scale_number)
    
    test_sample = test_data[0]
    predicted_label = test_single_entry(contract, scaling_factor, test_sample, dtw_window=10)
    print("Final predicted label for the test sample:", predicted_label)

if __name__ == "__main__":
    main()
