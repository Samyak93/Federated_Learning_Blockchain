"""
Aggregator Script for FedAverage Global Model Update

This script performs the following:
  1. Connects to Ganache.
  2. Pulls updated global models from multiple edge nodes (each stored in a contract).
  3. Aggregates these models using a weighted FedAvg algorithm that incorporates the sample counts.
  4. Pushes the aggregated global model (flattened and scaled) along with aggregated sample counts to a public aggregator contract.
  
Assumptions:
  - Each edge-node's global model is stored as a flat vector of 240 scaled integer parameters.
  - The sample counts for each edge node are stored as an array of length 3.
  - The scaling factor used originally is 1e6.
  - Models should be reshaped into a (3×80) matrix (i.e., 3 clusters, each of length 80).

Update the lists of edge node contract addresses, the aggregator's contract address,
and the private key accordingly.
"""

import numpy as np
from web3 import Web3
from configparser import ConfigParser
import json

# =============================================================================
# === Global: Initializing Configurations======================================
# =============================================================================
config = ConfigParser()
config.read('../config/config.ini')
SCALING_FACTOR = 1e6
N_CLUSTERS = 3
MODEL_LENGTH = 80  # each centroid has 80 samples
TOTAL_PARAMS = N_CLUSTERS * MODEL_LENGTH  # should be 240


# =============================================================================
# === FedAverage Algorithm ====================================================
# =============================================================================
def fed_average_weighted(model_list, sample_counts_list):
    """
    Given a list of model arrays (each a flat vector of length TOTAL_PARAMS)
    and a corresponding list of sample counts (each an array of length N_CLUSTERS),
    perform a weighted FedAvg as follows:
    For each cluster (i), compute:
         aggregated_centroid[i] = ( sum_j (model_j[i] * sample_count_j[i]) ) / ( sum_j sample_count_j[i] )
    The function returns:
         aggregated_model_flat: a flat vector of length TOTAL_PARAMS.
         aggregated_sample_counts: a list of aggregated sample counts per cluster.
    """
    weighted_sum = np.zeros((N_CLUSTERS, MODEL_LENGTH), dtype=np.float64)
    total_counts = np.zeros((N_CLUSTERS,), dtype=np.float64)
    
    for model_flat, sample_counts in zip(model_list, sample_counts_list):
        model_reshaped = model_flat.reshape((N_CLUSTERS, MODEL_LENGTH))
        # sample_counts is assumed to be a list/array of length N_CLUSTERS for this edge node.
        weighted_sum += model_reshaped * np.array(sample_counts)[:, None]
        total_counts += np.array(sample_counts)
        
    aggregated_model = weighted_sum / total_counts[:, None]
    return aggregated_model.flatten(), total_counts.tolist()

# =============================================================================
# === Main Script =============================================================
# =============================================================================
def main():
    # List of edge node contract addresses from configurations
    edge_node_addresses = [
        config["CONTRACT_ADDRESS"]["account1"],   # First edge node contract address (ACCOUNT 1)
        config["CONTRACT_ADDRESS"]["account2"],   # Second edge node contract address (ACCOUNT 2)
        config["CONTRACT_ADDRESS"]["account3"]    # Third edge node contract address (ACCOUNT 3)
    ]
    
    aggregator_contract_address = config["CONTRACT_ADDRESS"]["global_model"]  # Global Model
    
    # Connect to Ganache.
    ganache_url = config["URLS"]["ganache_url"]
    w3 = Web3(Web3.HTTPProvider(ganache_url))
    if not w3.is_connected():
        raise Exception("Failed to connect to Ganache. Check the RPC URL and Ganache status.")
    
    abi = json.loads(config["ABI"]["abi"])
    # Retrieve models and sample counts from each edge node contract.
    edge_models = []
    edge_sample_counts = []
    for addr in edge_node_addresses:
        contract = w3.eth.contract(address=addr, abi=abi)
        stored_weights, stored_counts = contract.functions.getGlobalModel().call()
        model_flat = np.array(stored_weights, dtype=np.float64) / SCALING_FACTOR
        if len(model_flat) != TOTAL_PARAMS:
            raise ValueError("Model from {} does not have the expected {} parameters.".format(addr, TOTAL_PARAMS))
        edge_models.append(model_flat)
        # Here we assume stored_counts is an array of length N_CLUSTERS.
        # Use it directly as the sample counts for the corresponding model.
        edge_sample_counts.append(stored_counts)
        
    print("LATEST MODELS AND SAMPLE COUNTS FOUND:")
    for model, counts in zip(edge_models, edge_sample_counts):
        print(model)
        print("Sample counts:", counts)
        print("#" * 50)
    
    # Aggregate models using the weighted FedAvg algorithm.
    aggregated_model_flat, aggregated_counts = fed_average_weighted(edge_models, edge_sample_counts)
    
    # Reshape aggregated_model_flat to (3,80) for display.
    aggregated_model = aggregated_model_flat.reshape((N_CLUSTERS, MODEL_LENGTH))
    
    print("Aggregated Global Model (reshaped to 3x80):")
    print(aggregated_model)
    print("Aggregated Sample Counts (per cluster):")
    print(aggregated_counts)
    
    # Prepare the aggregated model for blockchain update: flatten and scale back.
    aggregated_model_flat_scaled = [int(x * SCALING_FACTOR) for x in aggregated_model_flat]
    
    # Convert aggregated_counts to integers.
    aggregated_counts_int = list(map(int, aggregated_counts))
    
    # Create the aggregator contract instance.
    aggregator_contract = w3.eth.contract(address=aggregator_contract_address, abi=abi)
    
    # Build the transaction for updating the aggregator's global model.
    account = w3.eth.accounts[0]  # 0 is the global account
    tx = aggregator_contract.functions.updateGlobalModel(aggregated_model_flat_scaled, aggregated_counts_int).build_transaction({
        'from': account,
        'nonce': w3.eth.get_transaction_count(account),
        'gas': 20000000,
        'gasPrice': w3.to_wei('20', 'gwei')
    })

    
    # Sign and send the transaction (update the private key accordingly).
    private_key = config["PRIVATE_KEYS"]["global_model"]
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    print("Aggregator Blockchain transaction receipt:")
    print(receipt)
    
    # Retrieve and print the updated aggregated global model for verification.
    stored_updated_weights, stored_updated_counts = aggregator_contract.functions.getGlobalModel().call()
    updated_aggregated_model = np.array(stored_updated_weights, dtype=np.float64) / SCALING_FACTOR
    updated_aggregated_model = updated_aggregated_model.reshape((N_CLUSTERS, MODEL_LENGTH))
    print("Updated Aggregated Global Model on Blockchain (reshaped to 3x80):")
    print(updated_aggregated_model)
    print("Updated Aggregated Sample Counts on Blockchain:")
    print(stored_updated_counts)

if __name__ == "__main__":
    main()
