
import os
import numpy as np
import torch
import Configrations
import ConvNet
import Iterative_BP_CNN as ibd
import LinearBlkCodes as lbc
import DataIO

def main():
    # 1. Initialization
    print("--- Initialization ---")
    top_config = Configrations.TopConfig()

    # Fix paths and parameters to match generated file (0.80 correlation, current dir)
    print("--- Adjusting Noise Configuration ---")
    top_config.corr_para = 0.8
    top_config.cov_1_2_file = 'cov_1_2_corr_para0.80.dat'
    top_config.corr_para_simu = 0.8
    top_config.cov_1_2_file_simu = 'cov_1_2_corr_para0.80.dat'

    print(f"Correlation Parameter: {top_config.corr_para}")
    print(f"Covariance File: {top_config.cov_1_2_file}")

    # Initialize sub-configs
    train_config = Configrations.TrainingConfig(top_config)
    net_config = Configrations.NetConfig(top_config)

    # 2. Configuration Overrides for Demo
    print("--- Overriding Configurations for Demo ---")
    train_config.training_sample_num = 56000
    train_config.test_sample_num = 7000
    train_config.epoch_num = 6

    print(f"Training Samples: {train_config.training_sample_num}")
    print(f"Test Samples: {train_config.test_sample_num}")
    print(f"Epochs: {train_config.epoch_num}")

    # 3. Data Generation

    code = lbc.LDPC(top_config.N_code, top_config.K_code, top_config.file_G, top_config.file_H)

    try:
        noise_io = DataIO.NoiseIO(top_config.N_code, False, None, top_config.cov_1_2_file)
    except Exception as e:
        print(f"Error initializing NoiseIO: {e}")
        raise e

    os.makedirs('./TrainingData', exist_ok=True)
    os.makedirs('./TestData', exist_ok=True)

    # Force regeneration of data

    ibd.generate_noise_samples(
        code, top_config, net_config, train_config,
        top_config.BP_iter_nums_gen_data,
        top_config.currently_trained_net_id,
        'Training',
        noise_io,
        top_config.model_id
    )

    ibd.generate_noise_samples(
        code, top_config, net_config, train_config,
        top_config.BP_iter_nums_gen_data,
        top_config.currently_trained_net_id,
        'Test',
        noise_io,
        top_config.model_id
    )

    # 4. Training
 
    os.makedirs(net_config.model_folder, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on {device}")

    net_id = top_config.currently_trained_net_id
    conv_net = ConvNet.ConvNet(net_config, train_config, net_id)

    # Train
    conv_net.train_network(top_config.model_id, device=device)

    # 5. Residual Noise Analysis (Crucial Step)
   
    # Use a small number of samples for analysis in this demo
    analysis_batch_size = 500
    analysis_sim_times = 2000 # Enough to get some stats

    print(f"Analyzing residual noise with {analysis_sim_times} samples...")
    ibd.analyze_residual_noise(
        code, top_config, net_config,
        analysis_sim_times, analysis_batch_size
    )

    # 6. Simulation
    
    simutimes_range = np.array([10000, 20000], dtype=np.int32)
    target_err_bits_num = 100
    batch_size = 500

    ibd.simulation_colored_noise(
        code, top_config, net_config, simutimes_range,
        target_err_bits_num, batch_size
    )

    

if __name__ == "__main__":
    main()
