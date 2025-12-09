# This file contains some functions used to implement an iterative BP and denoising system for channel decoding.
# The denoising is implemented through CNN.
# The system architecture can be briefly denoted as BP-CNN-BP-CNN-BP...
import numpy as np
import datetime
import torch
from BP_Decoder import BP_NetDecoder
from ConvNet import ConvNet
import LinearBlkCodes as lbc
from DataIO import NoiseIO


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def stat_prob(x, prob):
    """Calculate empirical probability distribution"""
    qstep = 0.01
    min_v = -10
    x = np.reshape(x, [1, np.size(x)])
    hist, _ = np.histogram(x, np.int32(np.round(2*(-min_v) / qstep)), [min_v, -min_v])
    
    if np.size(prob) == 0:
        prob = hist
    else:
        prob = prob + hist
    
    return prob


def calc_LLR_epdf(prob, s_mod_plus_res_noise):
    """Calculate LLR using empirical PDF"""
    qstep = 0.01
    min_v = -10
    
    # Calculate indices for p0 (bit=0, symbol=+1)
    id_p0 = ((s_mod_plus_res_noise - 1 - min_v) / qstep).astype(np.int32)
    id_p0 = np.clip(id_p0, 0, np.size(prob) - 1)
    p0 = prob[id_p0]
    
    # Calculate indices for p1 (bit=1, symbol=-1)
    id_p1 = ((s_mod_plus_res_noise + 1 - min_v) / qstep).astype(np.int32)
    id_p1 = np.clip(id_p1, 0, np.size(prob) - 1)
    p1 = prob[id_p1]
    
    # LLR = log(p0/p1)
    LLR = np.log(np.divide(p0 + 1e-7, p1 + 1e-7))
    
    return LLR


def denoising_and_calc_LLR_awgn(res_noise_power, y_receive, output_pre_decoder, 
                                 conv_net, device):
    """
    Denoise using CNN and calculate LLR assuming AWGN residual noise.
    
    Args:
        res_noise_power: Residual noise power after denoising
        y_receive: Received signal
        output_pre_decoder: Previous decoder output
        conv_net: CNN denoiser model
        device: 'cuda' or 'cpu'
    
    Returns:
        LLR for next BP iteration
    """
    # Estimate noise with CNN denoiser
    noise_before_cnn = y_receive - (output_pre_decoder * (-2) + 1)
    
    # Convert to torch and denoise
    noise_before_cnn_torch = torch.from_numpy(noise_before_cnn).float().to(device)
    
    with torch.no_grad():
        noise_after_cnn_torch = conv_net(noise_before_cnn_torch)
    
    noise_after_cnn = noise_after_cnn_torch.cpu().numpy()
    
    # Calculate LLR for next BP decoding
    s_mod_plus_res_noise = y_receive - noise_after_cnn
    LLR = s_mod_plus_res_noise * 2.0 / res_noise_power
    
    return LLR


def denoising_and_calc_LLR_epdf(prob, y_receive, output_pre_decoder, 
                                conv_net, device):
    """
    Denoise using CNN and calculate LLR using empirical PDF.
    
    Args:
        prob: Empirical probability distribution
        y_receive: Received signal
        output_pre_decoder: Previous decoder output
        conv_net: CNN denoiser model
        device: 'cuda' or 'cpu'
    
    Returns:
        LLR for next BP iteration
    """
    # Estimate noise with CNN denoiser
    noise_before_cnn = y_receive - (output_pre_decoder * (-2) + 1)
    
    # Convert to torch and denoise
    noise_before_cnn_torch = torch.from_numpy(noise_before_cnn).float().to(device)
    
    with torch.no_grad():
        noise_after_cnn_torch = conv_net(noise_before_cnn_torch)
    
    noise_after_cnn = noise_after_cnn_torch.cpu().numpy()
    
    # Calculate LLR for next BP decoding
    s_mod_plus_res_noise = y_receive - noise_after_cnn
    LLR = calc_LLR_epdf(prob, s_mod_plus_res_noise)
    
    return LLR


# ============================================================================
# SIMULATION
# ============================================================================

def simulation_colored_noise(linear_code, top_config, net_config, simutimes_range, 
                             target_err_bits_num, batch_size):
    """
    Simulate BER performance of iterative BP-CNN system.
    
    Args:
        linear_code: LDPC code object
        top_config: Top-level configuration
        net_config: Network configuration
        simutimes_range: [min_simutimes, max_simutimes]
        target_err_bits_num: Stop when this many errors collected
        batch_size: Batch size for simulation
    """
    # Load configurations
    SNRset = top_config.eval_SNRs
    bp_iter_num = top_config.BP_iter_nums_simu
    noise_io = NoiseIO(top_config.N_code, False, None, 
                      top_config.cov_1_2_file_simu, rng_seed=0)
    denoising_net_num = top_config.cnn_net_number
    model_id = top_config.model_id
    
    G_matrix = linear_code.G_matrix
    H_matrix = linear_code.H_matrix
    K, N = np.shape(G_matrix)
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Build BP decoder
    if np.size(bp_iter_num) != denoising_net_num + 1:
        print('Error: length of bp_iter_num is incorrect!')
        exit(0)
    
    bp_decoder = BP_NetDecoder(H_matrix, batch_size, device=device)
    # bp_decoder.to(device)  # Removed: handled in init
    
    # Build denoising networks
    conv_net = {}
    for net_id in range(denoising_net_num):
        if top_config.same_model_all_nets and net_id > 0:
            conv_net[net_id] = conv_net[0]
        else:
            conv_net[net_id] = ConvNet(net_config, None, net_id)
            conv_net[net_id].to(device)
            conv_net[net_id].eval()
            # Load trained weights
            conv_net[net_id].load_network(model_id[0:(net_id+1)])
    
    # Initialize simulation parameters
    max_simutimes = simutimes_range[1]
    min_simutimes = simutimes_range[0]
    max_batches, residual_times = divmod(max_simutimes, batch_size)
    max_batches = int(max_batches)
    residual_times = int(residual_times)
    
    if residual_times != 0:
        max_batches += 1
    
    # Generate output BER file name
    bp_str = np.array2string(bp_iter_num, separator='_', 
                            formatter={'int': lambda d: "%d" % d})
    bp_str = bp_str[1:(len(bp_str) - 1)]
    ber_file = f'{net_config.model_folder}BER({N}_{K})_BP({bp_str})'
    
    if top_config.corr_para != top_config.corr_para_simu:
        ber_file += f'_SimuCorrPara{top_config.corr_para_simu:.2f}'
    if top_config.same_model_all_nets:
        ber_file += '_SameModelAllNets'
    if top_config.update_llr_with_epdf:
        ber_file += '_llrepdf'
    if denoising_net_num > 0:
        model_id_str = np.array2string(model_id, separator='_',
                                       formatter={'int': lambda d: "%d" % d})
        model_id_str = model_id_str[1:(len(model_id_str)-1)]
        ber_file += f'_model{model_id_str}'
    if np.size(SNRset) == 1:
        ber_file += f'_{SNRset[0]:.1f}dB'
    
    ber_file += '.txt'
    fout_ber = open(ber_file, 'wt')
    
    # Simulation starts
    start = datetime.datetime.now()
    
    for SNR in SNRset:
        print(f'\n=== Simulating SNR = {SNR} dB ===')
        
        real_batch_size = batch_size
        bit_errs_iter = np.zeros(denoising_net_num + 1, dtype=np.int32)
        actual_simutimes = 0
        rng = np.random.RandomState(0)
        noise_io.reset_noise_generator()
        
        for ik in range(max_batches):
            print(f'Batch {ik+1}/{max_batches}', end=' ')
            
            if ik == max_batches - 1 and residual_times != 0:
                real_batch_size = residual_times
            
            # Generate transmitted signal and add noise
            x_bits, _, s_mod, ch_noise, y_receive, LLR = lbc.encode_and_transmission(
                G_matrix, SNR, real_batch_size, noise_io, rng
            )
            
            noise_power = np.mean(np.square(ch_noise))
            practical_snr = 10 * np.log10(1 / (noise_power * 2.0))
            print(f'Practical EbN0: {practical_snr:.2f} dB')
            
            # Iterative BP-CNN decoding
            for iter_idx in range(denoising_net_num + 1):
                # BP decoding
                u_BP_decoded = bp_decoder.decode(LLR.astype(np.float32), bp_iter_num[iter_idx])
                
                # CNN denoising (if not last iteration)
                if iter_idx < denoising_net_num:
                    if top_config.update_llr_with_epdf:
                        prob = conv_net[iter_idx].get_res_noise_pdf(model_id).get(np.float32(SNR))
                        LLR = denoising_and_calc_LLR_epdf(
                            prob, y_receive, u_BP_decoded, 
                            conv_net[iter_idx], device
                        )
                    else:
                        res_noise_power = conv_net[iter_idx].get_res_noise_power(
                            model_id, SNRset
                        ).get(np.float32(SNR))
                        LLR = denoising_and_calc_LLR_awgn(
                            res_noise_power, y_receive, u_BP_decoded,
                            conv_net[iter_idx], device
                        )
                
                # Count bit errors
                output_x = linear_code.dec_src_bits(u_BP_decoded)
                bit_errs_iter[iter_idx] += np.sum(output_x != x_bits)
            
            actual_simutimes += real_batch_size
            
            # Check if enough errors collected
            if (bit_errs_iter[denoising_net_num] >= target_err_bits_num and 
                actual_simutimes >= min_simutimes):
                break
        
        print(f'\n{actual_simutimes * K} bits simulated')
        
        # Calculate and save BER
        ber_iter = np.zeros(denoising_net_num + 1, dtype=np.float64)
        fout_ber.write(f'{SNR}\t')
        
        for iter_idx in range(denoising_net_num + 1):
            ber_iter[iter_idx] = bit_errs_iter[iter_idx] / float(K * actual_simutimes)
            fout_ber.write(f'{ber_iter[iter_idx]}\t')
        
        fout_ber.write('\n')
        print(f'BER after final iteration: {ber_iter[denoising_net_num]:.6e}')
    
    fout_ber.close()
    end = datetime.datetime.now()
    
    print(f'\nTotal simulation time: {(end-start).seconds}s')
    print(f'Results saved to: {ber_file}')


# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_noise_samples(linear_code, top_config, net_config, train_config,
                          bp_iter_num, net_id_data_for, generate_data_for,
                          noise_io, model_id):
    """
    Generate training/test data for CNN denoiser.
    
    Args:
        linear_code: LDPC code object
        top_config: Top-level configuration
        net_config: Network configuration
        train_config: Training configuration
        bp_iter_num: BP iteration numbers
        net_id_data_for: Which CNN network to generate data for
        generate_data_for: 'Training' or 'Test'
        noise_io: Noise generator
        model_id: Model IDs
    """
    G_matrix = linear_code.G_matrix
    H_matrix = linear_code.H_matrix
    
    SNRset = train_config.SNR_set_gen_data
    
    if generate_data_for == 'Training':
        batch_size_each_SNR = int(
            train_config.training_minibatch_size // np.size(SNRset)
        )
        total_batches = int(
            train_config.training_sample_num // train_config.training_minibatch_size
        )
    elif generate_data_for == 'Test':
        batch_size_each_SNR = int(
            train_config.test_minibatch_size // np.size(SNRset)
        )
        total_batches = int(
            train_config.test_sample_num // train_config.test_minibatch_size
        )
    else:
        print('Invalid data generation objective!')
        exit(0)
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Build BP decoder
    if np.size(bp_iter_num) != net_id_data_for + 1:
        print('Error: length of bp_iter_num is incorrect!')
        exit(0)
    
    bp_decoder = BP_NetDecoder(H_matrix, batch_size_each_SNR, device=device)
    # bp_decoder.to(device)  # Removed: handled in init
    
    # Build previous CNN denoisers (if any)
    conv_net = {}
    for net_id in range(net_id_data_for):
        conv_net[net_id] = ConvNet(net_config, None, net_id)
        conv_net[net_id].to(device)
        conv_net[net_id].eval()
        conv_net[net_id].load_network(model_id[0:(net_id+1)])
    
    # Open output files
    if generate_data_for == 'Training':
        fout_est_noise = open(train_config.training_feature_file, 'wb')
        fout_real_noise = open(train_config.training_label_file, 'wb')
    else:
        fout_est_noise = open(train_config.test_feature_file, 'wb')
        fout_real_noise = open(train_config.test_label_file, 'wb')
    
    start = datetime.datetime.now()
    print(f'Generating {generate_data_for} data...')
    
    # Generate data
    for ik in range(total_batches):
        if (ik + 1) % 10 == 0:
            print(f'Batch {ik+1}/{total_batches}')
        
        for SNR in SNRset:
            # Generate transmitted signal and add noise
            x_bits, _, _, channel_noise, y_receive, LLR = lbc.encode_and_transmission(
                G_matrix, SNR, batch_size_each_SNR, noise_io
            )
            
            # Iterative BP-CNN processing
            for iter_idx in range(net_id_data_for + 1):
                u_BP_decoded = bp_decoder.decode(LLR.astype(np.float32), bp_iter_num[iter_idx])
                
                # CNN denoising for intermediate iterations
                if iter_idx != net_id_data_for:
                    if top_config.update_llr_with_epdf:
                        prob = conv_net[iter_idx].get_res_noise_pdf(model_id).get(np.float32(SNR))
                        LLR = denoising_and_calc_LLR_epdf(
                            prob, y_receive, u_BP_decoded,
                            conv_net[iter_idx], device
                        )
                    else:
                        res_noise_power = conv_net[iter_idx].get_res_noise_power(
                            model_id
                        ).get(np.float32(SNR))
                        LLR = denoising_and_calc_LLR_awgn(
                            res_noise_power, y_receive, u_BP_decoded,
                            conv_net[iter_idx], device
                        )
            
            # Save estimated and real noise
            noise_before_cnn = y_receive - (u_BP_decoded * (-2) + 1)
            noise_before_cnn.astype(np.float32).tofile(fout_est_noise)
            channel_noise.astype(np.float32).tofile(fout_real_noise)
    
    fout_est_noise.close()
    fout_real_noise.close()
    
    end = datetime.datetime.now()
    print(f'Data generation completed in {(end-start).seconds}s')


# ============================================================================
# RESIDUAL NOISE ANALYSIS
# ============================================================================

def analyze_residual_noise(linear_code, top_config, net_config, simutimes, batch_size):
    """
    Analyze residual noise properties after CNN denoising.
    Calculates either power or empirical PDF.
    
    Args:
        linear_code: LDPC code object
        top_config: Top-level configuration
        net_config: Network configuration
        simutimes: Number of simulation times
        batch_size: Batch size
    """
    # Load configurations
    net_id_tested = top_config.currently_trained_net_id
    model_id = top_config.model_id
    bp_iter_num = top_config.BP_iter_nums_gen_data[0:(net_id_tested + 1)]
    noise_io = NoiseIO(top_config.N_code, False, None, top_config.cov_1_2_file)
    SNRset = top_config.eval_SNRs
    
    G_matrix = linear_code.G_matrix
    H_matrix = linear_code.H_matrix
    _, N = np.shape(G_matrix)
    
    max_batches, residual_times = divmod(simutimes, batch_size)
    max_batches = int(max_batches)
    residual_times = int(residual_times)
    
    print(f'Analyzing residual noise with {simutimes} samples')
    
    if residual_times != 0:
        max_batches += 1
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Build BP decoder
    if np.size(bp_iter_num) != net_id_tested + 1:
        print('Error: length of bp_iter_num is incorrect!')
        exit(0)
    
    bp_decoder = BP_NetDecoder(H_matrix, batch_size, device=device)
    # bp_decoder.to(device)  # Removed: handled in init
    
    # Build denoising networks
    conv_net = {}
    for net_id in range(net_id_tested + 1):
        conv_net[net_id] = ConvNet(net_config, None, net_id)
        conv_net[net_id].to(device)
        conv_net[net_id].eval()
        conv_net[net_id].load_network(model_id[0:(net_id+1)])
    
    # Prepare output file
    model_id_str = np.array2string(model_id, separator='_',
                                   formatter={'int': lambda d: "%d" % d})
    model_id_str = model_id_str[1:(len(model_id_str) - 1)]
    loss_file_name = (
        f"{net_config.residual_noise_property_folder}"
        f"residual_noise_property_netid{net_id_tested}_model{model_id_str}.txt"
    )
    fout_loss = open(loss_file_name, 'wt')
    
    start = datetime.datetime.now()
    
    for SNR in SNRset:
        print(f'\nAnalyzing SNR = {SNR} dB')
        noise_io.reset_noise_generator()
        
        real_batch_size = batch_size
        loss = 0.0
        prob = np.ones(0)
        
        for ik in range(max_batches):
            if (ik + 1) % 10 == 0:
                print(f'Batch {ik+1}/{max_batches}')
            
            if ik == max_batches - 1 and residual_times != 0:
                real_batch_size = residual_times
            
            # Generate data
            x_bits, _, s_mod, channel_noise, y_receive, LLR = lbc.encode_and_transmission(
                G_matrix, SNR, real_batch_size, noise_io
            )
            
            # Iterative BP-CNN processing
            for iter_idx in range(net_id_tested + 1):
                u_BP_decoded = bp_decoder.decode(LLR.astype(np.float32), bp_iter_num[iter_idx])
                
                # Denoise
                noise_before_cnn = y_receive - (u_BP_decoded * (-2) + 1)
                noise_before_cnn_torch = torch.from_numpy(noise_before_cnn).float().to(device)
                
                with torch.no_grad():
                    noise_after_cnn_torch = conv_net[iter_idx](noise_before_cnn_torch)
                
                noise_after_cnn = noise_after_cnn_torch.cpu().numpy()
                s_mod_plus_res_noise = y_receive - noise_after_cnn
                
                # Update LLR for next iteration
                if iter_idx < net_id_tested:
                    if top_config.update_llr_with_epdf:
                        prob_tmp = conv_net[iter_idx].get_res_noise_pdf(model_id).get(np.float32(SNR))
                        LLR = calc_LLR_epdf(prob_tmp, s_mod_plus_res_noise)
                    else:
                        res_noise_power = conv_net[iter_idx].get_res_noise_power(model_id).get(np.float32(SNR))
                        LLR = s_mod_plus_res_noise * 2.0 / res_noise_power
            
            # Accumulate statistics
            if top_config.update_llr_with_epdf:
                prob = stat_prob(s_mod_plus_res_noise - s_mod, prob)
            else:
                loss += np.sum(np.mean(np.square(s_mod_plus_res_noise - s_mod), 1))
        
        # Write results
        if top_config.update_llr_with_epdf:
            fout_loss.write(f'{SNR}\t')
            for i in range(np.size(prob)):
                fout_loss.write(f'{prob[i]}\t')
            fout_loss.write('\n')
        else:
            loss /= float(simutimes)
            fout_loss.write(f'{SNR}\t{loss}\n')
    
    fout_loss.close()
    end = datetime.datetime.now()
    
    print(f'\nAnalysis completed in {(end-start).seconds}s')
    print(f'Results saved to: {loss_file_name}')