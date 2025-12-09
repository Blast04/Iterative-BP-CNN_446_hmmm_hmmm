import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import datetime
from DataIO import TrainingDataset, TestDataset


class ConvNet(nn.Module):
    """
    Fully convolutional network for channel noise estimation.
    Architecture: Multiple 1D conv layers with ReLU, final layer has no activation.
    """
    def __init__(self, net_config, train_config, net_id):
        super(ConvNet, self).__init__()
        
        self.net_config = net_config
        self.train_config = train_config
        self.net_id = net_id
        
        # Build convolutional layers
        self.convs = nn.ModuleList()
        
        for layer in range(net_config.total_layers):
            if layer == 0:
                in_channels = 1
            else:
                in_channels = net_config.feature_map_nums[layer - 1]
            
            out_channels = net_config.feature_map_nums[layer]
            kernel_size = net_config.filter_sizes[layer]
            
            # 1D convolution (treating sequence as 1D signal)
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,  # 'SAME' padding
                bias=True
            )
            
            # Xavier initialization
            nn.init.xavier_uniform_(conv.weight)
            nn.init.xavier_uniform_(conv.bias.unsqueeze(0))
            
            self.convs.append(conv)
        
        # Dictionary to store residual noise properties
        self.res_noise_power_dict = {}
        self.res_noise_pdf_dict = {}
        
        print(f"ConvNet {net_id} constructed with {net_config.total_layers} layers")
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, feature_length)
        
        Returns:
            Output tensor of shape (batch_size, label_length)
        """
        # Reshape from (batch, length) to (batch, channels=1, length)
        x = x.unsqueeze(1)
        
        # Pass through convolutional layers
        for i, conv in enumerate(self.convs):
            x = conv(x)
            
            # ReLU activation for all layers except the last
            if i < len(self.convs) - 1:
                x = F.relu(x)
        
        # Reshape back to (batch, length)
        x = x.squeeze(1)
        
        return x
    
    def train_network(self, model_id, device='cuda'):
        """
        Train the denoising network.
        
        Args:
            model_id: Array of model IDs for saving
            device: 'cuda' or 'cpu'
        """
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            device = 'cpu'
        
        self.to(device)
        print(f"Training on device: {device}")
        
        start = datetime.datetime.now()
        
        # Create datasets
        train_dataset = TrainingDataset(
            self.train_config.training_feature_file,
            self.train_config.training_label_file,
            self.train_config.training_sample_num,
            self.net_config.feature_length,
            self.net_config.label_length
        )
        
        test_dataset = TestDataset(
            self.train_config.test_feature_file,
            self.train_config.test_label_file,
            self.train_config.test_sample_num,
            self.net_config.feature_length,
            self.net_config.label_length
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_config.training_minibatch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=(device == 'cuda')
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.train_config.test_minibatch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=(device == 'cuda')
        )
        
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters())
        
        # Calculate original loss before training
        print("Calculating initial test loss...")
        min_loss, orig_loss = self.test_network_online(
            test_loader, 
            calc_orig_loss=True,
            device=device
        )
        print(f"Initial test loss: {min_loss:.6f}, original loss: {orig_loss:.6f}")
        
        # Save best model state
        best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
        
        # Training loop
        count = 0
        epoch = 0
        print('Epoch\tLoss')
        
        while epoch < self.train_config.epoch_num:
            epoch += 1
            
            # Training step
            self.train()
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self(batch_features)
                
                # Calculate loss
                if self.train_config.normality_test_enabled:
                    # MSE loss
                    mse_loss = F.mse_loss(outputs, batch_labels)
                    
                    # Normality test loss
                    norm_loss = self.calc_normality_test(
                        batch_labels - outputs,
                        self.train_config.training_minibatch_size,
                        batch_size_for_norm_test=1
                    )
                    
                    if self.train_config.normality_lambda != np.inf:
                        loss = mse_loss + norm_loss * self.train_config.normality_lambda
                    else:
                        loss = norm_loss
                else:
                    loss = F.mse_loss(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            # Evaluation
            if epoch % 500 == 0 or epoch == self.train_config.epoch_num:
                print(f"{epoch}", end='\t')
                
                test_loss, _ = self.test_network_online(
                    test_loader,
                    calc_orig_loss=False,
                    device=device
                )
                
                if test_loss < min_loss:
                    min_loss = test_loss
                    # Save best model
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                    count = 0
                else:
                    count += 1
                    if count >= 8:  # Early stopping
                        print(f"\nEarly stopping at epoch {epoch}")
                        break
        
        # Load best model and save
        self.load_state_dict(best_state)
        self.save_network(model_id)
        
        end = datetime.datetime.now()
        print(f'\nFinal minimum loss: {min_loss:.6f}')
        print(f'Training time: {(end-start).seconds}s')
    
    def test_network_online(self, test_loader, calc_orig_loss=False, device='cuda'):
        """
        Evaluate network on test set.
        
        Args:
            test_loader: DataLoader for test data
            calc_orig_loss: Whether to calculate original loss (before denoising)
            device: 'cuda' or 'cpu'
        
        Returns:
            (test_loss, orig_loss) tuple
        """
        self.eval()
        
        total_loss = 0.0
        total_orig_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                batch_size = batch_features.size(0)
                
                # Forward pass
                outputs = self(batch_features)
                
                # Calculate losses
                if self.train_config.normality_test_enabled:
                    mse_loss = F.mse_loss(outputs, batch_labels, reduction='sum')
                    norm_loss = self.calc_normality_test(
                        batch_labels - outputs,
                        batch_size,
                        batch_size_for_norm_test=1
                    )
                    
                    if self.train_config.normality_lambda != np.inf:
                        loss = mse_loss + norm_loss * batch_size * self.train_config.normality_lambda
                    else:
                        loss = norm_loss * batch_size
                else:
                    loss = F.mse_loss(outputs, batch_labels, reduction='sum')
                
                total_loss += loss.item()
                
                if calc_orig_loss:
                    if self.train_config.normality_test_enabled:
                        orig_mse = F.mse_loss(batch_features, batch_labels, reduction='sum')
                        orig_norm = self.calc_normality_test(
                            batch_labels - batch_features,
                            batch_size,
                            batch_size_for_norm_test=1
                        )
                        
                        if self.train_config.normality_lambda != np.inf:
                            orig_loss = orig_mse + orig_norm * batch_size * self.train_config.normality_lambda
                        else:
                            orig_loss = orig_norm * batch_size
                    else:
                        orig_loss = F.mse_loss(batch_features, batch_labels, reduction='sum')
                    
                    total_orig_loss += orig_loss.item()
                
                total_samples += batch_size
        
        avg_loss = total_loss / total_samples
        avg_orig_loss = total_orig_loss / total_samples if calc_orig_loss else 0.0
        
        if calc_orig_loss:
            print(f"Test loss: {avg_loss:.6f}, orig loss: {avg_orig_loss:.6f}")
        else:
            print(f"{avg_loss:.6f}")
        
        return avg_loss, avg_orig_loss
    
    def calc_normality_test(self, residual_noise, batch_size, batch_size_for_norm_test):
        """
        Calculate normality test metric (skewness^2 + 0.25*(kurtosis-3)^2).
        
        Args:
            residual_noise: Tensor of shape (batch_size, feature_length)
            batch_size: Total batch size
            batch_size_for_norm_test: Batch size for each normality test group
        
        Returns:
            Scalar normality test loss
        """
        groups = batch_size // batch_size_for_norm_test
        residual_noise = residual_noise.view(groups, -1)
        
        # Calculate moments
        mean = residual_noise.mean(dim=1, keepdim=True)
        variance = ((residual_noise - mean) ** 2).mean(dim=1, keepdim=True)
        moment_3rd = ((residual_noise - mean) ** 3).mean(dim=1, keepdim=True)
        moment_4th = ((residual_noise - mean) ** 4).mean(dim=1, keepdim=True)
        
        # Skewness and kurtosis
        skewness = moment_3rd / (variance ** 1.5 + 1e-10)
        kurtosis = moment_4th / (variance ** 2 + 1e-10)
        
        # Normality test metric
        norm_test = (skewness ** 2 + 0.25 * (kurtosis - 3) ** 2).mean()
        
        return norm_test
    
    def save_network(self, model_id):
        """
        Save network weights to file.
        
        Args:
            model_id: Array of model IDs
        """
        model_id_str = np.array2string(model_id, separator='_', 
                                       formatter={'int': lambda d: "%d" % d})
        model_id_str = model_id_str[1:(len(model_id_str) - 1)]
        
        save_model_folder = f"{self.net_config.model_folder}netid{self.net_id}_model{model_id_str}"
        
        if not os.path.exists(save_model_folder):
            os.makedirs(save_model_folder)
        
        save_path = f"{save_model_folder}/model.pth"
        
        torch.save({
            'state_dict': self.state_dict(),
            'net_config': {
                'total_layers': self.net_config.total_layers,
                'feature_length': self.net_config.feature_length,
                'label_length': self.net_config.label_length,
                'filter_sizes': self.net_config.filter_sizes.tolist(),
                'feature_map_nums': self.net_config.feature_map_nums.tolist()
            }
        }, save_path)
        
        print(f"Model saved to {save_path}")
    
    def load_network(self, model_id):
        """
        Load network weights from file.
        
        Args:
            model_id: Array of model IDs (only first net_id+1 elements used)
        """
        model_id_subset = model_id[0:(self.net_id + 1)]
        model_id_str = np.array2string(model_id_subset, separator='_',
                                       formatter={'int': lambda d: "%d" % d})
        model_id_str = model_id_str[1:(len(model_id_str) - 1)]
        
        model_folder = f"{self.net_config.model_folder}netid{self.net_id}_model{model_id_str}"
        model_path = f"{model_folder}/model.pth"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'])
        
        print(f"Model loaded from {model_path}")
    
    def get_res_noise_power(self, model_id, SNRset=np.zeros(0)):
        """
        Load residual noise power from file.
        
        Args:
            model_id: Array of model IDs
            SNRset: Array of SNR values (unused, kept for compatibility)
        
        Returns:
            Dictionary mapping SNR to residual noise power
        """
        if len(self.res_noise_power_dict) == 0:
            model_id_subset = model_id[0:(self.net_id + 1)]
            model_id_str = np.array2string(model_id_subset, separator='_',
                                           formatter={'int': lambda d: "%d" % d})
            model_id_str = model_id_str[1:(len(model_id_str) - 1)]
            
            residual_noise_power_file = (
                f"{self.net_config.residual_noise_property_folder}"
                f"residual_noise_property_netid{self.net_id}_model{model_id_str}.txt"
            )
            
            data = np.loadtxt(residual_noise_power_file, dtype=np.float32)
            shape_data = np.shape(data)
            
            if len(shape_data) == 1:
                self.res_noise_power_dict[data[0]] = data[1:shape_data[0]]
            else:
                SNR_num = shape_data[0]
                for i in range(SNR_num):
                    self.res_noise_power_dict[data[i, 0]] = data[i, 1:shape_data[1]]
        
        return self.res_noise_power_dict
    
    def get_res_noise_pdf(self, model_id):
        """
        Load residual noise PDF from file.
        
        Args:
            model_id: Array of model IDs
        
        Returns:
            Dictionary mapping SNR to residual noise PDF
        """
        if len(self.res_noise_pdf_dict) == 0:
            model_id_subset = model_id[0:(self.net_id + 1)]
            model_id_str = np.array2string(model_id_subset, separator='_',
                                           formatter={'int': lambda d: "%d" % d})
            model_id_str = model_id_str[1:(len(model_id_str) - 1)]
            
            residual_noise_pdf_file = (
                f"{self.net_config.residual_noise_property_folder}"
                f"residual_noise_property_netid{self.net_id}_model{model_id_str}.txt"
            )
            
            data = np.loadtxt(residual_noise_pdf_file, dtype=np.float32)
            shape_data = np.shape(data)
            
            if len(shape_data) == 1:
                self.res_noise_pdf_dict[data[0]] = data[1:shape_data[0]]
            else:
                SNR_num = shape_data[0]
                for i in range(SNR_num):
                    self.res_noise_pdf_dict[data[i, 0]] = data[i, 1:shape_data[1]]
        
        return self.res_noise_pdf_dict
