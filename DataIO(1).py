import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TrainingDataset(Dataset):
    """
    PyTorch Dataset for loading training data from binary files.
    Replaces the old TrainingDataIO class with proper PyTorch integration.
    """
    def __init__(self, feature_filename, label_filename, total_samples, feature_length, label_length):
        print("Constructing PyTorch Training Dataset!")
        self.feature_length = feature_length
        self.label_length = label_length
        self.total_samples = total_samples
        
        # Load all data into memory (more efficient for repeated access)
        print(f"Loading {total_samples} training samples...")
        self.features = np.fromfile(feature_filename, dtype=np.float32)
        self.labels = np.fromfile(label_filename, dtype=np.float32)
        
        # Reshape to (num_samples, feature/label_length)
        self.features = self.features.reshape(total_samples, feature_length)
        self.labels = self.labels.reshape(total_samples, label_length)
        
        print(f"Features shape: {self.features.shape}")
        print(f"Labels shape: {self.labels.shape}")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """
        Get a single training sample.
        Returns PyTorch tensors ready for GPU.
        """
        feature = torch.from_numpy(self.features[idx]).float()
        label = torch.from_numpy(self.labels[idx]).float()
        return feature, label


# ============================================================================
# PYTORCH DATASET FOR TESTING
# ============================================================================
class TestDataset(Dataset):
    """
    PyTorch Dataset for loading test data from binary files.
    Replaces the old TestDataIO class.
    """
    def __init__(self, feature_filename, label_filename, test_sample_num, feature_length, label_length):
        print("Constructing PyTorch Test Dataset!")
        self.feature_length = feature_length
        self.label_length = label_length
        self.test_sample_num = test_sample_num
        
        # Load all test data into memory
        print(f"Loading {test_sample_num} test samples...")
        self.features = np.fromfile(feature_filename, dtype=np.float32)
        self.labels = np.fromfile(label_filename, dtype=np.float32)
        
        # Reshape
        self.features = self.features.reshape(test_sample_num, feature_length)
        self.labels = self.labels.reshape(test_sample_num, label_length)
        
        print(f"Features shape: {self.features.shape}")
        print(f"Labels shape: {self.labels.shape}")
    
    def __len__(self):
        return self.test_sample_num
    
    def __getitem__(self, idx):
        feature = torch.from_numpy(self.features[idx]).float()
        label = torch.from_numpy(self.labels[idx]).float()
        return feature, label


# ============================================================================
# NOISE GENERATOR (Kept as NumPy for compatibility)
# ============================================================================
class NoiseIO:
    """
    Generates correlated noise using pre-computed covariance matrix.
    Kept as NumPy since it's used during data generation, not training.
    """
    def __init__(self, blk_len, read_from_file, noise_file, cov_1_2_mat_file_gen_noise, rng_seed=None):
        self.read_from_file = read_from_file
        self.blk_len = blk_len
        self.rng_seed = rng_seed
        
        if read_from_file:
            self.fin_noise = open(noise_file, 'rb')
        else:
            self.rng = np.random.RandomState(rng_seed)
            
            # Load covariance matrix square root
            fin_cov_file = open(cov_1_2_mat_file_gen_noise, 'rb')
            cov_1_2_mat = np.fromfile(fin_cov_file, np.float32, blk_len * blk_len)
            cov_1_2_mat = np.reshape(cov_1_2_mat, [blk_len, blk_len])
            fin_cov_file.close()
            
            # Store as instance variable
            self.cov_1_2_mat = cov_1_2_mat
            
            # Output parts of correlation function for verification
            cov_func = np.matmul(cov_1_2_mat, cov_1_2_mat)
            print('Correlation function of channel noise:')
            print(cov_func[0, 0:10])

    def __del__(self):
        if self.read_from_file:
            self.fin_noise.close()

    def reset_noise_generator(self):
        """Reset to generate the same noise sequence"""
        if self.read_from_file:
            self.fin_noise.seek(0, 0)
        else:
            self.rng = np.random.RandomState(self.rng_seed)

    def generate_noise(self, batch_size):
        """
        Generate correlated noise samples.
        Returns: numpy array of shape (batch_size, blk_len)
        """
        if self.read_from_file:
            noise = np.fromfile(self.fin_noise, np.float32, batch_size * self.blk_len)
            noise = np.reshape(noise, [batch_size, self.blk_len])
        else:
            # Generate AWGN
            noise_awgn = self.rng.randn(batch_size, self.blk_len).astype(np.float32)
            # Apply correlation
            noise = np.matmul(noise_awgn, self.cov_1_2_mat)
        
        return noise


# ============================================================================
# HELPER FUNCTIONS FOR CREATING DATALOADERS
# ============================================================================
def create_train_loader(feature_file, label_file, total_samples, feature_length, 
                       label_length, batch_size, shuffle=True, num_workers=4):
    """
    Create a PyTorch DataLoader for training.
    
    Args:
        feature_file: Path to binary feature file
        label_file: Path to binary label file
        total_samples: Total number of training samples
        feature_length: Length of each feature vector
        label_length: Length of each label vector
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of parallel data loading workers
    
    Returns:
        DataLoader object
    """
    dataset = TrainingDataset(feature_file, label_file, total_samples, 
                             feature_length, label_length)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # Faster GPU transfer
    )
    
    return loader


def create_test_loader(feature_file, label_file, test_samples, feature_length,
                      label_length, batch_size, num_workers=2):
    """
    Create a PyTorch DataLoader for testing.
    
    Args:
        feature_file: Path to binary feature file
        label_file: Path to binary label file
        test_samples: Total number of test samples
        feature_length: Length of each feature vector
        label_length: Length of each label vector
        batch_size: Batch size for testing
        num_workers: Number of parallel data loading workers
    
    Returns:
        DataLoader object
    """
    dataset = TestDataset(feature_file, label_file, test_samples,
                         feature_length, label_length)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


# ============================================================================
# BACKWARD COMPATIBILITY WRAPPER (Optional)
# ============================================================================
class TrainingDataIO:
    """
    Backward compatibility wrapper that mimics the old TensorFlow interface.
    Use the Dataset classes above for new code.
    """
    def __init__(self, feature_filename, label_filename, total_training_samples, 
                 feature_length, label_length):
        print("WARNING: TrainingDataIO is deprecated. Use TrainingDataset + DataLoader instead.")
        self.dataset = TrainingDataset(feature_filename, label_filename, 
                                      total_training_samples, feature_length, label_length)
        self.current_idx = 0
    
    def load_next_mini_batch(self, mini_batch_size, factor_of_start_pos=1):
        """Legacy method - returns numpy arrays"""
        # Random starting position
        start_idx = np.random.randint(0, len(self.dataset))
        start_idx = (start_idx // factor_of_start_pos) * factor_of_start_pos
        
        features = []
        labels = []
        
        for i in range(mini_batch_size):
            idx = (start_idx + i) % len(self.dataset)
            feat, lab = self.dataset[idx]
            features.append(feat.numpy())
            labels.append(lab.numpy())
        
        features = np.stack(features)
        labels = np.stack(labels)
        
        return features, labels