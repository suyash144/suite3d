import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class SingleChannelCNN(nn.Module):
    """Lightweight 3D CNN for processing a single channel (1, 5, 20, 20) - only 3 layers"""
    
    def __init__(self, channel_feature_dim=64):
        super(SingleChannelCNN, self).__init__()
        
        # Three convolutional layers
        self.conv1 = nn.Conv3d(1, 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(4)
        
        self.conv2 = nn.Conv3d(4, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(8)
        
        # Final conv with mild downsampling
        self.conv3 = nn.Conv3d(8, 16, kernel_size=3, stride=(1, 2, 2), padding=1)
        self.bn3 = nn.BatchNorm3d(16)
        
        # Global pooling for this channel
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(16, channel_feature_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # x shape: (batch, 1, 5, 20, 20)
        x = F.relu(self.bn1(self.conv1(x)))  # (batch, 4, 5, 20, 20)
        x = F.relu(self.bn2(self.conv2(x)))  # (batch, 8, 5, 20, 20)
        x = F.relu(self.bn3(self.conv3(x)))  # (batch, 16, 5, 10, 10)

        x = self.global_pool(x)  # (batch, 16, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 16)
        x = self.dropout(x)
        x = self.fc(x)  # (batch, channel_feature_dim)
        
        return x


class MultiChannelCNN(nn.Module):
    """3D CNN with separate processing for each of the 3 channels"""
    
    def __init__(self, input_channels=3, channel_feature_dim=64, feature_dim=128):
        super(MultiChannelCNN, self).__init__()
        
        assert input_channels == 3, "This architecture is designed for exactly 3 channels"
        
        # Create separate CNNs for each channel
        self.channel_cnn_0 = SingleChannelCNN(channel_feature_dim)
        self.channel_cnn_1 = SingleChannelCNN(channel_feature_dim)
        self.channel_cnn_2 = SingleChannelCNN(channel_feature_dim)
        
        # Fusion layers to combine features from all channels
        self.fusion_fc1 = nn.Linear(3 * channel_feature_dim, 256)
        self.fusion_bn = nn.BatchNorm1d(256)
        self.fusion_dropout = nn.Dropout(0.3)
        self.fusion_fc2 = nn.Linear(256, feature_dim)
        
    def forward(self, x):
        # x shape: (batch, 3, 5, 20, 20)
        batch_size = x.size(0)
        
        # Split into individual channels
        channel_0 = x[:, 0:1, :, :, :]  # (batch, 1, 5, 20, 20)
        channel_1 = x[:, 1:2, :, :, :]  # (batch, 1, 5, 20, 20)
        channel_2 = x[:, 2:3, :, :, :]  # (batch, 1, 5, 20, 20)
        
        # Process each channel independently
        feat_0 = self.channel_cnn_0(channel_0)  # (batch, channel_feature_dim)
        feat_1 = self.channel_cnn_1(channel_1)  # (batch, channel_feature_dim)
        feat_2 = self.channel_cnn_2(channel_2)  # (batch, channel_feature_dim)
        
        # Concatenate channel features
        combined_features = torch.cat([feat_0, feat_1, feat_2], dim=1)  # (batch, 3*channel_feature_dim)
        
        # Fusion layers
        x = F.relu(self.fusion_bn(self.fusion_fc1(combined_features)))
        x = self.fusion_dropout(x)
        x = self.fusion_fc2(x)  # (batch, feature_dim)
        
        return x


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning"""
    
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.projection(x)


class ContrastiveModel(nn.Module):
    """Complete contrastive learning model"""
    
    def __init__(self, backbone_feature_dim=128, projection_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = MultiChannelCNN(channel_feature_dim=32, feature_dim=backbone_feature_dim)
        self.projection_head = ProjectionHead(
            input_dim=backbone_feature_dim,
            output_dim=projection_dim
        )
    
    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        return features, projections


class HDF5Dataset(Dataset):
    """Dataset class for loading 3D data from HDF5 file"""
    
    def __init__(self, hdf5_path, dataset_key='data', transform=None):
        self.hdf5_path = hdf5_path
        self.dataset_key = dataset_key
        self.transform = transform
        
        # Get dataset length without loading all data
        with h5py.File(hdf5_path, 'r') as f:
            self.length = len(f[dataset_key])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as f:
            sample = f[self.dataset_key][idx]  # Shape: (3, 5, 20, 20)
            
        sample = sample.astype(np.float32)

        view1 = sample.copy()
        view2 = sample.copy()
        
        if self.transform:
            view1 = self.transform(view1)
            view2 = self.transform(view2)
        
        return torch.from_numpy(view1), torch.from_numpy(view2)


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning"""
    
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        """
        Args:
            z1, z2: projections of shape (batch_size, projection_dim)
        """
        batch_size = z1.shape[0]
        
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)  # (2 * batch_size, projection_dim)

        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # (2 * batch_size, 2 * batch_size)
        
        labels = torch.arange(2 * batch_size, device=z.device)
        labels[:batch_size] += batch_size
        labels[batch_size:] -= batch_size
        
        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, device=z.device).bool()
        sim_matrix.masked_fill_(mask, -float('inf'))
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class ContrastiveLearner:
    """Training class for contrastive learning"""
    
    def __init__(self, model, device='cuda', lr=1e-3, temperature=0.1, log_dir=None, experiment_name=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = InfoNCELoss(temperature=temperature)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )

        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_name = experiment_name or f"contrastive_3d_{timestamp}"
            log_dir = f"runs/{exp_name}"
        
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        print(f"TensorBoard logging to: {log_dir}")
        print(f"View logs with: tensorboard --logdir={log_dir}")
        print(f"Using device: {device}")
    
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        batch_losses = []
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
        for batch_idx, (view1, view2) in enumerate(pbar):
            view1, view2 = view1.to(self.device), view2.to(self.device)
            
            features1, projections1 = self.model(view1)
            features2, projections2 = self.model(view2)
            
            loss = self.criterion(projections1, projections2)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            batch_loss = loss.item()
            total_loss += batch_loss
            batch_losses.append(batch_loss)
            num_batches += 1
            self.global_step += 1
            
            # Log to TensorBoard every 50 batches
            if batch_idx % 50 == 0:
                self.writer.add_scalar('Loss/Batch', batch_loss, self.global_step)
                self.writer.add_scalar('Learning_Rate', 
                                     self.optimizer.param_groups[0]['lr'], 
                                     self.global_step)
            
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}'
            })
        
        avg_epoch_loss = total_loss / num_batches
        
        # Log epoch-level metrics
        self.writer.add_scalar('Loss/Epoch', avg_epoch_loss, epoch)
        self.writer.add_scalar('Loss/Epoch_Std', np.std(batch_losses), epoch)
        self.writer.add_histogram('Loss/Batch_Distribution', np.array(batch_losses), epoch)
        
        # Log model parameters and gradients every 10 epochs
        if epoch % 10 == 0:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
                self.writer.add_histogram(f'Parameters/{name}', param, epoch)
        
        return avg_epoch_loss
    
    def extract_features(self, dataloader):
        """Extract features for downstream analysis (UMAP, etc.)"""
        self.model.eval()
        all_features = []
        
        with torch.no_grad():
            for view1, _ in tqdm(dataloader, desc='Extracting features'):
                view1 = view1.to(self.device)
                features, _ = self.model(view1)
                all_features.append(features.cpu().numpy())
        
        return np.concatenate(all_features, axis=0)
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'global_step' in checkpoint:
            self.global_step = checkpoint['global_step']
    
    def close_tensorboard(self):
        """Close TensorBoard writer"""
        self.writer.close()



