import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)

class GeneticDataset(Dataset):
    def __init__(self, genomic_seqs, ref_alleles=None, tumor_alleles=None, 
                 mutation_labels=None, cancer_labels=None, seq_length=101):
        """
        Dataset for genetic mutations and cancer types
        
        Args:
            genomic_seqs: List of genomic context sequences
            ref_alleles: List of reference alleles (optional)
            tumor_alleles: List of tumor alleles (optional)
            mutation_labels: Mutation type labels (optional)
            cancer_labels: Cancer type labels (optional)
            seq_length: Length to pad/truncate sequences to
        """
        self.genomic_seqs = list(genomic_seqs)
        self.ref_alleles = list(ref_alleles) if ref_alleles is not None else None
        self.tumor_alleles = list(tumor_alleles) if tumor_alleles is not None else None
        self.mutation_labels = np.array(mutation_labels) if mutation_labels is not None else None
        self.cancer_labels = np.array(cancer_labels) if cancer_labels is not None else None
        self.seq_length = seq_length
        
        # Validate data
        self.valid_indices = self._get_valid_indices()
        print(f"Initial samples: {len(self.genomic_seqs)}, Valid samples: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        
        # Process the genomic sequence
        seq = self._process_sequence(self.genomic_seqs[original_idx])
        
        # One-hot encode the sequence
        one_hot_seq = self.one_hot_encode(seq)
        
        # Additional features: reference and tumor alleles (if available)
        features = [one_hot_seq]
        
        if self.ref_alleles is not None and self.tumor_alleles is not None:
            ref_allele = self.ref_alleles[original_idx]
            tumor_allele = self.tumor_alleles[original_idx]
            
            # Create one-hot encoding for alleles (just one nucleotide)
            ref_one_hot = self.one_hot_encode_allele(ref_allele)
            tumor_one_hot = self.one_hot_encode_allele(tumor_allele)
            
            # Expand allele encodings to match sequence length
            ref_one_hot = np.repeat(ref_one_hot, self.seq_length, axis=1)  # Shape (6, 101)
            tumor_one_hot = np.repeat(tumor_one_hot, self.seq_length, axis=1)  # Shape (6, 101)
            
            features.append(ref_one_hot)
            features.append(tumor_one_hot)
        
        # Combine all features
        combined = np.concatenate(features, axis=0)  # Final shape (17, 101)
        tensor = torch.tensor(combined, dtype=torch.float32)
        
        # Return labels if available
        if self.mutation_labels is not None and self.cancer_labels is not None:
            mutation_label = int(self.mutation_labels[original_idx])
            cancer_label = int(self.cancer_labels[original_idx])
            return tensor, torch.tensor(mutation_label, dtype=torch.long), torch.tensor(cancer_label, dtype=torch.long)
        else:
            return tensor

    def _process_sequence(self, seq):
        """Process individual sequence to fixed length"""
        seq = str(seq)
        if len(seq) < self.seq_length:
            return seq + 'N' * (self.seq_length - len(seq))
        return seq[:self.seq_length]

    def _get_valid_indices(self):
        """Get indices of valid samples"""
        valid_indices = []
        
        for i in range(len(self.genomic_seqs)):
            # Check if sequence exists
            if not self.genomic_seqs[i]:
                continue
                
            # Skip invalid labels if provided
            if self.mutation_labels is not None and (pd.isna(self.mutation_labels[i])):
                continue
                
            if self.cancer_labels is not None and (pd.isna(self.cancer_labels[i])):
                continue
            
            valid_indices.append(i)
            
        return np.array(valid_indices)

    def one_hot_encode(self, sequence):
        """Convert DNA sequence to one-hot encoded matrix"""
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        one_hot = np.zeros((5, self.seq_length), dtype=np.float32)
        
        for i in range(min(len(sequence), self.seq_length)):
            char = sequence[i].upper()
            one_hot[mapping.get(char, 4), i] = 1.0
            
        return one_hot
    
    def one_hot_encode_allele(self, allele):
        """Encode a single nucleotide allele as one-hot vector"""
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, '-': 5}  # Added '-' for deletion
        one_hot = np.zeros((6, 1), dtype=np.float32)  # 6 categories, 1 position
        
        if allele and len(allele) > 0:
            char = allele[0].upper()  # Take first character
            one_hot[mapping.get(char, 4), 0] = 1.0
        else:
            one_hot[5, 0] = 1.0  # Mark as deletion/unknown
            
        return one_hot

class DualTaskClassifier(nn.Module):
    def __init__(self, num_mutation_classes, num_cancer_classes, input_channels=5):
        """
        Neural network for dual classification: mutation type and cancer type
        
        Args:
            num_mutation_classes: Number of mutation classes (Missense, Silent, etc.)
            num_cancer_classes: Number of cancer type classes (LUSC, KIRC, etc.)
            input_channels: Number of input channels (5 for sequence only, 
                           17 if including ref/tumor alleles)
        """
        super(DualTaskClassifier, self).__init__()
        
        self.input_channels = input_channels
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool1d(2)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        # Dropout
        self.dropout = nn.Dropout(0.3)

        # Attention
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
        
        # Task-specific layers
        self.fc_mutation1 = nn.Linear(256, 128)
        self.fc_mutation2 = nn.Linear(128, num_mutation_classes)
        
        self.fc_cancer1 = nn.Linear(256, 128)
        self.fc_cancer2 = nn.Linear(128, num_cancer_classes)

    def forward(self, x):
        batch_size, _, seq_len = x.shape  # dynamic length support

        # Conv block
        x = F.relu(self.bn1(self.conv1(x)))  # -> [batch, 64, seq_len]
        x = self.pool(x)                     # -> [batch, 64, seq_len/2]

        x = F.relu(self.bn2(self.conv2(x)))  # -> [batch, 128, ...]
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))  # -> [batch, 256, ...]
        x = self.pool(x)                     # final shape: [batch, 256, conv_out_len]

        # Attention
        x_attn = x.permute(0, 2, 1)          # [batch, seq_len, channels]
        attn_output, _ = self.attention(x_attn, x_attn, x_attn)
        x = x + attn_output.permute(0, 2, 1) # residual connection

        # Global average pooling
        x = torch.mean(x, dim=2)  # [batch, 256]
        
        # Shared representation
        shared_feat = self.dropout(x)
        
        # Task-specific branches
        # Mutation classification branch
        mutation_feat = F.relu(self.fc_mutation1(shared_feat))
        mutation_feat = self.dropout(mutation_feat)
        mutation_out = self.fc_mutation2(mutation_feat)
        
        # Cancer type classification branch
        cancer_feat = F.relu(self.fc_cancer1(shared_feat))
        cancer_feat = self.dropout(cancer_feat)
        cancer_out = self.fc_cancer2(cancer_feat)
        
        return mutation_out, cancer_out

def train_model(model, train_loader, val_loader, criterion_mutation, criterion_cancer, 
                optimizer, scheduler, num_epochs=20, patience=10):
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # For plotting
    train_losses = []
    val_losses = []
    train_mutation_accs = []
    train_cancer_accs = []
    val_mutation_accs = []
    val_cancer_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        mutation_correct = 0
        cancer_correct = 0
        total = 0
        
        for inputs, mutation_labels, cancer_labels in train_loader:
            inputs = inputs.to(device)
            mutation_labels = mutation_labels.to(device)
            cancer_labels = cancer_labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            mutation_outputs, cancer_outputs = model(inputs)
            
            # Calculate losses
            mutation_loss = criterion_mutation(mutation_outputs, mutation_labels)
            cancer_loss = criterion_cancer(cancer_outputs, cancer_labels)
            
            # Combined loss (equal weight for now)
            loss = mutation_loss + cancer_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, mutation_predicted = torch.max(mutation_outputs, 1)
            _, cancer_predicted = torch.max(cancer_outputs, 1)
            total += mutation_labels.size(0)
            mutation_correct += (mutation_predicted == mutation_labels).sum().item()
            cancer_correct += (cancer_predicted == cancer_labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_mutation_acc = mutation_correct / total
        epoch_train_cancer_acc = cancer_correct / total
        
        train_losses.append(epoch_train_loss)
        train_mutation_accs.append(epoch_train_mutation_acc)
        train_cancer_accs.append(epoch_train_cancer_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        mutation_correct = 0
        cancer_correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, mutation_labels, cancer_labels in val_loader:
                inputs = inputs.to(device)
                mutation_labels = mutation_labels.to(device)
                cancer_labels = cancer_labels.to(device)
                
                mutation_outputs, cancer_outputs = model(inputs)
                
                # Calculate losses
                mutation_loss = criterion_mutation(mutation_outputs, mutation_labels)
                cancer_loss = criterion_cancer(cancer_outputs, cancer_labels)
                
                # Combined loss
                loss = mutation_loss + cancer_loss
                
                val_loss += loss.item() * inputs.size(0)
                _, mutation_predicted = torch.max(mutation_outputs, 1)
                _, cancer_predicted = torch.max(cancer_outputs, 1)
                total += mutation_labels.size(0)
                mutation_correct += (mutation_predicted == mutation_labels).sum().item()
                cancer_correct += (cancer_predicted == cancer_labels).sum().item()
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_mutation_acc = mutation_correct / total
        epoch_val_cancer_acc = cancer_correct / total
        
        val_losses.append(epoch_val_loss)
        val_mutation_accs.append(epoch_val_mutation_acc)
        val_cancer_accs.append(epoch_val_cancer_acc)
        
        # Update learning rate
        scheduler.step(epoch_val_loss)
        
        # Print statistics
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {epoch_train_loss:.4f}, '
              f'Mutation Acc: {epoch_train_mutation_acc:.4f}, '
              f'Cancer Acc: {epoch_train_cancer_acc:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, '
              f'Val Mutation Acc: {epoch_val_mutation_acc:.4f}, '
              f'Val Cancer Acc: {epoch_val_cancer_acc:.4f}')
        
        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Plot training curves
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(2, 2, 2)
    plt.plot(train_mutation_accs, label='Training Mutation Accuracy')
    plt.plot(val_mutation_accs, label='Validation Mutation Accuracy')
    plt.plot(train_cancer_accs, label='Training Cancer Accuracy')
    plt.plot(val_cancer_accs, label='Validation Cancer Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.subplot(2, 2, 3)
    plt.plot(train_mutation_accs, label='Training Mutation Accuracy')
    plt.plot(val_mutation_accs, label='Validation Mutation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Mutation Type Accuracy')
    
    plt.subplot(2, 2, 4)
    plt.plot(train_cancer_accs, label='Training Cancer Accuracy')
    plt.plot(val_cancer_accs, label='Validation Cancer Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Cancer Type Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    return model

def evaluate_model(model, test_loader, mutation_labels_map, cancer_labels_map):
    device = torch.device('cpu')
    model = model.to(device)
    model.eval()
    
    all_mutation_preds = []
    all_mutation_labels = []
    all_cancer_preds = []
    all_cancer_labels = []
    
    with torch.no_grad():
        for inputs, mutation_labels, cancer_labels in test_loader:
            inputs = inputs.to(device)
            mutation_labels = mutation_labels.to(device)
            cancer_labels = cancer_labels.to(device)
            
            mutation_outputs, cancer_outputs = model(inputs)
            _, mutation_preds = torch.max(mutation_outputs, 1)
            _, cancer_preds = torch.max(cancer_outputs, 1)
            
            all_mutation_preds.extend(mutation_preds.cpu().numpy())
            all_mutation_labels.extend(mutation_labels.cpu().numpy())
            all_cancer_preds.extend(cancer_preds.cpu().numpy())
            all_cancer_labels.extend(cancer_labels.cpu().numpy())
    
    # Calculate metrics for mutation classification
    mutation_accuracy = accuracy_score(all_mutation_labels, all_mutation_preds)
    print(f"Mutation Classification Accuracy: {mutation_accuracy:.4f}")
    
    # Classification report for mutation types
    mutation_report = classification_report(all_mutation_labels, all_mutation_preds, 
                                          target_names=[mutation_labels_map[i] for i in sorted(set(all_mutation_labels))])
    print("Mutation Classification Report:")
    print(mutation_report)
    
    # Calculate metrics for cancer type classification
    cancer_accuracy = accuracy_score(all_cancer_labels, all_cancer_preds)
    print(f"Cancer Type Classification Accuracy: {cancer_accuracy:.4f}")
    
    # Classification report for cancer types
    cancer_report = classification_report(all_cancer_labels, all_cancer_preds,
                                         target_names=[cancer_labels_map[i] for i in sorted(set(all_cancer_labels))])
    print("Cancer Type Classification Report:")
    print(cancer_report)
    
    # Confusion matrices
    plt.figure(figsize=(16, 7))
    
    plt.subplot(1, 2, 1)
    mutation_cm = confusion_matrix(all_mutation_labels, all_mutation_preds)
    sns.heatmap(mutation_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[mutation_labels_map[i] for i in sorted(set(all_mutation_labels))],
                yticklabels=[mutation_labels_map[i] for i in sorted(set(all_mutation_labels))])
    plt.xlabel('Predicted Mutation Type')
    plt.ylabel('True Mutation Type')
    plt.title('Mutation Type Confusion Matrix')
    
    plt.subplot(1, 2, 2)
    cancer_cm = confusion_matrix(all_cancer_labels, all_cancer_preds)
    sns.heatmap(cancer_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[cancer_labels_map[i] for i in sorted(set(all_cancer_labels))],
                yticklabels=[cancer_labels_map[i] for i in sorted(set(all_cancer_labels))])
    plt.xlabel('Predicted Cancer Type')
    plt.ylabel('True Cancer Type')
    plt.title('Cancer Type Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.show()
    
    return mutation_accuracy, cancer_accuracy

def main():
    # Load the CSV data
    data = pd.read_csv('mutations/mutations_filtered.csv')
    
    # Check data
    print("Data shape:", data.shape)
    print("Columns:", data.columns.tolist())
    
    # Check for class imbalance
    print("Mutation type distribution:")
    print(data['Variant_Classification'].value_counts())
    
    print("Cancer type distribution:")
    print(data['Cancer_Type'].value_counts())
    
    # Create label encoders
    mutation_encoder = LabelEncoder()
    cancer_encoder = LabelEncoder()
    
    # Encode labels
    mutation_labels = mutation_encoder.fit_transform(data['Variant_Classification'])
    cancer_labels = cancer_encoder.fit_transform(data['Cancer_Type'])
    
    # Create maps for decoding
    mutation_labels_map = {i: label for i, label in enumerate(mutation_encoder.classes_)}
    cancer_labels_map = {i: label for i, label in enumerate(cancer_encoder.classes_)}
    
    print("Mutation classes:", mutation_encoder.classes_)
    print("Cancer classes:", cancer_encoder.classes_)
    
    # Split data into features and target
    genomic_seqs = data['Genomic_Context_Sequence'].values
    ref_alleles = data['Reference_Allele'].values
    tumor_alleles = data['Tumor_Seq_Allele2'].values
    
    # Get number of unique classes
    num_mutation_classes = len(mutation_encoder.classes_)
    num_cancer_classes = len(cancer_encoder.classes_)
    print(f"Number of mutation classes: {num_mutation_classes}")
    print(f"Number of cancer classes: {num_cancer_classes}")
    
    # Implement stratified cross-validation
    # We'll use cancer type for stratification as it's likely the more imbalanced
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_mutation_accs = []
    fold_cancer_accs = []
    fold_num = 1
    
    for train_index, test_index in skf.split(genomic_seqs, cancer_labels):
        print(f"\nFold {fold_num}/{skf.n_splits}")
        
        # Split the data
        X_train_seq = genomic_seqs[train_index]
        X_test_seq = genomic_seqs[test_index]
        
        X_train_ref = ref_alleles[train_index]
        X_test_ref = ref_alleles[test_index]
        
        X_train_tumor = tumor_alleles[train_index]
        X_test_tumor = tumor_alleles[test_index]
        
        y_train_mutation = mutation_labels[train_index]
        y_test_mutation = mutation_labels[test_index]
        
        y_train_cancer = cancer_labels[train_index]
        y_test_cancer = cancer_labels[test_index]
        
        # Further split training data to create a validation set
        (X_train_seq, X_val_seq, 
         X_train_ref, X_val_ref,
         X_train_tumor, X_val_tumor,
         y_train_mutation, y_val_mutation,
         y_train_cancer, y_val_cancer) = train_test_split(
            X_train_seq, X_train_ref, X_train_tumor, 
            y_train_mutation, y_train_cancer,
            test_size=0.2, random_state=42, stratify=y_train_cancer
        )
        
        # Create datasets
        train_dataset = GeneticDataset(
            X_train_seq, X_train_ref, X_train_tumor, 
            y_train_mutation, y_train_cancer
        )
        
        val_dataset = GeneticDataset(
            X_val_seq, X_val_ref, X_val_tumor, 
            y_val_mutation, y_val_cancer
        )
        
        test_dataset = GeneticDataset(
            X_test_seq, X_test_ref, X_test_tumor, 
            y_test_mutation, y_test_cancer
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Initialize model, loss function, and optimizer
        # 5 (sequence) + 6 (ref) + 6 (tumor)
        model = DualTaskClassifier(
            num_mutation_classes=num_mutation_classes,
            num_cancer_classes=num_cancer_classes,
            input_channels=17
        )
        
        criterion_mutation = nn.CrossEntropyLoss()
        criterion_cancer = nn.CrossEntropyLoss()
        
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # Train the model
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion_mutation=criterion_mutation,
            criterion_cancer=criterion_cancer,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=20,
            patience=10
        )
        
        # Evaluate the model
        mutation_acc, cancer_acc = evaluate_model(
            model, test_loader, mutation_labels_map, cancer_labels_map
        )
        
        fold_mutation_accs.append(mutation_acc)
        fold_cancer_accs.append(cancer_acc)
        
        fold_num += 1
    
    # Print cross-validation results
    print("\nCross-Validation Results:")
    for i in range(len(fold_mutation_accs)):
        print(f"Fold {i+1}: Mutation Accuracy = {fold_mutation_accs[i]:.4f}, Cancer Accuracy = {fold_cancer_accs[i]:.4f}")
    
    print(f"Average Mutation Accuracy: {np.mean(fold_mutation_accs):.4f} ± {np.std(fold_mutation_accs):.4f}")
    print(f"Average Cancer Accuracy: {np.mean(fold_cancer_accs):.4f} ± {np.std(fold_cancer_accs):.4f}")
    
    # Train on the full dataset for the final model
    print("\nTraining final model on the full dataset...")
    
    # Split data into train and test
    X_train_seq, X_test_seq, X_train_ref, X_test_ref, X_train_tumor, X_test_tumor, y_train_mutation, y_test_mutation, y_train_cancer, y_test_cancer = train_test_split(
        genomic_seqs, ref_alleles, tumor_alleles, mutation_labels, cancer_labels, 
        test_size=0.2, random_state=42, stratify=cancer_labels
    )
    
    # Create datasets
    train_dataset = GeneticDataset(
        X_train_seq, X_train_ref, X_train_tumor, 
        y_train_mutation, y_train_cancer
    )
    
    test_dataset = GeneticDataset(
        X_test_seq, X_test_ref, X_test_tumor, 
        y_test_mutation, y_test_cancer
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model, loss function, and optimizer
    final_model = DualTaskClassifier(
        num_mutation_classes=num_mutation_classes,
        num_cancer_classes=num_cancer_classes,
        input_channels=17
    )
    
    criterion_mutation = nn.CrossEntropyLoss()
    criterion_cancer = nn.CrossEntropyLoss()
    
    optimizer = Adam(final_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Train the model
    final_model = train_model(
        model=final_model,
        train_loader=train_loader,
        val_loader=test_loader,  # Using test set as validation for final evaluation
        criterion_mutation=criterion_mutation,
        criterion_cancer=criterion_cancer,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=20,
        patience=15
    )
    
    # Save the final model and encoders
    torch.save(final_model.state_dict(), 'cancer_and_mutation_type_classifier.pth')
    
    # Save label encoders
    import pickle
    with open('mutation_encoder.pkl', 'wb') as f:
        pickle.dump(mutation_encoder, f)
        
    with open('cancer_encoder.pkl', 'wb') as f:
        pickle.dump(cancer_encoder, f)
    
    print("Final model and encoders saved.")

main()