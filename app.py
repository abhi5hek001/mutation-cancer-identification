import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import matplotlib.pyplot as plt
import traceback

# Set page configuration
st.set_page_config(
    page_title="Gene Mutation & Cancer Classifier",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4361ee;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff4b4b;
        margin-bottom: 1rem;
        text-align: center;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 20px;
        background-color: #f8f9fa;
    }
    .dual-prediction {
        display: flex;
        flex-direction: column;
        gap: 20px;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        color: #6c757d;
        font-size: 0.8rem;
    }
    .tab-content {
        padding: 20px 0;
    }
    .input-section {
        margin-bottom: 30px;
    }
    .results-section {
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Load model architecture
class DualTaskClassifier(nn.Module):
    def __init__(self, num_mutation_classes, num_cancer_classes=10, input_channels=5):
        """
        Neural network for dual-task classification: mutation type and cancer type
        
        Args:
            num_mutation_classes: Number of mutation classes (Missense, Silent, etc.)
            num_cancer_classes: Number of cancer type classes
            input_channels: Number of input channels
        """
        super(DualTaskClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.has_cancer_classifier = num_cancer_classes > 0
        
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
        
        if self.has_cancer_classifier:
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
        
        # Mutation classification branch
        mutation_feat = F.relu(self.fc_mutation1(shared_feat))
        mutation_feat = self.dropout(mutation_feat)
        mutation_out = self.fc_mutation2(mutation_feat)

        # Cancer classification branch (if enabled)
        if self.has_cancer_classifier:
            cancer_feat = F.relu(self.fc_cancer1(shared_feat))
            cancer_feat = self.dropout(cancer_feat)
            cancer_out = self.fc_cancer2(cancer_feat)
            return mutation_out, cancer_out
        else:
            return mutation_out

# Utility functions for sequence processing
def process_sequence(seq, seq_length=101):
    """Process individual sequence to fixed length"""
    seq = str(seq)
    if len(seq) < seq_length:
        return seq + 'N' * (seq_length - len(seq))
    return seq[:seq_length]

def one_hot_encode(sequence, seq_length=101):
    """Convert DNA sequence to one-hot encoded matrix"""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    one_hot = np.zeros((5, seq_length), dtype=np.float32)
    
    for i in range(min(len(sequence), seq_length)):
        char = sequence[i].upper()
        one_hot[mapping.get(char, 4), i] = 1.0
        
    return one_hot

def one_hot_encode_allele(allele):
    """Encode a single nucleotide allele as one-hot vector"""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, '-': 5}  # Added '-' for deletion
    one_hot = np.zeros((6, 1), dtype=np.float32)  # 6 categories, 1 position
    
    if allele and len(allele) > 0:
        char = allele[0].upper()  # Take first character
        one_hot[mapping.get(char, 4), 0] = 1.0
    else:
        one_hot[5, 0] = 1.0  # Mark as deletion/unknown
        
    return one_hot

def prepare_input(genomic_seq, ref_allele=None, tumor_allele=None, seq_length=101):
    """Prepare input tensor for model prediction"""
    processed_seq = process_sequence(genomic_seq, seq_length)
    seq_one_hot = one_hot_encode(processed_seq, seq_length)
    
    features = [seq_one_hot]
    
    if ref_allele is not None and tumor_allele is not None:
        ref_one_hot = one_hot_encode_allele(ref_allele)
        tumor_one_hot = one_hot_encode_allele(tumor_allele)
        
        # Expand allele encodings to match sequence length
        ref_one_hot = np.repeat(ref_one_hot, seq_length, axis=1)  # Shape (6, 101)
        tumor_one_hot = np.repeat(tumor_one_hot, seq_length, axis=1)  # Shape (6, 101)
        
        features.append(ref_one_hot)
        features.append(tumor_one_hot)
    
    # Combine all features
    combined = np.concatenate(features, axis=0)  # Shape (5, 101) or (17, 101)
    tensor = torch.tensor(combined, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    return tensor

# Function to safely convert tensors to numpy arrays
def safe_tensor_to_numpy(tensor):
    """Safely convert PyTorch tensor to numpy array"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

# Load model and encoders
@st.cache_resource
def load_model_and_encoders():
    try:
        # Load model
        model = DualTaskClassifier(
            num_mutation_classes=3,   # We're focusing on 3 classes: Missense, Nonsense, Silent
            num_cancer_classes=3,    # Now using cancer classification with 10 cancer types
            input_channels=17         # 5 (sequence) + 6 (ref) + 6 (tumor)
        )
        
        try:
            # Load pre-trained weights
            model.load_state_dict(torch.load('cancer_and_mutation_type_classifier.pth', map_location=torch.device('cpu')))
        except FileNotFoundError:
            st.warning("Pre-trained model weights not found. Using a demonstration mode with random weights.")
            # If file not found, just use the random initialized weights for demonstration
            pass
            
        model.eval()
        
        # Create default mutation labels map
        default_mutation_labels = {0: "Missense", 1: "Nonsense", 2: "Silent"}
        mutation_labels_map = default_mutation_labels
        
        # Create default cancer labels map
        default_cancer_labels = {
            0: "Breast Cancer", 
            1: "Lung Cancer", 
            2: "Colorectal Cancer", 
        }
        cancer_labels_map = default_cancer_labels
        
        # Try to load mutation type encoder
        try:
            with open('mutation_encoder.pkl', 'rb') as f:
                mutation_encoder = pickle.load(f)
            
            # Create mapping from predicted indices to class names
            mutation_classes = mutation_encoder.classes_
            
            # Filter to just keep our three classes of interest
            target_classes = ["Missense", "Nonsense", "Silent"]
            temp_mutation_map = {}
            
            for i, class_name in enumerate(mutation_classes):
                if class_name in target_classes:
                    # Find position in our 3-class subset (0, 1, or 2)
                    target_idx = target_classes.index(class_name)
                    temp_mutation_map[target_idx] = class_name
            
            # Only update if we found all target classes
            if len(temp_mutation_map) == 3:
                mutation_labels_map = temp_mutation_map
        except FileNotFoundError:
            st.warning("Mutation encoder not found. Using default mutation labels.")
        
        # Try to load cancer type encoder
        try:
            with open('cancer_encoder.pkl', 'rb') as f:
                cancer_encoder = pickle.load(f)
            cancer_classes = cancer_encoder.classes_
            cancer_labels_map = {i: class_name for i, class_name in enumerate(cancer_classes)}
        except FileNotFoundError:
            st.warning("Cancer encoder not found. Using default cancer type labels.")
        
        return model, mutation_labels_map, cancer_labels_map
    except Exception as e:
        st.error(f"Error loading model or encoders: {str(e)}")
        st.error(traceback.format_exc())
        return None, None, None

# Main function
def main():
    st.markdown('<div class="main-header">Gene Mutation & Cancer Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict mutation type and associated cancer type from genomic data</div>', unsafe_allow_html=True)
    
    # Input section - full width
    st.subheader("Input Data")
    
    # Input fields
    genomic_seq = st.text_area("Genomic Context Sequence", 
                              placeholder="Enter DNA sequence (e.g., ATGCGTACGTAGCTAGCTAGCT...)",
                              height=150)
    
    col_a, col_b = st.columns(2)
    with col_a:
        ref_allele = st.text_input("Reference Allele", placeholder="e.g., A")
    with col_b:
        tumor_allele = st.text_input("Tumor Allele", placeholder="e.g., G")
    
    # Validate input
    if st.button("Predict Mutation & Cancer Type", type="primary"):
        if not genomic_seq:
            st.error("Please enter a genomic sequence")
        elif not ref_allele or not tumor_allele:
            st.error("Please enter both reference and tumor alleles")
        else:
            try:
                # Load model and encoders
                model, mutation_labels_map, cancer_labels_map = load_model_and_encoders()
                
                if model and mutation_labels_map and cancer_labels_map:
                    # Prepare input
                    input_tensor = prepare_input(genomic_seq, ref_allele, tumor_allele)
                    
                    # Get prediction
                    with torch.no_grad():
                        try:
                            output = model(input_tensor)
                            
                            # Handle both mutation and cancer predictions
                            if isinstance(output, tuple) and len(output) == 2:
                                mutation_output, cancer_output = output
                                
                                # Mutation prediction - make sure to convert PyTorch tensors to numpy safely
                                filtered_mutation_output = safe_tensor_to_numpy(mutation_output[0, :3])  # First 3 indices
                                mutation_probs = safe_tensor_to_numpy(F.softmax(torch.tensor(filtered_mutation_output), dim=0))
                                mutation_pred_class = int(np.argmax(mutation_probs))
                                mutation_prediction = mutation_labels_map.get(mutation_pred_class, f"Class {mutation_pred_class}")
                                mutation_confidence = float(mutation_probs[mutation_pred_class] * 100)
                                
                                # Cancer prediction
                                cancer_output_np = safe_tensor_to_numpy(cancer_output[0])
                                cancer_probs = safe_tensor_to_numpy(F.softmax(torch.tensor(cancer_output_np), dim=0))
                                cancer_pred_class = int(np.argmax(cancer_probs))
                                cancer_prediction = cancer_labels_map.get(cancer_pred_class, f"Class {cancer_pred_class}")
                                cancer_confidence = float(cancer_probs[cancer_pred_class] * 100)
                                
                                # Store results for display
                                st.session_state.has_cancer_prediction = True
                            else:
                                # Only mutation prediction available
                                filtered_output = safe_tensor_to_numpy(output[0, :3])  # First 3 indices
                                mutation_probs = safe_tensor_to_numpy(F.softmax(torch.tensor(filtered_output), dim=0))
                                mutation_pred_class = int(np.argmax(mutation_probs))
                                mutation_prediction = mutation_labels_map.get(mutation_pred_class, f"Class {mutation_pred_class}")
                                mutation_confidence = float(mutation_probs[mutation_pred_class] * 100)
                                
                                st.session_state.has_cancer_prediction = False
                                
                                # Initialize empty cancer variables to avoid errors
                                cancer_probs = np.zeros(len(cancer_labels_map))
                                cancer_prediction = "N/A"
                                cancer_confidence = 0.0
                            
                            # Store mutation results
                            st.session_state.mutation_prediction = mutation_prediction
                            st.session_state.mutation_confidence = mutation_confidence
                            st.session_state.mutation_probs = mutation_probs
                            st.session_state.mutation_labels = mutation_labels_map
                            
                            # Store cancer results if available
                            if st.session_state.has_cancer_prediction:
                                st.session_state.cancer_prediction = cancer_prediction
                                st.session_state.cancer_confidence = cancer_confidence
                                st.session_state.cancer_probs = cancer_probs
                                st.session_state.cancer_labels = cancer_labels_map
                                
                        except Exception as e:
                            st.error(f"Error during model prediction: {str(e)}")
                            st.error(traceback.format_exc())
                else:
                    st.error("Failed to load model or encoders.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error(traceback.format_exc())
    
    # Prediction Results - now below the submit button
    if 'mutation_prediction' in st.session_state:
        st.subheader("Prediction Results")
        
        # Create tabs for mutation and cancer predictions
        tabs = st.tabs(["Mutation Type", "Cancer Type"])
        
        # Mutation Type Tab
        with tabs[0]:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown(f"### Predicted Mutation Type: <span style='color:#4361ee'>{st.session_state.mutation_prediction}</span>", unsafe_allow_html=True)
                st.markdown(f"**Confidence**: {st.session_state.mutation_confidence:.2f}%")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show detailed breakdown
                st.subheader("Detailed Confidence Scores:")
                for class_id, class_name in st.session_state.mutation_labels.items():
                    if class_id < len(st.session_state.mutation_probs):
                        st.write(f"{class_name}: {st.session_state.mutation_probs[class_id]*100:.2f}%")
                
            with col2:
                # Ensure mutation_labels and mutation_probs are compatible for plotting
                mutation_labels = list(st.session_state.mutation_labels.values())
                mutation_probs = st.session_state.mutation_probs
                
                if len(mutation_probs) == len(mutation_labels):
                    # Plot mutation prediction probabilities
                    fig, ax = plt.subplots(figsize=(8, 4))
                    bars = ax.bar(
                        mutation_labels, 
                        mutation_probs, 
                        color=['skyblue' for _ in range(len(mutation_labels))]
                    )
                    
                    # Highlight the predicted class
                    predicted_idx = int(np.argmax(mutation_probs))
                    if 0 <= predicted_idx < len(bars):
                        bars[predicted_idx].set_color('salmon')
                    
                    ax.set_title('Mutation Type Prediction Confidence')
                    ax.set_xlabel('Mutation Class')
                    ax.set_ylabel('Confidence')
                    plt.xticks(rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.warning(f"Label/probability size mismatch: {len(mutation_labels)} labels vs {len(mutation_probs)} probabilities")
        
        # Cancer Type Tab
        with tabs[1]:
            if st.session_state.has_cancer_prediction:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown(f"### Predicted Cancer Type: <span style='color:#4361ee'>{st.session_state.cancer_prediction}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Confidence**: {st.session_state.cancer_confidence:.2f}%")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Get cancer probabilities and labels safely
                    cancer_probs = st.session_state.cancer_probs
                    cancer_labels_map = st.session_state.cancer_labels
                    
                    # Show detailed breakdown - top predictions
                    num_to_show = min(5, len(cancer_probs))
                    top_indices = np.argsort(cancer_probs)[-num_to_show:][::-1]
                    top_indices = [idx for idx in top_indices if idx < len(cancer_probs)]
                    
                    st.subheader(f"Top {len(top_indices)} Cancer Types:")
                    for idx in top_indices:
                        class_name = cancer_labels_map.get(idx, f"Class {idx}")
                        st.write(f"{class_name}: {cancer_probs[idx]*100:.2f}%")
                
                with col2:
                    # Plot top 5 cancer prediction probabilities (or all if less than 5)
                    num_to_show = min(5, len(cancer_probs))
                    top_indices = np.argsort(cancer_probs)[-num_to_show:][::-1]
                    
                    # Safety check for indices
                    top_indices = [idx for idx in top_indices if idx < len(cancer_probs)]
                    
                    if len(top_indices) > 0:
                        top_probs = cancer_probs[top_indices]
                        top_labels = [cancer_labels_map.get(i, f"Class {i}") for i in top_indices]
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        bars = ax.bar(
                            top_labels, 
                            top_probs, 
                            color=['skyblue' for _ in range(len(top_labels))]
                        )
                        
                        # Highlight the predicted class
                        pred_class = int(np.argmax(cancer_probs))
                        if pred_class in top_indices:
                            pred_idx = list(top_indices).index(pred_class)
                            if 0 <= pred_idx < len(bars):
                                bars[pred_idx].set_color('salmon')
                        
                        ax.set_title(f'Top {num_to_show} Cancer Type Predictions')
                        ax.set_xlabel('Cancer Type')
                        ax.set_ylabel('Confidence')
                        plt.xticks(rotation=30, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("No valid cancer prediction probabilities available.")
            else:
                st.info("Cancer type prediction is not available with the current model configuration.")
    else:
        st.info("Enter a genomic sequence and alleles, then click 'Predict Mutation & Cancer Type' to see results.")
            
    # Example section
    with st.expander("See Example Input"):
        st.code("""
Genomic Context Sequence: ACTGGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC
Reference Allele: A
Tumor Allele: G
        """)
    
    # Info about mutation and cancer types
    with st.expander("About Mutation & Cancer Types"):
        tabs = st.tabs(["Mutation Types", "Cancer Types"])
        
        with tabs[0]:
            st.markdown("""
            - **Missense Mutation**: A mutation that changes a single nucleotide, resulting in a codon that codes for a different amino acid.
            - **Nonsense Mutation**: A mutation that changes a codon to a stop codon, resulting in a truncated protein.
            - **Silent Mutation**: A mutation that changes a codon but still codes for the same amino acid, resulting in no change to the protein.
            """)
            
        with tabs[1]:
            st.markdown("""
            Cancer types that might be predicted include:
            
            - **BLCA**: Bladder Urothelial Carcinoma
            - **LUSC**: Lung Squamous Cell Carcinoma
            - **KIRC**: Kidney Renal Clear Cell Carcinoma
            
            > Note: The specific cancer types predicted may vary based on the model's training data.
            """)
    
    # Demo mode notification
    if 'warning' in st.session_state.keys():
        st.warning("⚠️ Running in demonstration mode with randomly initialized weights. Predictions will not be accurate.")
    
    # Footer
    st.markdown('<div class="footer">Gene Mutation & Cancer Type Classifier | Created with PyTorch and Streamlit</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()