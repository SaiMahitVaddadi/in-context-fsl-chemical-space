"""
Model loader for chemistry language models (LLMs).

Supports loading ChemBERTa2, ChemBERTa3, MolFormer, and IBM models
from HuggingFace transformers library.
"""

from typing import Optional, Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM


class ModelLoader:
    """Load and manage chemistry language models."""
    
    # Supported models with their HuggingFace identifiers
    SUPPORTED_MODELS = {
        'chemberta2': 'DeepChem/ChemBERTa-77M-MLM',
        'chemberta3': 'ibm/chemberta-base-v3',
        'molformer': 'ibm/MolFormer-XL-both-10pct',
        'molformer-large': 'ibm/MolFormer-Large-10pct',
        'chemberta-mlm-only': 'seyonec/ChemBERTa-zinc-base-v1',
        'chemberta-mtr': 'seyonec/ChemBERTa_zinc250k_v2_40k',
    }
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        use_mlm: bool = False
    ):
        """
        Initialize model loader.
        
        Args:
            model_name: Name of the model to load
            device: Device to load model on ('cuda', 'cpu', or None for auto)
            use_mlm: Whether to load as masked language model
        """
        self.model_name = model_name.lower()
        self.device = self._get_device(device)
        self.use_mlm = use_mlm
        
        # Get HuggingFace model identifier
        self.model_id = self._get_model_id()
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
    
    def _get_device(self, device: Optional[str]) -> str:
        """
        Determine device to use.
        
        Args:
            device: User-specified device or None for auto-detection
            
        Returns:
            Device string ('cuda' or 'cpu')
        """
        if device is not None:
            return device
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def _get_model_id(self) -> str:
        """
        Get HuggingFace model identifier.
        
        Returns:
            HuggingFace model ID
            
        Raises:
            ValueError: If model is not supported
        """
        if self.model_name in self.SUPPORTED_MODELS:
            return self.SUPPORTED_MODELS[self.model_name]
        else:
            # Allow custom model IDs
            print(f"Warning: '{self.model_name}' not in predefined models. Using as custom HuggingFace ID.")
            return self.model_name
    
    def load(self):
        """Load model and tokenizer from HuggingFace."""
        print(f"Loading model: {self.model_id}")
        print(f"Device: {self.device}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Load model
            if self.use_mlm:
                self.model = AutoModelForMaskedLM.from_pretrained(self.model_id)
            else:
                self.model = AutoModel.from_pretrained(self.model_id)
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"Successfully loaded {self.model_id}")
            
        except Exception as e:
            raise RuntimeError(f"Error loading model {self.model_id}: {e}")
    
    def encode(
        self,
        smiles: str,
        return_tensors: bool = True,
        max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Encode SMILES string to model input.
        
        Args:
            smiles: SMILES string
            return_tensors: Whether to return PyTorch tensors
            max_length: Maximum sequence length
            
        Returns:
            Dictionary of model inputs
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        
        encoding_kwargs = {
            'padding': True,
            'truncation': True,
            'return_tensors': 'pt' if return_tensors else None
        }
        
        if max_length is not None:
            encoding_kwargs['max_length'] = max_length
        
        inputs = self.tokenizer(smiles, **encoding_kwargs)
        
        if return_tensors:
            # Move tensors to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def get_embeddings(
        self,
        smiles: str,
        pooling: str = 'mean',
        max_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Get embeddings for SMILES string.
        
        Args:
            smiles: SMILES string
            pooling: Pooling method ('mean', 'max', 'cls')
            max_length: Maximum sequence length
            
        Returns:
            Embedding tensor
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        inputs = self.encode(smiles, max_length=max_length)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get hidden states
        hidden_states = outputs.last_hidden_state
        
        # Apply pooling
        if pooling == 'mean':
            # Mean pooling over sequence length
            embeddings = hidden_states.mean(dim=1)
        elif pooling == 'max':
            # Max pooling over sequence length
            embeddings = hidden_states.max(dim=1).values
        elif pooling == 'cls':
            # Use [CLS] token (first token)
            embeddings = hidden_states[:, 0, :]
        else:
            raise ValueError(f"Unsupported pooling method: {pooling}")
        
        return embeddings.squeeze()
    
    def batch_encode(
        self,
        smiles_list: List[str],
        return_tensors: bool = True,
        max_length: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Encode batch of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            return_tensors: Whether to return PyTorch tensors
            max_length: Maximum sequence length
            
        Returns:
            Dictionary of model inputs
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        
        encoding_kwargs = {
            'padding': True,
            'truncation': True,
            'return_tensors': 'pt' if return_tensors else None
        }
        
        if max_length is not None:
            encoding_kwargs['max_length'] = max_length
        
        inputs = self.tokenizer(smiles_list, **encoding_kwargs)
        
        if return_tensors:
            # Move tensors to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def batch_get_embeddings(
        self,
        smiles_list: List[str],
        pooling: str = 'mean',
        max_length: Optional[int] = None,
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Get embeddings for batch of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            pooling: Pooling method ('mean', 'max', 'cls')
            max_length: Maximum sequence length
            batch_size: Batch size for processing
            
        Returns:
            Stacked embedding tensors
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            inputs = self.batch_encode(batch, max_length=max_length)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            hidden_states = outputs.last_hidden_state
            
            # Apply pooling
            if pooling == 'mean':
                embeddings = hidden_states.mean(dim=1)
            elif pooling == 'max':
                embeddings = hidden_states.max(dim=1).values
            elif pooling == 'cls':
                embeddings = hidden_states[:, 0, :]
            else:
                raise ValueError(f"Unsupported pooling method: {pooling}")
            
            all_embeddings.append(embeddings)
        
        return torch.cat(all_embeddings, dim=0)
    
    @staticmethod
    def list_supported_models() -> Dict[str, str]:
        """
        Get list of supported models.
        
        Returns:
            Dictionary mapping model names to HuggingFace IDs
        """
        return ModelLoader.SUPPORTED_MODELS.copy()
