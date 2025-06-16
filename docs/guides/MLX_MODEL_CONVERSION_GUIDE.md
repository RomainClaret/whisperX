# MLX Model Conversion Guide

A practical guide to converting models to MLX format for Apple Silicon optimization.

## Should You Convert Your Model to MLX?

### Quick Decision Tree
```
Is your model < 100M parameters?
  ├─ No → Don't convert (too complex)
  └─ Yes → Continue ↓
  
Is it primarily CNN/RNN based?
  ├─ No (Transformer) → Maybe (check architecture)
  └─ Yes → Good candidate ↓
  
Do you need < 50ms inference?
  ├─ No → Current solution is fine
  └─ Yes → Consider conversion
```

## Understanding MLX Limitations

### What MLX Supports Well ✅
- Standard layers (Conv, Linear, LSTM, GRU)
- Common activations (ReLU, GELU, SiLU)
- Basic attention mechanisms
- Static computation graphs

### What MLX Struggles With ❌
- Dynamic shapes
- Custom attention variants
- Complex positional encodings
- Specialized operations (e.g., group convolutions)

## Step-by-Step Conversion Process

### 1. Analyze Your Model

```python
import torch
from transformers import AutoModel

# Load your model
model = AutoModel.from_pretrained("your-model")

# Analyze architecture
def analyze_model(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Check layer types
    layer_types = set()
    for name, module in model.named_modules():
        layer_types.add(type(module).__name__)
    
    print(f"Layer types: {layer_types}")
    
    # Check for problematic layers
    problematic = ['DynamicConv', 'MultiheadAttention', 'RelativePositionBias']
    issues = [l for l in layer_types if any(p in l for p in problematic)]
    
    if issues:
        print(f"⚠️ Problematic layers found: {issues}")
    else:
        print("✅ Architecture looks compatible!")

analyze_model(model)
```

### 2. Create MLX Architecture

```python
import mlx.core as mx
import mlx.nn as nn

class ConvertedModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Example: Converting a simple CNN
        self.conv1 = nn.Conv1d(
            in_channels=config.input_channels,
            out_channels=config.conv1_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        
        # For transformer layers
        if hasattr(config, 'num_attention_heads'):
            self.attention = nn.MultiHeadAttention(
                dims=config.hidden_size,
                num_heads=config.num_attention_heads,
                bias=True
            )
    
    def __call__(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
```

### 3. Convert Weights

```python
import numpy as np

def convert_weights(pytorch_model, mlx_model):
    """Convert PyTorch weights to MLX format"""
    
    # Get PyTorch state dict
    pt_state = pytorch_model.state_dict()
    
    # Create mapping between PyTorch and MLX parameter names
    # This often requires manual mapping due to naming differences
    param_mapping = {
        'conv1.weight': 'conv1.weight',
        'conv1.bias': 'conv1.bias',
        # Add more mappings as needed
    }
    
    # Convert each parameter
    mlx_state = {}
    for pt_name, mlx_name in param_mapping.items():
        if pt_name in pt_state:
            # Convert PyTorch tensor to numpy, then to MLX
            weight = pt_state[pt_name].detach().cpu().numpy()
            
            # MLX uses different weight format for some layers
            if 'Conv' in mlx_name and len(weight.shape) == 3:
                # PyTorch: [out_channels, in_channels, kernel_size]
                # MLX: [kernel_size, in_channels, out_channels]
                weight = weight.transpose(2, 1, 0)
            
            mlx_state[mlx_name] = mx.array(weight)
    
    # Load weights into MLX model
    mlx_model.load_weights(mlx_state)
    return mlx_model
```

### 4. Validate Conversion

```python
def validate_conversion(pt_model, mlx_model, sample_input):
    """Ensure MLX model produces same output as PyTorch"""
    
    # PyTorch inference
    pt_model.eval()
    with torch.no_grad():
        pt_output = pt_model(torch.tensor(sample_input))
        if isinstance(pt_output, tuple):
            pt_output = pt_output[0]
        pt_output = pt_output.numpy()
    
    # MLX inference
    mlx_input = mx.array(sample_input)
    mlx_output = mlx_model(mlx_input)
    mlx_output = np.array(mlx_output)
    
    # Compare outputs
    max_diff = np.max(np.abs(pt_output - mlx_output))
    mean_diff = np.mean(np.abs(pt_output - mlx_output))
    
    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    
    if max_diff < 1e-3:
        print("✅ Conversion successful!")
        return True
    else:
        print("❌ Conversion has errors")
        return False
```

## Real Example: Converting a Simple VAD Model

```python
# 1. Define original architecture (PyTorch)
class SimpleVAD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, 32, 5, padding=2)
        self.conv2 = torch.nn.Conv1d(32, 64, 5, padding=2)
        self.pool = torch.nn.MaxPool1d(2)
        self.fc = torch.nn.Linear(64 * 50, 2)  # Assuming input length 100
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 2. Create MLX version
class SimpleVAD_MLX(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.fc = nn.Linear(64 * 50, 2)
        
    def __call__(self, x):
        x = nn.relu(self.conv1(x))
        # MLX doesn't have MaxPool1d yet, use strided conv
        x = x[:, :, ::2]  # Simple downsampling
        x = nn.relu(self.conv2(x))
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

# 3. Convert weights
def convert_vad_model(pytorch_model_path):
    # Load PyTorch model
    pt_model = SimpleVAD()
    pt_model.load_state_dict(torch.load(pytorch_model_path))
    
    # Create MLX model
    mlx_model = SimpleVAD_MLX()
    
    # Manual weight conversion
    pt_state = pt_model.state_dict()
    
    # Convert conv1
    mlx_model.conv1.weight = mx.array(
        pt_state['conv1.weight'].numpy().transpose(2, 1, 0)
    )
    mlx_model.conv1.bias = mx.array(pt_state['conv1.bias'].numpy())
    
    # Convert conv2
    mlx_model.conv2.weight = mx.array(
        pt_state['conv2.weight'].numpy().transpose(2, 1, 0)
    )
    mlx_model.conv2.bias = mx.array(pt_state['conv2.bias'].numpy())
    
    # Convert fc
    mlx_model.fc.weight = mx.array(pt_state['fc.weight'].numpy().T)
    mlx_model.fc.bias = mx.array(pt_state['fc.bias'].numpy())
    
    return mlx_model

# 4. Save MLX model
def save_mlx_model(model, path):
    weights = model.parameters()
    mx.save(path, weights)
```

## Tools and Utilities

### 1. Automatic Conversion Script (Experimental)

```python
def auto_convert_simple_model(pt_model, input_shape):
    """Attempt automatic conversion for simple models"""
    
    # Trace the model
    example_input = torch.randn(input_shape)
    traced = torch.jit.trace(pt_model, example_input)
    
    # Extract graph
    graph = traced.graph
    
    # Parse and convert (simplified)
    mlx_layers = []
    for node in graph.nodes():
        if node.kind() == 'aten::conv1d':
            # Extract conv parameters
            # Create equivalent MLX layer
            pass
        elif node.kind() == 'aten::linear':
            # Convert linear layer
            pass
    
    # Note: This is highly simplified
    # Real implementation would be much more complex
```

### 2. Debugging Conversion Issues

```python
def debug_layer_outputs(pt_model, mlx_model, input_data):
    """Compare layer-by-layer outputs"""
    
    # Hook to capture PyTorch activations
    pt_activations = {}
    def get_activation(name):
        def hook(model, input, output):
            pt_activations[name] = output.detach().numpy()
        return hook
    
    # Register hooks
    for name, layer in pt_model.named_modules():
        layer.register_forward_hook(get_activation(name))
    
    # Run PyTorch model
    pt_output = pt_model(torch.tensor(input_data))
    
    # Manually get MLX activations
    mlx_activations = {}
    x = mx.array(input_data)
    
    # Run through MLX model step by step
    x = mlx_model.conv1(x)
    mlx_activations['conv1'] = np.array(x)
    
    # Compare activations
    for layer_name in pt_activations:
        if layer_name in mlx_activations:
            diff = np.mean(np.abs(
                pt_activations[layer_name] - mlx_activations[layer_name]
            ))
            print(f"{layer_name}: diff = {diff}")
```

## When NOT to Convert

### 1. Complex Architectures
```python
# These are too complex for manual conversion
models_to_avoid = [
    "transformers/wav2vec2",  # Custom attention, relative pos encoding
    "pyannote/segmentation",  # Multiple models, complex pipeline
    "openai/whisper",  # Already has MLX version
    "meta-llama/*",  # Too large, complex attention
]
```

### 2. Small Performance Gains
```python
# If your model already runs fast, don't bother
if inference_time < 0.05:  # 50ms
    print("Already fast enough on CPU")
```

### 3. Rapidly Evolving Models
```python
# If the model is frequently updated
# Maintaining MLX version becomes a burden
```

## Alternative Approaches

### 1. Use CoreML Instead
```bash
pip install coremltools

# Convert to CoreML (often easier than MLX)
python -m coremltools.converters.pytorch your_model.pt
```

### 2. Optimize PyTorch Model
```python
# Sometimes optimization is better than conversion
model = torch.jit.script(model)  # TorchScript
model = torch.quantization.quantize_dynamic(model)  # Quantization
```

### 3. Wait for Official Support
- Check MLX examples repo regularly
- Many models will get official implementations
- Community contributions are growing

## Conclusion

Converting models to MLX can provide significant speedups on Apple Silicon, but:

1. **It's not always necessary** - MLX already handles the heavy lifting (Whisper)
2. **It's not always possible** - Complex models don't map well
3. **It's not always worth it** - Time investment vs. performance gain

Focus on converting simple models that are bottlenecks in your pipeline. For WhisperX, the transcription is already MLX-optimized, so further conversions have diminishing returns.

Remember: **Perfect optimization is the enemy of good enough performance.**