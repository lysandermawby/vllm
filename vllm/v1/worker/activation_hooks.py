# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Activation extraction hooks for real-time probe computation."""

import threading
from collections import defaultdict

import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)


class ActivationStore:
    """Thread-safe store for activations extracted during inference.
    
    This class stores activations keyed by (request_id, layer_idx, token_position) and provides methods to retrieve activations for probe computation.
    """
    
    def __init__(self):
        self._store = {}
        self._lock = threading.Lock()
        self._request_layers = defaultdict(set)
        self._request_positions = defaultdict(set)
    
    def store_activation(
        self,
        request_id,
        layer_idx,
        token_position,
        activation: torch.Tensor):
        """Store an activation tensor."""
        with self._lock:
            key = (request_id, layer_idx, token_position)
            # Detach and clone to avoid keeping references to computation graph
            if activation.requires_grad:
                activation = activation.detach()
            self._store[key] = activation.clone()
            self._request_layers[request_id].add(layer_idx)
            self._request_positions[request_id].add(token_position)
    
    def get_activation(
        self,
        request_id,
        layer_idx,
        token_position):
        """Retrieve an activation tensor."""
        with self._lock:
            key = (request_id, layer_idx, token_position)
            return self._store.get(key)
    
    def get_activations_for_request(
        self,
        request_id,
        layer_idx=None):
        """Get all activations for a request, optionally filtered by layer."""
        with self._lock:
            result = {}
            for (req_id, l_idx, pos), activation in self._store.items():
                if req_id == request_id and (layer_idx is None or l_idx == layer_idx):
                    result[(l_idx, pos)] = activation
            return result
    
    def get_layers_for_request(self, request_id):
        """Get all layer indices that have activations for a request."""
        with self._lock:
            return self._request_layers.get(request_id, set()).copy()
    
    def get_positions_for_request(self, request_id):
        """Get all token positions that have activations for a request."""
        with self._lock:
            return self._request_positions.get(request_id, set()).copy()
    
    def clear_request(self, request_id):
        """Clear all activations for a specific request."""
        with self._lock:
            keys_to_remove = [
                key for key in self._store.keys() if key[0] == request_id
            ]
            for key in keys_to_remove:
                del self._store[key]
            if request_id in self._request_layers:
                del self._request_layers[request_id]
            if request_id in self._request_positions:
                del self._request_positions[request_id]
    
    def clear_all(self):
        """Clear all stored activations."""
        with self._lock:
            self._store.clear()
            self._request_layers.clear()
            self._request_positions.clear()
    
    def get_stats(self):
        """Get statistics about stored activations."""
        with self._lock:
            return {
                "total_activations": len(self._store),
                "num_requests": len(self._request_layers),
                "requests": list(self._request_layers.keys()),
            }


class ActivationHookManager:
    """Manages forward hooks for activation extraction."""
    
    def __init__(
        self,
        activation_store,
        extract_layers=None):
        """Initialize the hook manager."""
        self.activation_store = activation_store
        self.extract_layers = set(extract_layers) if extract_layers else None
        self.hooks = []
        self.layer_mapping = {}
        self._current_request_ids = None
        self._current_token_positions = None
    
    def set_request_context(
        self,
        request_ids,
        token_positions=None):
        """Set the current request context for activation extraction."""
        self._current_request_ids = request_ids
        self._current_token_positions = token_positions or {}
    
    def clear_request_context(self):
        """Clear the current request context."""
        self._current_request_ids = None
        self._current_token_positions = None
    
    def _create_hook_fn(self, layer_idx, layer_name):
        """Create a forward hook function for a specific layer."""
        def hook_fn(module, input, output):
            # Skip if no request context
            if self._current_request_ids is None:
                return
            
            # Check if we should extract from this layer
            if self.extract_layers is not None and layer_idx not in self.extract_layers:
                return
            
            # Handle different output formats
            if isinstance(output, tuple):
                # Usually (hidden_states,) or (hidden_states, ...)
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Remove batch dimension if present [1, num_tokens, hidden_size] -> [num_tokens, hidden_size]
            if hidden_states.dim() == 3:
                hidden_states = hidden_states[0]
            
            # Store activations for each request
            # In batched inference, we need to split by request
            # For now, we'll store the full batch and let the caller handle splitting
            # This is a simplified version - in production you'd need proper batching logic
            num_tokens = hidden_states.shape[0]
            
            # If we have multiple requests, we need to map tokens to requests
            # This is simplified - in practice you'd use scheduler_output to map properly
            for i, request_id in enumerate(self._current_request_ids):
                # Get token position for this request
                token_pos = self._current_token_positions.get(request_id, i)
                
                # For now, store the activation at the first token position
                # In a full implementation, you'd iterate through all token positions
                # and map them correctly using scheduler_output
                if num_tokens > 0:
                    # Store the last token's activation (most recent)
                    activation = hidden_states[-1].clone()
                    self.activation_store.store_activation(
                        request_id=request_id,
                        layer_idx=layer_idx,
                        token_position=token_pos,
                        activation=activation,
                    )
        
        return hook_fn
    
    def register_hooks(self, model, layer_names=None):
        """Register forward hooks on transformer layers."""
        self.remove_hooks()
        
        # Find transformer layers
        layers = self._find_transformer_layers(model, layer_names)
        
        # Register hooks
        for layer_idx, (layer_module, layer_name) in enumerate(layers):
            hook_fn = self._create_hook_fn(layer_idx, layer_name)
            handle = layer_module.register_forward_hook(hook_fn)
            self.hooks.append(handle)
            self.layer_mapping[layer_module] = (layer_idx, layer_name)
        
        logger.info(
            f"Registered {len(self.hooks)} activation extraction hooks "
            f"on layers: {[idx for idx, _ in layers]}"
        )
    
    def _find_transformer_layers(
        self,
        model,
        layer_names=None):
        """Find transformer decoder layers in the model."""
        layers = []
        
        # Common patterns for transformer layers
        if layer_names:
            # Use provided layer names
            for name in layer_names:
                module = dict(model.named_modules()).get(name)
                if module is not None:
                    layers.append((module, name))
        else:
            # Auto-detect layers
            # Look for common patterns: model.layers, model.model.layers, etc.
            for name, module in model.named_modules():
                # Check if this looks like a transformer layer
                if self._is_transformer_layer(module, name):
                    layers.append((module, name))
        
        # Sort by name to ensure consistent ordering
        layers.sort(key=lambda x: x[1])
        
        return layers
    
    def _is_transformer_layer(self, module, name):
        """Check if a module is a transformer decoder layer."""
        # Common patterns for layer names
        layer_patterns = [
            "layers.",
            ".layer.",
            "transformer.h.",
            "model.layers.",
        ]
        
        # Check name patterns
        for pattern in layer_patterns:
            if pattern in name and name.endswith(("layer", "block")):
                # Check if it has attention and MLP components
                has_attn = hasattr(module, "self_attn") or hasattr(module, "attention")
                has_mlp = hasattr(module, "mlp") or hasattr(module, "feed_forward")
                if has_attn and has_mlp:
                    return True
        
        return False
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
        self.layer_mapping.clear()
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()

