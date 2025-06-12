#!/usr/bin/env python
# coding: utf-8


# In[2]:
#### Import Libraries ####
##########################

import json
from sae_lens import SAE, HookedSAETransformer
from functools import partial
import einops
import os
from huggingface_hub import snapshot_download
import gc
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer
from transformer_lens.hook_points import (
    HookPoint,
)
import numpy as np
import pandas as pd
from pprint import pprint as pp
from typing import Tuple
from torch import Tensor
from functools import lru_cache
from typing import TypedDict, Optional, Tuple, Union
from tqdm import tqdm
import random
from beta_copula_mask import SAECircuitMasker
from collections import defaultdict
import mem

# In[3]:
#### Load Model ####
####################

# Prompt user for Hugging Face token (optional)
# token = input("Enter your Hugging Face token (or leave blank for anonymous access): ").strip()
# if token:
#     os.environ["HF_TOKEN"] = token
# else:
#     print("No token provided. Proceeding with anonymous access.")

# These are in ~/.profile now, which gets loaded on OS startup, not Cursor startup. So can switch 
# to this after next reboot.
#token = os.environ.get("HUGGING_FACE_TOKEN", None)
token = os.environ.get("HF_TOKEN", None)
if token is None or token == "":
    raise ValueError("HUGGING_FACE_TOKEN environment variable is not set")

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Set Hugging Face cache directory (default to /content for Colab)
#hf_cache = input("Enter cache directory for Hugging Face (default: /content/hf_cache): ").strip()
#if not hf_cache:
#    hf_cache = "/content/hf_cache"
hf_cache = "hf_cache"
os.makedirs(hf_cache, exist_ok=True)
os.environ["HF_HOME"] = hf_cache
print(f"Using Hugging Face cache directory: {hf_cache}")

# Load the model
#model_name = "google/gemma-2-9b"  # Replace with the desired model name
model_name = "gpt2-small"  # Replace with the desired model name
print(f"Loading model: {model_name}...")
model = HookedSAETransformer.from_pretrained(model_name, device=device, cache_dir=hf_cache, torch_dtype=torch.bfloat16)

# Set pad_token_id and freeze model parameters
pad_token_id = model.tokenizer.pad_token_id
for param in model.parameters():
    param.requires_grad_(False)

print("Model loaded and parameters frozen.")


# # Helper functions and classes

# In[4]:
#### Memory Functions ####
##########################

def clear_memory():
    for sae in saes:
        for param in sae.parameters():
            param.grad = None
        for param in sae.mask.parameters():
            param.grad = None
    for param in model.parameters():
        param.grad = None
    cleanup_cuda()


# Explanation for the memory free calculation
# The memory free calculation is given as:
# torch.cuda.memory_reserved(i) - torch.cuda.memory_allocated(i) / 1024 ** 3
# This is incorrect because the division by 1024 ** 3 is only applied to torch.cuda.memory_allocated(i).
# To fix this, we need to ensure that both memory_reserved and memory_allocated are divided by 1024 ** 3 before subtracting.

def check_cuda_memory():
    """
    Function to check CUDA memory of different sorts.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")
        print(f"  Max Memory Allocated: {torch.cuda.max_memory_allocated(i) / 1024 ** 3:.2f} GB")
        print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB")
        print(f"  Max Memory Cached: {torch.cuda.max_memory_reserved(i) / 1024 ** 3:.2f} GB")
        # Corrected memory free calculation
        memory_reserved_gb = torch.cuda.memory_reserved(i) / 1024 ** 3
        memory_allocated_gb = torch.cuda.memory_allocated(i) / 1024 ** 3
        print(f"  Memory Free: {memory_reserved_gb - memory_allocated_gb:.2f} GB")
        print(f"  Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.2f} GB")
        print()

# Example usage
check_cuda_memory()

# In[4]:


# class IGMask(nn.Module):
#     # igscores is seq x num_sae_latents
#     def __init__(self, ig_scores):
#         super().__init__()
#         self.ig_scores = ig_scores

#     def forward(self, x, threshold, mean_ablation = None):
#         censored_activations = torch.ones_like(x)
#         if mean_ablation != None:
#             censored_activations = censored_activations * mean_ablation
#         else:
#             censored_activations = censored_activations * 0

#         mask = (self.ig_scores.abs() > threshold).float()

#         diff_to_x = x - censored_activations
#         return censored_activations + diff_to_x * mask

#     def get_threshold_info(self, threshold):
#         mask = (self.ig_scores.abs() > threshold).float()

#         total_latents = mask.sum()
#         avg_latents_per_tok = mask.sum()/mask.shape[0]
#         latents_per_tok = mask.sum(dim=-1)
#         return {"total_latents":total_latents,
#                 "avg_latents_per_tok":avg_latents_per_tok,
#                 "latents_per_tok":latents_per_tok}

#     def get_binarized_mask(self, threshold):
#         return (self.ig_scores.abs()>threshold).float()

# def refresh_class():
#     for sae in saes:
#         if hasattr(sae, 'igmask'):
#             sae.igmask = IGMask(sae.igmask.ig_scores)

# try:
#     refresh_class()
# except Exception as e:
#     print(e)

# refresh_class()

# def produce_ig_binary_masks(threshold=0.01):
#     hook_points = []
#     masks = []

#     for sae in saes:
#         hook_point = sae.cfg.hook_name
#         mask = sae.igmask.get_binarized_mask(threshold=threshold)
#         hook_points.append(hook_point)
#         masks.append(mask)

#     return SAEMasks(
#         hook_points=hook_points,
#         masks=masks
#     )



# In[5]:
#### Data Setup ####
####################

# Define dropdown for task selection
dropdown = "codereason/index/len6_digit1"  # @param ["sva/rc", "codereason/key/len5_digit1", "codereason/index/len6_digit1", "ioi/baba21"]

# Define the task configurations
tasks = {
    "sva/rc": {"file_path": "data/sva/rc_train.json", "example_length": 7},
    "codereason/key/len5_digit1": {
        "file_path": "data/codereason/index/data_len5_digit1_errOnly_output.json",
        "example_length": 24,
    },
    "codereason/index/len6_digit1": {
        "file_path": "data/codereason/key/data_len5_digit1.json",
        "example_length": 41,
    },
    "ioi/baba21": {
        "file_path": "data/custom_task.json",  # Replace with the actual file path
        "example_length": 10,  # Replace with the actual example length
    },
}

task = dropdown
# Fetch the selected task details
task_info = tasks[dropdown]
file_path = task_info["file_path"]
example_length = task_info["example_length"]

# Load the data based on the selection
try:
    if "rc_train" in file_path or file_path.endswith(".jsonl"):
        # Handle JSON lines file
        with open(file_path, "r") as file:
            data = [json.loads(line) for line in file]
    else:
        # Handle standard JSON file
        with open(file_path, "r") as file:
            data = json.load(file)

    # Display information
    print(f"Loaded Data for: {dropdown}")
    print(f"File Path: {file_path}")
    print(f"Example Length: {example_length}")
    print("\nFirst Entry:")
    print(data[0] if isinstance(data, list) else data)

except FileNotFoundError:
    print(f"File not found: {file_path}")
except json.JSONDecodeError:
    print(f"Error decoding JSON in file: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")


batch_size = 16
# How many examples from the training data to use?
N = 3000

clean_data = []
corr_data = []
clean_labels = []
corr_labels = []
n_lens_clean = defaultdict(int)
n_lens_corr = defaultdict(int)
for entry in data:
    clean_len = len(model.tokenizer(entry['clean_prefix']).input_ids)
    corr_len = len(model.tokenizer(entry['patch_prefix']).input_ids)
    n_lens_clean[clean_len] += 1
    n_lens_corr[corr_len] += 1
    if clean_len == corr_len == example_length:
        clean_data.append(entry['clean_prefix'])
        corr_data.append(entry['patch_prefix'])
        clean_labels.append(entry['clean_answer'])
        corr_labels.append(entry['patch_answer'])
print(f"{len(clean_data)=}")
print(f"{example_length=}")
print(n_lens_clean)
print(n_lens_corr)

clean_tokens = model.to_tokens(clean_data[:N], prepend_bos=False)
corr_tokens = model.to_tokens(corr_data[:N], prepend_bos=False)
clean_label_tokens = model.to_tokens(clean_labels[:N], prepend_bos=False).squeeze(-1)
corr_label_tokens = model.to_tokens(corr_labels[:N], prepend_bos=False).squeeze(-1)
print(f"{clean_tokens.shape=}")
print(f"{corr_tokens.shape=}")
print(f"{clean_label_tokens.shape=}")
print(f"{corr_label_tokens.shape=}")

def logit_diff_fn(logits, clean_labels, corr_labels, token_wise=False):
    clean_logits = logits[torch.arange(logits.shape[0]), -1, clean_labels]
    corr_logits = logits[torch.arange(logits.shape[0]), -1, corr_labels]
    return (clean_logits - corr_logits).mean() if not token_wise else (clean_logits - corr_logits)

clean_tokens = clean_tokens[:batch_size*(len(clean_tokens)//batch_size)]
corr_tokens = corr_tokens[:batch_size*(len(corr_tokens)//batch_size)]
clean_label_tokens = clean_label_tokens[:batch_size*(len(clean_label_tokens)//batch_size)]
corr_label_tokens = corr_label_tokens[:batch_size*(len(corr_label_tokens)//batch_size)]

clean_tokens = clean_tokens.reshape(-1, batch_size, clean_tokens.shape[-1])
corr_tokens = corr_tokens.reshape(-1, batch_size, corr_tokens.shape[-1])
clean_label_tokens = clean_label_tokens.reshape(-1, batch_size)
corr_label_tokens = corr_label_tokens.reshape(-1, batch_size)

print(f"{clean_tokens.shape=}")
print(f"{corr_tokens.shape=}")
print(f"{clean_label_tokens.shape=}")
print(f"{corr_label_tokens.shape=}")



# In[8]:
#### Setup SAEs ####
####################

device = "cuda"
## For Gemma-2-9b
#layers= [7, 14, 21, 40]
#l0s = [92, 67, 129, 125]
# saes = [SAE.from_pretrained(release="gemma-scope-9b-pt-res",
#                             sae_id=f"layer_{layers[i]}/width_16k/average_l0_{l0s[i]}", 
#                             device=device)[0] for i in range(len(layers))]

## For GPT2-Small
#layers= [3, 5, 7, 9]
layers= [5, 9]


saes = [SAE.from_pretrained(release="jbloom/GPT2-Small-SAEs-Reformatted",
                            sae_id=f"blocks.{layer}.hook_resid_pre", 
                            device=device)[0] for layer in layers]

## For now just set the parameter values here. Upgrade to config file or the like at some point. 
import importlib
import beta_copula_mask
importlib.reload(beta_copula_mask)
from beta_copula_mask import SAECircuitMasker

SCM = SAECircuitMasker(saes=saes,
                       seq_len=example_length,
                       model=model,
                       lambda_e=4.0,
                       lambda_beta=0.1,
                       stretch_right=(1 - 1e-5),
                       per_token_mask=True, #### NOTE!! Need to make sure False works too!!! ####
                       device=device,
                       batch_size=16,
                       binary_threshold=0.5,
                       mean_tokens=corr_tokens)

print(SCM.lambda_e_idx_dict)
SCM.create_beta_icdf_lookup_table()


#%%
#### True vs SAE Reconstructed Logit Diff Sanity Check ####
###########################################################

use_mask = False
mean_mask = False
avg_logit_diff = 0
SCM.cleanup_cuda()
with torch.no_grad():
    for i in range(10):
        logits = model(
            clean_tokens[i]
            )
        ld = logit_diff_fn(logits, clean_label_tokens[i], corr_label_tokens[i])
        print(ld)
        avg_logit_diff += ld
        del logits
        SCM.cleanup_cuda()
model.reset_hooks(including_permanent=True)
model.reset_saes()
avg_model_diff = (avg_logit_diff / 10).item()
print("True Average LD: ", avg_model_diff)


use_mask = False
mean_mask = False
avg_logit_diff = 0
SCM.cleanup_cuda()
with torch.no_grad():
    for i in range(10):
        logits = model.run_with_hooks(
            clean_tokens[i],
            return_type="logits",
            prepend_bos=False,
            fwd_hooks=SCM.build_hooks_list(clean_tokens[i], use_mask=False, mean_mask=False)
            )
        ld = logit_diff_fn(logits, clean_label_tokens[i], corr_label_tokens[i])
        print(ld)
        avg_logit_diff += ld
        del logits
        SCM.cleanup_cuda()
model.reset_hooks(including_permanent=True)
model.reset_saes()
avg_logit_diff = (avg_logit_diff / 10).item()
print("SAE Average LD: ", avg_logit_diff)

ld_plain  = torch.tensor([ 0.3575, -0.0926, -0.3996,  0.5070, -0.1311,
                          -0.0271,  0.4333, -0.1432, -0.1939,  0.4285])
ld_recon  = torch.tensor([ 0.9971, -0.2210, -0.6975,  0.6570, -0.1431,
                          -0.6729,  1.0169, -0.2781, -1.2534,  0.5531])

r = torch.corrcoef(torch.vstack([ld_plain, ld_recon]))[0,1]
flip_rate = (ld_plain.sign() != ld_recon.sign()).float().mean()
print("Mean Absolute Difference: ", (ld_plain - ld_recon).abs().mean())
print("Pearson r =", r.item())        # expect ~0.99
print("Sign‑flip rate =", flip_rate)  # expect 0.0


# In[20]:


check_cuda_memory()
#%%
SCM.create_beta_icdf_lookup_table()


#%%
SCM.sample_joint_masks()
print("mask stats:", SCM.gcm.current_mask.min().item(),
                      SCM.gcm.current_mask.max().item())
check_cuda_memory()

#%%
check_cuda_memory()


#%%
SCM.set_sae_means(corr_tokens)

#%%
for i, sae in enumerate(SCM.saes):
    print(f"SAE {i}: d_sae =", sae.cfg.d_sae)

print(f"Full mask shape: {SCM.gcm.current_mask.shape}")
print(f"Full mask stats: min={SCM.gcm.current_mask.min().item():.4f}, max={SCM.gcm.current_mask.max().item():.4f}")

# Show masks for each SAE layer
for i, sae in enumerate(SCM.saes):
    layer_mask = SCM.get_masks(layer=sae.cfg.hook_layer)
    print(f"Layer {sae.cfg.hook_layer} mask shape: {layer_mask.shape}")
    print(f"Layer {sae.cfg.hook_layer} mask head (first batch, first 10 tokens, first 5 neurons): {layer_mask[0, :10, :5]}")




# In[19]:

logits = model.run_with_hooks(
    clean_tokens[0],
    return_type="logits",
    prepend_bos=False,
    fwd_hooks=SCM.build_hooks_list(
        clean_tokens[0],
        use_mask=True,       # <- uses the mask you just sampled
        mean_mask=True))     # <- replaces "off" latents with their mean

ld = logit_diff_fn(logits, clean_label_tokens[0], corr_label_tokens[0])
print("Logit‑diff with mask:", ld.item())


# # Test Beta Copula Training

# In[21]:

print("="*50)
print("TESTING BETA COPULA TRAINING")
print("="*50)

# Test the new training system with just a few iterations
print("Running a short training test with the new Beta Copula masking system...")

# Define hyperparameters for the test
test_hyperparams = {
    "sparsity_multiplier": 0.1,  # Low sparsity for quick test
    "learning_rate": 1.0,
        "batch_size": 16,
    "total_steps": 5  # Just 5 steps for testing
}

print(f"Test hyperparameters: {test_hyperparams}")
print(f"Mask shape would be: {SCM.gcm.current_mask.shape if hasattr(SCM.gcm, 'current_mask') else 'Not set'}")
print(f"Total neurons: {len(SCM.saes) * SCM.saes[0].cfg.d_sae}")
print(f"Seq len: {SCM.seq_len}")
print(f"Batch size: {SCM.batch_size}")

# Run a short training test
try:
    SCM.run_training(
        token_dataset=clean_tokens[:5],  # Just first 5 batches
        labels_dataset=clean_label_tokens[:5],
        corr_labels_dataset=corr_label_tokens[:5],
        task=dropdown,
        loss_function='logit_diff',
        portion_of_data=1.0,  # Use all of the limited data
        learning_rate=1.0
    )
    print("\n✅ Training test completed successfully!")
    print("The Beta Copula masking system is working correctly.")
    
except Exception as e:
    print(f"\n❌ Training test failed with error: {e}")
    import traceback
    traceback.print_exc()

print("="*50)

#%%
mem.check_memory()

# Simple training test with hyperparams override
print("="*50)
print("TESTING BETA COPULA TRAINING WITH HYPERPARAMS")
print("="*50)

# Hyperparameters for a short training test
# hyperparams = {
#     'lambda_e': 0.1,        # Lower for faster convergence
#     'lambda_beta': 0.01,    # Reduced complexity penalty
#     'lambda_sim': 0.01,     # Reduced similarity penalty  
#     'lambda_diag': 0.01,    # Reduced diagonal penalty
#     'lambda_Q': 0.001       # Reduced Q sparsity
# }

learning_rate = 1.0
total_steps = 3  # Very short test

#print(f"Using hyperparameters: {hyperparams}")
print(f"Learning rate: {learning_rate}")
print(f"Total steps: {total_steps}")

try:
    # Use smaller dataset for quick test
    test_tokens = clean_tokens[:total_steps]
    test_clean_labels = clean_label_tokens[:total_steps] 
    test_corr_labels = corr_label_tokens[:total_steps]
    
    print(f"Test data shapes:")
    print(f"  Tokens: {test_tokens.shape}")
    print(f"  Clean labels: {test_clean_labels.shape}")
    print(f"  Corrupted labels: {test_corr_labels.shape}")
    
    # Run training with hyperparams override
    print("\nStarting training with hyperparams override...")
    result = SCM.run_training(
        test_tokens,
        test_clean_labels, 
        test_corr_labels,
        #hyperparams=hyperparams,  # Override specific hyperparameters
        task="hyperparams_test",
        loss_function='logit_diff',
        portion_of_data=1.0,  # Use all test data
        learning_rate=learning_rate
    )
    
    print("\nTraining with hyperparams completed successfully!")
    print(f"Final results: {result}")
    
except Exception as e:
    print(f"Error during training test: {e}")
    import traceback
    traceback.print_exc()

#%%
mem.check_memory()