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

#%%
import inspect, beta_copula_mask as bcm

###################################
#### Params #######################
###################################


lambda_e      = 4.0
lambda_beta   = 0.5
lambda_sim    = 5.0
lambda_diag   = 1.0
lambda_Q      = 15.0
lambda_copula = None


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

os.environ["TRANSFORMERS_OFFLINE"]="1"
os.environ["HF_HUB_OFFLINE"]="1"
os.environ["HF_DATASETS_OFFLINE"]="1"

# Load the model
#model_name = "google/gemma-2-9b"  # Replace with the desired model name
model_name = "gpt2-small"  # Replace with the desired model name
print(f"Loading model: {model_name}...")
model = HookedSAETransformer.from_pretrained_no_processing(model_name, device=device, cache_dir=hf_cache, torch_dtype=torch.bfloat16, )

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
print("Loading Data...")
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
# print(f"{len(clean_data)=}")
# print(f"{example_length=}")
# print(n_lens_clean)
# print(n_lens_corr)

clean_tokens = model.to_tokens(clean_data[:N], prepend_bos=False)
corr_tokens = model.to_tokens(corr_data[:N], prepend_bos=False)
clean_label_tokens = model.to_tokens(clean_labels[:N], prepend_bos=False).squeeze(-1)
corr_label_tokens = model.to_tokens(corr_labels[:N], prepend_bos=False).squeeze(-1)
# print(f"{clean_tokens.shape=}")
# print(f"{corr_tokens.shape=}")
# print(f"{clean_label_tokens.shape=}")
# print(f"{corr_label_tokens.shape=}")

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
layers= [3, 5, 7, 9]
# layers= [5, 9]


saes = [SAE.from_pretrained(release="jbloom/GPT2-Small-SAEs-Reformatted",
                            sae_id=f"blocks.{layer}.hook_resid_pre", 
                            device=device)[0] for layer in layers]


SCM = SAECircuitMasker(saes=saes,
                       seq_len=example_length,
                       model=model,
                       lambda_e=lambda_e,
                       lambda_beta=lambda_beta,
                       lambda_sim=lambda_sim,
                       lambda_diag=lambda_diag,
                       lambda_Q=lambda_Q,
                       lambda_copula=lambda_copula,
                       stretch_right=(1 - 1e-5),
                       per_token_mask=True, #### NOTE!! Need to make sure False works too!!! ####
                       device=device,
                       batch_size=16,
                       binary_threshold=0.5,
                       mean_tokens=corr_tokens)

print("type MRO:", SCM.gcm.__class__.mro())
print("instance dict keys that shadow class attrs:",
      [k for k in SCM.gcm.__dict__.keys() if k in ("alpha_o", "beta_o")])

#print(SCM.lambda_e_idx_dict)
SCM.create_beta_icdf_lookup_table(verbose=True)


#%%
#### True vs SAE Reconstructed Logit Diff Sanity Check ####
###########################################################

# use_mask = False
# mean_mask = False
# avg_logit_diff = 0
# SCM.cleanup_cuda()
# with torch.no_grad():
#     for i in range(10):
#         logits = model(
#             clean_tokens[i]
#             )
#         ld = logit_diff_fn(logits, clean_label_tokens[i], corr_label_tokens[i])
#         print(ld)
#         avg_logit_diff += ld
#         del logits
#         SCM.cleanup_cuda()
# model.reset_hooks(including_permanent=True)
# model.reset_saes()
# avg_model_diff = (avg_logit_diff / 10).item()
# print("True Average LD: ", avg_model_diff)


# use_mask = False
# mean_mask = False
# avg_logit_diff = 0
# SCM.cleanup_cuda()
# with torch.no_grad():
#     for i in range(10):
#         logits = model.run_with_hooks(
#             clean_tokens[i],
#             return_type="logits",
#             prepend_bos=False,
#             fwd_hooks=SCM.build_hooks_list(clean_tokens[i], use_mask=False, mean_mask=False)
#             )
#         ld = logit_diff_fn(logits, clean_label_tokens[i], corr_label_tokens[i])
#         print(ld)
#         avg_logit_diff += ld
#         del logits
#         SCM.cleanup_cuda()
# model.reset_hooks(including_permanent=True)
# model.reset_saes()
# avg_logit_diff = (avg_logit_diff / 10).item()
# print("SAE Average LD: ", avg_logit_diff)

# ld_plain  = torch.tensor([ 0.3575, -0.0926, -0.3996,  0.5070, -0.1311,
#                           -0.0271,  0.4333, -0.1432, -0.1939,  0.4285])
# ld_recon  = torch.tensor([ 0.9971, -0.2210, -0.6975,  0.6570, -0.1431,
#                           -0.6729,  1.0169, -0.2781, -1.2534,  0.5531])

# r = torch.corrcoef(torch.vstack([ld_plain, ld_recon]))[0,1]
# flip_rate = (ld_plain.sign() != ld_recon.sign()).float().mean()
# print("Mean Absolute Difference: ", (ld_plain - ld_recon).abs().mean())
# print("Pearson r =", r.item())        # expect ~0.99
# print("Sign‑flip rate =", flip_rate)  # expect 0.0

#%%
import inspect, importlib, beta_copula_mask as bcm
print("id(BetaMask in module):", id(bcm.BetaMask))
print("id(gcm base class)    :", id(SCM.gcm.__class__.__mro__[1]))




#%%
SCM.sample_joint_masks()
print("mask stats:", SCM.gcm.current_mask.min().item(),
                      SCM.gcm.current_mask.max().item())
SCM.set_sae_means(corr_tokens)


#%%
# # Test Beta Copula Training

print("="*50)
print("TESTING BETA COPULA TRAINING")
print("="*50)

# Test the new training system with just a few iterations
print("Running a short training test with the new Beta Copula masking system...")

# Define hyperparameters for the test
test_hyperparams = {
    "learning_rate": 0.01,
        "batch_size": 16,
    "total_steps": 7  # Just 5 steps for testing
}

print(f"Test hyperparameters: {test_hyperparams}")
print(f"Mask shape would be: {SCM.gcm.current_mask.shape if hasattr(SCM.gcm, 'current_mask') else 'Not set'}")
print(f"Total neurons: {len(SCM.saes) * SCM.saes[0].cfg.d_sae}")
print(f"Seq len: {SCM.seq_len}")
print(f"Batch size: {SCM.batch_size}")
check_cuda_memory()

# SCM.enable_grad_debug(n_steps=999, log_every=1)

# Run a short training test
try:
    SCM.run_training(
        token_dataset=clean_tokens,  # Just first 7 batches
        labels_dataset=clean_label_tokens,
        corr_labels_dataset=corr_label_tokens,
        task=dropdown,
        loss_function='logit_diff',
        portion_of_data=1.0,  # Use all of the limited data
        learning_rate=0.01,
        epochs=1,
        verbose=False
    )
    print("\n✅ Training test completed successfully!")
    print("The Beta Copula masking system is working correctly.")
    
except Exception as e:
    print(f"\n❌ Training test failed with error: {e}")
    import traceback
    traceback.print_exc()

print("="*50)



#%%
# ────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────
# Analyze the learned parameters
# ────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────
print("="*50)
print("ANALYZING LEARNED PARAMETERS")
print("="*50)

import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# ────────────────────────────────────────────────────────────────
# Memory-efficient diagnostics for the low-rank covariance QQᵀ
# ────────────────────────────────────────────────────────────────
# Plot 1: Main loss components
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

import matplotlib as mpl
mpl.rcParams['axes.formatter.limits'] = (-999, 999) 

# Plot 1: Main loss components
axes[0,0].plot(SCM.complexity_losses, label='Complexity Loss', color='blue')
axes[0,0].plot(SCM.lhood_losses, label='Likelihood Loss', color='red')
axes[0,0].plot(SCM.task_losses, label='Task Loss', color='green')
axes[0,0].set_title('Main Loss Components')
axes[0,0].set_xlabel('Iteration')
axes[0,0].set_ylabel('Loss Value')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 1: Main loss components
axes[0,1].plot(SCM.task_losses, label='Task Loss', color='green')
axes[0,1].set_title('Task (CE or Logit Diff) Loss Components')
axes[0,1].set_xlabel('Iteration')
axes[0,1].set_ylabel('Loss Value')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)


# Plot 2: Individual loss components (unweighted)
axes[1,0].plot(SCM.beta_log_losses, label='Beta Log Loss', color='purple')
axes[1,0].plot(SCM.beta_sim_losses, label='Beta Sim Loss', color='orange')
axes[1,0].plot(SCM.diag_penalties, label='Diagonal Penalty', color='brown')
axes[1,0].plot(SCM.Q_sparsities, label='Q Sparsity', color='pink')
axes[1,0].plot(SCM.copula_losses, label='Copula Loss', color='cyan')
axes[1,0].set_title('Individual Loss Components (Unweighted)')
axes[1,0].set_xlabel('Iteration')
axes[1,0].set_ylabel('Loss Value')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Plot 3: Individual loss components (weighted by tuning parameters)
axes[1,1].plot([x * SCM.gcm.lambda_sim for x in SCM.beta_sim_losses], label=f'Beta Sim Loss × {SCM.gcm.lambda_sim}', color='orange')
axes[1,1].plot([x * SCM.gcm.lambda_diag for x in SCM.diag_penalties], label=f'Diagonal Penalty × {SCM.gcm.lambda_diag}', color='brown')
axes[1,1].plot([x * SCM.gcm.lambda_Q for x in SCM.Q_sparsities], label=f'Q Sparsity × {SCM.gcm.lambda_Q}', color='pink')
axes[1,1].plot([x * SCM.gcm.lambda_beta for x in SCM.beta_log_losses], label=f'Beta Log Loss × {SCM.gcm.lambda_beta}', color='purple')
axes[1,1].plot([x * SCM.gcm.lambda_copula for x in SCM.copula_losses], label=f'Copula Loss × {SCM.gcm.lambda_copula}', color='cyan')
axes[1,1].set_title('Individual Loss Components (Weighted by Tuning Parameters)')
axes[1,1].set_xlabel('Iteration')
axes[1,1].set_ylabel('Loss Value')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/loss_components.png', dpi=300, bbox_inches='tight')
if plt.isinteractive():
    plt.show()

# Plot 4: Effective parameters tracking (all in one graph)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Mean effective alpha trend
axes[0].plot(SCM.eff_alphas_mean, label='Mean Effective Alpha', color='blue', linewidth=2)
axes[0].set_title('Mean Effective Alpha Trend')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Mean Alpha Value')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: High parameter counts
axes[1].plot(SCM.eff_alphas_high, label='High Alpha Count (>75% of lambda_e)', color='red', linewidth=2)
axes[1].plot(SCM.eff_betas_high, label='High Beta Count (>75% of lambda_e)', color='green', linewidth=2)
axes[1].set_title('High Parameter Counts')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Count')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plots/effective_parameters.png', dpi=300, bbox_inches='tight')
if plt.isinteractive():
    plt.show()

# Print summary statistics for effective parameters
print("\nEFFECTIVE PARAMETERS SUMMARY:")
print("-" * 40)
print(f"Final Mean Alpha: {SCM.eff_alphas_mean[-1]:.6f}")
print(f"Final High Alpha Count: {SCM.eff_alphas_high[-1]:.0f}")
print(f"Final High Beta Count: {SCM.eff_betas_high[-1]:.0f}")
print(f"Alpha Range: [{min(SCM.eff_alphas_mean):.6f}, {max(SCM.eff_alphas_mean):.6f}]")
print(f"High Alpha Range: [{min(SCM.eff_alphas_high):.0f}, {max(SCM.eff_alphas_high):.0f}]")
print(f"High Beta Range: [{min(SCM.eff_betas_high):.0f}, {max(SCM.eff_betas_high):.0f}]")



# Print summary statistics for the loss components
print("\nLOSS COMPONENT SUMMARY:")
print("-" * 40)
print(f"Final Task Loss: {SCM.task_losses[-1]:.6f}")
print(f"Final Complexity Loss: {SCM.complexity_losses[-1]:.6f}")
print(f"Final Likelihood Loss: {SCM.lhood_losses[-1]:.6f}")
print(f"Final Beta Log Loss: {SCM.beta_log_losses[-1]:.6f}")
print(f"Final Beta Sim Loss: {SCM.beta_sim_losses[-1]:.6f}")
print(f"Final Diagonal Penalty: {SCM.diag_penalties[-1]:.6f}")
print(f"Final Q Sparsity: {SCM.Q_sparsities[-1]:.6f}")
print(f"Final Copula Loss: {SCM.copula_losses[-1]:.6f}")

# Q : (N, k)
Q = SCM.gcm.Q.detach().cpu().float()
print(f"Q shape: {Q.shape}")

# ---------- helpers -------------------------------------------
def sample_dot_products(Q_: torch.Tensor, n_pairs: int = 50_000):
    """Return *n_pairs* random off-diagonal dot-products of rows of *Q_*."""
    N = Q_.shape[0]
    idx_i = torch.randint(0, N, (n_pairs,))
    idx_j = torch.randint(0, N, (n_pairs,))
    return (Q_[idx_i] * Q_[idx_j]).sum(dim=1)

# ---------- basic stats ---------------------------------------
row_var     = (Q ** 2).sum(dim=1)                # diag of QQᵀ
off_diag    = sample_dot_products(Q, 100_000)    # sample off-diag entries

# Spectrum via small Gram matrix
G       = Q.T @ Q                                # (k, k)
eigvals = torch.linalg.eigvalsh(G)               # ascending order

# Effective α/β params (for completeness)
alpha_eff, beta_eff = SCM.gcm.get_effective_params()
alpha_eff = alpha_eff.flatten().detach().cpu().float()
beta_eff  = beta_eff.flatten().detach().cpu().float()

# ---------- plots ---------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1) row variance
axes[0].hist(row_var.numpy(), bins=100, color='black', alpha=0.7)
axes[0].set_title('Row variance (diag QQᵀ)')
axes[0].set_xlabel('‖qᵢ‖²')

# 2) sampled off-diagonal dot products
axes[1].hist(off_diag.numpy(), bins=200, color='red', alpha=0.7)
axes[1].set_title('Sampled off-diag dot products')
axes[1].set_xlabel('qᵢ · qⱼ,  i≠j')

# 3) eigen-spectrum
axes[2].semilogy(eigvals.flip(0).numpy())
axes[2].set_title('Eigenvalues of QQᵀ (via QᵀQ)')
axes[2].set_xlabel('rank index')

plt.tight_layout()
plt.savefig('plots/Q_matrix_analysis.png', dpi=300, bbox_inches='tight')
if plt.isinteractive():
    plt.show()

# ---------- summary print -------------------------------------
print("\nSUMMARY STATISTICS:")
print("-" * 30)
# Calculate quantiles for off-diagonal dot products
quantiles = [99.9, 99, 97, 95, 90, 75, 60, 50, 40, 25, 10, 5, 3, 1, 0.1]
off_diag_quantiles = torch.quantile(off_diag, torch.tensor([q/100 for q in quantiles]))

print(f"Off-diagonal dot product quantiles:")
for i, q in enumerate(quantiles):
    print(f"  {q:4.1f}th percentile: {off_diag_quantiles[i]:8.4f}")

print(f"Q matrix:  shape {Q.shape},  mean {Q.mean():.4f},  std {Q.std():.4f}, "
      f"range [{Q.min():.4f}, {Q.max():.4f}]")

print(f"Row variance  –  mean {row_var.mean():.4f},  std {row_var.std():.4f}, "
      f"range [{row_var.min():.4f}, {row_var.max():.4f}]")

print(f"Sampled off-diag dot products  –  mean {off_diag.mean():.4f},  "
      f"std {off_diag.std():.4f},  range [{off_diag.min():.4f}, {off_diag.max():.4f}]")

print(f"Eigenvalue spectrum  –  min {eigvals[0]:.4e},  max {eigvals[-1]:.4e}, "
      f"condition {eigvals[-1]/eigvals[0]:.2e}")

print(f"Effective α  –  mean {alpha_eff.mean():.4f}, std {alpha_eff.std():.4f}, "
      f"range [{alpha_eff.min():.4f}, {alpha_eff.max():.4f}]")
print(f"Effective β  –  mean {beta_eff.mean():.4f}, std {beta_eff.std():.4f}, "
      f"range [{beta_eff.min():.4f}, {beta_eff.max():.4f}]")

# Histogram of effective alpha and beta parameters
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Alpha histogram
ax1.hist(alpha_eff.numpy(), bins=50, color='blue', alpha=0.7)
ax1.set_title('Effective Alpha Parameters')
ax1.set_xlabel('Alpha')
ax1.set_ylabel('Frequency')

# Beta histogram
ax2.hist(beta_eff.numpy(), bins=50, color='red', alpha=0.7)
ax2.set_title('Effective Beta Parameters')
ax2.set_xlabel('Beta')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('plots/effective_params_histogram.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate percentages
alpha_lt_025 = (alpha_eff < 0.25).float().mean() * 100
alpha_gt_075 = (alpha_eff > 0.75).float().mean() * 100
alpha_lt_05 = (alpha_eff < 0.5).float().mean() * 100

beta_lt_025 = (beta_eff < 0.25).float().mean() * 100
beta_gt_075 = (beta_eff > 0.75).float().mean() * 100
beta_lt_05 = (beta_eff < 0.5).float().mean() * 100

print("\nEFFECTIVE PARAMETER STATISTICS:")
print("-" * 40)
print(f"Alpha < 0.25:  {alpha_lt_025:.2f}%")
print(f"Alpha > 0.75:  {alpha_gt_075:.2f}%")
print(f"Alpha < 0.5:   {alpha_lt_05:.2f}%")
print(f"Beta < 0.25:   {beta_lt_025:.2f}%")
print(f"Beta > 0.75:   {beta_gt_075:.2f}%")
print(f"Beta < 0.5:    {beta_lt_05:.2f}%")



print("=" * 50)

### --- END REPLACEMENT ---

# %%
print(SCM.gcm.Q.numel())
print(SCM.gcm.Q.shape)
-7e6 / SCM.gcm.Q.numel()

# %%
SCM.gcm.lambda_copula