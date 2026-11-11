# One run: choose an optimizer, its learning rate, and an output folder.
optimizer_name = 'sgdm'         # 'adamw', 'muon', or 'sgdm'
out_dir = 'runs/sgdm-default'

# Default learning rates. Edit the one for your chosen optimizer.
learning_rate = 3e-4            # AdamW
muon_learning_rate = 1e-2       # Muon
sgdm_learning_rate = 0.1        # SGDM

# Fixed course settings.
model_preset = 'gpt2-medium'
block_size = 256
batch_size = 72
gradient_accumulation_steps = 1
max_tokens = 92160000           # 5,000 updates
max_seconds = 0.0
eval_interval = 500
eval_iters = 8

warmup_iters = 100
lr_schedule = 'cosine'
lr_decay_iters = 5000
min_lr_ratio = 0.1
lr_multiplier = 1.0

dropout = 0.2
weight_decay = 0.0
grad_clip = 1.0
beta1 = 0.9
beta2 = 0.99
muon_momentum = 0.95
sgdm_momentum = 0.9
sgdm_nesterov = False
seed = 42
eval_seed = 1729
dtype = 'auto'
compile = True
