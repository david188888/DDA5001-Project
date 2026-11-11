"""Single-device nanoGPT training: same GPT, explicit token/time experiment budgets."""
import csv
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import platform
import sys
import time

import numpy as np
import torch
from configurator import load_config
from evaluate import language_model_scores
from model import GPT, GPTConfig, resolve_model_preset
from checkpoint_io import atomic_torch_save, assert_kernel_healthy
from exercise_helpers import todo

ROOT = Path(__file__).resolve().parent
DEFAULTS = dict(out_dir='', data_dir=str(ROOT / 'data/shakespeare_char'), group_id='pilot',
    model_preset='', lr_schedule='constant', lr_decay_iters=5000, min_lr_ratio=.1,
    n_layer=6, n_head=6, n_embd=384, block_size=256, bias=False, dropout=.2,
    batch_size=16, gradient_accumulation_steps=4, optimizer_name='adamw',
    learning_rate=1e-3, muon_learning_rate=1e-3, lr_multiplier=1.0, muon_momentum=.95,
    sgdm_learning_rate=.1, sgdm_momentum=.9, sgdm_nesterov=False,
    weight_decay=0.0, beta1=.9, beta2=.99, grad_clip=1.0,
    max_tokens=8192000, max_seconds=0.0, warmup_iters=20, eval_interval=100, eval_iters=20,
    seed=42, eval_seed=1729, device='cuda', dtype='auto', compile=False, resume='',
    cpu_threads=4)


def prediction_windows(data, starts, block_size):
    """Given 1-D token data and start indices, return aligned [B,T] inputs/targets."""
    offsets = torch.arange(block_size)
    # TODO shifted_labels: Return [B,T] inputs and next-token targets shifted by exactly one position.
    # x and y must index the SAME original token sequence. Fill the offset
    # for the next token position; do not add a number to token IDs themselves.
    x = data[starts[:, None] + offsets]
    y = data[starts[:, None] + offsets + 1]
    return x, y


def step_optimizers(optimizers):
    """Step each provided optimizer once; backward/accumulation/clipping are elsewhere."""
    for optimizer in optimizers:
        # TODO optimizer_step: Update all optimizers after backpropagation and gradient clipping.
        # Replace this ONE placeholder with optimizer.step(), with no arguments.
        # The enclosing loop is provided: every optimizer must step exactly once.
        # Do not add backward, zero_grad, clipping, closures or another loop here.
        # This helper is called once AFTER all gradient accumulation and clipping.
        optimizer.step()


def get_batch(data, generator, config, device):
    starts = torch.randint(len(data) - config['block_size'], (config['batch_size'],), generator=generator)
    x, y = prediction_windows(data, starts, config['block_size'])
    if device.type == 'cuda':
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def validate(config):
    for name in ('n_layer', 'n_head', 'n_embd', 'block_size', 'batch_size',
                 'gradient_accumulation_steps', 'eval_interval', 'eval_iters', 'cpu_threads'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if config['n_embd'] % config['n_head']:
        raise ValueError('Embedding width must be divisible by attention heads')
    if not all(math.isfinite(config[k]) for k in ('max_seconds', 'lr_multiplier', 'learning_rate',
                                                   'muon_learning_rate', 'sgdm_learning_rate', 'sgdm_momentum',
                                                   'dropout', 'grad_clip')):
        raise ValueError('Settings must be finite')
    if config['max_tokens'] < 0 or config['max_seconds'] < 0 or config['warmup_iters'] < 0:
        raise ValueError('Budgets/warmup must be nonnegative')
    if (config['max_tokens'] > 0) == (config['max_seconds'] > 0):
        raise ValueError('Set exactly one budget: max_tokens>0 OR max_seconds>0; set the other to zero')
    per_step = config['batch_size'] * config['block_size'] * config['gradient_accumulation_steps']
    if config['max_tokens'] and config['max_tokens'] % per_step:
        raise ValueError(f'max_tokens must be a multiple of {per_step} tokens per completed update')
    if min(config['learning_rate'], config['muon_learning_rate'], config['sgdm_learning_rate'], config['lr_multiplier']) <= 0:
        raise ValueError('Learning rates must be positive')
    if not 0 < config['sgdm_momentum'] < 1 or type(config['sgdm_nesterov']) is not bool:
        raise ValueError('SGDM requires 0 < sgdm_momentum < 1 and a boolean sgdm_nesterov')
    if not 0 <= config['dropout'] < 1 or config['grad_clip'] < 0:
        raise ValueError('Invalid dropout/gradient clipping')
    if config['lr_schedule'] not in ('constant','cosine'):
        raise ValueError('lr_schedule must be constant or cosine')
    if not math.isfinite(config['min_lr_ratio']) or not 0 <= config['min_lr_ratio'] <= 1:
        raise ValueError('min_lr_ratio must lie between 0 and 1')
    if config['lr_schedule']=='cosine' and (type(config['lr_decay_iters']) is not int
                                           or config['lr_decay_iters'] <= config['warmup_iters']):
        raise ValueError('Cosine decay requires lr_decay_iters > warmup_iters')
    return per_step


def lr_factor(step, config):
    """nanoGPT warmup/cosine shape, scaled separately by every optimizer's base LR."""
    warmup=config['warmup_iters']
    if config['lr_schedule']=='constant':
        return min(1.0,(step+1)/max(1,warmup))  # preserve the earlier constant-LR experiments
    if step < warmup:
        return (step+1)/(warmup+1)  # same warmup convention as reference/nanoGPT/train.py
    if step >= config['lr_decay_iters']:
        return config['min_lr_ratio']
    progress=(step-warmup)/(config['lr_decay_iters']-warmup)
    return config['min_lr_ratio']+(1-config['min_lr_ratio'])*.5*(1+math.cos(math.pi*progress))


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def run(config):
    started = time.perf_counter()
    host_taint=assert_kernel_healthy()
    config = resolve_model_preset(config)
    per_step = validate(config)
    torch.set_num_threads(config['cpu_threads'])
    device = torch.device(config['device'])
    if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
        raise ValueError('Use a working CUDA GPU, or device=cpu for small correctness checks')
    if device.type == 'cuda':
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        torch.cuda.set_device(device)
    precision = config['dtype']
    if precision == 'auto':
        precision = 'bfloat16' if device.type == 'cuda' and torch.cuda.is_bf16_supported(including_emulation=False) else 'float32'
    if precision not in ('float32', 'bfloat16'):
        raise ValueError('This teaching trainer supports float32/bfloat16; use float32 on the RTX 2060')
    if precision == 'bfloat16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported(including_emulation=False)):
        raise ValueError('Requested BF16 requires native GPU support; choose float32')
    def autocast():
        return torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bfloat16' else nullcontext()
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    torch.manual_seed(config['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])
    train_generator = torch.Generator().manual_seed(config['seed'] + 1)
    data_dir = Path(config['data_dir'])
    with (data_dir / 'meta.pkl').open('rb') as stream:
        meta = pickle.load(stream)  # supplied course tokenizer
    data = {split: torch.from_numpy(np.fromfile(data_dir / f'{split}.bin', dtype=np.uint16).astype(np.int64))
            for split in ('train', 'val')}
    if any(len(values) <= config['block_size'] for values in data.values()):
        raise ValueError('Dataset is shorter than the context window')
    model_args = {key: config[key] for key in ('n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'dropout')}
    model_args['vocab_size'] = meta['vocab_size']
    model = GPT(GPTConfig(**model_args)).to(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    optimizers = model.configure_optimizers(config['weight_decay'], config['learning_rate'],
        (config['beta1'], config['beta2']), device.type, config['optimizer_name'],
        config['muon_learning_rate'], config['muon_momentum'],
        config['sgdm_learning_rate'], config['sgdm_momentum'], config['sgdm_nesterov'])
    step, tokens, train_seconds, best_loss = 0, 0, 0.0, float('inf')
    if config['resume']:
        checkpoint = torch.load(config['resume'], map_location='cpu', weights_only=True)
        if config['optimizer_name'] == 'muon' and checkpoint.get('muon_implementation') != 'torch.optim.Muon/original':
            raise ValueError('Legacy custom Muon checkpoint is not a native-Muon continuation; start a new run')
        changeable = {'out_dir', 'resume', 'max_tokens', 'max_seconds', 'group_id', 'eval_interval'}
        if config['optimizer_name'] != 'muon':
            changeable.update(('muon_learning_rate', 'muon_momentum'))
        if config['optimizer_name'] != 'sgdm':
            changeable.update(('sgdm_learning_rate', 'sgdm_momentum', 'sgdm_nesterov'))
        # Old AdamW/Muon checkpoints predate the optional SGDM settings.
        checkpoint_config = {**DEFAULTS, **checkpoint['config']}
        changed = [key for key in config if key not in changeable and config[key] != checkpoint_config[key]]
        if changed or model_args != checkpoint['model_args']:
            raise ValueError(f'Resume would change training settings: {changed}')
        model.load_state_dict(checkpoint['model'])
        if len(optimizers) != len(checkpoint['optimizers']):
            raise ValueError('Checkpoint optimizer count differs')
        for optimizer, state in zip(optimizers, checkpoint['optimizers']):
            optimizer.load_state_dict(state)
        step, tokens = checkpoint['step'], checkpoint['tokens']
        train_seconds, best_loss = checkpoint['train_seconds'], checkpoint['best_loss']
        if (config['max_tokens'] and tokens > config['max_tokens']) or (config['max_seconds'] and train_seconds > config['max_seconds']):
            raise ValueError('Requested cumulative budget is smaller than the checkpoint progress')
        torch.set_rng_state(checkpoint['torch_rng'])
        train_generator.set_state(checkpoint['data_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    forward_model = torch.compile(model) if config['compile'] else model
    out = Path(config['out_dir'] or f"runs/{config['optimizer_name']}-{datetime.now().strftime('%Y%m%dT%H%M%S%f')}")
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'config.json', config)
    environment = {'python': platform.python_version(), 'torch': str(torch.__version__),
        'kernel_taint_at_start':host_taint, 'checkpoint_io':'fsync-clean-cache-hint-atomic-replace-directory-fsync-v2',
        'initialization':'resume' if config['resume'] else 'random_from_scratch',
        'model_preset':config['model_preset'],
        'model_parameters':model.get_num_params(non_embedding=False),
        'model_parameters_without_positions':model.get_num_params(),
        'cuda_runtime': torch.version.cuda, 'device': str(device), 'dtype': precision,
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'muon_implementation': 'torch.optim.Muon/original' if config['optimizer_name'] == 'muon' else None,
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},
        'dataset_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data_dir.glob('*.bin')}}
    save_json(out / 'environment.json', environment)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        result = {}
        for split_id, split in enumerate(('train', 'val')):
            # Fixed windows, independent of training RNG and evaluation frequency.
            generator = torch.Generator().manual_seed(config['eval_seed'] + split_id)
            nll, count = 0.0, 0
            for _ in range(config['eval_iters']):
                x, y = get_batch(data[split], generator, config, device)
                with autocast():
                    _, loss = forward_model(x, y)
                nll += loss.item() * y.numel()
                count += y.numel()
            result[split] = language_model_scores(nll, count)
        model.train()
        return result

    checkpoint_seconds = 0.0
    def save_checkpoint(filename):
        nonlocal checkpoint_seconds
        state = dict(model=model.state_dict(), optimizers=[o.state_dict() for o in optimizers],
            muon_implementation='torch.optim.Muon/original' if config['optimizer_name'] == 'muon' else None,
            model_args=model_args, config=config, step=step, tokens=tokens,
            train_seconds=train_seconds, best_loss=best_loss, torch_rng=torch.get_rng_state(),
            data_rng=train_generator.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            tokenizer={'itos': meta['itos'], 'stoi': meta['stoi']})
        checkpoint_seconds += atomic_torch_save(state,out/filename,log)

    fields = ['step', 'tokens', 'train_seconds', 'wall_seconds', 'train_loss', 'val_loss', 'adamw_lr', 'muon_lr', 'sgdm_lr']
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream, (out / 'run.log').open('w', encoding='utf-8') as log_stream:
        def log(message):
            print(message, flush=True)
            log_stream.write(message + '\n')
            log_stream.flush()
        log(f"Starting {config['optimizer_name']}; config.json and environment.json record this run.")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        last_evaluated = -1
        scores = None
        def evaluate_and_record():
            nonlocal best_loss, last_evaluated, scores
            scores = estimate_loss()
            val_loss = scores['val']['loss']
            if not math.isfinite(val_loss):
                raise RuntimeError('Nonfinite validation loss')
            writer.writerow(dict(step=step, tokens=tokens, train_seconds=train_seconds,
                wall_seconds=time.perf_counter()-started,
                train_loss=scores['train']['loss'], val_loss=val_loss,
                adamw_lr=optimizers[0].param_groups[0]['lr'] if step and config['optimizer_name'] != 'sgdm' else 0,
                muon_lr=optimizers[1].param_groups[0]['lr'] if step and len(optimizers) == 2 else 0,
                sgdm_lr=optimizers[0].param_groups[0]['lr'] if step and config['optimizer_name'] == 'sgdm' else 0))
            stream.flush()
            log(f'step {step}: tokens={tokens} train_s={train_seconds:.3f} val_loss={val_loss:.4f}')
            if val_loss < best_loss:
                best_loss = val_loss
                save_checkpoint('best.pt')
            last_evaluated = step
        try:
            evaluate_and_record()
            while (tokens < config['max_tokens'] if config['max_tokens'] else train_seconds < config['max_seconds']):
                synchronize()
                tick = time.perf_counter()
                factor = lr_factor(step,config) * config['lr_multiplier']
                for optimizer in optimizers:
                    for group in optimizer.param_groups:
                        group['lr'] = group['base_lr'] * factor
                    optimizer.zero_grad(set_to_none=True)
                for _ in range(config['gradient_accumulation_steps']):
                    x, y = get_batch(data['train'], train_generator, config, device)
                    with autocast():
                        _, loss = forward_model(x, y)
                        loss = loss / config['gradient_accumulation_steps']
                    loss.backward()
                # error_if_nonfinite ensures no optimizer is updated after nonfinite gradients.
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'] or float('inf'), error_if_nonfinite=True)
                step_optimizers(optimizers)
                synchronize()
                train_seconds += time.perf_counter() - tick
                step += 1
                tokens += per_step
                if step % config['eval_interval'] == 0:
                    evaluate_and_record()
                    save_checkpoint('last.pt')
            if last_evaluated != step:
                evaluate_and_record()
            save_checkpoint('last.pt')
            summary = dict(status='completed', step=step, tokens=tokens, train_seconds=train_seconds,
                checkpoint_seconds=checkpoint_seconds,
                wall_seconds=time.perf_counter() - started, endpoint_scores=scores['val'], best_val_loss=best_loss,
                tokens_per_second=tokens / train_seconds if train_seconds else None,
                peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                budget='tokens' if config['max_tokens'] else 'time', output=str(out.resolve()))
            save_json(out / 'summary.json', summary)
            log('Completed; endpoint saved as last.pt and summary.json.')
            return summary
        except BaseException as exc:
            log(f'Failed: {type(exc).__name__}: {exc}')
            save_json(out / 'summary.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}',
                                                step=step, tokens=tokens, train_seconds=train_seconds))
            raise


if __name__ == '__main__':
    print(json.dumps(run(load_config(DEFAULTS, sys.argv[1:])), indent=2))
