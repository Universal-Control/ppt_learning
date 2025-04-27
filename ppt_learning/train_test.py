from typing import Tuple
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import hydra
from collections import OrderedDict

import numpy as np
import copy

import yaml
from collections import deque
import wandb
import logging

from ppt_learning.utils.utils import dict_apply, batchify, sample_pcd_data, unbatchify

try:
    from ppt_learning.utils.video import save_video
except:
    pass
from ppt_learning.utils.model_utils import module_max_gradient, module_mean_param

info_key = [
    "loss",
    "max_action",
    "max_label_action",
    "max_gradient",
    "max_stem_gradient",
    "max_trunk_gradient",
    "max_head_gradient",
    "batch_time",
    "mean_param",
    "lr",
]

Loss = F.mse_loss


def log_stat(
    info_log,
    train_step,
    log_interval,
    log_name,
    domain,
    target,
    loss,
    output,
    model,
    optimizer,
    start_time,
    rank=0,
    world_size=1,
    use_wandb=True,
):
    # Only rank 0 logs to avoid duplicates
    if rank != 0:
        return

    if domain + "_loss" not in info_log:
        info_log[domain + "_loss"] = deque([], maxlen=50)

    # Clone loss to avoid modifying the original tensor
    loss_item = loss.item()

    # Aggregate metrics in DDP mode
    if dist.is_initialized() and world_size > 1:
        # Aggregate loss (average)
        loss_tensor = torch.tensor(loss_item, device=loss.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        loss_item = loss_tensor.item() / world_size

        # Aggregate max_label_action (maximum across processes)
        max_label_action = torch.tensor(target.max().item(), device=loss.device)
        dist.all_reduce(max_label_action, op=dist.ReduceOp.MAX)

        # Aggregate max_action (maximum, if output is not None)
        max_action = torch.tensor(output.max().item() if output is not None else 0.0, device=loss.device)
        if output is not None:
            dist.all_reduce(max_action, op=dist.ReduceOp.MAX)

        # Aggregate gradients and mean_param (maximum for gradients, average for mean_param)
        max_gradient = torch.tensor(module_max_gradient(model), device=loss.device)
        max_stem_gradient = torch.tensor(module_max_gradient(model.stems), device=loss.device)
        max_trunk_gradient = torch.tensor(module_max_gradient(model.trunk), device=loss.device)
        max_head_gradient = torch.tensor(module_max_gradient(model.heads), device=loss.device)
        mean_param = torch.tensor(module_mean_param(model), device=loss.device)

        dist.all_reduce(max_gradient, op=dist.ReduceOp.MAX)
        dist.all_reduce(max_stem_gradient, op=dist.ReduceOp.MAX)
        dist.all_reduce(max_trunk_gradient, op=dist.ReduceOp.MAX)
        dist.all_reduce(max_head_gradient, op=dist.ReduceOp.MAX)
        dist.all_reduce(mean_param, op=dist.ReduceOp.SUM)
        mean_param = mean_param.item() / world_size
    else:
        # Single-process mode: Use local values
        max_label_action = target.max().item()
        max_action = output.max().item() if output is not None else 0.0
        max_gradient = module_max_gradient(model)
        max_stem_gradient = module_max_gradient(model.stems)
        max_trunk_gradient = module_max_gradient(model.trunk)
        max_head_gradient = module_max_gradient(model.heads)
        mean_param = module_mean_param(model)

    # Store metrics in info_log
    info_log[domain + "_loss"].append(loss_item)
    info_log["loss"].append(loss_item)
    info_log["max_label_action"].append(max_label_action)
    info_log["max_action"].append(max_action)
    info_log["max_gradient"].append(max_gradient)
    info_log["max_stem_gradient"].append(max_stem_gradient)
    info_log["max_trunk_gradient"].append(max_trunk_gradient)
    info_log["max_head_gradient"].append(max_head_gradient)
    info_log["mean_param"].append(mean_param)
    info_log["batch_time"].append(time.time() - start_time)
    info_log["lr"].append(optimizer.param_groups[0]["lr"])

    # Log to wandb (only rank 0)
    if use_wandb and (train_step % log_interval == 0):
        wandb_metrics = {
            f"{log_name}/{k}": np.mean(v) for k, v in info_log.items() if len(v) > 0
        }
        wandb.log({"train_step": train_step, **wandb_metrics})


def train(
    log_interval,
    model,
    device,
    train_loader,
    optimizer,
    scheduler,
    epoch,
    rank=0,
    world_size=1,
    pcd_npoints=8192,
    in_channels=4,
    log_name="train",
    debug=True,
):
    """training for one epoch"""
    info_log = {k: deque([], maxlen=20) for k in info_key}
    model.train()
    start_time = time.time()

    # combined_dataloader = train_loader  # WeightedDataLoader(train_loaders)
    epoch_size = len(train_loader)
    assert epoch_size > 0, "empty dataloader"
    pbar = tqdm(train_loader, position=1, leave=True, disable=(rank != 0))  # Disable pbar for non-rank-0

    # randomly sample a dataloader with inverse probability square root to the number of data
    for batch_idx, batch in enumerate(pbar):
        batch["data"] = dict_apply(
            batch["data"], lambda x: x.to(device, non_blocking=True).float()
        )

        batch["data"] = batchify(batch["data"], exclude=["action"])

        if "pointcloud" in batch["data"]:
            sample_pcd_data(
                batch["data"]["pointcloud"],
                npoints=pcd_npoints,
                in_channels=in_channels,
            )

        data_time = time.time() - start_time
        start_time = time.time()
        if isinstance(model, DDP):
            output = model.module.forward_train(batch)
        else:
            output = model.forward_train(batch)
        target = batch["data"]["action"]
        if rank == 0 and target.shape[1] == 0:
            print("empty target:", target.shape)
            continue

        if isinstance(output, (dict, OrderedDict)):
            domain_loss = output["loss"]
            output = None
        else:
            output = output.reshape(target.shape)
            domain_loss = Loss(output, target)

        # backward
        optimizer.zero_grad()
        domain_loss.backward()
        optimizer.step()
        scheduler.step()
        train_step = len(train_loader) * epoch + batch_idx

        model_ = model.module if isinstance(model, DDP) else model

        # Log stats (only rank 0, with aggregated metrics)
        log_stat(
            info_log,
            train_step,
            log_interval,
            log_name,
            batch["domain"][0],
            target,
            domain_loss,
            output,
            model_,
            optimizer,
            start_time,
            rank=rank,
            world_size=world_size,
            use_wandb=not debug,
        )

        if rank == 0:
            step_time = time.time() - start_time
            pbar.set_description(
                f"Epoch: {epoch} Step: {batch_idx}/{epoch_size} Time: {step_time:.3f} {data_time:.3f} Loss: {info_log[batch['domain'][0] + '_loss'][-1]:.3f} Grad: {info_log['max_gradient'][-1]:.3f}"
            )
        start_time = time.time()

    return {k: np.mean(v) for k, v in info_log.items() if len(v) > 1}


@torch.no_grad()
def test(
    model,
    device,
    test_loader,
    epoch,
    rank=0,
    world_size=1,
    pcd_npoints=8192,
    in_channels=4,
    log_name="test",
    debug=True,
):
    """evaluate imitation losses on the test sets"""
    model.eval()
    test_loss, num_examples = 0, 0
    pbar = tqdm(test_loader, position=2, leave=False, disable=rank != 0)

    for batch_idx, batch in enumerate(pbar):
        batch["data"] = dict_apply(
            batch["data"], lambda x: x.to(device, non_blocking=True).float()
        )

        batch["data"] = batchify(batch["data"], exclude=["action"])

        if "pointcloud" in batch["data"]:
            sample_pcd_data(
                batch["data"]["pointcloud"],
                npoints=pcd_npoints,
                in_channels=in_channels,
            )

        if isinstance(model, DDP):
            output = model.module.forward_train(batch)
        else:
            output = model.forward_train(batch)
        target = batch["data"]["action"].to(device)
        if target.shape[1] == 0:
            continue
        if isinstance(output, (dict, OrderedDict)):
            loss = output["loss"]
            output = None
        else:
            output = output.reshape(target.shape)
            loss = Loss(output, target)

        # Accumulate local loss and examples
        test_loss += loss.item() * target.size(0)  # Weighted by batch size
        num_examples += target.size(0)

    # Aggregate test_loss and num_examples across processes
    if dist.is_initialized() and world_size > 1:
        loss_tensor, num_examples_tensor = torch.tensor([test_loss, num_examples], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_examples_tensor, op=dist.ReduceOp.SUM)
        test_loss = loss_tensor.item()
        num_examples = num_examples_tensor.item()
    # Compute global average loss (only rank 0 logs)
    if rank == 0:
        global_loss = test_loss / num_examples if num_examples > 0 else 0.0
        pbar.set_description(
            f"Test Epoch: {epoch} Step: {batch_idx} Domain: {batch['domain'][0]} Loss: {global_loss:.3f}"
        )

    return test_loss / (num_examples + 1)


# sequential evaluation
def eval_policy(policy, cfg, env_name=None, seed=111, rollout_runner=None):
    # run the rollout function from each environment
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    policy.eval()
    if rollout_runner is None:
        rollout_runner = hydra.utils.instantiate(cfg.rollout_runner)
    success, reward, images = rollout_runner.run(
        policy=policy, env_name=env_name, seed=seed
    )
    if "RLBench" in cfg.rollout_runner._target_:
        tr = images
        tr.save(f"{cfg.output_dir}/{env_name}.mp4")
    else:
        if len(images) > 0:
            save_video(images, env_name, cfg.output_dir)

    return success, reward


def eval_policy_sequential(policy, cfg):
    env_names = [env_name.strip() for env_name in cfg.env_names]
    rollout_runner = None
    if "RLBench" in cfg.rollout_runner._target_:
        # from gensim2.env.utils.rlbench import ENV_DICT
        # env_names = [env_name for env_name in ENV_DICT.keys()]
        rollout_runner = hydra.utils.instantiate(cfg.rollout_runner)
    total_reward_list = []
    total_success_list = []

    for env_name in env_names:
        success, rewards = eval_policy(
            policy, cfg, env_name, rollout_runner=rollout_runner, seed=cfg.seed
        )
        total_success_list.append(success)

    total_success = {
        env_name: rew for env_name, rew in zip(env_names, total_success_list)
    }
    return total_success


def eval_policy_parallel(policy, cfg, seed=233):
    policy.eval()

    # run the rollout function from each environment
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    policy.eval()
    rollout_runner = hydra.utils.instantiate(cfg.rollout_runner)
    rollout_runner.initiate(policy=policy)
    rollout_runner.join()

    print("collecting results...")
    env_names, total_success_list, total_reward_list, images_list = (
        rollout_runner.collect_results()
    )

    if len(images_list) > 0:
        for i, images in enumerate(images_list):
            if len(images) > 0:
                save_video(images, env_names[i], cfg.output_dir)

    total_success = {
        env_name: rew for env_name, rew in zip(env_names, total_success_list)
    }
    return total_success
