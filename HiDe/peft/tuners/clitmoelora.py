# -*- encoding: utf-8 -*-
# here put the import lib
import importlib
import re
import warnings
import math
from dataclasses import dataclass, field
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers.pytorch_utils import Conv1D
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union, List
from ..utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
    ModulesToSaveWrapper,
)
from .lora import (
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
    Embedding,
    Conv2d,
)

from ..import_utils import is_bnb_4bit_available, is_bnb_available

if is_bnb_available():
    import bitsandbytes as bnb
    from .lora import Linear8bitLt
else:
    Linear8bitLt = None

if is_bnb_4bit_available():
    from .lora import Linear4bit
else:
    Linear4bit = None

@dataclass
class HiDeMOELoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MOE_LORA_HiDe`]
    """
    task_embedding_dim: int = field(default=64)
    expert_num: int = field(default=4)
    cur_task: int = field(default=4)
    consensus_enable: bool = field(default=False)
    consensus_rank: int = field(default=32)
    consensus_rank_shared: int = field(default=32)
    consensus_eta: float = field(default=0.5)
    consensus_sample_limit: int = field(default=128)
    consensus_samples_per_forward: int = field(default=4)
    consensus_oversample: int = field(default=8)

    def __post_init__(self):
        self.peft_type = PeftType.MOE_LORA_HiDe
        if not 0.0 <= self.consensus_eta <= 1.0:
            raise ValueError("consensus_eta must be in [0, 1]")
        if self.consensus_rank <= 0 or self.consensus_rank_shared <= 0:
            raise ValueError("consensus ranks must be positive")
        if self.consensus_rank_shared > self.consensus_rank:
            raise ValueError("consensus_rank_shared must not exceed consensus_rank")
        if self.consensus_sample_limit < self.consensus_rank:
            raise ValueError("consensus_sample_limit must be at least consensus_rank")
        if self.consensus_samples_per_forward <= 0:
            raise ValueError("consensus_samples_per_forward must be positive")


class HiDeMOELoraModel(LoraModel):
    """
    Create MMOELoRA (MMOE based LoRA) model from a pretrained transformers model.
    """
    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])
        self.expert_weight = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        print(self.model.training)

    def add_adapter(self, adapter_name, config=None):
        if config is not None:  # get the lora config
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_hidemoelora_config(config, model_config)   # load config
            self.peft_config[adapter_name] = config # subsititue the original config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "MMOELoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )

        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def finalize_consensus_task(self):
        summary = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "finalize_consensus_task"):
                stats = module.finalize_consensus_task()
                if stats is not None:
                    summary[name] = stats
        return summary

    def export_consensus_state(self):
        state = {}
        for name, module in self.model.named_modules():
            if hasattr(module, "export_consensus_state"):
                module_state = module.export_consensus_state()
                if module_state is not None:
                    state[name] = module_state
        return state

    def load_consensus_state(self, state):
        if not state:
            return
        modules = dict(self.model.named_modules())
        missing = []
        for name, module_state in state.items():
            module = modules.get(name)
            if module is not None and hasattr(module, "load_consensus_state"):
                module.load_consensus_state(module_state)
            else:
                missing.append(name)
        if missing:
            warnings.warn(
                f"Could not restore consensus state for {len(missing)} LoRA modules."
            )


    def _find_and_replace(self, adapter_name):
        """Replace the target `Linear` module with LoRA layer (Linear+LoRA)"""
        lora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]   # all module in raw model
        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name, layer = _get_submodules(self.model, key)

            if isinstance(target, LoraLayer) and isinstance(target, torch.nn.Conv2d):
                target.update_layer_conv2d(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer) and isinstance(target, torch.nn.Embedding):
                target.update_layer_embedding(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )

            elif isinstance(target, LoraLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            else:
                new_module = self._create_new_module(lora_config, adapter_name, target, self.model.training, layer, self.expert_weight)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _create_new_module(self, lora_config, adapter_name, target, training, layer, expert_weight):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "task_embedding_dim": lora_config.task_embedding_dim,
            "expert_num": lora_config.expert_num,
            "cur_task": lora_config.cur_task,
            "consensus_enable": lora_config.consensus_enable,
            "consensus_rank": lora_config.consensus_rank,
            "consensus_rank_shared": lora_config.consensus_rank_shared,
            "consensus_eta": lora_config.consensus_eta,
            "consensus_sample_limit": lora_config.consensus_sample_limit,
            "consensus_samples_per_forward": lora_config.consensus_samples_per_forward,
            "consensus_oversample": lora_config.consensus_oversample,
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            if lora_config.consensus_enable:
                raise ValueError(
                    "Consensus-aware HiDe supports 4-bit or 16-bit training, not 8-bit."
                )
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
            )
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            module_cls = (
                HiDeMOELoraLinear4bit
                if lora_config.consensus_enable
                else Linear4bit
            )
            new_module = module_cls(adapter_name, target.in_features, target.out_features, bias=bias, train_signal=training, layer=layer, expert_weight=expert_weight, **fourbit_kwargs)
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = HiDeMOELoraLinear(adapter_name, in_features, out_features, 
                                                    bias=bias, train_signal=training, layer=layer, expert_weight=expert_weight, **kwargs)

        return new_module

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)


    @staticmethod
    def _prepare_hidemoelora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def _unload_and_optionally_merge(self, merge=True):
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name, _ = _get_submodules(self.model, key)
            except IndexError:
                continue
            if isinstance(target, LoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                else:
                    bias = target.bias is not None
                    if getattr(target, "is_target_conv_1d_layer", False):
                        new_module = Conv1D(target.out_features, target.in_features)
                    else:
                        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    target.merge()
                # self._replace_module(parent, target_name, new_module, target)

            # save any additional trainable modules part of `modules_to_save`
            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model

class HiDeMOELoraLayer(LoraLayer):

    def __init__(self, in_features: int, out_features: int, expert_num: int, cur_task: int, training: bool, layer: int, expert_weight: list):
        
        super().__init__(in_features, out_features)
        self.expert_num = expert_num
        self.cur_task = cur_task
        self.training = training
        self.layer = layer
        self.expert_weight = expert_weight

    
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: HiDeMOELinearA(self.in_features, r, self.expert_num, self.cur_task, self.training, self.layer, self.expert_weight)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: HiDeMOELinearB(r, self.out_features, self.expert_num, self.cur_task, self.training, self.layer, self.expert_weight)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)
    
    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            for i in range(self.expert_num):
                nn.init.normal_(self.lora_A[adapter_name].loraA[i].mlp.weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.lora_B[adapter_name].loraB[i].mlp.weight)

class HiDeMOELoraLinear(nn.Linear, HiDeMOELoraLayer):
    # Lora implemented in a dense layer
    # nn.Linear is the pretrained weights in LLM, MMOELoraLayer is the designed trainable Lora 
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        train_signal: bool = False,
        layer: int = 0,
        expert_weight: list = [],
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.expert_num = kwargs.pop("expert_num", True)
        self.te_dim = kwargs.pop("task_embedding_dim", True)
        self.cur_task = kwargs.pop("cur_task", True)
        self.consensus_enable = kwargs.pop("consensus_enable", False)
        self.consensus_rank = kwargs.pop("consensus_rank", 32)
        self.consensus_rank_shared = kwargs.pop("consensus_rank_shared", 32)
        self.consensus_eta = kwargs.pop("consensus_eta", 0.5)
        self.consensus_sample_limit = kwargs.pop("consensus_sample_limit", 128)
        self.consensus_samples_per_forward = kwargs.pop("consensus_samples_per_forward", 4)
        self.consensus_oversample = kwargs.pop("consensus_oversample", 8)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        HiDeMOELoraLayer.__init__(self, in_features=in_features, 
                               out_features=out_features, 
                               expert_num=self.expert_num,
                               cur_task=self.cur_task,
                               training=train_signal,
                               layer=layer,
                               expert_weight=expert_weight)

        self.layer = layer
        self.expert_weight = expert_weight
        self.training = train_signal
        self._consensus_samples = []
        self._consensus_seen = 0
        self._consensus_task_bases = []
        self._consensus_basis = None
        self._consensus_collect = self.consensus_enable and int(self.layer) != 31
        
        # init the Gate network
        self.lora_router = nn.ModuleDict({})
        self.lora_router.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, self.expert_num, bias=False)}))

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name

    def _collect_consensus_activations(self, x):
        if not self._consensus_collect or self.consensus_sample_limit <= 0:
            return
        if len(self._consensus_samples) >= self.consensus_sample_limit:
            return

        with torch.no_grad():
            rows = x.detach().reshape(-1, x.shape[-1])
            take = min(self.consensus_samples_per_forward, rows.shape[0])
            if take <= 0:
                return
            indices = torch.randperm(rows.shape[0], device=rows.device)[:take]
            sampled = rows.index_select(0, indices).float().cpu()

            for row in sampled:
                if len(self._consensus_samples) >= self.consensus_sample_limit:
                    break
                self._consensus_seen += 1
                self._consensus_samples.append(row)

    def finalize_consensus_task(self):
        if not self.consensus_enable or int(self.layer) == 31:
            return None
        if len(self._consensus_samples) < 2:
            self._consensus_collect = False
            return None

        activations = torch.stack(self._consensus_samples, dim=0)
        max_rank = min(activations.shape[0], activations.shape[1])
        q = min(self.consensus_rank + self.consensus_oversample, max_rank)
        if q <= 0:
            return None

        _, singular_values, right_vectors = torch.pca_lowrank(
            activations, q=q, center=False, niter=2
        )
        rank = min(self.consensus_rank, right_vectors.shape[1])
        task_basis = right_vectors[:, :rank].contiguous()

        overlap = None
        if self._consensus_task_bases:
            previous = self._consensus_task_bases[-1].float()
            overlap = torch.linalg.svdvals(previous.T @ task_basis).mean().item()

        self._consensus_task_bases.append(task_basis.half().cpu())
        stacked = torch.cat([basis.float() for basis in self._consensus_task_bases], dim=1)
        shared_rank = min(self.consensus_rank_shared, stacked.shape[0], stacked.shape[1])
        consensus, _, _ = torch.pca_lowrank(
            stacked, q=shared_rank, center=False, niter=2
        )
        self._consensus_basis = consensus[:, :shared_rank].half().cpu().contiguous()
        self._consensus_basis_runtime = None
        self._consensus_samples = []
        self._consensus_seen = 0
        self._consensus_collect = False

        retained = singular_values[:rank].square().sum()
        total = singular_values.square().sum().clamp_min(1e-12)
        return {
            "samples": int(activations.shape[0]),
            "rank": int(rank),
            "captured_energy": float((retained / total).item()),
            "previous_task_overlap": overlap,
        }

    def export_consensus_state(self):
        if not self.consensus_enable or int(self.layer) == 31:
            return None
        return {
            "task_bases": self._consensus_task_bases,
            "consensus_basis": self._consensus_basis,
        }

    def load_consensus_state(self, state):
        if not state:
            return
        self._consensus_task_bases = [basis.half().cpu() for basis in state.get("task_bases", [])]
        basis = state.get("consensus_basis")
        self._consensus_basis = None if basis is None else basis.half().cpu()
        self._consensus_basis_runtime = None

    def _consensus_filter(self, x):
        if self._consensus_basis is None:
            return x
        runtime = getattr(self, "_consensus_basis_runtime", None)
        if runtime is None or runtime.device != x.device or runtime.dtype != x.dtype:
            runtime = self._consensus_basis.to(device=x.device, dtype=x.dtype)
            self._consensus_basis_runtime = runtime
        projected = torch.matmul(torch.matmul(x, runtime), runtime.T)
        return self.consensus_eta * x + (1.0 - self.consensus_eta) * projected


    def merge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            # for i in range(self.expert_num):
            #     lora_A_weights = self.lora_A[self.active_adapter].loraA[i].mlp.weight
            #     lora_B_weights = self.lora_B[self.active_adapter].loraB[i].mlp.weight
            #     self.weight.data += (
            #         transpose(
            #             lora_B_weights @ lora_A_weights,
            #             self.fan_in_fan_out,
            #         )
            #         * self.scaling[self.active_adapter]
            #     )
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            # for i in range(self.expert_num):
            #     lora_A_weights = self.lora_A[self.active_adapter].loraA[i].mlp.weight
            #     lora_B_weights = self.lora_B[self.active_adapter].loraB[i].mlp.weight
            #     self.weight.data -= (
            #         transpose(
            #             lora_B_weights @ lora_A_weights,
            #             self.fan_in_fan_out,
            #         )
            #         * self.scaling[self.active_adapter]
            #     )
            self.merged = False

    def forward(self, x: torch.Tensor, **kwargs):
        previous_dtype = x.dtype

        if self.training:
            self._collect_consensus_activations(x)
        if self.active_adapter not in self.lora_A.keys():   # No adapter, directly use linear
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:   # No adapter
            if self.r[self.active_adapter] > 0 and self.merged: # merge the adapter to linear
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0:   # general lora process
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            x = x.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)

            if self.training:
                lora_a_output = self.lora_A[self.active_adapter].loraA[self.cur_task](self.lora_dropout[self.active_adapter](x))
                lora_b_output = self.lora_B[self.active_adapter].loraB[self.cur_task](lora_a_output)
                result += lora_b_output * self.scaling[self.active_adapter]
            else:
                if int(self.layer) != 31:
                    if self.consensus_enable and self._consensus_basis is not None:
                        filtered = self._consensus_filter(x)
                        filtered = self.lora_dropout[self.active_adapter](filtered)
                        learned_experts = min(self.cur_task + 1, self.expert_num)
                        for i in range(learned_experts):
                            result += (
                                self.lora_B[self.active_adapter].loraB[i](
                                    self.lora_A[self.active_adapter].loraA[i](filtered)
                                )
                                * self.scaling[self.active_adapter]
                            )
                    else:
                        lora_a_output = self.lora_A[self.active_adapter](
                            self.lora_dropout[self.active_adapter](x)
                        )
                        lora_b_output = self.lora_B[self.active_adapter](lora_a_output)
                        result += lora_b_output * self.scaling[self.active_adapter]
                else:
                    for i in range(len(self.expert_weight)):
                        result += ( # lora process
                            self.lora_B[self.active_adapter].loraB[i](
                                self.lora_A[self.active_adapter].loraA[i](self.lora_dropout[self.active_adapter](x)),
                            )
                            * self.scaling[self.active_adapter]
                            * self.expert_weight[i]
                        )
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result
    


if is_bnb_4bit_available():
    class HiDeMOELoraLinear4bit(bnb.nn.Linear4bit, HiDeMOELoraLayer):
        """HiDe expert LoRA over a frozen bitsandbytes 4-bit base layer."""

        def __init__(self, adapter_name, in_features, out_features, r=0,
                     lora_alpha=1, lora_dropout=0.0, train_signal=False,
                     layer=0, expert_weight=None, **kwargs):
            bnb.nn.Linear4bit.__init__(
                self, in_features, out_features,
                bias=kwargs.get("bias", True),
                compute_dtype=kwargs.get("compute_dtype", torch.float32),
                compress_statistics=kwargs.get("compress_statistics", True),
                quant_type=kwargs.get("quant_type", "nf4"),
            )
            expert_num = kwargs.pop("expert_num")
            cur_task = kwargs.pop("cur_task")
            HiDeMOELoraLayer.__init__(
                self, in_features=in_features, out_features=out_features,
                expert_num=expert_num, cur_task=cur_task,
                training=train_signal, layer=layer,
                expert_weight=expert_weight or [],
            )
            self.consensus_enable = kwargs.pop("consensus_enable", False)
            self.consensus_rank = kwargs.pop("consensus_rank", 32)
            self.consensus_rank_shared = kwargs.pop("consensus_rank_shared", 32)
            self.consensus_eta = kwargs.pop("consensus_eta", 0.5)
            self.consensus_sample_limit = kwargs.pop("consensus_sample_limit", 128)
            self.consensus_samples_per_forward = kwargs.pop("consensus_samples_per_forward", 4)
            self.consensus_oversample = kwargs.pop("consensus_oversample", 8)
            self._consensus_samples = []
            self._consensus_seen = 0
            self._consensus_task_bases = []
            self._consensus_basis = None
            self._consensus_collect = self.consensus_enable and int(self.layer) != 31
            self.weight.requires_grad = False
            self.update_layer(
                adapter_name, r, lora_alpha, lora_dropout,
                kwargs.pop("init_lora_weights", True),
            )
            self.active_adapter = adapter_name

        _collect_consensus_activations = HiDeMOELoraLinear._collect_consensus_activations
        finalize_consensus_task = HiDeMOELoraLinear.finalize_consensus_task
        export_consensus_state = HiDeMOELoraLinear.export_consensus_state
        load_consensus_state = HiDeMOELoraLinear.load_consensus_state
        _consensus_filter = HiDeMOELoraLinear._consensus_filter

        def forward(self, x):
            if self.training:
                self._collect_consensus_activations(x)
            result = bnb.nn.Linear4bit.forward(self, x)
            if self.disable_adapters or self.active_adapter not in self.lora_A:
                return result
            if self.r[self.active_adapter] <= 0:
                return result

            result = result.clone()
            expected_dtype = result.dtype
            adapter_dtype = self.lora_A[self.active_adapter].loraA[0].weight.dtype
            adapter_input = x.to(adapter_dtype)
            if self.training:
                update = self.lora_B[self.active_adapter].loraB[self.cur_task](
                    self.lora_A[self.active_adapter].loraA[self.cur_task](
                        self.lora_dropout[self.active_adapter](adapter_input)
                    )
                )
                result += update.to(expected_dtype) * self.scaling[self.active_adapter]
            elif int(self.layer) != 31:
                filtered = self._consensus_filter(adapter_input)
                filtered = self.lora_dropout[self.active_adapter](filtered)
                for i in range(min(self.cur_task + 1, self.expert_num)):
                    update = self.lora_B[self.active_adapter].loraB[i](
                        self.lora_A[self.active_adapter].loraA[i](filtered)
                    )
                    result += update.to(expected_dtype) * self.scaling[self.active_adapter]
            else:
                for i, weight in enumerate(self.expert_weight):
                    update = self.lora_B[self.active_adapter].loraB[i](
                        self.lora_A[self.active_adapter].loraA[i](adapter_input)
                    )
                    result += update.to(expected_dtype) * self.scaling[self.active_adapter] * weight
            return result


class HiDeMOELinearA(nn.Module):
    '''MMOE based LoRA block'''
    def __init__(self, in_features, out_features, expert_num, cur_task, training, layer, weight) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.cur_task = cur_task
        self.in_features, self.out_features = in_features, out_features
        self.loraA = nn.ModuleList([])
        self.training = training
        self.layer = layer
        self.expert_weight = weight

        assert self.out_features % self.expert_num == 0  # lora rank should be divided by expert number
        self.r = self.out_features // self.expert_num
        
        for _ in range(self.expert_num):
            self.loraA.append(HiDeMOEExpert(self.in_features, self.r))

    
    def forward(self, x):
        '''input x is a vector, return output is a list'''
        if self.training:
            assert 0 <= self.cur_task < self.expert_num, "Invalid current_task value"
            output = self.loraA[self.cur_task](x)
            return output
        else:
            merge_weight = 1.0
            if int(self.layer) != 31:
                temp_mlp = nn.Linear(self.in_features, self.r, bias=False).to(x.device)
                
                fused_weight = torch.zeros((self.r, self.in_features), device=x.device)
            
                for i in range(self.cur_task + 1):
                    fused_weight += merge_weight * self.loraA[i].weight

                with torch.no_grad(): 
                    temp_mlp.weight.copy_(fused_weight)

                output = temp_mlp(x)

                return output
            else:
                outputs = []
                for i in range(self.expert_num):
                    outputs.append(self.loraA[i](x))

                return outputs
                


    
class HiDeMOELinearB(nn.Module):
    '''MMOE based LoRA block'''
    def __init__(self, in_features, out_features, expert_num, cur_task, training, layer, weight) -> None:

        super().__init__()

        self.expert_num = expert_num
        self.cur_task = cur_task
        self.in_features, self.out_features = in_features, out_features
        self.loraB = nn.ModuleList([])
        self.training = training
        self.layer = layer
        self.expert_weight = weight

        assert self.in_features % self.expert_num == 0
        self.r = self.in_features // self.expert_num
        
        for _ in range(self.expert_num):
            self.loraB.append(HiDeMOEExpert(self.r, self.out_features))

    
    def forward(self, x):
        '''input x is a list, return output is also a list'''
        if self.training:
            assert 0 <= self.cur_task < self.expert_num, "Invalid current_task value"
            output = self.loraB[self.cur_task](x)
        else:
            merge_weight = 1.0
            if int(self.layer) != 31:
                temp_mlp = nn.Linear(self.r, self.out_features, bias=False).to(x.device)
                
                fused_weight = torch.zeros((self.out_features, self.r), device=x.device)
            
                for i in range(self.cur_task + 1):
                    fused_weight += merge_weight * self.loraB[i].weight

                with torch.no_grad(): 
                    temp_mlp.weight.copy_(fused_weight)

                output = temp_mlp(x)

                return output
            else:
                outputs = []
                for i in range(self.expert_num):
                    outputs.append(self.loraB[i](x[i]))

                return outputs



class HiDeMOEExpert(nn.Module):

    def __init__(self, in_features, out_features):
        
        super().__init__()

        self.in_features, self.out_features = in_features, out_features
        self.mlp = nn.Linear(self.in_features, self.out_features, bias=False)
        self.weight = self.mlp.weight
    

    def forward(self, x):
        # LoRA A or B block
        y = self.mlp(x)

        return y



class HiDeMOEGate(nn.Module):

    def __init__(self, input_size, expert_num):

        super().__init__()
        # 使用embedding来代替线性层
        self.GateL = nn.Linear(input_size, expert_num, bias=False)
        self.act = nn.Softmax(dim=1)    # 第0维为batch size
    
    def forward(self, x):

        y = self.GateL(x)
        y = self.act(y)

        return y


class HiDeMOERouter(nn.Module):
    """
    Router using tokens choose top-1 experts assignment.

    This router uses the same mechanism as in Switch Transformer (https://arxiv.org/abs/2101.03961) and V-MoE
    (https://arxiv.org/abs/2106.05974): tokens choose their top experts. Items are sorted by router_probs and then
    routed to their choice of expert until the expert's expert_capacity is reached. **There is no guarantee that each
    token is processed by an expert**, or that each expert receives at least one token.

    """

    def __init__(self, config: HiDeMOELoraConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
        self.jitter_noise = config.router_jitter_noise
        self.ignore_padding_tokens = config.router_ignore_padding_tokens
        self.dtype = getattr(torch, config.router_dtype)

    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        self.input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.dtype)

        if self.training and self.jitter_noise > 0:
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        # Shape: [num_groups, tokens_per_group, num_experts]
        self._cast_classifier()
        router_logits = self.classifier(hidden_states)

        # Apply Softmax and cast back to the original `dtype`
        router_probabilities = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(self.input_dtype)
        return router_probabilities, router_logits

    def _cast_classifier(self):
        if not (hasattr(self.classifier, "SCB") or hasattr(self.classifier, "CB")):
            self.classifier = self.classifier.to(self.dtype)

    def forward(self, hidden_states: torch.Tensor) -> Tuple:
        router_probs, router_logits = self._compute_router_probabilities(hidden_states)

        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)

        # Mask tokens outside expert capacity. Sum over each sequence
        token_priority = torch.cumsum(expert_index, dim=-2)
        # mask if the token routed to to the expert will overflow
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_index = expert_index * expert_capacity_mask

        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return expert_index, router_probs, router_logits
