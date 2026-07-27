#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class RunningStatList(nn.Module):
    """A buffer-backed list with ParameterList-compatible state-dict keys."""

    def __init__(self, tensors):
        super().__init__()
        for index, tensor in enumerate(tensors):
            self.register_buffer(str(index), tensor)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(self)[index]
        return self._buffers[str(index)]

    def __iter__(self):
        return iter(self._buffers.values())

    def __len__(self):
        return len(self._buffers)

    def _apply(self, fn):
        super()._apply(fn)
        for name, buffer in self._buffers.items():
            if buffer is not None and torch.is_floating_point(buffer):
                self._buffers[name] = buffer.float()
        return self


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.training = False
        self.cur_task = 0
        self.expert_num = 6

        # Initialize anchors
        self.image_anchors = RunningStatList(
            [torch.zeros(1, 768, dtype=torch.float32) for _ in range(10)]
        )
        self.text_anchors = RunningStatList(
            [torch.zeros(1, 768, dtype=torch.float32) for _ in range(10)]
        )
        self.image_boundary = RunningStatList(
            [torch.zeros(1, dtype=torch.float32) for _ in range(10)]
        )
        self.text_boundary = RunningStatList(
            [torch.zeros(1, dtype=torch.float32) for _ in range(10)]
        )

        self.expert_weight = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]

    def set_cur_task(self, cur_task, expert_num):
        self.cur_task = cur_task
        self.expert_num = expert_num

        # Anchors are running statistics, not optimizer parameters. Updating them
        # both manually and through gradients makes them numerically unstable.
        for param in self.image_anchors.parameters():
            param.requires_grad = False
        for param in self.text_anchors.parameters():
            param.requires_grad = False
        for param in self.image_boundary.parameters():
            param.requires_grad = False
        for param in self.text_boundary.parameters():
            param.requires_grad = False

    def reset_task_statistics(self, task_id):
        if not 0 <= task_id < len(self.image_anchors):
            raise ValueError(f"Invalid task_id={task_id}")
        with torch.no_grad():
            self.image_anchors[task_id].zero_()
            self.text_anchors[task_id].zero_()
            self.image_boundary[task_id].zero_()
            self.text_boundary[task_id].zero_()

    def assert_running_statistics_fp32(self):
        statistics = (
            *self.image_anchors,
            *self.text_anchors,
            *self.image_boundary,
            *self.text_boundary,
        )
        invalid = [stat.dtype for stat in statistics if stat.dtype != torch.float32]
        if invalid:
            raise RuntimeError(
                "HiDe running statistics must remain FP32; "
                f"found dtypes: {sorted(set(map(str, invalid)))}"
            )

    def set_boundary_for_save(self):
        # Kept for compatibility with older entrypoints. Checkpoint collectors
        # must include these frozen running statistics explicitly.
        return

    def get_model(self):
        return self.model

    def set_clip_tokenizer(self, tokenizer):
        self.clip_tokenizer = tokenizer

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

try:
    AutoConfig.register("llava", LlavaConfig, exist_ok=True)
except TypeError:
    try:
        AutoConfig.register("llava", LlavaConfig)
    except ValueError:
        pass

try:
    AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
except ValueError:
    pass
