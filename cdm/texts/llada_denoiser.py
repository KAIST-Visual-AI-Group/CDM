import copy
import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from cdm.texts.modeling_llada import LLaDAModelLM

# this tokenizer gives warning about parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add the parent directory to Python path to access dplm
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


class TransformerTwistHead(nn.Module):
    """Small transformer encoder over token-level hidden states → scalar.

    Applies an optional input projection, a few TransformerEncoder layers
    with key-padding masking from ``attention_mask``, masked mean pooling,
    then a zero-initialized linear layer to a single scalar per sequence.
    """

    def __init__(
        self,
        backbone_hidden_size: int,
        hidden_size: int,
        n_layers: int,
        n_heads: int,
        ffn_size: int,
        dropout: float,
    ):
        super().__init__()
        if backbone_hidden_size != hidden_size:
            self.input_proj = nn.Linear(backbone_hidden_size, hidden_size)
        else:
            self.input_proj = nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=ffn_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, hidden_states, attention_mask=None):
        # hidden_states: [B, L, H_backbone]  (float32)
        x = self.input_proj(hidden_states)

        key_padding_mask = None
        if attention_mask is not None:
            # TransformerEncoder expects True at positions to ignore.
            key_padding_mask = attention_mask == 0

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        if attention_mask is not None:
            w = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
            denom = w.sum(dim=1).clamp(min=1.0)
            pooled = (x * w).sum(dim=1) / denom
        else:
            pooled = x.mean(dim=1)

        return self.out(pooled).squeeze(-1)


class LLaDADenoiser:

    def __init__(self, device = 'cuda', chunk_b_size=None, call_chunk_size=None):
        self.device = device
        self.chunk_b_size = chunk_b_size
        self.call_chunk_size = call_chunk_size

        self.name = "LLaDA-8B-Instruct"
        # Load via the local modeling/configuration files instead of the
        # remote `trust_remote_code=True` path. The pretrained weights are
        # still pulled from the HF hub (or cache) under 'GSAI-ML/<name>'.
        self.model = LLaDAModelLM.from_pretrained(
            'GSAI-ML/' + self.name,
            torch_dtype=torch.bfloat16,
        ).to(device).eval()

        self.vocab_size = self.model.config.embedding_size or self.model.config.vocab_size
        self.logits_dtype = next(self.model.parameters()).dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            'GSAI-ML/'+self.name, trust_remote_code=True
        )
        # self.model.to(self.device).eval()

        self.mask_token = 126336
        self.length = None # works with flexible lengths, depending on input_seq in sampling

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("LLaDA tokenizer is missing both pad_token_id and eos_token_id")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id == self.mask_token:
            raise ValueError(
                "LLaDA tokenizer pad_token_id must differ from mask_token for batched inference"
            )

        # Optional twist head — attached lazily via `attach_twist_head`.
        # When present, `forward(..., return_value=True)` projects the
        # backbone's last hidden state to a scalar value ψ^θ(x_t) used by
        # the CDM twist. The backbone is shared with the LM
        # forward path, so a single backbone pass yields both logits and
        # the value.
        self.head = None
        self.head_type = None
        self.mlp_n_layers = None
        self.mlp_hidden_size = None
        self.tf_n_layers = None
        self.tf_n_heads = None
        self.tf_hidden_size = None
        self.tf_ffn_size = None
        self.tf_dropout = None

        self.ema_heads: list[nn.Module] = []
        self.ema_decays: list[float] = []

    def attach_twist_head(
        self,
        head_type="linear",
        mlp_n_layers=3,
        mlp_hidden_size=256,
        tf_n_layers=2,
        tf_n_heads=8,
        tf_hidden_size=512,
        tf_ffn_size=2048,
        tf_dropout=0.1,
    ):
        """Install a trainable scalar reward head on top of the backbone.

        Backbone parameters are not touched (they are kept frozen by callers
        that train this head). Returns `self` so the call chains.
        """
        hidden_size = self.model.config.d_model

        if head_type == "linear":
            head = nn.Linear(hidden_size, 1)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        elif head_type == "mlp":
            layers = []
            in_dim = hidden_size
            for _ in range(mlp_n_layers - 1):
                lin = nn.Linear(in_dim, mlp_hidden_size)
                nn.init.kaiming_uniform_(lin.weight, nonlinearity="relu")
                nn.init.zeros_(lin.bias)
                layers.extend([lin, nn.ReLU()])
                in_dim = mlp_hidden_size
            final = nn.Linear(in_dim, 1)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            layers.append(final)
            head = nn.Sequential(*layers)
        elif head_type == "transformer":
            head = TransformerTwistHead(
                backbone_hidden_size=hidden_size,
                hidden_size=tf_hidden_size,
                n_layers=tf_n_layers,
                n_heads=tf_n_heads,
                ffn_size=tf_ffn_size,
                dropout=tf_dropout,
            )
        else:
            raise ValueError(
                f"Unknown head_type '{head_type}'. Choose 'linear', 'mlp', or 'transformer'."
            )

        self.head = head.to(self.device)
        self.head_type = head_type
        self.mlp_n_layers = mlp_n_layers
        self.mlp_hidden_size = mlp_hidden_size
        self.tf_n_layers = tf_n_layers
        self.tf_n_heads = tf_n_heads
        self.tf_hidden_size = tf_hidden_size
        self.tf_ffn_size = tf_ffn_size
        self.tf_dropout = tf_dropout
        return self

    def _unwrap_head(self):
        if isinstance(self.head, torch.nn.parallel.DistributedDataParallel):
            return self.head.module
        return self.head

    def attach_ema_heads(self, decays):
        if self.head is None:
            raise RuntimeError("attach_twist_head must be called before attach_ema_heads")
        head_module = self._unwrap_head()

        self.ema_heads = list()
        self.ema_decays = list(decays)

        for _ in self.ema_decays:
            ema_head = copy.deepcopy(head_module).to(self.device).eval()
            for p in ema_head.parameters():
                p.requires_grad = False
            self.ema_heads.append(ema_head)
        return self

    @torch.no_grad()
    def update_ema_heads(self):
        if not self.ema_heads:
            return
        head_module = self._unwrap_head()

        for ema_head, decay in zip(self.ema_heads, self.ema_decays):
            for p_ema, p in zip(ema_head.parameters(), head_module.parameters()):
                if p.requires_grad:
                    p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

    def apply_lm_head(self, hidden_b):
        cfg = self.model.model.config
        transformer = self.model.model.transformer
        if cfg.weight_tying:
            logits = F.linear(hidden_b, transformer.wte.weight, None)
        else:
            logits = transformer.ff_out(hidden_b)
        if cfg.scale_logits:
            logits.mul_(1.0 / math.sqrt(cfg.d_model))
        return logits

    def _apply_head(self, head_module, hidden_states, attention_mask):
        if self.head_type == "transformer":
            return head_module(hidden_states, attention_mask=attention_mask)

        if attention_mask is not None:
            weights = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
            denom = weights.sum(dim=1).clamp(min=1.0)
            pooled = (hidden_states * weights).sum(dim=1) / denom
        else:
            pooled = hidden_states.mean(dim=1)
        return head_module(pooled).squeeze(-1)

    def _set_length(self, length):
        self.length = length
        return

    # llada expects this prompt format for Instruct model 
    def apply_prompt_template(self, prompt_list):

        if "Instruct" not in self.name:
            return prompt_list
        
        # process for instruct model 
        new_prompts = []
        
        # Add special tokens for the Instruct model. The Base model does not require the following two lines.
        for p in prompt_list:
            m = [{"role": "user", "content": p}, ]
            new_prompt = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            new_prompts.append(new_prompt)

        return new_prompts

    def encode_prompt_list(self, prompt_list):
        input_ids = self.tokenizer(
            prompt_list,
            add_special_tokens=False,
            padding=True,
        )['input_ids']

        input_ids = torch.tensor(input_ids).to(self.device)

        return input_ids 

    # adds length number of 
    def add_mask_tokens(self, input_ids):
        x = torch.full((input_ids.shape[0], input_ids.shape[1] + self.length), self.mask_token, dtype=torch.long).to(self.device)
        x[:, :input_ids.shape[1]] = input_ids.clone()
        return x

    def get_input_ids_template(self, prompt_list):
        prompt_list = self.apply_prompt_template(prompt_list)
        input_ids = self.encode_prompt_list(prompt_list)

        return input_ids

    # apply template, encode and add mask tokens of length self.length
    def prepare_init_seq(self, prompt_list, gen_length):
        self._set_length(gen_length)

        prompt_list = self.apply_prompt_template(prompt_list)
        input_ids = self.encode_prompt_list(prompt_list)
        init_seq = self.add_mask_tokens(input_ids)

        return init_seq

    def __call__(self, input_seq, attention_mask=None, return_logits=True,
                 return_value=False, return_ema_value=False, ema_idx=0, return_hidden=False):
        """Single backbone forward pass returning logits / live value /
        EMA value (or any subset).

        The backbone runs at most ONCE per call. When both `return_value`
        and `return_ema_value` are True, the live head and the chosen EMA
        head are both applied to the same hidden states — so EMA-based
        weighting in the negative loss does not require a second backbone
        forward.

        Args:
            input_seq: [B, L] int64 token ids.
            attention_mask: optional [B, L] tensor marking non-pad positions.
            return_logits: include the LM logits in the output.
            return_value: include the live twist value ψ^θ(x_t).
                Requires `attach_twist_head(...)` first.
            return_ema_value: include ψ^{ema}_θ(x_t) from the chosen EMA head.
                Requires `attach_ema_heads(...)` first.
            ema_idx: which EMA tracker to use (default 0).

        Returns the requested outputs in the order [logits, value, ema_value],
        unwrapped to a scalar when only one is requested.
        """
        if not (return_logits or return_value or return_ema_value or return_hidden):
            raise ValueError(
                "At least one of return_logits / return_value / return_ema_value / return_hidden must be True"
            )
        if return_value and self.head is None:
            raise RuntimeError(
                "twist head not attached; call `attach_twist_head(...)` first"
            )
        if return_ema_value:
            if not self.ema_heads:
                raise RuntimeError(
                    "EMA heads not attached; call `attach_ema_heads(...)` first"
                )
            if not (0 <= ema_idx < len(self.ema_heads)):
                raise IndexError(
                    f"ema_idx={ema_idx} out of range (have {len(self.ema_heads)})"
                )

        need_hidden_for_head = return_value or return_ema_value
        need_hidden = need_hidden_for_head or return_hidden

        B = input_seq.shape[0]
        L = input_seq.shape[1]
        cs = self.call_chunk_size if self.call_chunk_size is not None else B

        logits_out = None
        if return_logits:
            logits_out = torch.empty(B, L, self.vocab_size, dtype=self.logits_dtype, device=input_seq.device)

        hidden_cache_dtype = self.logits_dtype if (return_hidden and not need_hidden_for_head) else torch.float32
        hidden_out = None
        if need_hidden:
            d_model = self.model.model.config.d_model
            hidden_out = torch.empty(B, L, d_model, dtype=hidden_cache_dtype, device=input_seq.device)

        for b0 in range(0, B, cs):
            chunk = input_seq[b0 : b0 + cs]
            attn_chunk = None if attention_mask is None else attention_mask[b0 : b0 + cs]
            net_out = self.model(chunk, attention_mask=attn_chunk, return_last_hidden_only=need_hidden)

            if return_logits:
                logits_out[b0 : b0 + chunk.shape[0]].copy_(net_out.logits)
            if need_hidden:
                hidden_out[b0 : b0 + chunk.shape[0]].copy_(net_out.hidden_states[-1])
            del net_out

        values_out = None
        ema_values_out = None
        if need_hidden_for_head:
            if return_value:
                values_out = self._apply_head(self.head, hidden_out, attention_mask)
            if return_ema_value:
                with torch.no_grad():
                    ema_values_out = self._apply_head(self.ema_heads[ema_idx], hidden_out, attention_mask)
            if not return_hidden:
                del hidden_out
                hidden_out = None
            else:
                hidden_out = hidden_out.to(self.logits_dtype)

        outs = list()
        if return_logits:
            outs.append(logits_out)
        if return_value:
            outs.append(values_out)
        if return_ema_value:
            outs.append(ema_values_out)
        if return_hidden:
            outs.append(hidden_out)
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)



class LoRALinear(nn.Module):
    """
    Replaces a standard nn.Linear layer with a LoRA-adapted linear layer.
    """
    def __init__(self, linear_layer: nn.Linear, rank: int, lora_alpha: float = 1.0):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        if lora_alpha is None:
            lora_alpha = rank

        self.scaling = lora_alpha / rank
        
        # 1. Keep the original pre-trained weight matrix frozen
        self.linear = linear_layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        device = self.linear.weight.device
        dtype = self.linear.weight.dtype
            
        # 2. Inject trainable rank decomposition matrices A and B
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, self.rank, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(self.rank, self.out_features, device=device, dtype=dtype))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize A with random Gaussian and B with zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)) 
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # h = W_0 x + B A x * (\alpha / r)
        frozen_out = self.linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return frozen_out + lora_out

def inject_lora_layers(model: nn.Module, rank: int, lora_alpha: float, target_modules: list):
    """
    Recursively replaces nn.Linear layers matching the target_modules list with LoRALinear.
    """
    for name, module in model.named_children():
        # Check if the module name contains any of the target module strings
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            # Replace the layer
            setattr(model, name, LoRALinear(module, rank, lora_alpha))
        else:
            # Recurse down the children
            inject_lora_layers(module, rank, lora_alpha, target_modules)
            
    # Ensure only LoRA parameters require gradients
    for name, param in model.named_parameters():
        if 'lora_A' not in name and 'lora_B' not in name:
            param.requires_grad = False
