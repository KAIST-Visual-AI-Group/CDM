"""
DNA (MPRA/Gosai) data loading and tokenization for discrete diffusion.

Vocabulary: A=0, C=1, G=2, T=3 (vocab_size=4).
Mask token is appended as index 4 by the Diffusion model.
"""

import typing

import numpy as np
import torch
import transformers

DNA_ALPHABET = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
INDEX_TO_DNA = {v: k for k, v in DNA_ALPHABET.items()}
_LOOKUP_ARRAY = np.array([INDEX_TO_DNA[i] for i in range(len(INDEX_TO_DNA))])

# ══════════════════════════════════════════════════════════════
#  Tokenizer
# ══════════════════════════════════════════════════════════════

class GosaiTokenizer(transformers.PreTrainedTokenizer):
    def __init__(
        self,
        bos_token='[BOS]',
        eos_token='[EOS]',
        sep_token='[SEP]',
        cls_token='[CLS]',
        pad_token='[PAD]',
        mask_token='[MASK]',
        unk_token='[UNK]',
        **kwargs,
    ):
        self.characters = list('ACGT')
        self._vocab_str_to_int = {
            'A': 0,
            'C': 1,
            'G': 2,
            'T': 3,
            '[MASK]': 4,
            '[UNK]': 5,
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return 4

    def _tokenize(self, text, **kwargs):
        return list(text.upper())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int['[UNK]'])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return ''.join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int

    def batch_decode(self, tokens, **kwargs):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        # Clamp to valid range for lookup (mask/special tokens → 'A' placeholder)
        tokens_clamped = np.clip(tokens, 0, 3)
        detokenized = _LOOKUP_ARRAY[tokens_clamped]
        return [''.join(seq) for seq in detokenized]


# Alias for backward compatibility with duo checkpoints that pickled MPRATokenizer
MPRATokenizer = GosaiTokenizer


def get_tokenizer(config=None):
    return GosaiTokenizer()


# ══════════════════════════════════════════════════════════════
#  Batch tokenize / detokenize utilities
# ══════════════════════════════════════════════════════════════

def batch_dna_tokenize(batch_seq):
    """Convert list of DNA strings to numpy array [B, L]."""
    return np.array([[DNA_ALPHABET[c] for c in seq] for seq in batch_seq])


def batch_dna_detokenize(batch_seq):
    """Convert numpy/tensor array [B, L] to list of DNA strings."""
    if isinstance(batch_seq, torch.Tensor):
        batch_seq = batch_seq.cpu().numpy()
    batch_seq = np.clip(batch_seq, 0, 3)
    detokenized = _LOOKUP_ARRAY[batch_seq]
    return [''.join(seq) for seq in detokenized]
