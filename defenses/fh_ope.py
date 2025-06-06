# fh_ope_argmax_li2021_fixed.py

import torch
import hmac
import hashlib
import os
import secrets
from collections import defaultdict
from typing import List, Optional
import torch.nn.functional as F
import numpy as np

class FH_OPE:
    SCALE  = 2**31 - 1
    OFFSET = 0

    def __init__(self, key: Optional[bytes] = None):
        # Use Optional[bytes] instead of “bytes | None”
        self.key = key or os.urandom(16)

        # Pick a random slope between [2^30, 2^31 − 1)
        low, high = 2**30, 2**31 - 1
        self._slope = secrets.randbelow(high - low) + low

        # A counter per plaintext code
        self._counters = defaultdict(int)

    def _prf(self, pt_code: int, dup_idx: int) -> int:
        """
        Pseudorandom function: HMAC(key, pt_code ∥ dup_idx), truncated to 64 bits,
        then mod _slope.
        """
        msg = pt_code.to_bytes(4, "big") + dup_idx.to_bytes(4, "big")
        h   = hmac.new(self.key, msg, hashlib.sha256).digest()
        # Take the first 8 bytes, interpret as big‐endian integer, mod _slope
        return int.from_bytes(h[:8], "big") % self._slope

    def encrypt_row(self, row: torch.Tensor) -> List[int]:
        """
        Input: 1‐D tensor of probabilities in [0,1].
        Output: A list of integers (one per class) representing the OPE ciphertexts.
        """
        if row.ndim != 1:
            raise ValueError("encrypt_row expects a 1‐D tensor")

        if row.min() < 0 or row.max() > 1:
            raise ValueError("values must lie in [0,1]")

        cts: List[int] = []
        for v in row.tolist():
            # Convert probability ∈ [0,1] to an integer in [0, SCALE] by rounding
            pt_code = int(round(v * self.SCALE))

            # How many times we’ve already seen this exact pt_code
            k = self._counters[pt_code]
            self._counters[pt_code] += 1

            # Ciphertext = slope * pt_code + PRF(pt_code, k)
            ct = self._slope * pt_code + self._prf(pt_code, k)
            cts.append(ct)

        return cts


def random_confidence_scores(batch_size: int, num_classes: int, device: str = 'cpu') -> torch.Tensor:
    logits = torch.randn(batch_size, num_classes, device=device)
    true_labels = torch.randint(0, num_classes, (batch_size,), device=device)
    boost = torch.abs(torch.randn(batch_size, device=device)) * 2.0
    logits[torch.arange(batch_size), true_labels] += boost
    return F.softmax(logits, dim=1)


def encrypt_confidence_batch(
    conf_batch: torch.Tensor,
    ope: FH_OPE,
    device: Optional[torch.device] = None
) -> torch.Tensor:

    batch_size, num_classes = conf_batch.shape
    ciphertexts = torch.empty((batch_size, num_classes), dtype=torch.long)

    # Loop row by row
    for i in range(batch_size):
        # Detach and move to CPU because encrypt_row expects a CPU tensor
        row_i_cpu = conf_batch[i].detach().cpu()
        ct_list = ope.encrypt_row(row_i_cpu)  # List[int] of length num_classes

        # Write into the i-th row of ciphertexts
        ciphertexts[i] = torch.tensor(ct_list, dtype=torch.long)

    if device is not None:
        ciphertexts = ciphertexts.to(device)
    return ciphertexts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
num_classes = 10

conf_batch = random_confidence_scores(batch_size, num_classes, device)
ope = FH_OPE()
encrypted_tensor = encrypt_confidence_batch(conf_batch, ope, device=None)


