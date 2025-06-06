#!/usr/bin/env python3
"""
pp_topk_tenseal.py  – Two-server top-k with TenSEAL BFV and HMAC auth
Dependencies: tenseal, cryptography
"""

import tenseal as ts
from cryptography.hazmat.primitives import hashes, hmac
import os, random
import torch

# ─────────────────── TenSEAL BFV context ───────────────────
context = ts.context(
    ts.SCHEME_TYPE.BFV,
    poly_modulus_degree=8192,
    plain_modulus=65537
)
context.generate_galois_keys()

# ─────────────────── HMAC helpers ───────────────────
HMAC_KEY = os.urandom(32)
def mac(data: bytes) -> bytes:
    h = hmac.HMAC(HMAC_KEY, hashes.SHA256()); h.update(data); return h.finalize()
def mac_ok(data: bytes, tag: bytes) -> bool:
    h = hmac.HMAC(HMAC_KEY, hashes.SHA256()); h.update(data)
    try: h.verify(tag); return True
    except: return False

# ─────────────────── KeyServer (S2) ───────────────────
class KeyServer:
    def __init__(self, l_bits=16):
        self.l = l_bits
    def handle_cmp(self, payload: bytes, tag: bytes):
        if not mac_ok(payload, tag): raise ValueError("Bad MAC at S2")
        ct_z = ts.BFVVector.load(context, payload)
        val = ct_z.decrypt()[0]
        bit = 1 if (val >> self.l) else 0
        ct_bit = ts.bfv_vector(context, [bit])
        resp = ct_bit.serialize()
        return resp, mac(resp)

# ─────────────────── DataServer (S1) ───────────────────
class DataServer:
    def __init__(self, ks: KeyServer, l_bits=16):
        self.S2 = ks; self.l = l_bits; self.cts = []
    def load(self, enc_list): self.cts = enc_list
    def _compare_swap(self, i, j):
        a, b = self.cts[i], self.cts[j]
        two_l = ts.bfv_vector(context, [1 << self.l])
        z = a - b + two_l
        req = z.serialize(); tag = mac(req)
        resp, tagr = self.S2.handle_cmp(req, tag)
        if not mac_ok(resp, tagr): raise ValueError("Bad MAC resp")
        ct_bit = ts.BFVVector.load(context, resp)
        inv = ts.bfv_vector(context, [1]) - ct_bit
        new_a = ct_bit * a + inv * b
        new_b = ct_bit * b + inv * a
        self.cts[i], self.cts[j] = new_a, new_b
    def sort_and_topk(self, k):
        n = len(self.cts)
        size = 1 << ((n-1).bit_length())
        def bitonic(lo, cnt, up):
            if cnt>1:
                m=cnt//2; bitonic(lo,m,True); bitonic(lo+m,m,False)
                merge(lo,cnt,up)
        def merge(lo,cnt,up):
            if cnt>1:
                m=cnt//2
                for x in range(lo, lo+cnt-m):
                    if x<n and x+m<n:
                        (self._compare_swap(x,x+m) if up else self._compare_swap(x+m,x))
                merge(lo,m,up); merge(lo+m,cnt-m,up)
        bitonic(0, size, True)
        vals = [c.decrypt()[0] for c in self.cts]
        idxs = sorted(range(n), key=lambda i: vals[i], reverse=True)
        return [(i, vals[i]) for i in idxs[:k]]

# ─────────────────── Demo ───────────────────
def topkHE(n=10, k=1, l_bits=16):
    ratings = torch.tensor([random.randint(0,2**l_bits-1) for _ in range(n)])
    encs = [ts.bfv_vector(context, [r]) for r in ratings]
    ks = KeyServer(l_bits)
    ds = DataServer(ks, l_bits)
    ds.load(encs)
    topk = ds.sort_and_topk(k)
    print(topk)
    return topk
    

if __name__=="__main__":
    for i in range(256):
        topkHE()