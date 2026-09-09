#!/usr/bin/env python3
"""Independent generator + verifier for the negacyclic polymul solution.

Writes a point directory in exactly the layout the harness (main.cpp) reads,
computes the expected answer with Python big integers, and checks what the
solution left in out/. This is a standalone oracle for local testing — not the
platform bundle — so it can be trusted as a second opinion on correctness.

    python3 oracle.py make  <dir> <N> <W> [seeds]
    python3 oracle.py verify <dir>
"""
import json
import os
import random
import struct
import sys


def ceil_div(a, b):
    return (a + b - 1) // b


def is_prime(n, rounds=40):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    rng = random.Random(0xC0FFEE)
    for _ in range(rounds):
        a = rng.randrange(2, n - 1)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def negacyclic_prime(N, W):
    """The largest prime below 2^W with q == 1 (mod 2N), as the spec derives q."""
    m = 2 * N
    q = ((1 << W) - 1)
    q -= (q - 1) % m  # largest value <= 2^W - 1 congruent to 1 mod 2N
    while q > 1:
        if is_prime(q):
            return q
        q -= m
    raise RuntimeError("no prime found")


def to_limbs(x, L):
    return [(x >> (32 * i)) & 0xFFFFFFFF for i in range(L)]


def from_limbs(limbs):
    return sum(int(v) << (32 * i) for i, v in enumerate(limbs))


def pack(values, L):
    out = bytearray()
    for x in values:
        for limb in to_limbs(x, L):
            out += struct.pack("<I", limb)
    return bytes(out)


def unpack(raw, N, L):
    words = struct.unpack("<%dI" % (N * L), raw)
    return [from_limbs(words[i * L:(i + 1) * L]) for i in range(N)]


def negacyclic_mul(a, b, q, N):
    c = [0] * N
    for i in range(N):
        for j in range(N):
            t = i + j
            p = a[i] * b[j] % q
            if t < N:
                c[t] = (c[t] + p) % q
            else:
                c[t - N] = (c[t - N] - p) % q
    return c


def make(root, N, W, seeds):
    L = ceil_div(W, 32)
    q = negacyclic_prime(N, W)
    os.makedirs(os.path.join(root, "point"), exist_ok=True)
    with open(os.path.join(root, "point", "q.bin"), "wb") as f:
        f.write(pack([q], L))
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump({"N": N, "W": W, "L": L, "cases": seeds}, f)

    for s in range(seeds):
        rng = random.Random(1000 + s)
        a = [rng.randrange(0, q) for _ in range(N)]
        b = [rng.randrange(0, q) for _ in range(N)]
        case = os.path.join(root, "cases", "%06d" % s)
        exp = os.path.join(root, "expected", "%06d" % s)
        os.makedirs(case, exist_ok=True)
        os.makedirs(exp, exist_ok=True)
        with open(os.path.join(case, "a.bin"), "wb") as f:
            f.write(pack(a, L))
        with open(os.path.join(case, "b.bin"), "wb") as f:
            f.write(pack(b, L))
        c = negacyclic_mul(a, b, q, N)
        with open(os.path.join(exp, "c.bin"), "wb") as f:
            f.write(pack(c, L))
    print("made N=%d W=%d L=%d q=%#x cases=%d at %s" % (N, W, L, q, seeds, root))


def verify(root):
    with open(os.path.join(root, "manifest.json")) as f:
        m = json.load(f)
    N, L, cases = m["N"], m["L"], m["cases"]
    ok = True
    for s in range(cases):
        got_path = os.path.join(root, "out", "%06d" % s, "c.bin")
        exp_path = os.path.join(root, "expected", "%06d" % s, "c.bin")
        if not os.path.exists(got_path):
            print("case %06d: no output at %s" % (s, got_path))
            ok = False
            continue
        got = unpack(open(got_path, "rb").read(), N, L)
        exp = unpack(open(exp_path, "rb").read(), N, L)
        if got == exp:
            print("case %06d: OK (%d coefficients match)" % (s, N))
        else:
            ok = False
            diffs = [(i, got[i], exp[i]) for i in range(N) if got[i] != exp[i]]
            print("case %06d: MISMATCH in %d/%d coefficients" % (s, len(diffs), N))
            for i, g, e in diffs[:5]:
                print("   c[%d]: got %#x  want %#x" % (i, g, e))
    print("VERIFY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "make":
        seeds = int(sys.argv[5]) if len(sys.argv) > 5 else 1
        make(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), seeds)
    elif len(sys.argv) == 3 and sys.argv[1] == "verify":
        verify(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(2)
