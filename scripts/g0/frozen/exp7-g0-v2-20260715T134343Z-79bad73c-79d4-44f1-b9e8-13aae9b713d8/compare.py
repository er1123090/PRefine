"""Verified exact-comparison helper."""
import hashlib


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def exact_equal(left, right):
    return isinstance(left, bytes) and isinstance(right, bytes) and left == right
