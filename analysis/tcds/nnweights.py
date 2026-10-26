#!/usr/bin/env python3
"""
Read the trained coupling networks (.pth) without PyTorch and evaluate them with NumPy.

The networks are the ones used by the Gazebo controller
(ros/Attachment-Model/src/attachment_model_control.py): PhysicalDistanceNN (tanh) and
EmotionalDistanceNN (SiLU), each with three hidden layers.
"""
import os
import pickle
import zipfile
from collections import OrderedDict

import numpy as np

SRC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "ros", "Attachment-Model", "src")

_DTYPES = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64}


class _Storage:
    def __init__(self, dtype, key):
        self.dtype, self.key = dtype, key


def _rebuild_tensor(storage, offset, size, stride, *args):
    flat = storage
    if not size:
        return flat[offset].copy()
    itemsize = flat.itemsize
    return np.lib.stride_tricks.as_strided(flat[offset:], shape=tuple(size),
                                           strides=tuple(s * itemsize for s in stride)).copy()


def load_state_dict(path):
    """Return {parameter name: ndarray} from a torch.save(state_dict) zip archive."""
    with zipfile.ZipFile(path) as z:
        root = z.namelist()[0].split("/")[0]
        blobs = {}

        class Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "collections" and name == "OrderedDict":
                    return OrderedDict
                if module == "torch._utils" and name == "_rebuild_tensor_v2":
                    return _rebuild_tensor
                if module == "torch" and name in _DTYPES:
                    return _DTYPES[name]
                raise pickle.UnpicklingError(f"unexpected {module}.{name}")

            def persistent_load(self, pid):
                _, dtype, key, _location, _numel = pid
                if key not in blobs:
                    raw = z.read(f"{root}/data/{key}")
                    blobs[key] = np.frombuffer(raw, dtype=dtype)
                return blobs[key]

        return dict(Unpickler(z.open(f"{root}/data.pkl")).load())


def _silu(x):
    return x / (1.0 + np.exp(-x))


class MLP:
    """Four linear layers with tanh (physical) or SiLU (emotional) activations."""

    def __init__(self, state, activation):
        self.layers = [(state[f"fc{i}.weight"].astype(float), state[f"fc{i}.bias"].astype(float))
                       for i in range(1, 5)]
        self.act = np.tanh if activation == "tanh" else _silu

    def __call__(self, x):
        h = np.asarray(x, dtype=float)
        for i, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if i < 3:
                h = self.act(h)
        return h[..., 0]


def load_networks(src_dir=SRC_DIR):
    """The four trained networks: {(kind, agent): MLP}, kind in physical/emotional."""
    nets = {}
    for kind, act in (("physical", "tanh"), ("emotional", "silu")):
        for agent in ("parent", "child"):
            state = load_state_dict(os.path.join(src_dir, f"{kind}_{agent}_coupling_model.pth"))
            nets[(kind, agent)] = MLP(state, act)
    return nets


if __name__ == "__main__":
    for key, net in load_networks().items():
        shapes = [w.shape for w, _ in net.layers]
        print(key, shapes, net(np.zeros(shapes[0][1])))
