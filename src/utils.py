import os
from pathlib import Path

import faiss
import h5py
import numpy as np


def merge_knn_results(
    distances_all: list[np.ndarray], indices_all: list[np.ndarray], keep_max: bool
) -> tuple[np.ndarray, np.ndarray]:
    """DOES NOT WORK CORRECTLY WITH DUPLICATES!"""
    return faiss.merge_knn_results(
        np.stack(distances_all), np.stack(indices_all), keep_max
    )


def store_results(dst, algo, dataset, task, D, I, buildtime, querytime, params):
    os.makedirs(Path(dst).parent, exist_ok=True)
    f = h5py.File(dst, "w")
    f.attrs["algo"] = algo
    f.attrs["dataset"] = dataset
    f.attrs["task"] = task
    f.attrs["buildtime"] = buildtime
    f.attrs["querytime"] = querytime
    f.attrs["params"] = params
    f.create_dataset("knns", I.shape, dtype=I.dtype)[:] = I
    f.create_dataset("dists", D.shape, dtype=D.dtype)[:] = D
    f.close()


def _mips_transformation_data(data: np.ndarray) -> np.ndarray:
    max_norm = float(np.max(np.linalg.norm(data, axis=1)))
    data_scaled = data / max_norm if max_norm > 0.0 else np.zeros_like(data)
    norms = np.sum(data_scaled**2, axis=1)
    extra_col = np.sqrt(np.maximum(1.0 - norms, 0.0))
    return np.column_stack([data_scaled, extra_col])


def _mips_transformation_queries(queries: np.ndarray) -> np.ndarray:
    extra_dim = np.zeros((queries.shape[0], 1), dtype=queries.dtype)
    return np.hstack((queries, extra_dim))


def mips_transformation(X: np.ndarray, is_query: bool = False) -> np.ndarray:
    if is_query:
        return _mips_transformation_queries(X)

    return _mips_transformation_data(X)
