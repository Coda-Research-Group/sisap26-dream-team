#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import h5py
from scipy.sparse import csr_matrix

from src.task1 import run_task1
from src.task2 import run_task2


def load_task_config(task_description_path):
    """Load task configuration from a config.json file."""
    with open(task_description_path) as f:
        return json.load(f)


def load_data_from_input(input_path, task_cfg):
    """Load data directly from the given HDF5 input file using config."""

    def get_h5_item(f, path):
        if isinstance(path, list):
            cur = f
            for p in path:
                cur = cur[p]
            return cur
        cur = f
        for p in path.split("/"):
            cur = cur[p]
        return cur

    def load_sparse_matrix(h5_group):
        indptr = h5_group["indptr"][:]
        indices = h5_group["indices"][:]
        data = h5_group["data"][:]
        shape = tuple(h5_group.attrs["shape"])
        return csr_matrix((data, indices, indptr), shape=shape)

    with h5py.File(input_path) as f:
        data_item = get_h5_item(f, task_cfg["data"])
        task_name = task_cfg["task"]
        if task_cfg.get("sparse"):
            data = load_sparse_matrix(data_item)
        else:
            data = data_item[()]

        queries = None
        if "queries" in task_cfg:
            q_item = get_h5_item(f, task_cfg["queries"])
            if task_cfg.get("sparse"):
                queries = load_sparse_matrix(q_item)
            else:
                queries = q_item[()]

    return data, queries, task_cfg, task_name


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input HDF5 benchmark file (e.g. benchmark-dev-gooaq-small.h5)",
        type=Path,
    )
    parser.add_argument(
        "--task-description",
        required=True,
        help="Path to the task config JSON file (e.g. config.json)",
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where result HDF5 files will be written",
        type=Path,
    )

    args = parser.parse_args()

    cfg = load_task_config(args.task_description)
    data, queries, task_cfg, task_type = load_data_from_input(args.input, cfg)

    k = task_cfg.get("k", 10)
    dataset = task_cfg["dataset_name"]
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    if task_type == "task1":
        run_task1(data, task_type, k, output_dir, dataset)
    elif task_type == "task2":
        run_task2(data, queries, task_type, k, output_dir, dataset)
    elif task_type == "task3":
        print("Task 3 is not implemented.")
        exit(1)
    else:
        print(f"Unknown task type '{task_type}' in config.")
        exit(1)
