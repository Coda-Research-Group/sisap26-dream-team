import logging
import time

import faiss
import numpy as np

from .utils import mips_transformation, store_results

logger = logging.getLogger(__name__)


def run_task2(data, queries, task, k, output_dir, dataset="unknown"):
    print(f"Running {task} on {dataset}")

    data = mips_transformation(np.array(data), is_query=False)
    queries = mips_transformation(np.array(queries), is_query=True)

    # ids = np.arange(1, len(data) + 1)  # 1-indexed IDs for the data points

    n, d = data.shape

    M = 128
    ef_construction = 500
    index_identifier = f"HNSW{M}"
    index = faiss.index_factory(d, index_identifier, faiss.METRIC_L2)
    index.hnsw.efConstruction = ef_construction

    print(f"Training index on {data.shape} with {data.dtype}")
    start = time.time()
    index.add(data)
    elapsed_build = time.time() - start
    print(f"Done training in {elapsed_build}s.")
    
    ef_searches = [10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 750, 1000]

    for ef_search in ef_searches:
        print(f"Starting search on {queries.shape} with efSearch={ef_search}")
        start = time.time()
        index.hnsw.efSearch = ef_search
        D, I = index.search(queries, k)
        elapsed_search = time.time() - start
        print(f"Done searching in {elapsed_search}s.")

        I += 1  # FAISS is 0-indexed, groundtruth is 1-indexed

        identifier = f"index=({index_identifier},efConstruction={ef_construction}),query=(efSearch={ef_search})"

        store_results(
            output_dir / f"{identifier}.h5",
            "faissHNSW",
            dataset,
            task,
            D,
            I,
            elapsed_build,
            elapsed_search,
            identifier,
        )
