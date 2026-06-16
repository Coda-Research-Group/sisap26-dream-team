import gc
import logging
import time

import faiss
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from .utils import merge_knn_results, store_results

logger = logging.getLogger(__name__)


class _LMIDataset(Dataset):
    def __init__(self, X: Tensor, y: Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.X[index], self.y[index]


class LearnedMetricIndexIVF:
    kmeans: faiss.Kmeans
    indices: list[faiss.Index]
    n_buckets: int
    dim: int
    epochs: int
    lr: float
    model: nn.Sequential
    loss_fn: CrossEntropyLoss
    optimizer: Adam
    nlist: int
    quantize: bool

    def __init__(
        self,
        dim: int,
        n_buckets: int = 128,
        nlist: int = 32,
        hidden_layers: list[int] = [],
        epochs: int = 5,
        lr: float = 0.00098,
        quantize: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.n_buckets = n_buckets
        self.dim = dim
        self.epochs = epochs
        self.lr = lr
        self.nlist = nlist
        self.quantize = quantize

        self._kmeans = faiss.Kmeans(
            dim, n_buckets, niter=10, verbose=False, spherical=True
        )

        factory_string = f"IVF{nlist},SQ8"

        self._indices = [
            faiss.index_factory(dim, factory_string, faiss.METRIC_INNER_PRODUCT)
            for _ in range(n_buckets)
        ]

        input_dim = dim
        layers = []
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, n_buckets))

        self.model = nn.Sequential(*layers)

        self.loss_fn = CrossEntropyLoss()
        self.optimizer = Adam(params=self.model.parameters(), lr=self.lr)

    @classmethod
    def trainable(cls) -> bool:
        return True

    def train(self, data: np.ndarray) -> None:
        logger.debug("Training k-means...")
        self._kmeans.train(data)
        logger.debug("Assigning data to clusters...")
        _, assignment = self._kmeans.assign(data)

        logger.debug("Training bucket indices...")
        for i in range(self.n_buckets):
            bucket_data = data[assignment == i]
            if bucket_data.shape[0] > 0:
                self._indices[i].train(bucket_data)  # pyright: ignore[reportCallIssue]

        train_loader = DataLoader(
            dataset=_LMIDataset(data, assignment), batch_size=128, shuffle=True
        )

        # Train the model
        logger.debug("Training the model...")
        self.model.train()

        logger.debug(f"Epochs: {self.epochs}")

        for epoch in range(self.epochs):
            for X_batch, y_batch in train_loader:
                loss = self.loss_fn(self.model(X_batch.to(torch.float32)), y_batch)

                # Do the backpropagation
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            logger.debug(f"Epoch {epoch} | Loss {loss.item():.5f}")  # type: ignore

        if self.quantize:
            self.model = torch.ao.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )

    def _predict(self, X: Tensor, top_k: int) -> tuple[Tensor, Tensor]:
        self.model.eval()
        with torch.no_grad():
            logits = F.softmax(self.model(X), dim=-1)

        return logits.topk(top_k)

    def add(self, data: np.ndarray, ids: np.ndarray):
        logger.debug("Predicting buckets for data to add...")
        classes = self._predict(torch.from_numpy(data), 1)[1].reshape(-1).numpy()

        logger.debug("Adding data to bucket indices...")
        for i in range(self.n_buckets):
            class_mask = classes == i
            bucket_data = data[class_mask]
            bucket_ids = ids[class_mask]

            if bucket_data.shape[0] > 0:
                self._indices[i].add_with_ids(
                    bucket_data, bucket_ids
                )  # pyright: ignore[reportCallIssue]

    def search(
        self,
        queries: np.ndarray,
        k: int = 16,
        nprobe: int = 4,
        ivf_nprobe_total: int = 8,
        discard: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        remaining_subclusters = np.full(
            (queries.shape[0],), ivf_nprobe_total, dtype=np.uint32
        )

        logger.debug("Predicting buckets for queries...")
        bucket_probabilities, buckets_to_visit = self._predict(
            torch.from_numpy(queries), nprobe
        )
        bucket_weights = bucket_probabilities.clone().numpy()

        similarities = []
        indices_list = []
        buckets_to_visit = buckets_to_visit.numpy()

        logger.debug("Searching in bucket indices...")
        for i in range(nprobe):
            similarity = np.empty((queries.shape[0], k), dtype=np.float32)
            indices = np.empty((queries.shape[0], k), dtype=np.int64)
            subclusters_to_search = np.ceil(
                bucket_weights[:, 0] * remaining_subclusters
            ).astype(np.uint32)
            new_weights = bucket_weights[:, 1:]
            if new_weights.shape[1] > 0:
                new_weights = new_weights / new_weights.sum(axis=1, keepdims=True)
            bucket_weights = new_weights

            for bucket in np.unique(buckets_to_visit[:, i], sorted=False):
                query_mask = buckets_to_visit[:, i] == bucket
                index = self._indices[bucket]
                for subclusters_to_search_bucket in np.unique(
                    subclusters_to_search[query_mask], sorted=False
                ):
                    subcluster_no_mask = (
                        subclusters_to_search == subclusters_to_search_bucket
                    ) & query_mask

                    bucket_subcluster_no_queries = queries[subcluster_no_mask]
                    if bucket_subcluster_no_queries.shape[0] == 0:
                        continue
                    if subclusters_to_search_bucket == 0:
                        similarity[subcluster_no_mask] = -np.inf
                        indices[subcluster_no_mask] = -1
                        continue
                    index.nprobe = subclusters_to_search_bucket.item()
                    (
                        bucket_subcluster_no_similarity,
                        bucket_subcluster_no_indices,
                    ) = index.search(
                        bucket_subcluster_no_queries, k
                    )  # pyright: ignore[reportCallIssue]
                    similarity[subcluster_no_mask] = bucket_subcluster_no_similarity
                    indices[subcluster_no_mask] = bucket_subcluster_no_indices

            remaining_subclusters -= (
                subclusters_to_search
                if discard
                else np.clip(subclusters_to_search, a_min=0, a_max=self.nlist)
            )

            similarities.append(similarity)
            indices_list.append(indices)

        merged_similarities, merged_indices = merge_knn_results(
            similarities, indices_list, keep_max=True
        )

        return merged_indices, -merged_similarities


def run_task1(data, task, k, output_dir, dataset="unknown"):
    print(f"Running {task} on {dataset}")

    ids = np.arange(1, len(data) + 1)  # 1-indexed IDs for the data points
    n, d = data.shape
    k_search = k + 1  # query for one extra to guarantee k non-self neighbours

    if dataset == "gooaq-small":
        # Use smaller parameters for gooaq-small
        # This is done because otherwise there is not enough data to train the index
        n_buckets = 3
        nlist = 4
        hidden_layers = []
        epochs = 5
        lr = 0.00098
        quantize = True
    else:
        # Actual parameters we want to use
        n_buckets = 32
        nlist = 256
        hidden_layers = []
        epochs = 5
        lr = 0.00098
        quantize = True

    index_identifier = f"LMI{n_buckets}+IVF{nlist}SQ8"

    index = LearnedMetricIndexIVF(
        d,
        n_buckets,
        nlist,
        hidden_layers,
        epochs,
        lr,
        quantize,
    )

    train_size = min(n_buckets * nlist * 40, len(data))
    batch_size_add = 100_000
    batch_size_search = 100_000

    print(f"Training index on {data.shape} with {data.dtype}")
    start_time = time.time()
    rng = np.random.default_rng(seed=42)
    train_indices = np.sort(rng.choice(len(ids), size=train_size, replace=False))
    train_data = np.array(data[train_indices], dtype=np.float32)
    index.train(train_data)

    for start in range(0, len(data), batch_size_add):
        end = min(start + batch_size_add, len(data))
        logger.info(f"Adding batch {start}:{end}...")
        data_batch = np.array(data[start:end], dtype=np.float32)
        ids_batch = ids[start:end]
        index.add(data_batch, ids_batch)
        del data_batch, ids_batch
        gc.collect()

    elapsed_build = time.time() - start_time
    print(f"Done training in {elapsed_build}s.")

    discard = True

    if dataset == "gooaq-small":
        nprobes = [1, 2, 3]
        ivf_nprobe_totals = [1, 2, 4, 8]
    else:
        nprobes = [4, 8, 16]
        ivf_nprobe_totals = [8, 16, 32, 64, 128]

    for nprobe in nprobes:
        for ivf_nprobe_total in ivf_nprobe_totals:
            print(
                f"Starting search on {data.shape} with nprobe={nprobe} and ivf_nprobe_total={ivf_nprobe_total}"
            )
            start_time = time.time()
            I = np.empty((data.shape[0], k_search), dtype=np.int64)
            D = np.empty((data.shape[0], k_search), dtype=np.float32)

            for start in range(0, len(data), batch_size_search):
                end = min(start + batch_size_search, len(data))
                logger.info(f"Searching batch {start}:{end}...")
                data_batch = np.array(data[start:end], dtype=np.float32)
                indices, distances = index.search(
                    data_batch,
                    k=k_search,
                    nprobe=nprobe,
                    ivf_nprobe_total=ivf_nprobe_total,
                    discard=discard,
                )
                I[start:end] = indices
                D[start:end] = distances
                del data_batch, indices, distances
                gc.collect()

            elapsed_search = time.time() - start_time
            print(f"Done searching in {elapsed_search}s.")

            identifier = f"index=({index_identifier}),query=(nprobe={nprobe},ivf_nprobe_total={ivf_nprobe_total},discard={discard})"

            store_results(
                output_dir / f"{identifier}.h5",
                "LMIIVF",
                dataset,
                task,
                D,
                I,
                elapsed_build,
                elapsed_search,
                identifier,
            )
