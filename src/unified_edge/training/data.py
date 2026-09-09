"""Verified relocatable binary manifests, document windows and deterministic cursors."""

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import torch

from unified_edge.training.config import canonical_hash


def file_bytes(root: Path, relative: str) -> bytes:
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError("document identity must be a relative POSIX path")
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or any(p in (".", "..") for p in path.parts):
        raise ValueError("document identity must stay inside data root")
    if path.as_posix() != relative:
        raise ValueError("document path must be canonical")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("document resolves outside data root")
    return resolved.read_bytes()


@dataclass(frozen=True)
class Document:
    path: str
    sha256: str
    byte_count: int
    document_count: int
    split: str
    domain: str


@dataclass(frozen=True)
class DatasetManifest:
    schema: str
    corpus_id: str
    documents: tuple[Document, ...]

    def __post_init__(self):
        if self.schema != "1" or not isinstance(self.corpus_id, str) or not self.corpus_id:
            raise ValueError("invalid dataset schema/corpus ID")
        if not self.documents or not all(isinstance(d, Document) for d in self.documents):
            raise ValueError("manifest requires documents")
        if len({d.path for d in self.documents}) != len(self.documents):
            raise ValueError("duplicate manifest document")
        if {d.split for d in self.documents} != {"train", "validation"}:
            raise ValueError("manifest requires distinct train and validation splits")
        for d in self.documents:
            if (
                type(d.byte_count) is not int
                or d.byte_count <= 0
                or type(d.document_count) is not int
                or d.document_count != 1
            ):
                raise ValueError("each nonempty file must represent exactly one document")
            if not isinstance(d.domain, str) or not d.domain:
                raise ValueError("document domain is required")
            if not isinstance(d.sha256, str) or len(d.sha256) != 64:
                raise ValueError("invalid document SHA-256")
            try:
                int(d.sha256, 16)
            except ValueError as error:
                raise ValueError("invalid document SHA-256") from error
        train = {d.sha256 for d in self.documents if d.split == "train"}
        valid = {d.sha256 for d in self.documents if d.split == "validation"}
        if train & valid:
            raise ValueError("train and validation cannot contain identical documents")

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "corpus_id": self.corpus_id,
            "documents": [asdict(d) for d in self.documents],
        }

    @property
    def sha256(self) -> str:
        return canonical_hash(self.to_dict())

    def verify(self, root: Path) -> dict[str, bytes]:
        result = {}
        for d in self.documents:
            payload = file_bytes(root, d.path)
            if len(payload) != d.byte_count or hashlib.sha256(payload).hexdigest() != d.sha256:
                raise ValueError(f"dataset mutation detected: {d.path}")
            result[d.path] = payload
        return result

    @classmethod
    def create(cls, root: Path, corpus_id: str, files: list[tuple[str, str, str]]):
        documents = []
        for relative, split, domain in sorted(files):
            payload = file_bytes(root, relative)
            documents.append(
                Document(
                    relative, hashlib.sha256(payload).hexdigest(), len(payload), 1, split, domain
                )
            )
        return cls("1", corpus_id, tuple(documents))

    @classmethod
    def from_dict(cls, value: dict):
        if not isinstance(value, dict) or set(value) != {"schema", "corpus_id", "documents"}:
            raise ValueError("manifest has missing or unknown fields")
        if not isinstance(value["documents"], list):
            raise ValueError("manifest documents must be a list")
        documents = []
        for d in value["documents"]:
            if not isinstance(d, dict) or set(d) != set(Document.__dataclass_fields__):
                raise ValueError("document has missing or unknown fields")
            documents.append(Document(**d))
        return cls(value["schema"], value["corpus_id"], tuple(documents))


@dataclass(frozen=True)
class Window:
    document: str
    start: int
    payload: bytes


class WindowDataset:
    def __init__(self, manifest: DatasetManifest, root: Path, split: str, length: int):
        if (
            split not in ("train", "validation")
            or type(length) is not int
            or not 1 <= length <= 256
        ):
            raise ValueError("invalid split or bounded window length")
        self.manifest, self.root = manifest, Path(root)
        documents = manifest.verify(self.root)
        self.windows = tuple(
            Window(d.path, start, documents[d.path][start : start + length])
            for d in manifest.documents
            if d.split == split
            for start in range(0, d.byte_count, length)
        )

    def __len__(self):
        return len(self.windows)

    def verify(self):
        self.manifest.verify(self.root)


def batches_by_length(windows: list[Window]) -> list[torch.Tensor]:
    groups = {}
    for window in windows:
        groups.setdefault(len(window.payload), []).append(list(window.payload))
    return [torch.tensor(rows, dtype=torch.long) for rows in groups.values()]


class BatchStream:
    def __init__(self, dataset: WindowDataset, seed: int, batch_size: int):
        if type(seed) is not int or seed < 0 or type(batch_size) is not int or batch_size < 1:
            raise ValueError("invalid stream seed/batch size")
        self.dataset, self.seed, self.batch_size = dataset, seed, batch_size
        self.epoch, self.offset = 0, 0
        self.order = self._order(0)

    def _order(self, epoch):
        order = list(range(len(self.dataset)))
        random.Random(self.seed + epoch).shuffle(order)
        return order

    def next_batch(self) -> list[Window]:
        result = []
        for _ in range(self.batch_size):
            if self.offset == len(self.order):
                self.epoch += 1
                self.offset = 0
                self.order = self._order(self.epoch)
            result.append(self.dataset.windows[self.order[self.offset]])
            self.offset += 1
        return result

    def state_dict(self):
        return {
            "schema": "1",
            "epoch": self.epoch,
            "offset": self.offset,
            "order": list(self.order),
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {"schema", "epoch", "offset", "order"}:
            raise ValueError("invalid data cursor fields")
        if state["schema"] != "1" or type(state["epoch"]) is not int or state["epoch"] < 0:
            raise ValueError("invalid data cursor epoch/schema")
        if type(state["offset"]) is not int or not 0 <= state["offset"] <= len(self.dataset):
            raise ValueError("invalid data cursor offset")
        if state["order"] != self._order(state["epoch"]) or any(
            type(i) is not int for i in state["order"]
        ):
            raise ValueError("data ordering does not match deterministic permutation")
        self.epoch, self.offset, self.order = state["epoch"], state["offset"], list(state["order"])


def create_tiny_fixture(root: Path) -> DatasetManifest:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    (root / "train.raw").write_bytes(b"edge400: abcdef\n" * 64)
    (root / "validation.raw").write_bytes(b"edge401: uvwxyz\n" * 16)
    manifest = DatasetManifest.create(
        root,
        "edge400-controlled-v1",
        [
            ("train.raw", "train", "synthetic-pattern"),
            ("validation.raw", "validation", "held-out-synthetic-pattern"),
        ],
    )
    with (root / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2)
    return manifest
