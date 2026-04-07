import os
import re
import random
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.utils.data as data

# ── Amino acid alphabet & BLOSUM62 encoding ──────────────────────────────────

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
WINDOW_SIZE = 9

_BLOSUM62_STD_ORDER = "ARNDCQEGHILKMFPSTWYV"
_BLOSUM62_MATRIX = [
#   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
  [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],  # A
  [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
  [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],  # N
  [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],  # D
  [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
  [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],  # Q
  [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],  # E
  [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],  # G
  [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],  # H
  [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],  # I
  [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],  # L
  [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],  # K
  [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],  # M
  [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],  # F
  [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],  # P
  [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],  # S
  [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],  # T
  [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],  # W
  [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],  # Y
  [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4],  # V
]

_std_idx   = {aa: i for i, aa in enumerate(_BLOSUM62_STD_ORDER)}
_reorder   = [_std_idx[aa] for aa in AMINO_ACIDS]
_blosum_raw = torch.tensor([_BLOSUM62_MATRIX[i] for i in _reorder], dtype=torch.float32)
BLOSUM62_ENCODING = _blosum_raw[:, _reorder]   # (20, 20), both axes in AMINO_ACIDS order


# ── Encoding helpers ─────────────────────────────────────────────────────────

def transform_sequence(seq: str) -> torch.Tensor:
    """Encode a 9-mer via BLOSUM62 rows → flat vector of shape (180,)."""
    if isinstance(seq, bytes):
        seq = seq.decode()
    seq = seq.strip().upper()
    if len(seq) != WINDOW_SIZE:
        raise ValueError(f"Expected length {WINDOW_SIZE}, got {len(seq)}: {seq}")
    rows = [
        BLOSUM62_ENCODING[AA_TO_INDEX[aa]] if aa in AA_TO_INDEX
        else torch.zeros(len(AMINO_ACIDS))
        for aa in seq
    ]
    return torch.stack(rows).view(-1)   # (180,)


def encode_label_one_hot(label: int, num_classes: int) -> torch.Tensor:
    one_hot = torch.zeros(num_classes, dtype=torch.float32)
    one_hot[label] = 1.0
    return one_hot


# ── Dataset classes ──────────────────────────────────────────────────────────

class SequenceFileDataset(data.Dataset):
    """
    One file per class under a directory.
    Each line is a raw peptide string (no header).
    File names become class labels (sorted alphabetically → class index).

    Use the classmethod `from_dir` to construct from a directory, with optional
    in-memory cleaning that removes any negative sequence that overlaps a positive.
    """

    def __init__(
        self,
        samples: list[tuple[str, int]],
        classes: list[str],
        transform=None,
        target_transform=None,
    ):
        self.samples          = samples
        self.classes          = classes
        self.class_to_idx     = {c: i for i, c in enumerate(classes)}
        self.num_classes      = len(classes)
        self.transform        = transform
        self.target_transform = target_transform

    @classmethod
    def from_dir(cls, root_dir: str, clean: bool = False, transform=None, target_transform=None):
        """
        Load sequences from root_dir (one file per class).
        If clean=True, any negative sequence that also appears in a positive file
        is silently dropped — no intermediate files are written.
        """
        filenames = sorted(
            f for f in os.listdir(root_dir)
            if os.path.isfile(os.path.join(root_dir, f))
        )

        pos_seqs: set[str] = set()
        if clean:
            for fname in filenames:
                if fname != "negs.txt":
                    with open(os.path.join(root_dir, fname), encoding="utf-8") as f:
                        for line in f:
                            seq = line.strip()
                            if seq:
                                pos_seqs.add(seq)

        samples: list[tuple[str, int]] = []
        for idx, fname in enumerate(filenames):
            with open(os.path.join(root_dir, fname), encoding="utf-8") as f:
                for line in f:
                    seq = line.strip()
                    if not seq:
                        continue
                    if clean and fname == "negs.txt" and seq in pos_seqs:
                        continue
                    samples.append((seq, idx))

        if clean:
            print(f"Cleaned: dropped sequences from negs.txt that overlap positives")

        return cls(samples, filenames, transform=transform, target_transform=target_transform)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq, label = self.samples[idx]
        if self.transform is not None:
            seq = self.transform(seq)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return seq, label


class AugmentedTrainDataset(data.Dataset):
    """
    Wraps a Subset; applies random single-AA BLOSUM62-guided substitution at train time.

    p_augment   : probability of augmenting each sample per call
    temperature : controls conservation of substitution (low = more similar AA)
    """

    def __init__(self, subset: data.Subset, p_augment: float = 0.5, temperature: float = 1.0):
        self.subset      = subset
        self.p_augment   = p_augment
        self.temperature = temperature

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        real_idx = self.subset.indices[idx]
        base_ds  = self.subset.dataset
        seq, label = base_ds.samples[real_idx]

        if random.random() < self.p_augment:
            seq = blosum_substitute(seq, temperature=self.temperature)

        if base_ds.transform is not None:
            seq = base_ds.transform(seq)
        if base_ds.target_transform is not None:
            label = base_ds.target_transform(label)
        return seq, label


class PeptideDataset(data.Dataset):
    """Sliding-window peptide dataset for inference on a raw protein sequence."""

    def __init__(self, peptides: list[str], transform):
        self.peptides  = peptides
        self.transform = transform

    def __len__(self) -> int:
        return len(self.peptides)

    def __getitem__(self, idx: int):
        return self.transform(self.peptides[idx]), torch.tensor(0)


# ── Augmentation ─────────────────────────────────────────────────────────────

def blosum_substitute(seq: str, temperature: float = 1.0) -> str:
    """Replace one random position with a BLOSUM62-weighted amino acid (never self)."""
    seq = list(seq)
    pos = random.randint(0, len(seq) - 1)
    aa  = seq[pos]
    if aa not in AA_TO_INDEX:
        return "".join(seq)
    scores = BLOSUM62_ENCODING[AA_TO_INDEX[aa]].clone()
    scores[AA_TO_INDEX[aa]] = -1e9                       # exclude self
    probs   = F.softmax(scores / temperature, dim=0)
    new_idx = torch.multinomial(probs, 1).item()
    seq[pos] = AMINO_ACIDS[new_idx]
    return "".join(seq)


# ── Splitting ─────────────────────────────────────────────────────────────────

def stratified_train_test_split(
    dataset: SequenceFileDataset,
    test_size: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Stratified split preserving class proportions in both splits."""
    label_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        label_to_indices[label].append(idx)

    train_indices, test_indices = [], []
    rng = random.Random(seed)
    for label, inds in label_to_indices.items():
        if shuffle:
            rng.shuffle(inds)
        if len(inds) == 1:
            train_indices.append(inds[0])
        else:
            n_test = max(1, min(int(round(len(inds) * test_size)), len(inds) - 1))
            test_indices.extend(inds[:n_test])
            train_indices.extend(inds[n_test:])

    return train_indices, test_indices


# ── High-level builders ───────────────────────────────────────────────────────

def build_dataset(
    data_dir: str = "ex1_data",
    clean: bool = True,
    test_size: float = 0.1,
    p_augment: float = 0.5,
    aug_temperature: float = 1.0,
    seed: int = 42,
) -> tuple[SequenceFileDataset, AugmentedTrainDataset, data.Subset]:
    """
    Returns (dataset, aug_train_dataset, test_dataset).

    If clean=True, negative sequences overlapping any positive are removed
    in-memory — no intermediate files are written.
    aug_train_dataset applies BLOSUM62 augmentation at train time.
    test_dataset is plain (no augmentation).
    """
    dataset = SequenceFileDataset.from_dir(data_dir, clean=clean, transform=transform_sequence)
    dataset.target_transform = lambda label: encode_label_one_hot(label, dataset.num_classes)

    train_indices, test_indices = stratified_train_test_split(
        dataset, test_size=test_size, seed=seed
    )
    train_subset = data.Subset(dataset, train_indices)
    test_dataset = data.Subset(dataset, test_indices)
    aug_train_dataset = AugmentedTrainDataset(
        train_subset, p_augment=p_augment, temperature=aug_temperature
    )
    return dataset, aug_train_dataset, test_dataset


def build_protein_loader(
    fasta_path: str,
    batch_size: int = 512,
) -> tuple[data.DataLoader, list[str], list[int]]:
    """
    Parse a raw protein file, slide a 9-mer window, and return
    (loader, peptides, positions).
    """
    with open(fasta_path) as f:
        raw = f.read()
    sequence  = re.sub(r"[^A-Za-z]", "", raw).upper()
    total_length = len(sequence)
    print(f"Protein length: {total_length} amino acids")

    valid_aas = set(AMINO_ACIDS)
    peptides  = [sequence[i:i + WINDOW_SIZE] for i in range(total_length - WINDOW_SIZE + 1)]
    positions = list(range(1, total_length - WINDOW_SIZE + 2))

    keep      = [all(aa in valid_aas for aa in p) for p in peptides]
    peptides  = [p for p, m in zip(peptides,  keep) if m]
    positions = [p for p, m in zip(positions, keep) if m]
    print(f"Valid 9-mer windows: {len(peptides)}")

    loader = data.DataLoader(
        PeptideDataset(peptides, transform_sequence),
        batch_size=batch_size,
        shuffle=False,
    )
    return loader, peptides, positions, total_length
