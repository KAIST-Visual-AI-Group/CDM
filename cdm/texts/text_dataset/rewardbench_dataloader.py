import json
import math
import os
import random
from collections import defaultdict

REWARDBENCH_CATEGORY_SUBSETS = {
    "chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "chat-hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}
REWARDBENCH_SUBSET_TO_CATEGORY = {
    subset: category
    for category, subsets in REWARDBENCH_CATEGORY_SUBSETS.items()
    for subset in subsets
}


class RewardBenchPromptDataset:
    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def infer_rewardbench_category(subset):
    if subset is None:
        return None
    return REWARDBENCH_SUBSET_TO_CATEGORY.get(str(subset))


def validate_rewardbench_record(record, index=None):
    if not isinstance(record, dict):
        where = f" at line {index}" if index is not None else ""
        raise ValueError(f"RewardBench record{where} must be a JSON object")

    if "prompt" not in record:
        where = f" at line {index}" if index is not None else ""
        raise ValueError(f"RewardBench record{where} is missing required field 'prompt'")

    prompt = record["prompt"]
    if prompt is None or str(prompt).strip() == "":
        where = f" at line {index}" if index is not None else ""
        raise ValueError(f"RewardBench record{where} has empty 'prompt'")

    return True


def normalize_rewardbench_record(record, index=None):
    validate_rewardbench_record(record, index=index)

    normalized = dict(record)
    normalized["prompt"] = str(record["prompt"])

    subset = normalized.get("subset")
    if subset is not None:
        normalized["subset"] = str(subset)

    category = normalized.get("category")
    if category is None:
        category = infer_rewardbench_category(subset)
    if category is not None:
        normalized["category"] = str(category)

    if "id" in normalized and normalized["id"] is not None:
        try:
            normalized["id"] = int(normalized["id"])
        except (TypeError, ValueError):
            pass

    return normalized


def read_rewardbench_jsonl(jsonl_path, validate=True, normalize=True):
    if not os.path.isfile(jsonl_path):
        raise FileNotFoundError(f"RewardBench jsonl not found: {jsonl_path}")

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            if validate:
                validate_rewardbench_record(record, index=line_idx)
            if normalize:
                record = normalize_rewardbench_record(record, index=line_idx)
            records.append(record)

    return records


def write_rewardbench_jsonl(jsonl_path, records, validate=True):
    out_dir = os.path.dirname(jsonl_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    count = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, record in enumerate(records, start=1):
            if validate:
                record = normalize_rewardbench_record(record, index=idx)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def stratified_split_by_subset(records, train_ratio=0.8, seed=42):
    train_ratio = float(train_ratio)
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError(f"train_ratio must be in [0.0, 1.0], got {train_ratio}")

    normalized_records = [
        normalize_rewardbench_record(record, index=idx)
        for idx, record in enumerate(records, start=1)
    ]
    if not normalized_records:
        return [], []

    groups = defaultdict(list)
    for record in normalized_records:
        subset_key = str(record.get("subset", "__missing_subset__"))
        groups[subset_key].append(record)

    rng = random.Random(seed)
    for subset_rows in groups.values():
        rng.shuffle(subset_rows)

    target_train = int(round(len(normalized_records) * train_ratio))
    subset_keys = sorted(groups.keys())

    train_counts = {}
    fractional_parts = []
    assigned_train = 0

    for subset in subset_keys:
        n_subset = len(groups[subset])
        exact = n_subset * train_ratio
        count = int(math.floor(exact))
        count = max(0, min(count, n_subset))
        train_counts[subset] = count
        assigned_train += count
        fractional_parts.append((exact - count, subset))

    if assigned_train < target_train:
        for _, subset in sorted(fractional_parts, key=lambda x: (-x[0], x[1])):
            if assigned_train >= target_train:
                break
            if train_counts[subset] < len(groups[subset]):
                train_counts[subset] += 1
                assigned_train += 1

    elif assigned_train > target_train:
        for _, subset in sorted(fractional_parts, key=lambda x: (x[0], x[1])):
            if assigned_train <= target_train:
                break
            if train_counts[subset] > 0:
                train_counts[subset] -= 1
                assigned_train -= 1

    train_records = []
    test_records = []
    for subset in subset_keys:
        n_train = train_counts[subset]
        subset_rows = groups[subset]
        train_records.extend(subset_rows[:n_train])
        test_records.extend(subset_rows[n_train:])

    rng.shuffle(train_records)
    rng.shuffle(test_records)
    return train_records, test_records


def load_rewardbench_records(jsonl_path):
    """Load RewardBench prompt records from a jsonl file on disk.

    Each line must include at least `prompt`. Optional metadata fields such as
    `category`, `subset`, and `id` are preserved.
    """
    return read_rewardbench_jsonl(jsonl_path, validate=True, normalize=True)


def create_rewardbench_prompt_dataloader(jsonl_path, batch_size, shuffle=False):
    try:
        from torch.utils.data import DataLoader
    except Exception as e:
        raise ImportError(
            "create_rewardbench_prompt_dataloader requires torch. "
            "Install torch in the runtime environment."
        ) from e

    records = load_rewardbench_records(jsonl_path)
    dataset = RewardBenchPromptDataset(records)

    def collate_fn(batch):
        return batch

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=True,
    )
    return loader
