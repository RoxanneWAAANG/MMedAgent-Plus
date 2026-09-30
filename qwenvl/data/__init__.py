import os
import re

# Root of the downloaded training data (see README, "Data"). Expected layout:
#   $MMEDAGENT_DATA_ROOT/train/*.jsonl
#   $MMEDAGENT_DATA_ROOT/tool_image/train/...
# Image paths inside the annotation files are relative to this root.
DATA_ROOT = os.environ.get("MMEDAGENT_DATA_ROOT", "./data")


def _dataset(annotation_file):
    return {
        "annotation_path": os.path.join(DATA_ROOT, "train", annotation_file),
        "data_path": DATA_ROOT,
    }


data_dict = {
    "single_tool_multi_round": _dataset("single_tool_multi_round.jsonl"),
    "multi_tool_multi_round": _dataset("multi_tool_multiround.jsonl"),
    "multi_tool_single_round": _dataset("multi_tool_single_round.jsonl"),
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list
