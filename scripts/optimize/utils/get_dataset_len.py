import os
import sys
import json

dataset_base_dir = "metagpt/ext/aflow/data"

def get_dataset_len(dataset_name):
    dataset_file = os.path.join(dataset_base_dir, f'{dataset_name.lower()}_test.jsonl')
    with open(dataset_file, 'r') as f:
        for i, line in enumerate(f):
            pass
    return i + 1


def get_dataset(dataset_name):
    dataset_file = os.path.join(dataset_base_dir, f'{dataset_name.lower()}_test.jsonl')
    data = []
    with open(dataset_file, 'r') as f:
        for i, line in enumerate(f):
            data.append(json.loads(line))
    return data

    

    