import os
from pathlib import Path

processor_type = "jaccard"
data_dir = Path(f"./dataset/{processor_type}")
train_path = data_dir / 'train.json'
dev_path = data_dir / 'dev.json'
test_path = data_dir / 'test.json'
output_dir = Path('./outputs')

pretrain_path = "pretrain_model/bert-base-chinese"