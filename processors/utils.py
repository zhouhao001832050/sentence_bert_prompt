import csv
import json
import torch


class DataProcessor(object):
    """Base class for data converters for sentence bert data sets"""

    def get_train_examples(self,data_dir):
        return NotImplementedError()

    
    def get_dev_examples(self,data_dir):
        return NotImplementedError()


    def get_test_examples(self,data_dir):
        return NotImplementedError()


    @classmethod
    def _read_json(self, input_file):
        lines = []
        with open(input_file, "r") as f:
            for line in f.readlines():
                line = json.loads(line.strip())
                corpus = line["corpus"]
                entity = line["entity"]
                label = line["label"]
                lines.append({"corpus":corpus,"entity":entity,"label":label})
        return lines