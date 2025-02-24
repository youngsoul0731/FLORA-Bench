import json
from pathlib import Path
from abc import ABC, abstractmethod


class Reader(ABC):
    @abstractmethod
    def parse(self, file_path: Path) -> str:
        """ To be overriden by the descendant class """


class JSONLReader(Reader):
    def parse_file(file_path: Path) -> list:
        print(f"Reading JSON Lines file from {file_path}.")
        with open(file_path, "r", encoding='utf-8') as f:
            lines = [json.loads(line) for line in f]
            # text = '\n'.join([str(line) for line in lines])
        return lines  # text

    def parse(file_path: Path) -> str:
        print(f"Reading JSON Lines file from {file_path}.")
        with open(file_path, "r", encoding='utf-8') as f:
            lines = [json.loads(line) for line in f]
            text = '\n'.join([str(line) for line in lines])
        return text