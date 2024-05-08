import os
from pathlib import Path
from typing import Iterable

import ujson


def read_jsonl(file_path: Path):
    # Taken from prodigy support
    """Read a .jsonl file and yield its contents line by line.
    file_path (unicode / Path): The file path.
    YIELDS: The loaded JSON contents of each line.
    """
    with Path(file_path).open(encoding="utf8") as f:
        for line in f:
            try:  # hack to handle broken jsonl
                yield ujson.loads(line.strip())
            except ValueError:
                continue


def write_jsonl(file_path: Path, lines: Iterable):
    # Taken from prodigy
    """Create a .jsonl file and dump contents.
    file_path (unicode / Path): The path to the output file.
    lines (list): The JSON-serializable contents of each line.
    """
    data = [ujson.dumps(line, escape_forward_slashes=False) for line in lines]
    Path(file_path).open("w", encoding="utf-8").write("\n".join(data))


def file_list_folders(rootdir, fileend):
    file_list = []
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(fileend):
                filename = [os.path.join(subdir, file).replace("\\", "/")]
                file_list += filename
    return file_list


def create_batch(iterable, n=1):
    lr = len(iterable)
    for ndx in range(0, lr, n):
        yield iterable[ndx : min(ndx + n, lr)]
