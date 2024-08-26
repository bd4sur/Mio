import torch

from typing import Union, Dict
from dataclasses import is_dataclass
import logging
from pathlib import Path
import os
import hashlib
from mmap import mmap, ACCESS_READ

class Logger:
    def __init__(self, logger=logging.getLogger(Path(__file__).parent.name)):
        self.logger = logger

    def set_logger(self, logger: logging.Logger):
        self.logger = logger

    def get_logger(self) -> logging.Logger:
        return self.logger

logger = Logger()

def select_device(min_memory=2047, experimental=False):
    if torch.cuda.is_available():
        selected_gpu = 0
        max_free_memory = -1
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free_memory = props.total_memory - torch.cuda.memory_reserved(i)
            if max_free_memory < free_memory:
                selected_gpu = i
                max_free_memory = free_memory
        free_memory_mb = max_free_memory / (1024 * 1024)
        if free_memory_mb < min_memory:
            logger.get_logger().warning(
                f"GPU {selected_gpu} has {round(free_memory_mb, 2)} MB memory left. Switching to CPU."
            )
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{selected_gpu}")
    elif torch.backends.mps.is_available():
        """
        Currently MPS is slower than CPU while needs more memory and core utility,
        so only enable this for experimental use.
        """
        if experimental:
            # For Apple M1/M2 chips with Metal Performance Shaders
            logger.get_logger().warning("experimantal: found apple GPU, using MPS.")
            device = torch.device("mps")
        else:
            logger.get_logger().info("found Apple GPU, but use CPU.")
            device = torch.device("cpu")
    else:
        logger.get_logger().warning("no GPU found, use CPU instead")
        device = torch.device("cpu")

    return device


def del_all(d: Union[dict, list]):
    if is_dataclass(d):
        for k in list(vars(d).keys()):
            x = getattr(d, k)
            if isinstance(x, dict) or isinstance(x, list) or is_dataclass(x):
                del_all(x)
            del x
            delattr(d, k)
    elif isinstance(d, dict):
        lst = list(d.keys())
        for k in lst:
            x = d.pop(k)
            if isinstance(x, dict) or isinstance(x, list) or is_dataclass(x):
                del_all(x)
            del x
    elif isinstance(d, list):
        while len(d):
            x = d.pop()
            if isinstance(x, dict) or isinstance(x, list) or is_dataclass(x):
                del_all(x)
            del x
    else:
        del d



def sha256(fileno: int) -> str:
    data = mmap(fileno, 0, access=ACCESS_READ)
    h = hashlib.sha256(data).hexdigest()
    del data
    return h


def check_model(
    dir_name: Path, model_name: str, hash: str, remove_incorrect=False
) -> bool:
    target = dir_name / model_name
    relname = target.as_posix()
    logger.get_logger().debug(f"checking {relname}...")
    if not os.path.exists(target):
        logger.get_logger().warning(f"{target} not exist.")
        return False
    with open(target, "rb") as f:
        digest = sha256(f.fileno())
        bakfile = f"{target}.bak"
        if digest != hash:
            logger.get_logger().warning(f"{target} sha256 hash mismatch.")
            logger.get_logger().info(f"expected: {hash}")
            logger.get_logger().info(f"real val: {digest}")
            if remove_incorrect:
                if not os.path.exists(bakfile):
                    os.rename(str(target), bakfile)
                else:
                    os.remove(str(target))
            return False
        if remove_incorrect and os.path.exists(bakfile):
            os.remove(bakfile)
    return True


def check_all_assets(model_dir: Path, sha256_map: Dict[str, str], update=False) -> bool:
    logger.get_logger().info("checking assets...")
    names = [
        "Decoder.pt",
        "DVAE_full.pt",
        "GPT.pt",
        "spk_stat.pt",
        "tokenizer.pt",
        "Vocos.pt",
    ]
    for model in names:
        menv = model.replace(".", "_")
        if not check_model(
            model_dir, model, sha256_map[f"sha256_asset_{menv}"], update
        ):
            return False

    logger.get_logger().info("all assets are already latest.")
    return True

