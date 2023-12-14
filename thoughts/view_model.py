import torch
from pathlib import Path

from safetensors import safe_open
from enum import StrEnum

class ModelType(StrEnum):
    SAFETENSORS = ".safetensors"
    PTH = ".pth"
    CKPT = ".ckpt"


class ModelViewer:
    def __init__(self, model_path: Path):
        if not model_path.exists(): raise FileExistsError(f"Can't find file {model_file}")

        self.model_path = model_path

    def view_weights(self):
        match self.model_path.suffix:
            case ModelType.SAFETENSORS:
                with safe_open(self.model_path.as_posix(), framework="pt", device="cpu") as f: # type: ignore
                    table = "{key:<{key_width}} {dtype:<15} {shape:<20}"
                    longest_key = max(map(lambda x: len(x), f.keys()))
                    for key in f.keys():
                        value = f.get_tensor(key)
                        print(table.format(
                            key=key, 
                            dtype=str(value.dtype),
                            shape=str(value.shape),
                            key_width=longest_key + 5
                            ))

            case ModelType.PTH:
                state_dict = torch.load(model_file.as_posix(), map_location="meta", mmap=True, weights_only=True)
                table = "{key:<{key_width}} {dtype:<15} {shape:<20}"
                longest_key = max(map(lambda x: len(x), state_dict.keys()))
                print(longest_key)
                for key, value in state_dict.items():
                    print(table.format(
                        key=key, 
                        dtype=str(value.dtype),
                        shape=str(value.shape),
                        key_width=longest_key + 5
                        ))
            case _:
                print(f"unknown suffix: {model_file.suffix}")

    def get_file_metadata(self):
        full_path = self.model_path.resolve().as_posix()
        stats = self.model_path.stat()
        TABLE_FORMAT = "{:<{label_size}} {:<{field_size}}"
        table_data = [
                ("file_size", sizeof_fmt(stats.st_size)),
                ("full_path", full_path),
                ("file_name", self.model_path.name),
                ("suffix", self.model_path.suffix),
                ]
        print("="*60)
        for label, field in table_data:
            print(TABLE_FORMAT.format(
                label + ":", field, 
                label_size = 12, 
                field_size = 20)
                )
        print("="*60)

def sizeof_fmt(num, suffix="B"):
    """ Returns a human readable string representation of bytes """
    for unit in ("", "Ki", "Mi", "Gi", "Ti"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Ti{suffix}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            prog='ModelViewer',
            description="Reads model files and prints visual representations of the layout")

    parser.add_argument('FILENAME')
    parser.add_argument('-v', '--verbose',
                        action='store_true')

    args = parser.parse_args()
    
    model_file = Path(args.FILENAME)
    model = ModelViewer(model_file)
    model.get_file_metadata()
    model.view_weights()
