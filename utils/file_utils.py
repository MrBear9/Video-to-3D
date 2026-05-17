
from pathlib import Path
import shutil
from typing import List


def get_file_extension(path: str) -&gt; str:
    return Path(path).suffix.lower()


def ensure_dir(path: str) -&gt; Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_files(directory: str, extensions: List[str] = None) -&gt; List[Path]:
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    
    files = list(dir_path.iterdir())
    if extensions:
        extensions = [ext.lower() for ext in extensions]
        files = [f for f in files if f.is_file() and f.suffix.lower() in extensions]
    else:
        files = [f for f in files if f.is_file()]
    return files


def copy_file(src: str, dst: str) -&gt; bool:
    try:
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


def delete_file(path: str) -&gt; bool:
    try:
        Path(path).unlink()
        return True
    except Exception:
        return False

