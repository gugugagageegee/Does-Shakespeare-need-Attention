from pathlib import Path
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATA_PATH = DATA_DIR / "tinyshakespeare.txt"

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_tiny_shakespeare(force=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists() and DATA_PATH.stat().st_size > 0 and not force:
        print("Dataset already exists.")
        return DATA_PATH

    print("Downloading Tiny Shakespeare...")

    try:
        with urlopen(DATA_URL, timeout=30) as response:
            content = response.read()

        if not content:
            raise ValueError("Downloaded file is empty.")

        DATA_PATH.write_bytes(content)

    except Exception as error:
        raise RuntimeError(
            "Failed to download Tiny Shakespeare. "
            "Please check your internet connection."
        ) from error

    print(f"Downloaded successfully: {DATA_PATH}")

    return DATA_PATH


def load_text(path=None):
    if path is None:
        path = DATA_PATH

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            "Dataset not found. "
            "Run download_tiny_shakespeare() first."
        )

    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

    return text