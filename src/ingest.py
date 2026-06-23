import kagglehub
import shutil
from pathlib import Path

path = kagglehub.dataset_download("martj42/international-football-results-from-1872-to-2017")
dest = Path("data/raw/international_results")
dest.mkdir(parents=True, exist_ok=True)
for f in Path(path).glob("*"):
    shutil.copy(f, dest / f.name)
print(f"Downloaded dataset to {dest}")