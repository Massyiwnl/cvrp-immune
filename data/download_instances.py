"""Scarica le 10 istanze del progetto (+ soluzioni ottime) dal sito
ufficiale CVRPLIB """

import urllib.request
from pathlib import Path

BASE = "https://galgos.inf.puc-rio.br/cvrplib/index.php/en/download"


IDS = {
    "A-n45-k7": 17, "A-n60-k9": 23, "A-n80-k10": 31,
    "B-n56-k7": 45, "B-n66-k9": 50, "B-n78-k10": 53,
    "E-n76-k8": 62, "E-n101-k14": 66,
    "P-n50-k10": 86, "P-n101-k4": 98,
}


def fetch(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        dest.write_bytes(r.read())
    print("scaricato:", dest.name)


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    for name, i in IDS.items():
        fetch(f"{BASE}/instance/{i}", here / f"{name}.vrp")
        fetch(f"{BASE}/bks/{i}", here / f"{name}.sol")
    print("Fatto: 10 .vrp + 10 .sol")
