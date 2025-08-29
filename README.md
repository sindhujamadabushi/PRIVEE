# PRIVEE

*A modular framework for privacy‑aware **Vertical Federated Learning** (VFL) during inference*


## Key Features

| Category       | Highlights                                                           |
| -------------- | -------------------------------------------------------------------- |
| **Datasets**   | MNIST, Fashion‑MNIST, CIFAR‑10/100, Credit‑Card, Adult, custom CSV   |
| **Models**     | `BottomModel` per organisation, `LearningCoordinator*` top networks  |
| **Attacks**    | GIA, GRNA, ESA, active/passive label & feature inference                  |
| **Defenses**   | Rounding, raw Gaussian noise, FH‑OPE, PRIVEE, PRIVEE++            |

---

## Quick Start

```bash
# clone
git clone git@github.com:sindhujamadabushi/PRIVEE.git
cd PRIVEE

# optional: create conda env
conda env create -f environment.yml
conda activate PERFACY   # or any name

# install dependencies
pip install -r requirements_PERFACY.txt
```

> **Note:** The framework automatically uses CUDA if available; otherwise it falls back to CPU.

---

## Running Experiments

### Vanilla VFL baseline (MNIST, 2 parties)

```bash
python -m privee.main \
  --dname MNIST 
```

### GIA + PRIVEE defence (ε = 0.1)

```bash
python -m privee.main \
  --dname CIFAR10 \
  --attack gia \
  --defense privee \
  --epsilon 0.1
```
### GRNA + PRIVEE++ defence 

```bash
python -m privee.main \
  --dname MNIST \
  --attack grna \
  --defense priveeplus \
  --epsilon 0.1
```


All CLI flags:

| Flag | Type | Accepted Values / Purpose | Default |
|------|------|---------------------------|---------|
| `--attack` | `str` | `grna`, `gia`, `esa` | `grna` |
| `--defense` | `str` | `rounding`, `noising`, `fh-ope`, `privee`, `priveeplus` | **(required)** |
| `--decimals` | `int` | Number of decimal places for rounding (when `--defense rounding`) | `1` |
| `--epsilon` | `float` | Privacy ε (when `--defense noising/privee/priveeplus`) | `0.1` |
| `--attack_strength` | `float` | Strength parameter for the selected attack | `0.5` |
| `--dataset` | `str` | Dataset name (e.g., `DRIVE`, `CIFAR10`) | `DRIVE` |
| `--organization_num` | `int` | Number of passive parties | 2 |

---

## Defenses & Attacks

| Defence      | CLI flag     | Key parameters |
| ------------ | ------------ | -------------- |
| Rounding     | `rounding`   | `--decimals N` |
| Gaussian raw | `noising`    | `--epsilon E`  |
| FH‑OPE       | `fh-ope`     | *none*         |
| PriVEE       | `privee`     | `--epsilon E`  |
| PriVEE‑Plus  | `priveeplus` | `--epsilon E`  |

Select an attack with `--attack {gia,grna,lia,fia,none}`.

## Multi-party setting:

```bash
python multi_party_vfl.py \
  --defense priveeplus \
  --epsilon 0.1 \
  --organization_num 10 \
  --attack grna \
  --attack_strength 0.4
```
organization_num can range from 2 up to 25. All other settings (defenses, ε values, attack types, etc.) remain identical to the 2-party case, including how attacks are run.

---
Note : Dataset CSV files are not bundled with this repository because of GitHub’s size constraints.
After you preprocess the raw data on your side, please create a datasets/ directory in the project root and place the resulting .csv files there before running the experiments.
