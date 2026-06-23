# SISAP 2026 Challenge: Dream Team implementation

This repository contains the code for our submission to the SISAP 2026 Indexing Challenge.

**Members:**

- [David Procházka](https://github.com/ProchazkaDavid), Masaryk University
- [Emma Sommerová](https://github.com/emmaSommer), Masaryk University
- [Jan Mach](https://github.com/machous02), Masaryk University
- [Vlastislav Dohnal](https://github.com/dohnal), Masaryk University

## Installation & Setup

### Install Python requirements

```bash
pip install -r requirements.txt
```

Make sure Python 3.11 is available on your system, or, preferably, create a virtual environment.

### Build Docker image

```bash
docker build -t sisap26 .
```

### Download datasets

```bash
bash download_datasets.sh --small-only
```

Skip the `--small-only` flag if you want to download all of the datasets, not just the ones for development.

## Running the Code

### Run search

```bash
bash run_search.sh
```

The `run_search.sh` scripts evaluates the implementation on a smaller development subset of the wikipedia dataset for task 1, and on the full llama-dev dataset for task 2. Feel free to modify the script to run on different datasets, just make sure they are downloaded, and accompanied by their respective `config.json` file.

### Evaluation

```bash
python eval.py results.csv
```

will produce a summary file of the results with the computed recall against the ground truth data.

This csv file can be further processed to create plots (using `python plot.py --task {task1, task2, task3} res.csv`) and show the fastest solutions above a certain recall threshold (using `python show_operating_points.py`).

## GitHub Actions: Continuous integration

There are two CI pipelines specified in this repository. `ci.yml` performs search and evaluation on example datasets. `upload-software-to-tira.yml` submits the implementation into the TIRA evaluation system.
