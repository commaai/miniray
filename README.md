# miniray

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue)](./pyproject.toml)
[![License](https://img.shields.io/github/license/commaai/miniray)](./LICENSE)
[![Stars](https://img.shields.io/github/stars/commaai/miniray?style=social)](https://github.com/commaai/miniray)

A minimal Python library for distributed compute across a datacenter.

`miniray` dispatches arbitrary Python tasks through Redis and exposes a workflow that feels familiar if you already use `concurrent.futures`. It is a good fit when you want to spread CPU or GPU work across multiple machines without rewriting your application around a brand-new programming model.

## Table of Contents

- [Why miniray](#why-miniray)
- [What it does](#what-it-does)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Running workers](#running-workers)
- [Development](#development)

## Why miniray

- Keeps a familiar `Executor` / `Future` style API
- Uses Redis as the coordination layer for task dispatch
- Supports local iteration and multi-machine execution patterns
- Includes worker scripts and utilities for queue inspection

## What it does

The repository includes:

- `executor.py` for submitting jobs and collecting results
- `worker.py` for processing queued tasks on worker machines
- `start_worker.sh` and helper scripts for bootstrapping workers
- `show_jobs.py`, `show_working.py`, and `wipe_queues.sh` for operations and debugging
- `tests/` for validating the task execution flow

## Repository layout

```text
.
├── executor.py
├── worker.py
├── start_worker.sh
├── show_jobs.py
├── show_working.py
├── wipe_queues.sh
├── lib/
├── tests/
└── pyproject.toml
```

## Requirements

- Python 3.12+
- A reachable Redis instance
- Optional GPU / Triton setup if you plan to run GPU-heavy workloads

## Installation

```bash
git clone https://github.com/commaai/miniray.git
cd miniray
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you want the extra tooling used in development:

```bash
pip install -e .[testing]
```

## Quick start

```python
from concurrent.futures import as_completed
import numpy as np
import miniray

def is_even(n):
    return n % 2 == 0

x = np.arange(100)
results_loop = [is_even(n) for n in x]

with miniray.Executor(job_name="miniray_example_map") as executor:
    results_map = executor.map(is_even, np.arange(100))

with miniray.Executor(job_name="miniray_example_submit") as executor:
    futures = [executor.submit(is_even, n) for n in x]
    results_submit = [future.result() for future in as_completed(futures)]

for a, b, c in zip(results_loop, results_map, results_submit):
    assert a == b == c
```

This preserves the same high-level pattern many Python users already know from `concurrent.futures`, while letting work run through the distributed queue.

## Running workers

A typical worker bootstrap starts with configuring Redis access and then launching the worker script:

```bash
export REDIS_HOST=<your-redis-host>
./start_worker.sh
```

Useful helper scripts in the repository:

- `python show_jobs.py` to inspect queued jobs
- `python show_working.py` to inspect active workers
- `./wipe_queues.sh` to clear queues during local testing

## Development

```bash
pytest
```

The repository also includes configuration for `ruff`, `pre-commit`, and additional test dependencies in `pyproject.toml`.

## Contributing

Issues and pull requests are welcome. If you are evaluating `miniray` for a production workflow and hit a missing capability, opening a short issue with your workload details will make the discussion easier.
