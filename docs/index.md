# NRDK: Neural Radar Development Kit

[![GitHub](https://img.shields.io/github/v/release/RadarML/nrdk)](https://github.com/RadarML/nrdk)
![MIT License](https://img.shields.io/badge/license-MIT-green)
![Supports Python 3.12+](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)
![Typed](https://img.shields.io/badge/types-typed-limegreen)
[![bear-ified](https://raw.githubusercontent.com/beartype/beartype-assets/main/badge/bear-ified.svg)](https://beartype.readthedocs.io)
[![CI](https://github.com/RadarML/nrdk/actions/workflows/ci.yml/badge.svg)](https://github.com/RadarML/nrdk/actions/workflows/ci.yml)
![GitHub issues](https://img.shields.io/github/issues/RadarML/nrdk)

The **Neural Radar Development Kit** (NRDK) is an open-source and MIT-licensed Python library and framework for developing, training, and evaluating machine learning models on radar spectrum and multimodal sensor data.

Built around typed, high modular interfaces, the NRDK is designed to reduce the barrier of entry to learning on spectrum via out-of-the-box reference implementations for [red-rover](https://radarml.github.io/red-rover/) data and the [I/Q-1M Dataset](https://radarml.github.io/red-rover/iq1m/), while also providing an easy path towards customization and extensions for other radar and data collection systems.

<div class="grid cards" markdown>

- :octicons-ai-model-16: [`nrdk`](design.md)

    ---

    neural radar development kit core framework

- :octicons-terminal-16: [`> nrdk ...`](cli.md)

    ---

    cli tools for working with NRDK output conventions

- :material-book-open-page-variant-outline: [`grt`](grt/index.md)

    ---

    reference implementation for [Towards Foundational Models for Single-Chip Radar](https://wiselabcmu.github.io/grt)

- :material-chart-line: [`nrdk.tss`](./tss/index.md)

    ---

    statistical testing for time series performance metrics

</div>

## Setup

The NRDK is intended to be used as a library; we recommend installing it as a submodule:

=== "As Submodule"

    ```sh
    git submodule add git@github.com:RadarML/nrdk.git
    uv pip install -e ./nrdk
    ```

=== "With Rover Support"

    ```sh
    git submodule add git@github.com:RadarML/nrdk.git
    uv pip install -e ./nrdk[roverd,grt]
    ```

=== "Development"

    ```sh
    git clone git@github.com:RadarML/nrdk.git
    cd nrdk
    uv sync --all-extras
    uv run pre-commit install
    ```

!!! tip

    If authenticating with a github token (i.e., `https`), translate all `ssh://git@github.com` authentication to `https://github.com` with
    ```sh
    git config --global url."https://github.com/".insteadOf ssh://git@github.com/
    ```

??? info "Extras"

    The NRDK library also includes the following extras:

    - `nrdk[roverd]`: support for loading and processing data using the [roverd](https://radarml.github.io/red-rover/roverd/) format (i.e., as collected by the [`red-rover`](https://radarml.github.io/red-rover/) system).
    - `nrdk[grt]`: extra dependencies for our GRT reference implementation, which uses a hydra-based configuration system.
    - `nrdk[docs]`: a mkdocs + mkdocs-material + mdocstrings-python documentation stack.
    - `nrdk[dev]`: linters, testing, pre-commit hooks, etc.

## See Also

<div class="grid cards" markdown>

- :material-golf-cart: [`red-rover`](https://radarml.github.io/red-rover/)

    ---

    a multimodal mmWave Radar spectrum ecosystem

- :material-antenna: [`xwr`](https://radarml.github.io/xwr/)

    ---

    python interface for collecting raw time signal data from TI mmWave radars

- :material-cube-outline: [`abstract_dataloader`](https://radarml.github.io/abstract-dataloader/)

    ---

    abstract interface for composable dataloaders and preprocessing pipelines

- :dart: [`dart`](https://wiselabcmu.github.io/dart/)

    ---

    *our prior work, DART: Implicit Doppler Tomography for Radar Novel View Synthesis*

- :material-video-wireless-outline: [`rover`](https://github.com/wiselabcmu/rover)

    ---

    *our previous data collection platform for radar time signal*

</div>
