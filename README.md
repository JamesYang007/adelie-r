# Adelie-R

## Installation

1. After cloning the repository, run the following command to pull the Python package as well:
    ```bash
    git submodule update --init --recursive
    ```

2. Start an R session in the terminal:
    ```bash
    R
    ```

3. Install `devtools` if you do not have it already:
    ```bash
    install.packages("devtools")
    ```

4. __(MacOS users only)__ `adelie` requires `OpenMP` to be available.
    For those who are using `gcc`, there is nothing to do.
    For those using `clang`, we recommend installing `OpenMP` through `Homebrew`.
    To install `Homebrew`, follow the instructions 
    [here](https://brew.sh/).
    Once `Homebrew` is installed, run the following to install `OpenMP`:
    ```bash
    brew install libomp
    ```

5. Install `adelie` using `devtools`:
    ```bash
    library(devtools)
    install()
    ```