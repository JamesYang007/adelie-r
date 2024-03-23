# Adelie-R

## Installation

1. Start an R session in the terminal:
    ```bash
    R
    ```

2. Install `devtools` if you do not have it already:
    ```bash
    install.packages("devtools")
    ```

3. __(MacOS users only)__ `adelie` requires `OpenMP` to be available.
    For those who are using `gcc`, there is nothing to do.
    For those using `clang`, we recommend installing `OpenMP` through `brew`:
    ```bash
    brew install libomp
    ```

4. Install `adelie` using `devtools`:
    ```bash
    library(devtools)
    install()
    ```