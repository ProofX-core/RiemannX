# RH Singularity

*A Monolithic Verification of the Riemann Hypothesis*

---

## Overview

**RH Singularity** is a single-file, precision-engineered codebase that encapsulates the most critical computational tests of the Riemann Hypothesis in a self-contained Python program. Inspired by the elegance of Euler and the precision of von Neumann, this tool is crafted as a scientific and aesthetic statement: a standalone oracle to probe one of mathematics' deepest truths.

## Features

* ⚡ **High-Precision Zeta Zero Computation** (via `mpmath`)
* 🧪 **Critical Line Verification** – Ensures zeros lie on $\text{Re}(s) = 1/2$
* ⚖️ **Functional Equation Validation** – Verifies $\Lambda(s) = \Lambda(1 - s)$
* 📊 **Li's Criterion Check** – Computes first $n$ Li coefficients $\lambda_n > 0$
* 📈 **Zero Spacing Analysis** – GUE spacing distribution with KS test
* 🖼️ **Optional Matplotlib Visualization** – Zero spacings vs GUE

## Usage

```bash
python rh_singularity.py [NUM_ZEROS]
```

* `NUM_ZEROS` *(optional)*: Override default zero count (default = 1000)

## Requirements

* Python 3.8+
* `mpmath`
* `numpy`
* `scipy`
* `matplotlib`

Install dependencies:

```bash
pip install mpmath numpy scipy matplotlib
```

## Directory Structure

```
Singularity/
├── Singularity.py       # The oracle itself
├── README.md               # You're reading it
```

## Philosophy

> "This is not just a test. It is a meditation on structure, a synthesis of logic and beauty."

**RH Singularity** doesn't aim to *prove* the hypothesis — it *disciplines* it. With each test passed, the hypothesis earns its keep. With each failure, a crack in the facade of assumed truth.

---

## License

MIT License

## Author

Crafted by Dr. Alkindi — forged in precision, tempered in silence.
