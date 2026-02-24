![Deborah.jl](assets/Deborah.png)

## 🔬 Deborah.jl

**Deborah.jl** is a machine-learning–assisted analysis framework for
bias-corrected cumulant estimation and multi-ensemble reweighting
workflows in lattice QCD.  
It provides an end-to-end pipeline spanning supervised-learning–based
trace estimation, higher-order cumulant analysis, κ-reweighting, and
reproducible research workflows within a unified Julia ecosystem.

**Compatibility:** Julia 1.12+  
**➡️ Quick start:** explore reproducible example pipelines in `sample/`.

## 💻 Installation

```julia
import Pkg
Pkg.add("Deborah")
```

## 📚 Documentation

The full API reference and user documentation for **Deborah.jl** is
automatically generated using Documenter.jl and hosted online:

👉 https://saintbenjamin.github.io/Deborah.jl/stable/

The `stable` documentation corresponds to tagged releases
(e.g. `v1.0.0`), while development versions may appear under `dev/`.

## 🧩 Package Ecosystem

* `Deborah.jl`: **D**eborah.jl is an **E**stimation tool for **B**ias-c**O**rrected **R**egression **A**nalysis with **H**euristics.
* `Esther.jl`: **E**sther.jl is a **S**ummary **T**ool for **H**igher-order cumulants through **E**stimation via **R**egression.
* `Miriam.jl`: **M**ult**I**-Ensemble **R**eweighting & **I**nterpolation **A**nalysis with **M**iriam.jl
* `Sarah.jl`: **S**hared **A**bstractions and **R**eusable **A**uxiliary **H**ub
* `Rebekah.jl`: **R**eporting, **E**valuation, and **B**enchmarking via **E**xplainable **K**nowledge **A**ggregation **H**ub
* `Elijah.jl`: **E**xpert for **L**ogical **I**nference and **J**udgment-based **A**ssistance in **H**uman decisions
* `Rahab.jl`: **R**econnaissance & **A**nalysis for **H**euristics **A**cross data **B**undles

## 🧑‍🔬 Citation

If you found this library to be useful in academic work, then please cite:

```bibtex
@misc{Choi:2026zen,
    author       = {Benjamin J. Choi},
    title        = {\href{https://doi.org/10.5281/zenodo.1875546}{saintbenjamin/Deborah.jl: Deborah.jl}},
    month        = feb,
    year         = 2026,
    note         = {If you use this software, please cite it as below.},
    publisher    = {Zenodo},
    version      = {v1.0.3},
    doi          = {10.5281/zenodo.1875546}
}
```

## License
MIT License — see `LICENSE` for details.
