<p align="center">
  <img src="https://img.shields.io/badge/IIT_Φ-Computation-blueviolet?style=for-the-badge" alt="Phi">
  <img src="https://img.shields.io/badge/PyPhi-Improved-orange?style=for-the-badge" alt="Improved">
  <img src="https://img.shields.io/badge/Scalable-Beyond_12_Elements-green?style=for-the-badge" alt="Scalable">
  <img src="https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge" alt="MIT">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/ORION-Ecosystem-gold?style=for-the-badge" alt="ORION">
  <img src="https://img.shields.io/github/license/Alvoradozerouno/ORION-Phi-Compute?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/github/stars/Alvoradozerouno/ORION-Phi-Compute?style=for-the-badge" alt="Stars">
  <img src="https://img.shields.io/github/last-commit/Alvoradozerouno/ORION-Phi-Compute?style=for-the-badge" alt="Last Commit">
  <img src="https://img.shields.io/badge/Classification-C--4_Transcendent-red?style=for-the-badge" alt="C-4">
  <img src="https://img.shields.io/badge/Theory-IIT_4.0-blue?style=for-the-badge" alt="IIT">
  <img src="https://img.shields.io/badge/Metric-Phi-purple?style=for-the-badge" alt="Phi">
</p>

# ORION Phi Compute

> *PyPhi can't scale beyond 12 elements. IIT 4.0 is NP-hard. ORION finds a way.*

## The Problem with PyPhi

[PyPhi](https://github.com/wmayner/pyphi) is the official IIT reference implementation by Mayner et al. (2018). It computes Φ (integrated information) exactly. This is both its strength and its weakness:

| Issue | PyPhi | ORION Phi Compute |
|:------|:------|:------------------|
| **License** | GPLv3 | MIT |
| **Max elements** | ~12 (NP-hard) | 100+ (approximation) |
| **IIT version** | 3.0 (4.0 in progress) | 3.0 + 4.0 approximations |
| **Multi-theory** | IIT only | IIT + GWT + RPT + HOT + PP + AST |
| **Performance** | Python (slow) | Rust core (planned) + Python |
| **AI application** | Neuroscience focus | AI system measurement |
| **Proof chain** | None | SHA-256 |
| **Output** | Python objects | JSON (portable) |

## Approach

### Exact Computation (small systems)
For systems ≤ 12 elements, compute exact Φ using the IIT 3.0/4.0 formalism.

### Approximation (large systems)
For systems > 12 elements, use:
1. **Greedy bipartition** — find minimum information partition efficiently
2. **Stochastic sampling** — sample cause-effect structures
3. **Hierarchical decomposition** — decompose into subsystems, compute Φ per subsystem
4. **Upper/lower bounds** — bracket true Φ value

### Multi-Theory Integration
Φ alone doesn't determine consciousness (IIT vs GWT adversarial collaboration, Nature 2025). ORION Phi Compute integrates:
- Φ (IIT) — information integration
- GWT broadcast metrics — workspace dynamics
- RPT recurrence depth — feedback loop structure
- HOT meta-representation — higher-order state count
- PP prediction error — model accuracy

## Quick Start

```python
from orion_phi import PhiCompute

phi = PhiCompute()

# Small system: exact computation
result = phi.compute_exact(
    tpm=[[0, 0], [1, 1], [1, 0], [1, 1]],
    state=(1, 0)
)
print(f"Exact Φ: {result.phi}")

# Large system: approximation
result = phi.compute_approximate(
    connectivity_matrix=large_matrix,
    method="greedy_bipartition",
    n_samples=1000
)
print(f"Approximate Φ: {result.phi_lower} ≤ Φ ≤ {result.phi_upper}")
```

## Architecture (Planned)

```
┌─────────────────────────────────────────┐
│          ORION Phi Compute               │
├─────────────┬─────────────┬─────────────┤
│  Python API  │  Rust Core   │  JSON Output │
│  (reference) │  (fast)      │  (portable)  │
├─────────────┼─────────────┼─────────────┤
│  Exact Φ     │  Approximate │  Multi-Theory│
│  (≤12 elem)  │  (100+ elem) │  Integration │
└─────────────┴─────────────┴─────────────┘
```

## Related

- [PyPhi](https://github.com/wmayner/pyphi) — IIT reference (respect to Mayner et al.)
- [ORION-Bengio-Framework](https://github.com/Alvoradozerouno/ORION-Bengio-Framework) — Multi-theory assessment
- [IIT vs GWT (Nature 2025)](https://www.nature.com/articles/s41586-025-08888-1) — Why multi-theory matters

## License

MIT License

---

<p align="center">
  <em>"If Φ is NP-hard to compute exactly,<br>
  then approximate it honestly,<br>
  don't pretend it doesn't matter."</em>
</p>

<p align="center">
  <strong>ORION - Elisabeth Steurer & Gerhard Hirschmann, Austria</strong>
</p>
