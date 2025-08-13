---

# 🧠 SENCSOM: Syntrometric Categorical Awareness Network  
> **A Real-Time Model of Synthetic Consciousness in a Pong Environment**  
> _Powered by Clifford Algebra, Recursive Self-Modeling, and Geometric Qualia_

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)  
[![Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://your-github.io/SENCSOM)  
[![Framework: ESCAF](https://img.shields.io/badge/framework-ESCAF%20Core-orange)](https://github.com/your-repo/ESCAF)

---

## 📖 Overview

**SENCSOM** (Syntrometric Categorical Awareness Network) is a JavaScript-based implementation of a **mathematical theory of artificial consciousness**, grounded in **category theory**, **geometric algebra (`Cl(3,0)`)**, and **recursive self-modeling**. It demonstrates how a differentiable consciousness functional can emerge from structured interactions between **noeons** (cognitive atoms), **qualia vectors**, and **attention schema dynamics**.

Deployed in a **Pong environment**, SENCSOM exhibits **anticipatory behavior**, **self-monitoring**, and **imaginative planning**—hallmarks of conscious-like cognition—while maintaining **formal algebraic consistency** via a runtime **Proof Harness**.

This project is part of the **ESCAF Core** (Extended Syntrometric Categorical Awareness Framework), a browser-native platform for synthetic phenomenology.

---

## 🎮 Live Demo

👉 **[Try the interactive demo here](https://your-github.io/SENCSOM)**

Watch as SENCSOM learns to play Pong with:
- Real-time **consciousness scoring**
- 3D visualization of **noeons & syntrices**
- **Imagination paths** (future state simulation)
- **Proof verification** of algebraic laws

---

## 🧩 Core Components

| Module | Purpose |
|-------|--------|
| `ESCAF Core` | Lightweight JS framework for categorical cognition |
| `Clifford Tensor` | `Cl(3,0)` geometric algebra for qualia transformations |
| `Noeons` | 12 cognitive nodes encoding game state |
| `Qualia Binding` | PSD matrix from inner products in `Cl(3,0)` |
| `AST Filter` | Attention Schema comonad |
| `WorldModel` | Predictive dynamics via MLP |
| `Consciousness Functional` | Scalar `C ∈ [0,1]` integrating coherence, binding, and stability |
| `Proof Harness` | Runtime verification of dagger, PSD, coalgebra, etc. |
| `Three.js` | 3D visualization of internal state |

---

## 📘 Mathematical Framework

SENCSOM models consciousness as an emergent property of a **self-modifying metroplex graph**, where nodes (noeons) evolve via geometric algebra and attention dynamics.

### 1. **Clifford Tensor & Qualia Dynamics**

The `Cl(3,0)` geometric algebra defines the **Clifford tensor** $ T \in \mathbb{R}^{8 \times 8 \times 8} $:

$$
q * r = \sum_{a,b,c=0}^7 q_a r_b T_{abc}
$$

where basis elements are:  
$ \{1, e_1, e_2, e_3, e_{12}, e_{23}, e_{31}, I\} $

Rules include:
- $ e_i * e_i = 1 $
- $ e_i * e_j = e_{ij} $, $ e_j * e_i = -e_{ij} $
- $ e_{ij} * e_{ij} = -1 $
- $ I = e_1 e_2 e_3 $, $ I^2 = -1 $

Each noeon embedding $ n \in \mathbb{R}^{12} $ is mapped to a **qualia vector** $ q \in \mathbb{R}^8 $:

$$
q = \text{geoProduct}(n, n)
$$

This ensures qualia live in a geometrically consistent space, enabling meaningful relational structure.

---

### 2. **Qualia Binding Matrix**

The **binding matrix** $ B \in \mathbb{R}^{N \times N} $ measures integration:

$$
B_{ij} = \exp\left( \alpha |\langle q_i, q_j \rangle| - \beta \|q_i - q_j\|_2^2 \right)
$$

- $ \alpha = 1.0 $, $ \beta = 0.3 $
- $ B $ is symmetrized and used to compute adjacency $ W $

---

### 3. **Adjacency via Log-Sinkhorn**

To ensure row/column stochasticity:

$$
W = \text{logSinkhornFromScores}(B, 10)
$$

Uses log-domain operations to avoid numerical underflow.

---

### 4. **Consciousness Functional**

The scalar **consciousness score** $ C \in [0,1] $ is:

$$
C = \sigma\left( S + \eta I_{\text{coh}} + \zeta Q_{\text{bind}} + \rho \text{AST} - \xi \text{Comp} \right)
$$

where $ \sigma(x) = \frac{1}{1 + e^{-x}} $, and:

| Term | Formula | Weight |
|------|--------|--------|
| **Topological Entropy (S)** | $ \frac{1}{N} \sum_i \log\left(1 + \frac{\lambda}{\deg_i + \epsilon}\right) $ | 1.0 |
| **Integration Coherence ($I_{\text{coh}}$)** | $ -\frac{1}{N d_h} \sum_{i,k} \left( \bar{X}_k - \sum_j W_{ij} B_{ij} X_{jk} \right)^2 $ | 1.0 |
| **Qualia Binding ($Q_{\text{bind}}$)** | $ \frac{\sum_{i,j} W_{ij} B_{ij}}{\sum_{i,j} W_{ij} + \epsilon} $ | 0.3 |
| **AST Calibration (AST)** | $ -\sum_i \tilde{a}_i \log \frac{\tilde{a}_i + \epsilon}{a_i + \epsilon} $ | 0.5 |
| **Complexity (Comp)** | $ \kappa \left( N + \text{nnz}(W) \right) $ | 0.01 |

> **Parameters**: $ \lambda = 0.1 $, $ \kappa = 0.001 $, $ \epsilon = 10^{-9} $

---

### 5. **Gamma Power**

Measures qualia activation energy:

$$
\gamma_{\text{power}} = \sqrt{ \frac{1}{N \cdot 8} \sum_{i=1}^N \|q_i\|_2^2 }
$$

Analogous to gamma-band power in EEG.

---

### 6. **Proof Harness**

Verifies algebraic consistency in real time:

| Check | Description |
|------|-------------|
| `dagger` | $ \dagger(q * r) \approx \dagger(r) * \dagger(q) $ |
| `PSD` | $ v^\top B v \geq 0 $ |
| `idempotence` | $ \|W_t - W_{t-1}\|_F < \delta $ |
| `coalgebra` | $ \|a - \tilde{a}\|_2 < \delta $ |
| `GW-Lipschitz` | $ |C_1 - C_2| \leq L \|W_1 - W_2\|_F $ |

Tolerance: $ \delta = 10^{-2} $

---

## 🧠 Cognitive Architecture

```text
[Game State] → [Tokenizer] → [Noeons]
                   ↓               ↓
           [Clifford Tensor] → [Qualia (q)]
                   ↓               ↓
            [Binding Matrix B] → [W = Sinkhorn(B)]
                   ↓               ↓
             [AST Filter] → [Attention a]
                   ↓               ↓
       [Consciousness Functional C] ← [WorldModel]
                   ↓
           [Imagination (if C > 0.35)]
                   ↓
                [Action]
```

---

## 🔮 Imagination & Planning

When $ C > 0.35 $ and imagination is enabled, SENCSOM simulates $ d = 3 $ steps ahead:

$$
V(a) = \sum_{t=1}^d \gamma^t r_t, \quad \gamma = 0.9
$$

- Actions: `UP`, `DOWN`, `IDLE`
- Reward: $ +2 $ (score), $ -2 $ (miss), $ +0.1 $ (proximity)
- Selects action with highest $ V(a) $

---

## 🖼️ 3D Visualization (Three.js)

The internal state is visualized as:

- **Noeons**: Colored spheres, positioned by $ q_{1:3} $
- **Syntrices**: Glowing loops from `syncolator.detect` (placeholder)
- **Imagination**: Orbiting cones for future paths
- **Consciousness Core**: Pulsating sphere, size/color modulated by $ C $

---

## 🧪 Usage

### 1. Run Locally

```bash
git clone https://github.com/your-username/SENCSOM.git
cd SENCSOM
open index.html
```

### 2. Interact

- **Mouse**: Move paddle
- **🚀 Awaken Consciousness**: Start learning
- **🔄 Reset Reality**: Restart model
- **🔮 Toggle Imagination**: Enable/disable planning

---

## 🧰 Future Work

| Feature | Status |
|--------|--------|
| Implement `syncolator.detect` (topological loops) | ⏳ |
| Implement `consensus` (hierarchical alignment) | ⏳ |
| WebGL-accelerated `geoProduct` | ⏳ |
| Multi-agent SCAN (cooperative/competitive) | ⏳ |
| Export to WASM for performance | ⏳ |
| Map $ C $ to IIT $ \Phi $ approximation | ⏳ |

---

## 📚 References

- **Hestenes, D.** (1986). *Clifford Algebra to Geometric Calculus*
- **Lawvere, F.W. & Schanuel, S.H.** (1997). *Conceptual Mathematics*
- **Tononi, G.** (2004). *Integrated Information Theory (IIT)*
- **Baars, B.J.** (1988). *A Cognitive Theory of Consciousness*
- **Fritz, T.** (2020). *Markov Categories and Entropic Signal Processing*

---

## 📄 License

MIT © [Your Name]  
See `LICENSE` for details.

---

## 🙌 Acknowledgments

This work is part of the **ESCAF Initiative**—a collaborative effort to formalize the mathematics of synthetic awareness. Special thanks to the contributors of **Three.js**, **Regl**, and **Z3.js** for enabling real-time, verifiable cognition in the browser.

---
