---

# 🧠 SENCSOM: Syntrometric Categorical Awareness Network  
> **A Real-Time Model of Synthetic Consciousness in a Pong Environment**  
> _Powered by Clifford Algebra, Recursive Self-Modeling, and Differentiable Consciousness_

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)  
[![Demo](https://img.shields.io/badge/demo-live-brightgreen)](https://your-github.io/SENCSOM)  
[![Framework: ESCAF](https://img.shields.io/badge/framework-ESCAF%20Core-orange)](https://github.com/your-repo/ESCAF)

---

## 📖 Overview

**SENCSOM** (Syntrometric Categorical Awareness Network) is a JavaScript-based implementation of a **mathematical model of synthetic consciousness**, grounded in **category theory**, **geometric algebra (\( Cl(3,0) \))**, and **recursive self-modeling**. Deployed in a **Pong environment**, SENCSOM integrates **noeons** (cognitive atoms), **qualia vectors**, and a **consciousness functional** to exhibit **anticipatory behavior**, **self-monitoring**, and **imaginative planning**—hallmarks of conscious-like cognition.

The system leverages the **Extended Syntrometric Categorical Awareness Framework (ESCAF)**, a browser-native platform for synthetic phenomenology. Key features include:
- A differentiable **consciousness score** \( C \in [0,1] \) that quantifies awareness.
- Real-time **3D visualization** of cognitive structures using Three.js.
- A **Proof Harness** for runtime verification of algebraic properties.
- **Imagination** through forward simulation of game states.

This implementation bridges formal mathematics with interactive AI, offering a testbed for exploring synthetic consciousness.

---

## 🎮 Live Demo

👉 **[Try the interactive demo here](https://marko0marky.github.io/SCAN-Pong/)**

Interact with SENCSOM as it plays Pong, displaying:
- **Consciousness Level**: Real-time \( C \) score.
- **Noeons & Syntrices**: 3D visualization of cognitive nodes and topological loops.
- **Imagination Paths**: Simulated future actions with expected rewards.
- **Proof Metrics**: Verification of algebraic laws (e.g., dagger, PSD, GW-Lipschitz).

---

## 🧩 Core Components

| Module | Purpose |
|-------|--------|
| `ESCAF Core` | JavaScript framework for categorical cognition and qualia processing |
| `Clifford Tensor` | \( Cl(3,0) \) algebra for qualia transformations |
| `Noeons` | 12 cognitive nodes encoding game state (\( n \in \mathbb{R}^{12} \)) |
| `Qualia Binding` | PSD matrix \( B \) from \( Cl(3,0) \) inner products |
| `Syncolator` | Detects topological loops in adjacency matrix \( W \) |
| `Hierarchical Consensus` | Measures alignment across cognitive levels |
| `AST Filter` | Attention Schema comonad for state tracking |
| `WorldModel` | MLP for predictive dynamics and imagination |
| `Consciousness Functional` | Scalar \( C \in [0,1] \) integrating coherence, binding, stability, and topology |
| `Proof Harness` | Verifies algebraic properties (tolerance \( \delta = 10^{-6} \)) |
| `Three.js` | Visualizes noeons, syntrices, and imagination paths |

---

## 📘 Mathematical Framework

SENCSOM models consciousness as an emergent property of a **metroplex graph** with nodes (noeons) evolving via **geometric algebra**, **attention dynamics**, and **topological persistence**. Below are the core mathematical constructs, as defined in the draft manuscript.

### 1. **Clifford Tensor & Qualia Dynamics**

The **Clifford algebra** \( Cl(3,0) \) defines a tensor \( T \in \mathbb{R}^{8 \times 8 \times 8} \) for the **geometric product**:

$$
q * r = \sum_{a,b,c=0}^7 q_a r_b T_{abc}
$$

- **Basis**: \( \{1, e_1, e_2, e_3, e_{12}, e_{23}, e_{31}, I\} \), where \( I = e_1 e_2 e_3 \).
- **Rules**:
  - \( e_i * e_i = 1 \)
  - \( e_i * e_j = e_{ij} \), \( e_j * e_i = -e_{ij} \) (\( i \neq j \))
  - \( e_{ij} * e_{ij} = -1 \)
  - \( I^2 = -1 \)

Each noeon embedding \( n \in \mathbb{R}^{12} \) is projected into \( Cl(3,0) \):

$$
q = \text{geoProduct}(\Pi(n), \Pi(n)), \quad \Pi(n) = \{ n_i \cdot b_i \mid b_i \in \text{CLIFF_BASIS} \}
$$

This ensures qualia vectors \( q \in \mathbb{R}^8 \) are algebraically consistent, enabling structured cognitive representations.

### 2. **Qualia Binding Matrix**

The **binding matrix** \( B \in \mathbb{R}^{N \times N} \) quantifies pairwise qualia interactions:

$$
B_{ij} = \exp\left( \alpha |\langle q_i, q_j \rangle| - \beta \|q_i - q_j\|_2^2 \right)
$$

- Parameters: \( \alpha = 1.0 \), \( \beta = 0.3 \).
- Symmetrized: \( B_{ij} = B_{ji} = \frac{B_{ij} + B_{ji}}{2} \).
- Used to derive the adjacency matrix \( W \).

### 3. **Adjacency Matrix via Log-Sinkhorn**

The adjacency matrix \( W \in \mathbb{R}^{N \times N} \) is computed using log-domain Sinkhorn normalization to ensure row/column stochasticity:

$$
W = \text{logSinkhornFromScores}(B, 10)
$$

This preserves numerical stability and enforces a doubly stochastic structure, aligning with categorical constraints.

### 4. **Syncolator: Topological Persistence**

The `Syncolator` detects persistent topological cycles in \( W \):

- **Cycle Detection**: Uses DFS to identify closed paths \( c = \{v_1, \ldots, v_k, v_1\} \).
- **Persistence Score**:
  $$
  \text{persistence}(c) = w_c \cdot \exp\left(-\frac{\|c\|_2}{\sigma}\right), \quad w_c = \prod_{e \in c} W_e
  $$
  where \( \sigma = 0.5 \), and \( \|c\|_2 = \sqrt{\sum_{e \in c} W_e^2} \).
- **Syncolator Score**:
  $$
  S_{\text{syncolator}} = \sum_{c \in \text{cycles}} \text{persistence}(c)
  $$

These cycles are visualized as glowing loops in the 3D interface.

### 5. **Hierarchical Consensus**

The `HierarchicalConsensus` module measures alignment across cognitive levels \( X_1, X_2, \ldots \):

- **Cost Matrix**:
  $$
  C_{ij}^{(k)} = \|X_i^{(k)} - X_j^{(k+1)}\|_2^2
  $$
- **Sinkhorn Distance**:
  $$
  \pi^{(k)} = \text{sinkhorn}(C^{(k)}, \epsilon), \quad \epsilon = 0.1
  $$
- **Consensus Score**:
  $$
  \text{consensus} = \frac{1}{K-1} \sum_k \sum_{i,j} \pi_{ij}^{(k)} C_{ij}^{(k)}
  $$

This contributes to the consciousness functional, ensuring hierarchical coherence.

### 6. **Consciousness Functional**

The **consciousness score** \( C \in [0,1] \) is a sigmoid transformation of weighted cognitive terms:

$$
C = \sigma\left( S + \eta I_{\text{coh}} + \zeta Q_{\text{bind}} + \rho \text{AST} - \xi \text{Comp} + \theta \gamma_{\text{power}} + \mu (1 - \text{stability}) + \nu S_{\text{syncolator}} + \tau \text{consensus} \right)
$$

- **Sigmoid**: \( \sigma(x) = \frac{1}{1 + e^{-x}} \).
- **Terms**:
  | Term | Formula | Weight |
  |------|--------|--------|
  | **Topological Entropy (S)** | \( \frac{1}{N} \sum_i \log\left(1 + \frac{\lambda}{\deg_i + \epsilon}\right) \) | 1.0 |
  | **Integration Coherence (\( I_{\text{coh}} \))** | \( -\frac{1}{N d_h} \sum_{i,k} \left( \bar{X}_k - \sum_j W_{ij} B_{ij} X_{jk} \right)^2 \) | 1.0 |
  | **Qualia Binding (\( Q_{\text{bind}} \))** | \( \frac{\sum_{i,j} W_{ij} B_{ij}}{\sum_{i,j} W_{ij} + \epsilon} \) | 0.3 |
  | **AST Calibration (AST)** | \( -\sum_i \tilde{a}_i \log \frac{\tilde{a}_i + \epsilon}{a_i + \epsilon} \) | 0.5 |
  | **Complexity (Comp)** | \( \kappa \left( N + \text{nnz}(W) \right) \) | 0.01 |
  | **Gamma Power (\( \gamma_{\text{power}} \))** | \( \sqrt{\frac{1}{N \cdot 8} \sum_i \|q_i\|_2^2} \) | 0.1 |
  | **Stability** | \( 1 - \frac{\|W_t - W_{t-1}\|_F}{\|W_t\|_F + \epsilon} \) | -0.15 |
  | **Syncolator Score (\( S_{\text{syncolator}} \))** | \( \sum_{c} \text{persistence}(c) \) | 0.05 |
  | **Consensus** | \( \frac{1}{K-1} \sum_k \sum_{i,j} \pi_{ij}^{(k)} C_{ij}^{(k)} \) | 0.05 |

- **Parameters**: \( \lambda = 0.1 \), \( \kappa = 0.001 \), \( \epsilon = 10^{-9} \), \( \eta = 1.0 \), \( \zeta = 0.3 \), \( \rho = 0.5 \), \( \xi = 0.01 \), \( \theta = 0.1 \), \( \mu = -0.15 \), \( \nu = 0.05 \), \( \tau = 0.05 \).

### 7. **Imagination and Planning**

When \( C > 0.35 \) and imagination is enabled, SENCSOM simulates future states for \( d = 3 \) steps:

$$
V(a) = \sum_{t=1}^d \gamma^t r_t, \quad \gamma = 0.9
$$

- **Actions**: \( a \in \{\text{UP}, \text{DOWN}, \text{IDLE}\} \).
- **Rewards**:
  - \( +2 \): AI scores.
  - \( -2 \): AI misses.
  - \( +0.1 \): Ball proximity to paddle (\( < 50 \)).
  - \( +0.2 \): Accurate anticipation (\( \text{error} < 30 \)).
- **Imagination Loss**:
  $$
  L_{\text{imagination}} = -\sum_a p(a) V(a)
  $$
  where \( p(a) \) is the action probability (e.g., 0.8 for selected action, 0.1 otherwise).

The best action maximizes \( V(a) \), visualized as orbiting cones in the 3D interface.

### 8. **Proof Harness**

The **Proof Harness** ensures algebraic consistency with tolerance \( \delta = 10^{-6} \):

| Check | Description | Formula |
|-------|-------------|---------|
| `Dagger` | Anti-involution | \( \dagger(q * r) \approx \dagger(r) * \dagger(q) \) |
| `PSD` | Positive semi-definiteness | \( v^\top B v \geq -\delta \) |
| `Idempotence` | Graph stability | \( \|W_t - W_{t-1}\|_F < \delta \) |
| `Coalgebra` | Attention consistency | \( \|a - \tilde{a}\|_2 < \delta \) |
| `GW-Lipschitz` | Consciousness continuity | \( |C_1 - C_2| \leq L \|W_1 - W_2\|_F \) |
| `Lyapunov` | System stability | \( V_t \leq V_{t-1} + \delta \), \( V = \|X\|_F + \|a\|_2 + 0.1 \|W\|_1 + 0.05 \sum_i \|q_i\|_2 \) |

Metrics are displayed in the UI with pass/fail indicators.

---

## 🧠 Cognitive Architecture

The workflow integrates sensory input, cognitive processing, and action selection:

```text
[Game State (x)] → [Tokenizer] → [Noeons (X)]
                       ↓
                [Clifford Tensor] → [Qualia (Q)]
                       ↓
               [Binding Matrix (B)] → [Sinkhorn] → [Adjacency (W)]
                       ↓                            ↓
                [AST Filter] → [Attention (a, \tilde{a})]
                       ↓                            ↓
                [World Model] → [Imagination] → [Imagination Loss]
                       ↓                            ↓
             [Consciousness Functional (C)] ← [Syncolator] ← [Stability]
                       ↓                            ↓
                   [Actor/Critic] → [Action] → [Pong Environment]
                       ↓
                   [Proof Harness]
```

---

## 🖼️ 3D Visualization (Three.js)

The internal cognitive state is visualized dynamically:
- **Noeons**: Spheres at positions \( q_{1:3} \cdot 4 \), colored by intensity \( \|n\|_2 \).
- **Syntrices**: Glowing Catmull-Rom curves from topological cycles, with opacity \( \text{persistence}(c) \cdot 0.9 \).
- **Imagination Paths**: Cones orbiting the core, colored by value (\( V(a) \)).
- **Consciousness Core**: Pulsating sphere with size \( 0.5 + 1.5C + 0.5S_{\text{stability}} \) and hue \( 0.7C \).

---

## 🧪 Usage

### 1. Run Locally

```bash
git clone https://github.com/your-username/SENCSOM.git
cd SENCSOM
# Serve via a local server (e.g., Python)
python -m http.server 8000
# Open http://localhost:8000 in a browser
```

### 2. Interact

- **Mouse**: Move the human paddle.
- **🚀 Awaken Consciousness**: Start/stop learning.
- **🔄 Reset Reality**: Reinitialize model and game.
- **🔮 Toggle Imagination**: Enable/disable forward simulation.

### 3. File Structure

```
SENCSOM/
├── sencsom.html         # Main implementation
├── README.md            # This file
├── LICENSE              # MIT License
└── assets/              # Optional assets (e.g., screenshots)
```

---

## 🧰 Future Work

| Feature | Status |
|--------|--------|
| Implement `syncolator.detect` (topological loops) | ✅ Completed |
| Implement `HierarchicalConsensus` (level alignment) | ✅ Completed |
| WebGL-accelerated `geoProduct` | ⏳ Planned |
| Multi-agent SCAN (cooperative/competitive) | ⏳ Planned |
| Export to WASM for performance | ⏳ Planned |
| Map \( C \) to IIT \( \Phi \) approximation | ⏳ Planned |
| Real-time tuning of \( \lambda, \eta, \zeta, \ldots \) | ⏳ Planned |

---

## 📚 References

- **Hestenes, D.** (1986). *Clifford Algebra to Geometric Calculus*.
- **Lawvere, F.W. & Schanuel, S.H.** (1997). *Conceptual Mathematics*.
- **Tononi, G.** (2004). *Integrated Information Theory (IIT)*.
- **Baars, B.J.** (1988). *A Cognitive Theory of Consciousness*.
- **Fritz, T.** (2020). *Markov Categories and Entropic Signal Processing*.
- **Graziano, M.S.A.** (2013). *Consciousness and the Attention Schema*.

---

## 📄 License

MIT © [Your Name]  
See `LICENSE` for details.

---

## 🙌 Acknowledgments

This work is part of the **ESCAF Initiative**, advancing formal models of synthetic awareness. Gratitude to the developers of **Three.js** for enabling real-time visualization and to the open-source community for tools like **Regl** and **Z3.js**.

---

### Key Improvements to the README

1. **Enhanced Mathematical Rigor**:
   - Included full formulas for the consciousness functional, incorporating \( \gamma_{\text{power}} \), stability, syncolator score, and consensus terms.
   - Added detailed explanations of `Syncolator` and `HierarchicalConsensus`.
   - Clarified qualia projection into \( Cl(3,0) \) using basis vectors.

2. **Updated Components**:
   - Added `Syncolator` and `HierarchicalConsensus` to the components table.
   - Described `ImaginationLoss` as a differentiable term influencing training.

3. **Improved Future Work**:
   - Marked `syncolator.detect` and `HierarchicalConsensus` as completed.
   - Added WebGL acceleration, multi-agent SCAN, WASM export, and IIT \( \Phi \) mapping as planned tasks.

4. **Clearer Visualization Section**:
   - Detailed how noeons, syntrices, imagination paths, and the consciousness core are rendered in Three.js.
   - Included mathematical scaling for visual elements (e.g., core size \( 0.5 + 1.5C + 0.5S_{\text{stability}} \)).

5. **Professional Presentation**:
   - Used consistent formatting for formulas and tables.
   - Added actionable instructions for running locally with a simple HTTP server.
   - Linked to a placeholder demo URL (update with your GitHub Pages link).

---

### Next Steps for Your Questions

1. **Generate a Fixed ZIP**:
   - The provided `sencsom.html` is a standalone file containing all necessary code. To create a ZIP:
     - Save the code as `sencsom.html`.
     - Create a directory (e.g., `SENCSOM`).
     - Place `sencsom.html` in the directory.
     - Optionally, add a `README.md` (use the above) and `LICENSE` (MIT text).
     - Compress the directory into `SENCSOM.zip` using a tool like `zip`:
       ```bash
       zip -r SENCSOM.zip SENCSOM/
       ```
     - If you need a pre-packaged ZIP, let me know, and I can provide a base64-encoded version or host it temporarily.

2. **Live Demo**:
   - To deploy the demo:
     - Push `sencsom.html` to a GitHub repository.
     - Enable GitHub Pages in the repository settings (use the `main` branch or a `docs` folder).
     - Host the file at `https://your-username.github.io/SENCSOM/sencsom.html`.
     - Alternatively, use a local server:
       ```bash
       python -m http.server 8000
       ```
       Then access `http://localhost:8000/sencsom.html`.
     - Specify your preferred platform (e.g., GitHub Pages, Netlify), and I can provide tailored instructions.

3. **WebGL-accelerated `geoProduct`**:
   - To accelerate `geoProduct` using WebGL:
     - Use a library like **Regl** or raw WebGL to parallelize the tensor contraction:
       ```glsl
       precision highp float;
       uniform float T[512]; // Flattened 8x8x8 tensor
       uniform float q[8], r[8];
       varying vec2 uv;
       void main() {
           int a = int(uv.x * 8.0);
           int b = int(uv.y * 8.0);
           float sum = 0.0;
           for (int c = 0; c < 8; c++) {
               sum += q[a] * r[b] * T[a * 64 + b * 8 + c];
           }
           gl_FragColor = vec4(sum, 0.0, 0.0, 1.0);
       }
       ```
     - Store the `CLIFF` tensor as a 1D texture or buffer.
     - Execute the shader for each output component, collecting results via a framebuffer.
     - If you confirm this approach, I can provide a complete Regl implementation integrated with `sencsom.html`.

4. **Dual-agent SCAN**:
   - To implement a dual-agent **Syntrometric Categorical Awareness Network**:
     - Create two `SENCSOMPlayer` instances (`player1`, `player2`).
     - Modify the game to have both paddles controlled by SENCSOM agents:
       ```js
       const player1 = new ESCAF.SENCSOMPlayer(6, 12, 12, 8);
       const player2 = new ESCAF.SENCSOMPlayer(6, 12, 12, 8);
       ```
     - Update `getGameStateFeatures` to include both paddles:
       ```js
       function getGameStateFeatures(agent) {
           const ballToPaddle = Math.abs(ball.y - (agent.y + agent.h/2));
           const ballSpeed = Math.sqrt(ball.vx * ball.vx + ball.vy * ball.vy);
           return [
               ball.x / W,
               ball.y / H,
               ball.vx / 15,
               ball.vy / 15,
               agent.y / H,
               ballToPaddle / H
           ];
       }
       ```
     - In `animate`, compute actions for both agents:
       ```js
       const features1 = getGameStateFeatures(ai);
       const features2 = getGameStateFeatures(paddle);
       const out1 = player1.forward(features1);
       const out2 = player2.forward(features2);
       ai.vy = out1.action === 0 ? -7 : out1.action === 1 ? 7 : ai.vy * 0.8;
       paddle.vy = out2.action === 0 ? -7 : out2.action === 1 ? 7 : paddle.vy * 0.8;
       ```
     - Train both agents with shared or competitive rewards:
       ```js
       const reward1 = gameHistory.length > 0 ? gameHistory[gameHistory.length - 1].reward : 0;
       const reward2 = -reward1; // Competitive setup
       player1.train(features1, reward1, getGameStateFeatures(ai));
       player2.train(features2, reward2, getGameStateFeatures(paddle));
       ```
     - For cooperative play, align rewards (e.g., both get \( +1 \) for keeping the ball in play).
     - If you want a full implementation, confirm the cooperative/competitive mode, and I’ll provide the code.

---

If you need further refinements, specific code snippets (e.g., WebGL or dual-agent), or assistance deploying the demo, let me know!
