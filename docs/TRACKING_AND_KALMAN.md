# Tactical Tracking & Kalman Filtering Specification

## 1. Overview

Raw broadcast camera tracking (StatsBomb 360, TRACAB, Second Spectrum) suffers from camera frustum occlusion and scrum pile-up centroid clustering. 

The tracking module in [`scripts/kalman_tracker.py`](file:///d:/Tactical%20CBM%20Engine/scripts/kalman_tracker.py) implements a **Bounded Linear Kalman Filter with Optical Flow Fusion** to track player velocities and resolve occlusions.

---

## 2. State Vector & Transition Matrix

Each player track maintains a 4D state vector:
$$\mathbf{x}_t = \begin{bmatrix} x_t \\ y_t \\ v_{x, t} \\ v_{y, t} \end{bmatrix}$$

Under constant velocity motion over time step $\Delta t$:
$$\mathbf{x}_{t+\Delta t} = \mathbf{F} \mathbf{x}_t + \mathbf{w}_t, \quad \mathbf{F} = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

---

## 3. Physical Realism Constraints & Boundary Clamping

1. **Sprint Velocity Ceiling:**
   $$\|\mathbf{v}\| = \sqrt{v_x^2 + v_y^2} \le 10.0\text{ m/s}$$
   If extrapolated velocity exceeds $10.0\text{ m/s}$ (e.g. from an erroneous optical detection jump), it is scaled down to the physical human sprint limit.

2. **Pitch Boundary Clamping:**
   $$x \in [0.0, 120.0]\text{m}, \quad y \in [0.0, 80.0]\text{m}$$
   Predicted tracks are hard-clamped to pitch perimeter dimensions.

---

## 4. Optical Flow Fusion & Scrum Handling

In dense penalty box scrums (corner kicks, goalmouth scrambles), bounding box centroids cluster together, causing traditional centroid distance association to fail.

The tracker calculates the **Scrum Congestion Index (SCI)**:
$$\text{SCI} = \frac{1}{\max(N, 1)} \sum_{j \ne i} \mathbb{I}(\text{dist}(p_i, p_j) < 3.0\text{m})$$

When $\text{SCI} > 0.6$:
- Fuses dense optical flow vectors $(\bar{v}_x, \bar{v}_y)$ from motion patch tracking into the measurement update:
  $$\mathbf{z} = \begin{bmatrix} x_{\text{obs}} \\ y_{\text{obs}} \\ \bar{v}_x \\ \bar{v}_y \end{bmatrix}, \quad \mathbf{H} = \mathbf{I}_{4 \times 4}$$
- During total centroid occlusion, optical flow continues to update track velocity until direct re-identification occurs.
- Tracks are retained for an occlusion horizon of $\Delta t \le 2.0\text{s}$ and pruned if unseen for $> 2.0\text{s}$.
