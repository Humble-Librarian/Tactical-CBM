/**
 * Tactical-CBM Engine - Computational Sports Physics & Kinematics Engine
 * Implements William Spearman (2018) Softmin Pitch Control & Andrew's Monotone Convex Hull.
 */

// Pitch Standard Dimensions (StatsBomb Coordinate System)
const PITCH_W = 120.0;
const PITCH_H = 80.0;

// Wave 4 & 5: Canonical 8 Position & Role Taxonomy (Zero Player Names)
const ROLE_CAPABILITIES = {
    "AM": { name: "AM · Advanced Playmaker", theta: 180, clearance: 1.4, gamma_box: 1.0, v_max: 7.8, a_max: 4.2, t_react: 0.55, turn_penalty_max: 0.30, desc: "Wide vision (180°), narrow clearance tolerance (1.4m)" },
    "CF": { name: "CF · Box Finisher",       theta: 90,  clearance: 2.2, gamma_box: 1.5, v_max: 8.5, a_max: 4.8, t_react: 0.60, turn_penalty_max: 0.35, desc: "Lethal 1.5x in-box finisher, heading & poach instinct" },
    "W":  { name: "W · Direct Winger",        theta: 110, clearance: 1.8, gamma_box: 1.1, v_max: 9.5, a_max: 5.2, t_react: 0.50, turn_penalty_max: 0.32, desc: "High sprint velocity (9.5 m/s), 1v1 flank isolation" },
    "DM": { name: "DM · Holding Pivot",       theta: 140, clearance: 1.9, gamma_box: 0.8, v_max: 7.5, a_max: 3.8, t_react: 0.60, turn_penalty_max: 0.34, desc: "Defensive central screen & tempo distribution" },
    "CM": { name: "CM · Box-to-Box",          theta: 120, clearance: 2.0, gamma_box: 1.2, v_max: 8.8, a_max: 4.5, t_react: 0.58, turn_penalty_max: 0.32, desc: "High work-rate engine & late penalty box runs" },
    "FB": { name: "FB · Overlapping Full-Back",theta: 100, clearance: 2.1, gamma_box: 0.9, v_max: 9.2, a_max: 5.0, t_react: 0.52, turn_penalty_max: 0.35, desc: "Flank width, recovery sprints & overlapping support" },
    "CB": { name: "CB · Central Stopper",     theta: 100, clearance: 2.4, gamma_box: 0.6, v_max: 8.2, a_max: 4.4, t_react: 0.65, turn_penalty_max: 0.38, desc: "Offside line anchor & step-up trap execution" },
    "GK": { name: "GK · Sweeper Keeper",      theta: 150, clearance: 2.5, gamma_box: 0.2, v_max: 7.0, a_max: 4.0, t_react: 0.45, turn_penalty_max: 0.28, desc: "Sweeping depth coverage behind high line" }
};

// Fast Sigmoid Look-Up Table (LUT)
const LUT_SIZE = 8192;
const LUT_MIN = -8.0;
const LUT_MAX = 8.0;
const SIGMOID_LUT = new Float32Array(LUT_SIZE);
for (let i = 0; i < LUT_SIZE; ++i) {
    const v = LUT_MIN + (i / LUT_SIZE) * (LUT_MAX - LUT_MIN);
    SIGMOID_LUT[i] = 1.0 / (1.0 + Math.exp(-v));
}

function fastSigmoid(x) {
    if (x <= LUT_MIN) return 0.0;
    if (x >= LUT_MAX) return 1.0;
    const idx = ((x - LUT_MIN) * (LUT_SIZE / (LUT_MAX - LUT_MIN))) | 0;
    return SIGMOID_LUT[idx];
}

// Andrew's Monotone Chain 2D Convex Hull Area (Exact Scipy/ConvexHull parity)
function computeConvexHullArea(points) {
    if (points.length < 3) return 0.0;
    const pts = points.slice().sort((a, b) => a.x === b.x ? a.y - b.y : a.x - b.x);
    const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    const lower = [];
    for (const p of pts) {
        while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
        lower.push(p);
    }
    const upper = [];
    for (let i = pts.length - 1; i >= 0; i--) {
        const p = pts[i];
        while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    const hull = lower.concat(upper);
    if (hull.length < 3) return 0.0;
    let area = 0.0;
    for (let i = 0; i < hull.length; i++) {
        const j = (i + 1) % hull.length;
        area += hull[i].x * hull[j].y;
        area -= hull[j].x * hull[i].y;
    }
    return Math.abs(area) * 0.5;
}

// Point to Line Segment Euclidean Distance
function getDistToSegment(p, a, b) {
    const dx = b.x - a.x, dy = b.y - a.y;
    const lenSq = dx * dx + dy * dy;
    if (lenSq < 1e-4) return Math.hypot(p.x - a.x, p.y - a.y);
    let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / lenSq;
    if (t < 0.05 || t > 0.95) return 999.0;
    return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

// Spearman (2018) 4-Phase Time-To-Intercept (TTI) Physics
function calculatePlayerTTI(p, v, tgtX, tgtY, roleTag, isGK) {
    const role = ROLE_CAPABILITIES[roleTag] || ROLE_CAPABILITIES["CM"];
    const tReact = role.t_react || 0.55;
    const vMax = role.v_max || 8.0;
    const aMax = role.a_max || 4.2;
    const turnMax = role.turn_penalty_max || 0.35;

    const rx = p.x + (v.x || 0) * tReact;
    const ry = p.y + (v.y || 0) * tReact;

    const dx = tgtX - rx;
    const dy = tgtY - ry;
    const dTotal = Math.hypot(dx, dy);

    let dEff = dTotal;
    if (isGK || roleTag === "GK") {
        const inBox = (rx >= 102.0 && ry >= 18.0 && ry <= 62.0);
        dEff = Math.max(0.0, dTotal - (inBox ? 2.5 : 0.8));
    } else {
        dEff = Math.max(0.0, dTotal - 0.4);
    }

    if (dEff <= 1e-4) return tReact;

    const vSpeed = Math.hypot(v.x || 0, v.y || 0);
    let tauTurn = 0.0;
    let vInit = 0.0;

    if (vSpeed > 0.1) {
        const dot = ((v.x || 0) * dx + (v.y || 0) * dy) / (vSpeed * dTotal + 1e-6);
        const clampedDot = Math.max(-1.0, Math.min(1.0, dot));
        const theta = Math.acos(clampedDot);
        tauTurn = turnMax * Math.pow(theta / Math.PI, 2);
        vInit = Math.max(0.0, vSpeed * clampedDot);
    }

    const dAccel = (vMax * vMax - vInit * vInit) / (2.0 * aMax);
    let tRun;
    if (dEff <= dAccel) {
        const disc = Math.max(0.0, vInit * vInit + 2.0 * aMax * dEff);
        tRun = (-vInit + Math.sqrt(disc)) / aMax;
    } else {
        const tAccel = (vMax - vInit) / aMax;
        const dRemain = dEff - dAccel;
        tRun = tAccel + (dRemain / vMax);
    }

    return tReact + tauTurn + tRun;
}

// Extrapolate Entity Positions into Future Horizon
function extrapolatePlayerList(list, dt) {
    const damping = Math.exp(-0.12 * dt);
    return list.map(p => {
        const vx = (p.vx || 0);
        const vy = (p.vy || 0);
        return {
            ...p,
            x: Math.max(0, Math.min(PITCH_W, p.x + vx * dt * damping)),
            y: Math.max(0, Math.min(PITCH_H, p.y + vy * dt * damping)),
            vx: vx * damping,
            vy: vy * damping
        };
    });
}
