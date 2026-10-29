/**
 * Tactical-CBM Engine - Concept Bottleneck Model & BCIT Synergy Tensor
 * Computes the 6 canonical concepts, pairwise synergies, and Beta-calibrated threat.
 */

const CONCEPT_NAMES = [
    "C1: Defensive Line Height",
    "C2: Defensive Compactness",
    "C3: Defensive Width Balance",
    "C4: Ball-Carrier Pressure",
    "C5: Central Lane Blockage",
    "C6: Box Overload Ratio"
];

// Wave 3 Canonical SpatioTemporalCBM Sidecar Parameters
let CBM_WEIGHTS = {
    linear_weights: [-0.9437, 0.3442, -0.4094, 0.7187, 0.0622, 0.9238],
    bias: -0.2044,
    pairwise_interactions: {
        "C1_x_C2": -0.7667,
        "C1_x_C3": -0.9850,
        "C1_x_C4": -0.2943,
        "C1_x_C5": -0.9954,
        "C1_x_C6": -0.0582,
        "C2_x_C3": 0.1437,
        "C2_x_C4": 1.0153,
        "C2_x_C5": 0.5557,
        "C2_x_C6": 1.0202,
        "C3_x_C4": 0.6182,
        "C3_x_C5": -0.1479,
        "C3_x_C6": 0.9241,
        "C4_x_C5": 0.7926,
        "C4_x_C6": 1.0410,
        "C5_x_C6": 0.9381
    },
    beta_calibration: {
        a: 0.8488,
        b: 1.1079,
        c: -2.8372
    },
    optimal_f05_threshold: 0.3662,
    threshold_at_p50: 0.3458
};

// Try dynamically loading latest exported sidecar weights
function loadLiveWeights() {
    const urls = ["models/spatiotemporal_cbm_weights.json", "../models/spatiotemporal_cbm_weights.json"];
    
    function tryFetch(index) {
        if (index >= urls.length) {
            console.log("[CBM Engine] Using embedded canonical weights fallback.");
            return;
        }
        fetch(urls[index])
            .then(res => {
                if (!res.ok) throw new Error("HTTP " + res.status);
                return res.json();
            })
            .then(data => {
                if (data && data.parameters) {
                    CBM_WEIGHTS.linear_weights = data.parameters.linear_weights || CBM_WEIGHTS.linear_weights;
                    CBM_WEIGHTS.bias = data.parameters.bias !== undefined ? data.parameters.bias : CBM_WEIGHTS.bias;
                    CBM_WEIGHTS.pairwise_interactions = data.parameters.pairwise_interactions || CBM_WEIGHTS.pairwise_interactions;
                    if (data.parameters.beta_calibration) {
                        CBM_WEIGHTS.beta_calibration = data.parameters.beta_calibration;
                    }
                    if (data.parameters.threshold_at_p50) {
                        CBM_WEIGHTS.threshold_at_p50 = data.parameters.threshold_at_p50;
                    }
                    console.log("[CBM Engine] Loaded live sidecar weights from " + urls[index]);
                    if (typeof recalculate === "function") recalculate();
                }
            })
            .catch(() => tryFetch(index + 1));
    }
    tryFetch(0);
}
loadLiveWeights();

// 6 Canonical Tactical Concept Extractors from Pitch Entities
function computeConceptsFromPitch() {
    const defs = (showLookaheadRollout && lookaheadHorizon > 0.0) ? extrapolatePlayerList(defenders, lookaheadHorizon) : defenders;
    const atts = (showLookaheadRollout && lookaheadHorizon > 0.0) ? extrapolatePlayerList(attackers, lookaheadHorizon) : attackers;
    const carrier = (showLookaheadRollout && lookaheadHorizon > 0.0 && ballCarrier) ? extrapolatePlayerList([ballCarrier], lookaheadHorizon)[0] : ballCarrier;
    const gk = (showLookaheadRollout && lookaheadHorizon > 0.0 && goalkeeper) ? extrapolatePlayerList([goalkeeper], lookaheadHorizon)[0] : goalkeeper;
    const allDefenders = gk ? [...defs, gk] : defs;

    // Geometric calculations
    const maxDefX = Math.max(...defs.map(d => d.x));
    const hullArea = computeConvexHullArea(defs);
    
    let minY = 80, maxY = 0;
    defs.forEach(d => { minY = Math.min(minY, d.y); maxY = Math.max(maxY, d.y); });
    const avgY = defs.reduce((s, d) => s + d.y, 0) / defs.length;
    const spread = maxY - minY;
    const yOffset = Math.abs(avgY - carrier.y) / 35.0;

    let minDefDist = 999.0;
    let defendersWithin5m = 0;
    allDefenders.forEach(d => {
        const dist = Math.hypot(d.x - carrier.x, d.y - carrier.y);
        if (dist < minDefDist) minDefDist = dist;
        if (dist <= 6.0) defendersWithin5m++;
    });

    // Dual-Engine Execution: WebAssembly C++ engine with transparent JIT fallback
    const effectiveDefX = (carrier && carrier.x > maxDefX) ? carrier.x : maxDefX;
    let c1, c2, c3, c4;
    if (typeof wasmGeometry !== 'undefined' && wasmGeometry) {
        c1 = wasmGeometry.c1_line_height(effectiveDefX);
        c2 = wasmGeometry.c2_compactness(hullArea);
        c3 = wasmGeometry.c3_width_balance(spread, avgY, carrier.y);
        c4 = wasmGeometry.c4_pressure(minDefDist, defendersWithin5m);
    } else {
        c1 = Math.min(Math.max((120.0 - effectiveDefX) / 70.0, 0.0), 1.0);
        c2 = Math.min(Math.max(1.0 - (hullArea / 1200.0), 0.0), 1.0);
        c3 = Math.min(Math.max((spread / 45.0) * (1.0 - yOffset * 0.5), 0.0), 1.0);
        c4 = Math.min(Math.max(Math.exp(-minDefDist / 4.0) * 0.7 + (Math.min(defendersWithin5m, 3) / 3.0) * 0.3, 0.0), 1.0);
    }

    // C5: Central Passing Lane Blockage to Forward Attackers
    const forwardAtt = atts.filter(a => a.x > carrier.x);
    let blockedCount = 0;
    const passerRole = ROLE_CAPABILITIES[carrier.role] || ROLE_CAPABILITIES["AM"];
    const requiredClearance = passerRole.clearance;

    forwardAtt.forEach(a => {
        let isBlocked = false;
        allDefenders.forEach(d => {
            const distToLine = getDistToSegment(d, carrier, a);
            if (distToLine <= requiredClearance) isBlocked = true;
        });
        if (isBlocked) blockedCount++;
    });
    const c5 = forwardAtt.length > 0 ? (blockedCount / forwardAtt.length) : 0.0;

    // C6: Box Overload Ratio (Penalty area X >= 102, Y in [18, 62] with smooth approach kernel)
    const offsideLineX = Math.max(maxDefX, carrier.x);
    let weightedAttInBox = 0.0;
    atts.forEach(a => {
        if (a.x >= 102.0 && a.y >= 18.0 && a.y <= 62.0 && a.x <= (offsideLineX + 0.5)) {
            const roleCap = ROLE_CAPABILITIES[a.role] || ROLE_CAPABILITIES["CF"];
            const sprintBoost = Math.max(0, (a.vx || 0) / (roleCap.v_max || 8.5)) * 0.3;
            weightedAttInBox += roleCap.gamma_box * (1.0 + sprintBoost);
        }
    });

    const isCarrierInBox = carrier.x >= 102.0 && carrier.y >= 18.0 && carrier.y <= 62.0;
    if (isCarrierInBox) {
        const carrierRole = ROLE_CAPABILITIES[carrier.role] || ROLE_CAPABILITIES["AM"];
        weightedAttInBox += carrierRole.gamma_box;
    } else if (carrier.x >= 92.0 && carrier.y >= 18.0 && carrier.y <= 62.0) {
        // Continuous approach kernel: smooth ramp into box threat to prevent artificial 102m data cliffs
        const carrierRole = ROLE_CAPABILITIES[carrier.role] || ROLE_CAPABILITIES["AM"];
        const approachWeight = Math.min(1.0, Math.max(0.0, (carrier.x - 92.0) / 10.0));
        weightedAttInBox += carrierRole.gamma_box * approachWeight;
    }

    const defInBox = allDefenders.filter(d => d.x >= 102.0 && d.y >= 18.0 && d.y <= 62.0).length;

    let c6 = 0.0;
    if (weightedAttInBox > 0) {
        const rawRatio = defInBox === 0 ? 1.0 : (weightedAttInBox / (weightedAttInBox + defInBox));
        
        let isConnectedToBox = isCarrierInBox || carrier.x >= 96.0;
        if (!isConnectedToBox) {
            atts.forEach(a => {
                if (a.x >= 102.0 && a.y >= 18.0 && a.y <= 62.0 && a.x <= (offsideLineX + 0.5)) {
                    let blocked = false;
                    allDefenders.forEach(d => {
                        if (getDistToSegment(d, carrier, a) <= requiredClearance) blocked = true;
                    });
                    if (!blocked) isConnectedToBox = true;
                }
            });
        }

        if (isConnectedToBox) {
            c6 = Math.min(1.0, rawRatio);
        } else {
            const boxProximity = Math.min(Math.max((carrier.x - 30.0) / 72.0, 0.0), 1.0);
            c6 = Math.min(1.0, rawRatio * (0.4 + 0.6 * boxProximity));
        }
    }

    return [c1, c2, c3, c4, c5, c6];
}

// Stage 2 Bilinear Concept Interaction Tensor (BCIT) & Threat Evaluation
function evaluateStage5(concepts) {
    const w_lin = CBM_WEIGHTS.linear_weights;
    const bias = CBM_WEIGHTS.bias;
    const interactions = CBM_WEIGHTS.pairwise_interactions;

    // 1. Linear dot product: w^T * c + bias
    let logit = bias;
    for (let i = 0; i < 6; ++i) {
        logit += concepts[i] * w_lin[i];
    }

    // 2. Bilinear pairwise interaction synergies: sum_{i < j} W_ij * c_i * c_j
    for (let i = 0; i < 6; ++i) {
        for (let j = i + 1; j < 6; ++j) {
            const key = `C${i+1}_x_C${j+1}`;
            if (interactions && interactions[key] !== undefined) {
                const synergyWeight = interactions[key];
                logit += concepts[i] * concepts[j] * synergyWeight;
            }
        }
    }

    // Wave 4: Kinematic Closing Speed Physics & Step-Up Offside Trap Detection
    isOffsideTrapSnapped = false;
    isBreakawayHazard = false;
    currentClosingSpeed = 0.0;
    let lastDef = null;
    let leadAtt = null;

    const defs = (showLookaheadRollout && lookaheadHorizon > 0.0) ? extrapolatePlayerList(defenders, lookaheadHorizon) : defenders;
    const atts = (showLookaheadRollout && lookaheadHorizon > 0.0) ? extrapolatePlayerList(attackers, lookaheadHorizon) : attackers;
    const carrier = (showLookaheadRollout && lookaheadHorizon > 0.0 && ballCarrier) ? extrapolatePlayerList([ballCarrier], lookaheadHorizon)[0] : ballCarrier;
    const gk = (showLookaheadRollout && lookaheadHorizon > 0.0 && goalkeeper) ? extrapolatePlayerList([goalkeeper], lookaheadHorizon)[0] : goalkeeper;

    if (defs.length > 0 && atts.length > 0) {
        lastDef = defs.reduce((prev, curr) => curr.x > prev.x ? curr : prev, defs[0]);
        leadAtt = atts.reduce((prev, curr) => curr.x > prev.x ? curr : prev, atts[0]);

        const dx = leadAtt.x - lastDef.x;
        const dy = leadAtt.y - lastDef.y;
        const dist = Math.hypot(dx, dy) + 1e-4;
        const dvx = (leadAtt.vx || 0) - (lastDef.vx || 0);
        const dvy = (leadAtt.vy || 0) - (lastDef.vy || 0);
        currentClosingSpeed = (dx * dvx + dy * dvy) / dist; // in m/s

        const isCarrierPastLine = carrier && lastDef && carrier.x > (lastDef.x + 0.5);
        const offsideCutoff = lastDef ? Math.max(lastDef.x, carrier ? carrier.x : 0) : 0;
        // IFAB Law 11: Only off-ball teammates ahead of BOTH the last defender AND the ball can be offside.
        const offsideTeammates = atts.filter(a => a !== carrier && a.x > (offsideCutoff + 0.5));

        if (isCarrierPastLine) {
            // Carrier broke past the defensive line into open space: BREAKAWAY HAZARD
            isBreakawayHazard = true;
            logit += 1.2;
        } else if (offsideTeammates.length > 0 && !isOpenNet && ((lastDef.vx || 0) < -0.3 || currentClosingSpeed > 0.8)) {
            // Teammates caught offside by coordinated defensive step-up: suppresses through-ball passing options
            isOffsideTrapSnapped = true;
            logit -= 1.0;
        }
    }

    // Update Kinematic HUD live
    const hudBadge = document.getElementById("hudBadge");
    const hudDetails = document.getElementById("hudDetails");
    if (hudBadge && hudDetails) {
        if (isOffsideTrapSnapped) {
            hudBadge.className = "hud-badge hud-badge-trap";
            hudBadge.innerHTML = "🚩 OFFSIDE TRAP SNAPPED";
            hudDetails.innerHTML = `Last Defender: CB #${lastDef ? lastDef.number : '4'} Step-Up | Closing Speed: +${currentClosingSpeed.toFixed(1)} m/s (Threat Neutralized)`;
        } else if (isBreakawayHazard) {
            hudBadge.className = "hud-badge hud-badge-breakaway";
            hudBadge.innerHTML = "⚡ BREAKAWAY 1v1 HAZARD";
            hudDetails.innerHTML = `Striker Pulling Away | Relative Velocity: ${currentClosingSpeed.toFixed(1)} m/s (Line Breached)`;
        } else {
            hudBadge.className = "hud-badge hud-badge-neutral";
            hudBadge.innerHTML = "🛡️ ACTIVE LINE";
            hudDetails.innerHTML = `Last Defender: CB #${lastDef ? lastDef.number : '4'} · Stopper | Relative Closing: ${currentClosingSpeed.toFixed(1)} m/s`;
        }
    }

    let minDistanceToGoal = Math.hypot(120.0 - carrier.x, 40.0 - carrier.y);
    const maxDefX = Math.max(...defs.map(d => d.x));
    const offsideLineX = Math.max(maxDefX, carrier.x);
    const allDef = gk ? [...defs, gk] : defs;
    const carrierRole = ROLE_CAPABILITIES[carrier.role] || ROLE_CAPABILITIES["AM"];
    const requiredClearance = carrierRole.clearance;

    atts.forEach(a => {
        if (a.x > carrier.x && a.x <= offsideLineX + 0.5) {
            let blocked = false;
            allDef.forEach(d => {
                if (getDistToSegment(d, carrier, a) <= requiredClearance) blocked = true;
            });
            if (!blocked) {
                const aDist = Math.hypot(120.0 - a.x, 40.0 - a.y);
                if (aDist < minDistanceToGoal) {
                    minDistanceToGoal = aDist;
                }
            }
        }
    });

    // Open-Net Keeper Displacement Check (Goalmouth Corridor Coverage)
    openNetHazard = 0.0;
    isOpenNet = false;
    if (gk && carrier) {
        const goalMouth = { x: 120.0, y: 40.0 };
        const distToGoal = Math.hypot(120.0 - carrier.x, 40.0 - carrier.y);
        
        // 1. Check if goalkeeper is positioned to cover the shooting corridor
        const gkDistToShotLine = getDistToSegment(gk, carrier, goalMouth);
        const effectiveGkReach = 2.4 + Math.max(0, (120.0 - gk.x) * 0.12);
        const isGkDisplaced = gkDistToShotLine > effectiveGkReach || gk.x < (carrier.x - 2.0) || Math.abs(gk.y - 40.0) > 8.0;

        // 2. Check if any outfield defender is covering the goal line or corridor
        let isDefCoveringGoal = false;
        defs.forEach(d => {
            if (d.x > carrier.x && d.x >= 112.0 && Math.abs(d.y - 40.0) <= 4.5) {
                isDefCoveringGoal = true;
            } else if (getDistToSegment(d, carrier, goalMouth) <= 1.4 && d.x > carrier.x) {
                isDefCoveringGoal = true;
            }
        });

        // 3. If goalmouth is uncovered and carrier has forward shot view within 75m
        if (isGkDisplaced && !isDefCoveringGoal && carrier.x >= 45.0 && distToGoal <= 75.0) {
            isOpenNet = true;
            const shotRangeFactor = Math.min(Math.max((75.0 - distToGoal) / 55.0, 0.35), 1.0);
            const lateralOffsetFactor = Math.min(1.0, Math.max(0.6, (gkDistToShotLine - effectiveGkReach) / 8.0));
            openNetHazard = shotRangeFactor * lateralOffsetFactor * 2.8;
            logit += openNetHazard;
        }
    }

    // Raw threat probability through sigmoid
    const rawProb = 1.0 / (1.0 + Math.exp(-logit));

    // 3. Post-Hoc Beta Calibration Layer (Kull et al., 2017)
    let calProb = rawProb;
    if (CBM_WEIGHTS.beta_calibration) {
        const a = CBM_WEIGHTS.beta_calibration.a;
        const b = CBM_WEIGHTS.beta_calibration.b;
        const c = CBM_WEIGHTS.beta_calibration.c;
        const s = Math.min(Math.max(rawProb, 1e-7), 1.0 - 1e-7);
        const logOdds = a * Math.log(s) - b * Math.log(1.0 - s) + c;
        calProb = 1.0 / (1.0 + Math.exp(-logOdds));
    }

    // Territorial Distance Scaling
    const progressRatio = Math.max(0.0, 120.0 - minDistanceToGoal) / 105.0;
    const territorialScale = 0.15 + 0.85 * Math.pow(Math.min(Math.max(progressRatio, 0.0), 1.0), 1.25);

    return Math.min(Math.max(calProb * territorialScale, 0.01), 0.99);
}
