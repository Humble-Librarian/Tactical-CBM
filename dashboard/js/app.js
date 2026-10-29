/**
 * Tactical-CBM Engine - Application Controller & Interaction Layer
 * Manages player state, formations, drag-and-aim kinematics, slider synchronization, and temporal rollout.
 */

// Application Global State
let autoMode = true;
let showPitchControlSurface = false;
let showLookaheadRollout = false;
let lookaheadHorizon = 0.0;
let isPlayingRollout = false;
let rolloutTimer = null;

let defenders = [];
let attackers = [];
let goalkeeper = null;
let ballCarrier = null;
let selectedPlayer = null;
let hoveredPlayer = null;

let isDraggingPlayer = false;
let isDraggingVelocity = false;
let activeHandlePlayer = null;

let openNetHazard = 0.0;
let isOpenNet = false;
let isOffsideTrapSnapped = false;
let isBreakawayHazard = false;
let currentClosingSpeed = 0.0;

// Initialize players with default formation
function initPlayers() {
    applyFormation("high-press");
}

// Preset Formations
function applyFormation(type) {
    // Defending goal is at X = 120, Y = 40 (Right side)
    goalkeeper = { x: 116, y: 40, vx: 0.0, vy: 0.0, isKeeper: true, role: "GK", number: 1 };

    if (type === "high-press") {
        // Aggressive High-Line: Back four pushed up to X=71-72m with active step-up offside trap
        defenders = [
            { x: 72, y: 18, vx: -1.0, vy: 0.0, role: "FB", number: 3 },
            { x: 71, y: 32, vx: -1.5, vy: 0.0, role: "CB", number: 4 }, // Step-up trap anchor
            { x: 71, y: 48, vx: -1.5, vy: 0.0, role: "CB", number: 5 }, // Step-up trap anchor
            { x: 72, y: 62, vx: -1.0, vy: 0.0, role: "FB", number: 2 },
            { x: 65, y: 28, vx: -0.5, vy: 0.2, role: "DM", number: 6 },
            { x: 64, y: 40, vx: -0.8, vy: 0.0, role: "CM", number: 8 },
            { x: 65, y: 52, vx: -0.5, vy: -0.2, role: "DM", number: 16 },
            { x: 57, y: 36, vx: -0.5, vy: 0.5, role: "W", number: 11 },
            { x: 56, y: 44, vx: -0.8, vy: 0.0, role: "CF", number: 9 },
            { x: 58, y: 22, vx: -0.5, vy: -0.5, role: "W", number: 7 }
        ];

        attackers = [
            { x: 54, y: 40, vx: 1.2, vy: 0.0, isActor: true, role: "AM", number: 10 }, // Ball carrier playmaker
            { x: 92, y: 30, vx: 4.5, vy: 0.2, isActor: false, role: "CF", number: 9 }, // Caught in offside trap
            { x: 94, y: 50, vx: 4.8, vy: -0.2, isActor: false, role: "CF", number: 19 }, // Caught in offside trap
            { x: 70, y: 15, vx: 3.2, vy: 0.5, isActor: false, role: "W", number: 11 },
            { x: 70, y: 65, vx: 3.2, vy: -0.5, isActor: false, role: "W", number: 7 }
        ];
    } else if (type === "5-4-1") {
        // Deep low block: 5 defenders packed inside / just outside penalty area
        defenders = [
            { x: 108, y: 18, vx: 0.0, vy: 0.0, role: "FB", number: 3 },
            { x: 109, y: 29, vx: 0.0, vy: 0.0, role: "CB", number: 4 },
            { x: 110, y: 40, vx: 0.0, vy: 0.0, role: "CB", number: 5 },
            { x: 109, y: 51, vx: 0.0, vy: 0.0, role: "CB", number: 14 },
            { x: 108, y: 62, vx: 0.0, vy: 0.0, role: "FB", number: 2 },
            { x: 98, y: 24, vx: 0.0, vy: 0.0, role: "CM", number: 11 },
            { x: 97, y: 34, vx: 0.0, vy: 0.0, role: "DM", number: 6 },
            { x: 97, y: 46, vx: 0.0, vy: 0.0, role: "DM", number: 8 },
            { x: 98, y: 56, vx: 0.0, vy: 0.0, role: "CM", number: 7 },
            { x: 86, y: 40, vx: 0.0, vy: 0.0, role: "CF", number: 9 }
        ];

        attackers = [
            { x: 80, y: 40, vx: 1.5, vy: 0.0, isActor: true, role: "AM", number: 10 },
            { x: 104, y: 35, vx: 2.0, vy: 0.2, isActor: false, role: "CF", number: 9 },
            { x: 104, y: 45, vx: 2.0, vy: -0.2, isActor: false, role: "CF", number: 19 },
            { x: 92, y: 20, vx: 1.8, vy: 0.4, isActor: false, role: "W", number: 11 },
            { x: 92, y: 60, vx: 1.8, vy: -0.4, isActor: false, role: "W", number: 7 }
        ];
    } else if (type === "breakdown") {
        // Broken structure: Low pressure, multiple attackers inside 18-yard box
        defenders = [
            { x: 96, y: 15, vx: 0.5, vy: 0.0, role: "FB", number: 3 },
            { x: 98, y: 30, vx: 0.8, vy: 0.0, role: "CB", number: 4 },
            { x: 108, y: 55, vx: 0.2, vy: 0.0, role: "CB", number: 5 },
            { x: 105, y: 68, vx: 0.4, vy: 0.0, role: "FB", number: 2 },
            { x: 82, y: 20, vx: 0.0, vy: 0.0, role: "DM", number: 6 },
            { x: 85, y: 60, vx: 0.0, vy: 0.0, role: "DM", number: 8 },
            { x: 70, y: 30, vx: 0.0, vy: 0.0, role: "W", number: 11 },
            { x: 72, y: 50, vx: 0.0, vy: 0.0, role: "W", number: 7 },
            { x: 65, y: 40, vx: 0.0, vy: 0.0, role: "CM", number: 10 },
            { x: 62, y: 20, vx: 0.0, vy: 0.0, role: "CF", number: 9 }
        ];

        attackers = [
            { x: 78, y: 40, vx: 2.2, vy: 0.0, isActor: true, role: "AM", number: 10 },
            { x: 106, y: 36, vx: 4.8, vy: 0.0, isActor: false, role: "CF", number: 9 }, // Lethal Box Finisher
            { x: 105, y: 44, vx: 4.5, vy: 0.0, isActor: false, role: "CF", number: 19 },
            { x: 104, y: 28, vx: 5.2, vy: 0.0, isActor: false, role: "W", number: 11 },
            { x: 85, y: 42, vx: 3.5, vy: 0.0, isActor: false, role: "CM", number: 7 }
        ];
    } else {
        // 4-3-3 Mid-block default
        defenders = [
            { x: 88, y: 18, vx: 0.0, vy: 0.0, role: "FB", number: 3 },
            { x: 89, y: 33, vx: 0.0, vy: 0.0, role: "CB", number: 4 },
            { x: 89, y: 47, vx: 0.0, vy: 0.0, role: "CB", number: 5 },
            { x: 88, y: 62, vx: 0.0, vy: 0.0, role: "FB", number: 2 },
            { x: 76, y: 28, vx: 0.0, vy: 0.0, role: "CM", number: 8 },
            { x: 75, y: 40, vx: 0.0, vy: 0.0, role: "DM", number: 6 },
            { x: 76, y: 52, vx: 0.0, vy: 0.0, role: "CM", number: 10 },
            { x: 62, y: 24, vx: 0.0, vy: 0.0, role: "W", number: 11 },
            { x: 60, y: 40, vx: 0.0, vy: 0.0, role: "CF", number: 9 },
            { x: 62, y: 56, vx: 0.0, vy: 0.0, role: "W", number: 7 }
        ];

        attackers = [
            { x: 68, y: 38, vx: 1.8, vy: 0.0, isActor: true, role: "AM", number: 10 },
            { x: 100, y: 34, vx: 3.2, vy: 0.1, isActor: false, role: "CF", number: 9 },
            { x: 101, y: 46, vx: 3.0, vy: -0.1, isActor: false, role: "CF", number: 19 },
            { x: 80, y: 18, vx: 2.5, vy: 0.3, isActor: false, role: "W", number: 11 },
            { x: 80, y: 62, vx: 2.5, vy: -0.3, isActor: false, role: "W", number: 7 }
        ];
    }

    ballCarrier = attackers[0];
    selectedPlayer = ballCarrier;
    updatePlayerControlPanel();
    recalculate();
}

function resetPitch() {
    const select = document.getElementById("formationSelect");
    applyFormation(select ? select.value : "high-press");
}

// Recalculates concepts, evaluates calibrated Stage 5 CBM threat, and updates UI
function recalculate() {
    let concepts;
    if (autoMode) {
        concepts = computeConceptsFromPitch();
        for (let i = 0; i < 6; ++i) {
            const slider = document.getElementById(`c${i+1}`);
            const valSpan = document.getElementById(`c${i+1}_val`);
            if (slider && valSpan) {
                slider.value = concepts[i].toFixed(2);
                valSpan.innerText = concepts[i].toFixed(2);
            }
        }
    } else {
        concepts = [];
        for (let i = 0; i < 6; ++i) {
            const slider = document.getElementById(`c${i+1}`);
            concepts.push(slider ? parseFloat(slider.value) : 0.5);
        }
    }

    const threat = evaluateStage5(concepts);
    updateDashboard(threat, concepts);
    drawPitch();
}

// Synchronizes the Threat Score, Badges, Attribution Rows, and BCIT Synergy Cards
function updateDashboard(threat, concepts) {
    const threatValEl = document.getElementById("threatValue");
    const threatStatusEl = document.getElementById("threatStatus");
    const threatCardEl = document.getElementById("threatCard");
    const threatLevelTagEl = document.getElementById("threatLevelTag");
    if (!threatValEl) return;

    const pct = (threat * 100).toFixed(1);
    threatValEl.innerText = `${pct}%`;

    let color, level, bgAlpha;
    if (threat < 0.20) {
        color = "#10b981"; // Safe Emerald
        level = "SAFE";
        bgAlpha = "rgba(16, 185, 129, 0.15)";
        if (threatStatusEl) threatStatusEl.innerText = "Defensive Structure Intact | Trap Active";
    } else if (threat < 0.40) {
        color = "#38bdf8"; // Controlled Cyan
        level = "CONTROLLED";
        bgAlpha = "rgba(56, 189, 248, 0.15)";
        if (threatStatusEl) threatStatusEl.innerText = "Controlled Territory | Safe Distances";
    } else if (threat < 0.65) {
        color = "#f59e0b"; // Contested Amber
        level = "CONTESTED";
        bgAlpha = "rgba(245, 158, 11, 0.15)";
        if (threatStatusEl) threatStatusEl.innerText = "Elevated Hazard | Central Lines Under Stress";
    } else {
        color = "#f43f5e"; // Critical Rose
        level = "CRITICAL";
        bgAlpha = "rgba(244, 63, 94, 0.15)";
        if (threatStatusEl) threatStatusEl.innerText = isOpenNet ? "CRITICAL HAZARD: UNGUARDED GOAL" : "High Shot Probability | Box Penetrated";
    }

    threatValEl.style.color = color;
    if (threatLevelTagEl) {
        threatLevelTagEl.innerText = level;
        threatLevelTagEl.style.color = color;
        threatLevelTagEl.style.background = bgAlpha;
    }

    // Dynamic Kinematic HUD Badges
    const hudBadge = document.getElementById("hudBadge");
    const hudDetails = document.getElementById("hudDetails");
    if (hudBadge) {
        if (isOpenNet) {
            hudBadge.className = "hud-badge hud-badge-opennet";
            hudBadge.innerText = "OPEN NET DETECTED";
            if (hudDetails) hudDetails.innerText = `Goal: +${(openNetHazard * 100).toFixed(0)}% logit`;
        } else if (isOffsideTrapSnapped) {
            hudBadge.className = "hud-badge hud-badge-trap";
            hudBadge.innerText = "OFFSIDE TRAP SPRUNG";
            if (hudDetails) hudDetails.innerText = `Closing: ${currentClosingSpeed.toFixed(1)} m/s`;
        } else if (isBreakawayHazard) {
            hudBadge.className = "hud-badge hud-badge-breakaway";
            hudBadge.innerText = "BREAKAWAY IN PROGRESS";
            if (hudDetails) hudDetails.innerText = `Separation: ${Math.abs(currentClosingSpeed).toFixed(1)} m/s`;
        } else {
            hudBadge.className = "hud-badge hud-badge-neutral";
            hudBadge.innerText = "STRUCTURE BALANCED";
            if (hudDetails) hudDetails.innerText = "Relative motion stable";
        }
    }

    // Update factor attribution rows
    const expList = document.getElementById("explanationsList");
    if (expList) {
        expList.innerHTML = "";

        if (isOpenNet) {
            const openNetRow = document.createElement("div");
            openNetRow.className = "exp-row";
            openNetRow.style.borderColor = "rgba(244, 63, 94, 0.5)";
            openNetRow.style.background = "rgba(244, 63, 94, 0.1)";
            openNetRow.innerHTML = `
                <span class="exp-label" style="color: #f43f5e; font-weight: 700;">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#f43f5e" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
                    Goal Exposure Hazard
                </span>
                <span class="exp-metric exp-danger">+${(openNetHazard * 100).toFixed(0)}% Logit (Stranded GK)</span>
            `;
            expList.appendChild(openNetRow);
        }

        const deepestDefX = defenders.length > 0 ? Math.max(...defenders.map(d => d.x)) : 0;
        const offsideLimit = ballCarrier ? Math.max(deepestDefX, ballCarrier.x) : deepestDefX;
        const offsideAttackers = attackers.filter(a => !a.isActor && a.x > offsideLimit + 0.5);

        if (offsideAttackers.length > 0) {
            const offsideRow = document.createElement("div");
            offsideRow.className = "exp-row";
            offsideRow.style.borderColor = "rgba(56, 189, 248, 0.3)";
            offsideRow.style.background = "rgba(56, 189, 248, 0.08)";
            offsideRow.innerHTML = `
                <span class="exp-label" style="color: var(--accent-blue); font-weight: 600;">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"></path><line x1="4" y1="22" x2="4" y2="15"></line></svg>
                    IFAB Offside Trap
                </span>
                <span class="exp-metric exp-safe">${offsideAttackers.length} Offside (Neutralized)</span>
            `;
            expList.appendChild(offsideRow);
        }

        for (let i = 0; i < 6; ++i) {
            const w_lin = CBM_WEIGHTS.linear_weights ? CBM_WEIGHTS.linear_weights[i] : -0.5;
            const contrib = concepts[i] * w_lin;
            const row = document.createElement("div");
            row.className = "exp-row";
            const isSafe = contrib < 0;
            row.innerHTML = `
                <span class="exp-label">${CONCEPT_NAMES[i]}</span>
                <span class="exp-metric ${isSafe ? 'exp-safe' : 'exp-danger'}">
                    ${contrib > 0 ? '+' : ''}${contrib.toFixed(2)} logit
                </span>
            `;
            expList.appendChild(row);
        }
    }

    // Populate Tactical Synergy Matrix (BCIT Pairwise Interactions)
    const synList = document.getElementById("synergyList");
    if (synList && CBM_WEIGHTS.pairwise_interactions) {
        synList.innerHTML = "";
        const synergies = [];
        for (let i = 0; i < 6; ++i) {
            for (let j = i + 1; j < 6; ++j) {
                const key = `C${i+1}_x_C${j+1}`;
                const w_ij = CBM_WEIGHTS.pairwise_interactions[key] || 0.0;
                const jointActivation = concepts[i] * concepts[j];
                const impact = jointActivation * w_ij;
                synergies.push({
                    pair: `C${i+1} x C${j+1}`,
                    name1: CONCEPT_NAMES[i].split(":")[1].trim(),
                    name2: CONCEPT_NAMES[j].split(":")[1].trim(),
                    w: w_ij,
                    activation: jointActivation,
                    impact: impact
                });
            }
        }

        // Sort by absolute impact descending
        synergies.sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));

        // Render top 6 active synergies
        synergies.slice(0, 6).forEach(s => {
            const card = document.createElement("div");
            const isThreat = s.impact > 0.05;
            card.className = `synergy-card ${isThreat ? 'active-threat' : 'active-neutral'}`;
            card.innerHTML = `
                <span class="synergy-name">
                    <span style="color: ${isThreat ? 'var(--accent-red)' : 'var(--accent-blue)'}; font-weight:700;">${s.pair}</span>
                    <span style="color: var(--text-muted); font-size:0.68rem;">${s.name1.split(' ')[0]} + ${s.name2.split(' ')[0]}</span>
                </span>
                <span class="synergy-score" style="color: ${s.impact > 0 ? '#f43f5e' : '#10b981'}; font-family: 'Fira Code', monospace;">
                    ${s.impact > 0 ? '+' : ''}${s.impact.toFixed(3)}
                </span>
            `;
            synList.appendChild(card);
        });
    }
}

// Concept Slider Manual Change
function onSliderChange() {
    if (autoMode) {
        autoMode = false;
        const btn = document.getElementById("modeBtn");
        if (btn) {
            btn.innerText = "Mode: Manual Slider (Frozen Pitch)";
            btn.style.borderColor = "#fbbf24";
            btn.style.color = "#fbbf24";
        }
    }
    for (let i = 0; i < 6; ++i) {
        const slider = document.getElementById(`c${i+1}`);
        const valSpan = document.getElementById(`c${i+1}_val`);
        if (slider && valSpan) {
            const val = parseFloat(slider.value);
            valSpan.innerText = val.toFixed(2);
        }
    }
    recalculate();
}

// Mode Toggle Button (Auto Live Pitch vs Manual Slider)
function toggleAutoMode() {
    autoMode = !autoMode;
    const btn = document.getElementById("modeBtn");
    if (btn) {
        if (autoMode) {
            btn.innerText = "Mode: Auto (Live Pitch)";
            btn.style.borderColor = "#334155";
            btn.style.color = "var(--text-main)";
        } else {
            btn.innerText = "Mode: Manual Slider (Frozen Pitch)";
            btn.style.borderColor = "#fbbf24";
            btn.style.color = "#fbbf24";
        }
    }
    recalculate();
}

// Pitch Control Surface Toggle
function togglePitchControl(checked) {
    showPitchControlSurface = checked;
    drawPitch();
}

// Lookahead Temporal Rollout Toggle
function toggleLookahead(checked) {
    showLookaheadRollout = checked;
    if (!checked && lookaheadHorizon > 0.0) {
        setLookaheadHorizon(0.0);
    }
    drawPitch();
    recalculate();
}

// Slider Input Listener
function onHorizonSliderInput(val) {
    setLookaheadHorizon(parseFloat(val));
}

// Updates Lookahead Horizon (dt in seconds)
function setLookaheadHorizon(dt) {
    lookaheadHorizon = dt;
    const slider = document.getElementById("horizonSlider");
    const readout = document.getElementById("horizonReadout");
    if (slider) slider.value = dt.toFixed(1);
    if (readout) readout.innerText = `+${dt.toFixed(1)}s ${dt === 0 ? "(Live)" : ""}`;
    if (dt > 0 && !showLookaheadRollout) {
        showLookaheadRollout = true;
        const toggle = document.getElementById("lookaheadToggle");
        if (toggle) toggle.checked = true;
    }
    updatePassingWindowBadge();
    recalculate();
}

// Play/Pause Automated Rollout Animation Loop (Clean SVG icons, zero emojis)
function toggleRolloutPlay() {
    isPlayingRollout = !isPlayingRollout;
    const btnIcon = document.getElementById("playRolloutIcon");
    if (isPlayingRollout) {
        if (btnIcon) {
            btnIcon.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>`;
        }
        if (!showLookaheadRollout) {
            showLookaheadRollout = true;
            const toggle = document.getElementById("lookaheadToggle");
            if (toggle) toggle.checked = true;
        }
        rolloutTimer = setInterval(() => {
            let next = lookaheadHorizon + 0.1;
            if (next > 2.05) next = 0.0;
            setLookaheadHorizon(next);
        }, 120);
    } else {
        if (btnIcon) {
            btnIcon.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"/></svg>`;
        }
        if (rolloutTimer) {
            clearInterval(rolloutTimer);
            rolloutTimer = null;
        }
    }
}

// Through-Ball Viability Passing Window Badge
function updatePassingWindowBadge() {
    const badge = document.getElementById("passingWindowBadge");
    if (!badge || !ballCarrier) return;

    const forwardAtts = attackers.filter(a => a.x > ballCarrier.x + 2.0);
    if (forwardAtts.length === 0) {
        badge.className = "window-badge window-badge-closed";
        badge.innerText = "Through-Ball: [NO RUNNER AHEAD]";
        return;
    }

    const primeTarget = forwardAtts[0];
    const dist = Math.hypot(primeTarget.x - ballCarrier.x, primeTarget.y - ballCarrier.y);
    const tauRec = calculatePlayerTTI({ x: primeTarget.x, y: primeTarget.y }, { x: primeTarget.vx, y: primeTarget.vy }, primeTarget.x, primeTarget.y, primeTarget.role, false);
    
    let minDefTau = 999.0;
    defenders.forEach(d => {
        const tauD = calculatePlayerTTI({ x: d.x, y: d.y }, { x: d.vx, y: d.vy }, primeTarget.x, primeTarget.y, d.role, false);
        if (tauD < minDefTau) minDefTau = tauD;
    });

    if (tauRec < minDefTau - 0.20) {
        badge.className = "window-badge";
        badge.innerText = `Through-Ball: [${lookaheadHorizon.toFixed(1)}s - ${(lookaheadHorizon + 0.9).toFixed(1)}s] (VIABLE)`;
    } else {
        badge.className = "window-badge window-badge-closed";
        badge.innerText = `Through-Ball: INTERCEPTED (+${(tauRec - minDefTau).toFixed(2)}s)`;
    }
}

// Mouse Drag & Kinematic Aim Detection Helpers
function getVelocityHandleAt(canvasPx, canvasPy) {
    const all = goalkeeper ? [...defenders, ...attackers, goalkeeper] : [...defenders, ...attackers];
    const s = PLAYER_SCALE || 1.0;
    for (const p of all) {
        const pPx = toPx(p.x, p.y);
        const speed = Math.hypot(p.vx || 0, p.vy || 0);
        const arrowLen = Math.min(speed * 3.8 * s, 46.0 * s);
        const angle = Math.atan2(p.vy || 0, p.vx || 0);
        const tipX = pPx.x + Math.cos(angle) * arrowLen;
        const tipY = pPx.y + Math.sin(angle) * arrowLen;
        if (Math.hypot(canvasPx - tipX, canvasPy - tipY) <= 12 * s) {
            return p;
        }
    }
    return null;
}

function getPlayerAt(mouseX, mouseY) {
    for (const a of attackers) {
        if (Math.hypot(a.x - mouseX, a.y - mouseY) < 3.2) return a;
    }
    for (const d of defenders) {
        if (Math.hypot(d.x - mouseX, d.y - mouseY) < 3.2) return d;
    }
    if (goalkeeper && Math.hypot(goalkeeper.x - mouseX, goalkeeper.y - mouseY) < 3.2) {
        return goalkeeper;
    }
    return null;
}

// Mouse Event Listeners Setup
function initMouseListeners() {
    const pCanvas = document.getElementById("pitchCanvas");
    if (!pCanvas) return;

    pCanvas.addEventListener("mousedown", e => {
        const rect = pCanvas.getBoundingClientRect();
        const canvasPx = (e.clientX - rect.left) * (pCanvas.width / rect.width);
        const canvasPy = (e.clientY - rect.top) * (pCanvas.height / rect.height);
        const pitchPos = fromPx(canvasPx, canvasPy);

        // 1. Check if user clicked on velocity handle tip
        const handlePlayer = getVelocityHandleAt(canvasPx, canvasPy);
        if (handlePlayer) {
            isDraggingVelocity = true;
            activeHandlePlayer = handlePlayer;
            selectedPlayer = handlePlayer;
            updatePlayerControlPanel();
            recalculate();
            return;
        }

        // 2. Check if user clicked on player body
        selectedPlayer = getPlayerAt(pitchPos.x, pitchPos.y);

        if (selectedPlayer) {
            isDraggingPlayer = true;
            if (attackers.includes(selectedPlayer)) {
                attackers.forEach(a => a.isActor = false);
                selectedPlayer.isActor = true;
                ballCarrier = selectedPlayer;
            }
            updatePlayerControlPanel();
            recalculate();
        }
    });

    window.addEventListener("mousemove", e => {
        if (!isDraggingPlayer && !isDraggingVelocity && !pCanvas) return;
        const rect = pCanvas.getBoundingClientRect();
        const canvasPx = (e.clientX - rect.left) * (pCanvas.width / rect.width);
        const canvasPy = (e.clientY - rect.top) * (pCanvas.height / rect.height);
        const pitchPos = fromPx(canvasPx, canvasPy);

        if (isDraggingVelocity && activeHandlePlayer) {
            const pPx = toPx(activeHandlePlayer.x, activeHandlePlayer.y);
            const dx = canvasPx - pPx.x;
            const dy = canvasPy - pPx.y;
            const s = PLAYER_SCALE || 1.0;
            let rawVx = (dx / (3.8 * s));
            let rawVy = (dy / (3.8 * s));

            const rawSpeed = Math.hypot(rawVx, rawVy);
            if (rawSpeed > 10.0) {
                rawVx = (rawVx / rawSpeed) * 10.0;
                rawVy = (rawVy / rawSpeed) * 10.0;
            }
            activeHandlePlayer.vx = rawVx;
            activeHandlePlayer.vy = rawVy;
            updatePlayerControlPanel();
            recalculate();
        } else if (isDraggingPlayer && selectedPlayer) {
            selectedPlayer.x = Math.max(1.5, Math.min(PITCH_W - 1.5, pitchPos.x));
            selectedPlayer.y = Math.max(1.5, Math.min(PITCH_H - 1.5, pitchPos.y));
            recalculate();
        } else {
            // Hover detection
            const prevHover = hoveredPlayer;
            const handleHover = getVelocityHandleAt(canvasPx, canvasPy);
            hoveredPlayer = handleHover || getPlayerAt(pitchPos.x, pitchPos.y);
            if (hoveredPlayer !== prevHover) {
                pCanvas.style.cursor = handleHover ? "crosshair" : (hoveredPlayer ? "grab" : "default");
                drawPitch();
            }
        }
    });

    window.addEventListener("mouseup", () => {
        if (isDraggingPlayer || isDraggingVelocity) {
            isDraggingPlayer = false;
            isDraggingVelocity = false;
            activeHandlePlayer = null;
            recalculate();
        }
    });
}

// Player Control Panel Updates
function updatePlayerControlPanel() {
    if (!selectedPlayer) return;
    const p = selectedPlayer;
    const info = document.getElementById("selectedPlayerInfo");
    const roleCap = ROLE_CAPABILITIES[p.role] || ROLE_CAPABILITIES["AM"];
    const speed = Math.hypot(p.vx || 0, p.vy || 0);

    if (info) {
        const team = attackers.includes(p) ? "Att" : (p.isKeeper ? "GK" : "Def");
        info.innerText = `#${p.number || '10'} ${p.role} (${team})`;
    }

    const velReadout = document.getElementById("velocityReadout");
    if (velReadout) {
        velReadout.innerText = `Sprint: ${speed.toFixed(1)} m/s`;
    }

    const chips = document.querySelectorAll(".role-chip");
    chips.forEach(chip => {
        if (chip.innerText.startsWith(p.role)) {
            chip.className = "role-chip active";
        } else {
            chip.className = "role-chip";
        }
    });
}

function setSelectedPlayerRole(roleTag) {
    if (!selectedPlayer) return;
    selectedPlayer.role = roleTag;
    updatePlayerControlPanel();
    recalculate();
}

function reverseSelectedVelocity() {
    if (!selectedPlayer) return;
    selectedPlayer.vx = -(selectedPlayer.vx || 0);
    selectedPlayer.vy = -(selectedPlayer.vy || 0);
    updatePlayerControlPanel();
    recalculate();
}

function zeroSelectedVelocity() {
    if (!selectedPlayer) return;
    selectedPlayer.vx = 0.0;
    selectedPlayer.vy = 0.0;
    updatePlayerControlPanel();
    recalculate();
}

function triggerStepUpTrap() {
    if (defenders.length === 0) return;
    const lastDef = defenders.reduce((prev, curr) => curr.x > prev.x ? curr : prev, defenders[0]);
    lastDef.vx = -3.2; // Step up aggressively
    lastDef.vy = 0.0;

    if (attackers.length > 0) {
        const leadAtt = attackers.reduce((prev, curr) => curr.x > prev.x ? curr : prev, attackers[0]);
        leadAtt.vx = 5.5; // Sprint forward toward goal
        leadAtt.vy = 0.0;
    }

    selectedPlayer = lastDef;
    updatePlayerControlPanel();
    recalculate();
}

function resetAllVelocities() {
    const all = goalkeeper ? [...defenders, ...attackers, goalkeeper] : [...defenders, ...attackers];
    all.forEach(p => { p.vx = 0.0; p.vy = 0.0; });
    updatePlayerControlPanel();
    recalculate();
}

// Bootstrap Application on DOM Ready
function bootstrap() {
    resizePitchCanvas();
    initMouseListeners();
    initPlayers();

    window.addEventListener("resize", () => {
        resizePitchCanvas();
        drawPitch();
    });

    if (window.ResizeObserver) {
        const wrapper = document.querySelector(".pitch-canvas-wrapper");
        if (wrapper) {
            new ResizeObserver(() => {
                resizePitchCanvas();
                drawPitch();
            }).observe(wrapper);
        }
    }
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootstrap);
} else {
    bootstrap();
}
