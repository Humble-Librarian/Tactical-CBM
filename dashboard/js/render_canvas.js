/**
 * Tactical-CBM Engine - High-Performance Broadcast Canvas Renderer
 * Renders high-fidelity stadium turf, precision pitch markings, William Spearman (2018)
 * pitch control raster, and authentic football jersey player tokens.
 */

let canvas = document.getElementById("pitchCanvas");
let ctx = canvas ? canvas.getContext("2d") : null;

function ensureCanvasContext() {
    if (!canvas) {
        canvas = document.getElementById("pitchCanvas");
    }
    if (canvas && !ctx) {
        ctx = canvas.getContext("2d");
    }
    return ctx;
}

// Pitch Dimensions & Responsive Coordinate Mapping (StatsBomb Regulatory Pitch)
let MARGIN_X = 24.0;
let MARGIN_Y = 20.0;
let FIELD_W = canvas ? canvas.width - (MARGIN_X * 2) : 1152;
let FIELD_H = canvas ? canvas.height - (MARGIN_Y * 2) : 760;
let PLAYER_SCALE = 1.0;

function toPx(x, y) {
    return {
        x: MARGIN_X + (x / PITCH_W) * FIELD_W,
        y: MARGIN_Y + (y / PITCH_H) * FIELD_H
    };
}

function fromPx(px, py) {
    return {
        x: Math.max(0, Math.min(PITCH_W, ((px - MARGIN_X) / FIELD_W) * PITCH_W)),
        y: Math.max(0, Math.min(PITCH_H, ((py - MARGIN_Y) / FIELD_H) * PITCH_H))
    };
}

function resizePitchCanvas() {
    ensureCanvasContext();
    if (!canvas) return;
    const wrapper = document.querySelector(".pitch-canvas-wrapper");
    if (!wrapper) return;
    const availW = wrapper.clientWidth - 8;
    const availH = wrapper.clientHeight - 8;
    if (availW <= 50 || availH <= 50) return;

    const PITCH_ASPECT = 1.5; // 120m / 80m
    let displayW, displayH;

    if (availW / availH > PITCH_ASPECT) {
        displayH = Math.floor(availH);
        displayW = Math.floor(displayH * PITCH_ASPECT);
    } else {
        displayW = Math.floor(availW);
        displayH = Math.floor(displayW / PITCH_ASPECT);
    }

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(displayW * dpr);
    canvas.height = Math.round(displayH * dpr);
    canvas.style.width = `${displayW}px`;
    canvas.style.height = `${displayH}px`;

    MARGIN_X = 24.0 * dpr;
    MARGIN_Y = 20.0 * dpr;
    FIELD_W = canvas.width - (MARGIN_X * 2);
    FIELD_H = canvas.height - (MARGIN_Y * 2);

    PLAYER_SCALE = Math.max(0.90, Math.min(1.30, displayW / 960.0)) * dpr;

    drawPitch();
}

// Wave 5 Dynamic Pitch Control Surface Offscreen Buffers (William Spearman 2018)
const PC_GRID_W = 60;
const PC_GRID_H = 40;
const offscreenPitchCanvas = document.createElement("canvas");
offscreenPitchCanvas.width = PC_GRID_W;
offscreenPitchCanvas.height = PC_GRID_H;
const offscreenPitchCtx = offscreenPitchCanvas.getContext("2d");
const pitchImageData = offscreenPitchCtx.createImageData(PC_GRID_W, PC_GRID_H);
const pitchPixelBuf32 = new Uint32Array(pitchImageData.data.buffer);

function renderPitchControlSurface() {
    const SIGMA_EFF = 0.248;
    const stepX = PITCH_W / PC_GRID_W;
    const stepY = PITCH_H / PC_GRID_H;

    const allDefs = [...defenders];
    if (goalkeeper) allDefs.push(goalkeeper);

    const allAtts = [...attackers];
    if (ballCarrier && !allAtts.includes(ballCarrier)) allAtts.push(ballCarrier);

    const defsToUse = lookaheadHorizon > 0.0 ? extrapolatePlayerList(allDefs, lookaheadHorizon) : allDefs;
    const attsToUse = lookaheadHorizon > 0.0 ? extrapolatePlayerList(allAtts, lookaheadHorizon) : allAtts;

    for (let r = 0; r < PC_GRID_H; ++r) {
        const tgtY = (r + 0.5) * stepY;
        for (let c = 0; c < PC_GRID_W; ++c) {
            const tgtX = (c + 0.5) * stepX;
            const idx = r * PC_GRID_W + c;

            let minAttTau = 999.0;
            let minDefTau = 999.0;

            for (let i = 0; i < attsToUse.length; ++i) {
                const a = attsToUse[i];
                const tau = calculatePlayerTTI({ x: a.x, y: a.y }, { x: a.vx, y: a.vy }, tgtX, tgtY, a.role, false);
                if (tau < minAttTau) minAttTau = tau;
            }

            for (let i = 0; i < defsToUse.length; ++i) {
                const d = defsToUse[i];
                const isGK = (d.role === "GK" || d === goalkeeper);
                const tau = calculatePlayerTTI({ x: d.x, y: d.y }, { x: d.vx, y: d.vy }, tgtX, tgtY, d.role, isGK);
                if (tau < minDefTau) minDefTau = tau;
            }

            const logit = (minDefTau - minAttTau) / SIGMA_EFF;
            const pAtt = fastSigmoid(logit);
            const certainty = Math.abs(pAtt - 0.5) * 2.0;
            const alpha = Math.min(135, Math.round(115 * Math.pow(certainty, 1.30)));

            if (pAtt >= 0.5) {
                // Attacking dominance: Rose / Crimson (#e11d48)
                pitchPixelBuf32[idx] = (alpha << 24) | (0x48 << 16) | (0x1d << 8) | 0xe1;
            } else {
                // Defensive dominance: Cobalt Blue (#2563eb)
                pitchPixelBuf32[idx] = (alpha << 24) | (0xeb << 16) | (0x63 << 8) | 0x25;
            }
        }
    }

    offscreenPitchCtx.putImageData(pitchImageData, 0, 0);
}

// Master Pitch Renderer
function drawPitch() {
    ensureCanvasContext();
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Stadium Grass Mowing Stripes (Broadcast Velvety Emerald)
    const numStripes = 14;
    const stripeW = FIELD_W / numStripes;
    for (let s = 0; s < numStripes; ++s) {
        ctx.fillStyle = s % 2 === 0 ? "#0a2e1b" : "#0d3822";
        ctx.fillRect(MARGIN_X + (s * stripeW), MARGIN_Y, stripeW, FIELD_H);
    }

    // Outer Grass Margin Border
    ctx.fillStyle = "#061f12";
    ctx.fillRect(0, 0, canvas.width, MARGIN_Y);
    ctx.fillRect(0, canvas.height - MARGIN_Y, canvas.width, MARGIN_Y);
    ctx.fillRect(0, 0, MARGIN_X, canvas.height);
    ctx.fillRect(canvas.width - MARGIN_X, 0, MARGIN_X, canvas.height);

    // Dynamic Pitch Control Surface
    if (showPitchControlSurface) {
        renderPitchControlSurface();
        ctx.save();
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        ctx.drawImage(offscreenPitchCanvas, MARGIN_X, MARGIN_Y, FIELD_W, FIELD_H);
        ctx.restore();
    }

    // 2. Precision Laser Pitch Markings
    ctx.strokeStyle = "rgba(255, 255, 255, 0.78)";
    ctx.lineWidth = 1.8;

    // Pitch Boundary Box
    ctx.strokeRect(MARGIN_X, MARGIN_Y, FIELD_W, FIELD_H);

    // Halfway Line
    const halfX = MARGIN_X + (FIELD_W / 2);
    ctx.beginPath();
    ctx.moveTo(halfX, MARGIN_Y);
    ctx.lineTo(halfX, canvas.height - MARGIN_Y);
    ctx.stroke();

    // Center Circle & Center Spot
    const centerY = MARGIN_Y + (FIELD_H / 2);
    const radius9m = (9.15 / PITCH_W) * FIELD_W;
    ctx.beginPath();
    ctx.arc(halfX, centerY, radius9m, 0, Math.PI * 2);
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(halfX, centerY, 2.5, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.fill();

    // Right Penalty Box (18-yard box: X in [102, 120], Y in [18, 62])
    const pBoxRight = toPx(102.0, 18.0);
    const pBoxRightW = (18.0 / PITCH_W) * FIELD_W;
    const pBoxRightH = (44.0 / PITCH_H) * FIELD_H;
    ctx.strokeRect(pBoxRight.x, pBoxRight.y, pBoxRightW, pBoxRightH);

    // Right 6-Yard Box
    const gBoxRight = toPx(114.0, 30.0);
    const gBoxRightW = (6.0 / PITCH_W) * FIELD_W;
    const gBoxRightH = (20.0 / PITCH_H) * FIELD_H;
    ctx.strokeRect(gBoxRight.x, gBoxRight.y, gBoxRightW, gBoxRightH);

    // Right Penalty Spot & Arc
    const penRight = toPx(109.0, 40.0);
    ctx.beginPath();
    ctx.arc(penRight.x, penRight.y, 2.4, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.arc(penRight.x, penRight.y, radius9m, Math.PI * 0.65, Math.PI * 1.35);
    ctx.stroke();

    // Left Penalty Box & 6-Yard Box
    const pBoxLeft = toPx(0.0, 18.0);
    ctx.strokeRect(pBoxLeft.x, pBoxLeft.y, pBoxRightW, pBoxRightH);

    const gBoxLeft = toPx(0.0, 30.0);
    ctx.strokeRect(gBoxLeft.x, gBoxLeft.y, gBoxRightW, gBoxRightH);

    const penLeft = toPx(11.0, 40.0);
    ctx.beginPath();
    ctx.arc(penLeft.x, penLeft.y, 2.4, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.arc(penLeft.x, penLeft.y, radius9m, -Math.PI * 0.35, Math.PI * 0.35);
    ctx.stroke();

    // Corner Arcs
    const cornerR = (1.5 / PITCH_W) * FIELD_W;
    ctx.beginPath(); ctx.arc(MARGIN_X, MARGIN_Y, cornerR, 0, Math.PI * 0.5); ctx.stroke();
    ctx.beginPath(); ctx.arc(MARGIN_X, canvas.height - MARGIN_Y, cornerR, -Math.PI * 0.5, 0); ctx.stroke();
    ctx.beginPath(); ctx.arc(canvas.width - MARGIN_X, MARGIN_Y, cornerR, Math.PI * 0.5, Math.PI); ctx.stroke();
    ctx.beginPath(); ctx.arc(canvas.width - MARGIN_X, canvas.height - MARGIN_Y, cornerR, Math.PI, Math.PI * 1.5); ctx.stroke();

    // Goal Nets
    drawGoalNet(ctx, canvas.width - MARGIN_X, toPx(120, 36).y, 10, (8.0 / PITCH_H) * FIELD_H, true);
    drawGoalNet(ctx, MARGIN_X - 10, toPx(0, 36).y, 10, (8.0 / PITCH_H) * FIELD_H, false);

    // 3. Defensive Compactness Convex Hull (C2)
    if (defenders.length >= 3) {
        const pts = defenders.slice().sort((a, b) => a.x === b.x ? a.y - b.y : a.x - b.x);
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

        if (hull.length >= 3) {
            ctx.beginPath();
            const p0 = toPx(hull[0].x, hull[0].y);
            ctx.moveTo(p0.x, p0.y);
            for (let i = 1; i < hull.length; ++i) {
                const pi = toPx(hull[i].x, hull[i].y);
                ctx.lineTo(pi.x, pi.y);
            }
            ctx.closePath();
            ctx.fillStyle = "rgba(56, 189, 248, 0.08)";
            ctx.fill();
            ctx.strokeStyle = "rgba(56, 189, 248, 0.45)";
            ctx.lineWidth = 1.4;
            ctx.stroke();
        }
    }

    // Offside Line (The deepest outfield defender)
    const deepestX = defenders.length > 0 ? Math.max(...defenders.map(d => d.x)) : 60;
    const offsidePx = toPx(deepestX, 0).x;
    ctx.beginPath();
    ctx.setLineDash([5, 4]);
    ctx.moveTo(offsidePx, MARGIN_Y);
    ctx.lineTo(offsidePx, canvas.height - MARGIN_Y);
    ctx.strokeStyle = "rgba(56, 189, 248, 0.85)";
    ctx.lineWidth = 2.0;
    ctx.stroke();
    ctx.setLineDash([]);

    // Offside Line Badge Pill
    ctx.fillStyle = "rgba(11, 17, 32, 0.88)";
    ctx.fillRect(offsidePx - 55, MARGIN_Y + 4, 110, 18);
    ctx.strokeStyle = "rgba(56, 189, 248, 0.50)";
    ctx.strokeRect(offsidePx - 55, MARGIN_Y + 4, 110, 18);
    ctx.fillStyle = "#38bdf8";
    ctx.font = "bold 8.5px 'Fira Code', monospace";
    ctx.textAlign = "center";
    ctx.fillText(`OFFSIDE: ${(120 - deepestX).toFixed(1)}m`, offsidePx, MARGIN_Y + 16);

    // Lookahead Extrapolated Offside Line
    if (showLookaheadRollout && lookaheadHorizon > 0.0) {
        const futureDefs = extrapolatePlayerList(defenders, lookaheadHorizon);
        const futureDeepestX = Math.max(...futureDefs.map(d => d.x));
        const futOffsidePx = toPx(futureDeepestX, 0).x;
        ctx.beginPath();
        ctx.setLineDash([4, 3]);
        ctx.moveTo(futOffsidePx, MARGIN_Y);
        ctx.lineTo(futOffsidePx, canvas.height - MARGIN_Y);
        const isStepUp = (futureDeepestX < deepestX);
        ctx.strokeStyle = isStepUp ? "rgba(16, 185, 129, 0.95)" : "rgba(244, 63, 94, 0.95)";
        ctx.lineWidth = 2.2;
        ctx.stroke();
        ctx.setLineDash([]);

        const pillColor = isStepUp ? "#10b981" : "#f43f5e";
        ctx.fillStyle = "rgba(11, 17, 32, 0.90)";
        ctx.fillRect(futOffsidePx - 60, MARGIN_Y + 26, 120, 16);
        ctx.strokeStyle = pillColor;
        ctx.strokeRect(futOffsidePx - 60, MARGIN_Y + 26, 120, 16);
        ctx.fillStyle = pillColor;
        ctx.font = "bold 8px 'Fira Code', monospace";
        ctx.textAlign = "center";
        ctx.fillText(`+${lookaheadHorizon.toFixed(1)}s: ${(120 - futureDeepestX).toFixed(1)}m`, futOffsidePx, MARGIN_Y + 38);
    }

    // Passing Rays from Ball Carrier to Forward Attackers
    if (ballCarrier) {
        const bPx = toPx(ballCarrier.x, ballCarrier.y);
        const allDef = goalkeeper ? [...defenders, goalkeeper] : defenders;

        attackers.forEach(a => {
            if (a.x > ballCarrier.x) {
                const isOffside = a.x > (Math.max(deepestX, ballCarrier.x) + 0.5);
                let isBlocked = false;
                allDef.forEach(d => {
                    if (getDistToSegment(d, ballCarrier, a) <= 2.2) isBlocked = true;
                });
                const aPx = toPx(a.x, a.y);
                ctx.beginPath();
                ctx.setLineDash(isOffside ? [2, 4] : [4, 4]);
                ctx.moveTo(bPx.x, bPx.y);
                ctx.lineTo(aPx.x, aPx.y);
                if (isOffside) {
                    ctx.strokeStyle = "rgba(239, 68, 68, 0.4)";
                    ctx.lineWidth = 1.3;
                } else {
                    ctx.strokeStyle = isBlocked ? "rgba(244, 63, 94, 0.75)" : "rgba(16, 185, 129, 0.85)";
                    ctx.lineWidth = isBlocked ? 1.8 : 2.2;
                }
                ctx.stroke();
                ctx.setLineDash([]);
            }
        });
    }

    // Open Net Hazard Ray
    if (isOpenNet && ballCarrier) {
        const bPx = toPx(ballCarrier.x, ballCarrier.y);
        const goalMouthPx = toPx(120.0, 40.0);
        ctx.beginPath();
        ctx.setLineDash([6, 4]);
        ctx.moveTo(bPx.x, bPx.y);
        ctx.lineTo(goalMouthPx.x, goalMouthPx.y);
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 2.8;
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = "rgba(239, 68, 68, 0.9)";
        ctx.font = "bold 10px 'Inter', sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("UNGUARDED GOAL", canvas.width - MARGIN_X - 60, goalMouthPx.y - 12);
    }

    // 4. Draw Tactical Players (Authentic Soccer Jerseys)
    if (goalkeeper) {
        const gkPx = toPx(goalkeeper.x, goalkeeper.y);
        drawTacticalPlayer(ctx, gkPx.x, gkPx.y, {
            type: "gk",
            primaryColor: "#eab308",
            secondaryColor: "#ca8a04",
            number: goalkeeper.number || 1,
            role: goalkeeper.role || "GK",
            vx: goalkeeper.vx || 0,
            vy: goalkeeper.vy || 0,
            isSelected: selectedPlayer === goalkeeper,
            isHovered: hoveredPlayer === goalkeeper,
            isGK: true,
            player: goalkeeper
        });
    }

    defenders.forEach(d => {
        const dPx = toPx(d.x, d.y);
        drawTacticalPlayer(ctx, dPx.x, dPx.y, {
            type: "def",
            primaryColor: "#2563eb",
            secondaryColor: "#1d4ed8",
            number: d.number,
            role: d.role || "CB",
            vx: d.vx || 0,
            vy: d.vy || 0,
            isSelected: selectedPlayer === d,
            isHovered: hoveredPlayer === d,
            player: d
        });
    });

    attackers.forEach(a => {
        const aPx = toPx(a.x, a.y);
        const offsideLimit = Math.max(deepestX, ballCarrier ? ballCarrier.x : 0);
        const isOff = !a.isActor && a.x > (offsideLimit + 0.5);

        drawTacticalPlayer(ctx, aPx.x, aPx.y, {
            type: "att",
            primaryColor: a.isActor ? "#f43f5e" : "#e11d48",
            secondaryColor: a.isActor ? "#e11d48" : "#be123c",
            number: a.number,
            role: a.role || "CF",
            vx: a.vx || 0,
            vy: a.vy || 0,
            isCarrier: a.isActor,
            isSelected: selectedPlayer === a,
            isHovered: hoveredPlayer === a,
            isOffside: isOff,
            player: a
        });
    });

    // 5. Lookahead Extrapolated Ghost Positions
    if (showLookaheadRollout && lookaheadHorizon > 0.0) {
        drawLookaheadGhosts(ctx);
    }
}

/**
 * Authentic Football Jersey Silhouette Player Token.
 * Restores the beloved sports jersey icon with shoulders, short sleeves, V-neck collar,
 * squad number, tactical role pill, aim vector, and 3D soccer ball.
 */
function drawTacticalPlayer(ctx, x, y, options) {
    const {
        type, primaryColor, secondaryColor, number, role, vx, vy,
        isGK, isCarrier, isSelected, isHovered, isOffside, player
    } = options;

    ctx.save();
    ctx.translate(x, y);

    const s = PLAYER_SCALE || 1.0;
    ctx.scale(s, s);

    // A. Dynamic Vision Cone for Ball Carrier
    if (isCarrier) {
        const roleCap = ROLE_CAPABILITIES[role] || ROLE_CAPABILITIES["AM"];
        const thetaRad = ((roleCap.theta || 180) * Math.PI) / 180.0;
        let aimAngle = 0.0;
        if (Math.hypot(vx, vy) > 0.4) {
            aimAngle = Math.atan2(vy, vx);
        }

        ctx.save();
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.arc(0, 0, 72, aimAngle - thetaRad / 2, aimAngle + thetaRad / 2);
        ctx.closePath();
        ctx.fillStyle = "rgba(56, 189, 248, 0.08)";
        ctx.fill();
        ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.font = "bold 7px 'Fira Code', monospace";
        ctx.fillStyle = "rgba(56, 189, 248, 0.85)";
        ctx.textAlign = "center";
        ctx.fillText(`VISION ${roleCap.theta}°`, 52 * Math.cos(aimAngle), 52 * Math.sin(aimAngle));
        ctx.restore();
    }

    // 1. Turf Drop Shadow
    ctx.beginPath();
    ctx.ellipse(0, 10, isSelected ? 14 : 10, isSelected ? 6 : 4.5, 0, 0, Math.PI * 2);
    ctx.fillStyle = isSelected ? "rgba(0, 0, 0, 0.65)" : "rgba(0, 0, 0, 0.4)";
    ctx.fill();

    // 2. Active Possession / Selection / Offside Auras
    if (isCarrier) {
        ctx.beginPath();
        ctx.arc(0, 0, 20, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(251, 191, 36, 0.18)";
        ctx.fill();
        ctx.strokeStyle = "#fbbf24";
        ctx.lineWidth = 1.8;
        ctx.stroke();
    }

    if (isSelected) {
        ctx.beginPath();
        ctx.arc(0, 0, 22, 0, Math.PI * 2);
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2;
        ctx.setLineDash([3, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
    }

    if (isOffside) {
        ctx.beginPath();
        ctx.arc(0, 0, 21, 0, Math.PI * 2);
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 2;
        ctx.setLineDash([3, 3]);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = "#ef4444";
        ctx.font = "bold 8px 'Fira Code', monospace";
        ctx.textAlign = "center";
        ctx.fillText("OFFSIDE", 0, -21);
    }

    // 3. Football Jersey Silhouette
    ctx.beginPath();
    ctx.moveTo(-5, -11);
    ctx.lineTo(-12, -8);
    ctx.lineTo(-16, -2);
    ctx.lineTo(-12, 1);
    ctx.lineTo(-9, -2);
    ctx.lineTo(-8, 9);
    ctx.quadraticCurveTo(0, 11, 8, 9);
    ctx.lineTo(9, -2);
    ctx.lineTo(12, 1);
    ctx.lineTo(16, -2);
    ctx.lineTo(12, -8);
    ctx.lineTo(5, -11);
    ctx.quadraticCurveTo(0, -8, -5, -11);
    ctx.closePath();

    // Fill Jersey with High-Quality Linear Gradient
    const grad = ctx.createLinearGradient(-12, -11, 12, 11);
    grad.addColorStop(0, primaryColor);
    grad.addColorStop(1, secondaryColor);
    ctx.fillStyle = grad;
    ctx.fill();

    // Jersey Border
    ctx.strokeStyle = isCarrier ? "#fef08a" : "rgba(255, 255, 255, 0.85)";
    ctx.lineWidth = 1.3;
    ctx.stroke();

    // Collar Detail Notch
    ctx.beginPath();
    ctx.arc(0, -10, 4, 0, Math.PI);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // 4. Squad Number on Jersey
    ctx.fillStyle = isGK ? "#1e293b" : "#ffffff";
    ctx.font = "bold 9px 'Fira Code', monospace, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(number !== undefined ? `${number}` : "", 0, 1);

    // 5. Tactical Role Pill Below Jersey
    if (role) {
        ctx.font = "700 8px 'Inter', sans-serif";
        ctx.fillStyle = isGK ? "#fef08a" : (type === "def" ? "#bae6fd" : "#fecdd3");
        ctx.textAlign = "center";
        ctx.fillText(role, 0, 18);
    }

    // 6. Draw Interactive Velocity Vector Arrow & Aim Handle
    const speed = Math.hypot(vx, vy);
    if (speed > 0.2 || isSelected || isHovered) {
        const arrowLen = Math.min(speed * 4.2, 52.0);
        const angle = Math.atan2(vy, vx);
        const tipX = Math.cos(angle) * arrowLen;
        const tipY = Math.sin(angle) * arrowLen;

        const isStepUp = type === "def" && vx < -0.3;
        const arrowColor = isStepUp ? "#10b981" : (type === "def" ? "#60a5fa" : (type === "gk" ? "#fbbf24" : "#f43f5e"));

        ctx.save();
        ctx.strokeStyle = arrowColor;
        ctx.fillStyle = arrowColor;
        ctx.lineWidth = 2.0;

        // Shaft
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(tipX, tipY);
        ctx.stroke();

        // Arrow head
        const headLen = 7;
        const headAngle = 0.45;
        ctx.beginPath();
        ctx.moveTo(tipX, tipY);
        ctx.lineTo(tipX - headLen * Math.cos(angle - headAngle), tipY - headLen * Math.sin(angle - headAngle));
        ctx.lineTo(tipX - headLen * Math.cos(angle + headAngle), tipY - headLen * Math.sin(angle + headAngle));
        ctx.closePath();
        ctx.fill();

        // Draggable Aim Handle Circle
        ctx.beginPath();
        ctx.arc(tipX, tipY, 4.5, 0, Math.PI * 2);
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Speed text badge with crisp pill background
        if ((speed >= 0.8 && (isSelected || isHovered)) || speed >= 1.5) {
            const velText = `${speed.toFixed(1)}m/s`;
            ctx.font = "bold 8px 'Fira Code', monospace";
            const textMetrics = ctx.measureText(velText);
            const textW = textMetrics.width;
            const badgeX = tipX + 12 * Math.cos(angle);
            const badgeY = tipY + 12 * Math.sin(angle);

            ctx.fillStyle = "rgba(10, 15, 29, 0.80)";
            ctx.fillRect(badgeX - textW / 2 - 3, badgeY - 5, textW + 6, 10);
            ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
            ctx.lineWidth = 0.8;
            ctx.strokeRect(badgeX - textW / 2 - 3, badgeY - 5, textW + 6, 10);

            ctx.fillStyle = "#ffffff";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(velText, badgeX, badgeY);
        }

        if (isStepUp) {
            ctx.fillStyle = "rgba(16, 185, 129, 0.18)";
            ctx.fillRect(-18, -26, 36, 10);
            ctx.strokeStyle = "rgba(16, 185, 129, 0.55)";
            ctx.lineWidth = 0.8;
            ctx.strokeRect(-18, -26, 36, 10);

            ctx.font = "bold 7px 'Fira Code', monospace";
            ctx.fillStyle = "#10b981";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("STEP-UP", 0, -21);
        }

        ctx.restore();
    }

    ctx.restore();

    // 7. Draw 3D Soccer Ball beside Ball Carrier
    if (isCarrier) {
        drawSoccerBall(ctx, x + (15 * s), y + (8 * s), s);
    }
}

// Realistic 3D Soccer Ball Renderer
function drawSoccerBall(ctx, bx, by, scale = 1.0) {
    ctx.save();
    ctx.translate(bx, by);
    ctx.scale(scale, scale);

    // Ball Grass Shadow
    ctx.beginPath();
    ctx.ellipse(1, 4, 5, 2.5, 0, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
    ctx.fill();

    // Ball Sphere with Radial Glare
    ctx.beginPath();
    ctx.arc(0, 0, 5.5, 0, Math.PI * 2);
    const ballGrad = ctx.createRadialGradient(-1.5, -1.5, 0.8, 0, 0, 5.5);
    ballGrad.addColorStop(0, "#ffffff");
    ballGrad.addColorStop(0.7, "#f1f5f9");
    ballGrad.addColorStop(1, "#94a3b8");
    ctx.fillStyle = ballGrad;
    ctx.fill();
    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 0.8;
    ctx.stroke();

    // Center Pentagon Patch
    ctx.beginPath();
    for (let i = 0; i < 5; i++) {
        const ang = (i * 2 * Math.PI) / 5 - Math.PI / 2;
        const px = 2.2 * Math.cos(ang);
        const py = 2.2 * Math.sin(ang);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
    }
    ctx.closePath();
    ctx.fillStyle = "#0f172a";
    ctx.fill();

    ctx.restore();
}

// Multi-Frame Lookahead Ghosts
function drawLookaheadGhosts(ctx) {
    const s = PLAYER_SCALE || 1.0;
    ctx.save();

    const all = goalkeeper ? [...defenders, ...attackers, goalkeeper] : [...defenders, ...attackers];
    const futures = extrapolatePlayerList(all, lookaheadHorizon);

    for (let i = 0; i < all.length; ++i) {
        const cur = all[i];
        const fut = futures[i];
        const speed = Math.hypot(cur.vx || 0, cur.vy || 0);

        if (speed > 0.3) {
            const curPx = toPx(cur.x, cur.y);
            const futPx = toPx(fut.x, fut.y);

            const isAtt = attackers.includes(cur);
            const isGK = (cur === goalkeeper || cur.isKeeper);
            const color = isGK ? "rgba(245, 158, 11, 0.85)" : (isAtt ? "rgba(244, 63, 94, 0.85)" : "rgba(56, 189, 248, 0.85)");

            // Trajectory Trail
            ctx.beginPath();
            ctx.setLineDash([3, 3]);
            ctx.moveTo(curPx.x, curPx.y);
            ctx.lineTo(futPx.x, futPx.y);
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.4;
            ctx.stroke();
            ctx.setLineDash([]);

            // Future Jersey Outline
            ctx.save();
            ctx.translate(futPx.x, futPx.y);
            ctx.scale(s, s);

            ctx.beginPath();
            ctx.moveTo(-5, -11);
            ctx.lineTo(-12, -8);
            ctx.lineTo(-16, -2);
            ctx.lineTo(-12, 1);
            ctx.lineTo(-9, -2);
            ctx.lineTo(-8, 9);
            ctx.quadraticCurveTo(0, 11, 8, 9);
            ctx.lineTo(9, -2);
            ctx.lineTo(12, 1);
            ctx.lineTo(16, -2);
            ctx.lineTo(12, -8);
            ctx.lineTo(5, -11);
            ctx.quadraticCurveTo(0, -8, -5, -11);
            ctx.closePath();

            ctx.strokeStyle = color;
            ctx.lineWidth = 1.2;
            ctx.stroke();

            ctx.fillStyle = color;
            ctx.font = "bold 8px 'Fira Code', monospace";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(`${cur.number || ""}`, 0, 1);

            ctx.restore();
        }
    }
    ctx.restore();
}

// 3D Goal Net
function drawGoalNet(ctx, x, y, width, height, isRight) {
    ctx.fillStyle = "rgba(255, 255, 255, 0.08)";
    ctx.fillRect(x, y, width, height);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.40)";
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, width, height);

    ctx.beginPath();
    for (let i = 0; i <= height; i += 5) {
        ctx.moveTo(x, y + i);
        ctx.lineTo(x + width, y + i);
    }
    for (let i = 0; i <= width; i += 4) {
        ctx.moveTo(x + i, y);
        ctx.lineTo(x + i, y + height);
    }
    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    const postX = isRight ? x : x + width;
    ctx.fillRect(postX - 1.5, y, 3, height);
}
