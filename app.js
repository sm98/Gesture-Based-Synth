import {
  HandLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";
import * as Tone from "https://cdn.jsdelivr.net/npm/tone@14.8.49/+esm";

const DB_KEY = "gesture-synth.gestures.v1";
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const waveformCanvas = document.getElementById("waveform");
const waveformCtx = waveformCanvas.getContext("2d");
const visualizerCanvas = document.getElementById("visualizer");
const visualizerCtx = visualizerCanvas.getContext("2d");
const statusEl = document.getElementById("status");
const labelInput = document.getElementById("labelInput");
const saveBtn = document.getElementById("saveBtn");
const gestureListEl = document.getElementById("gestureList");
const gestureTotalEl = document.getElementById("gestureTotal");
const clearDbBtn = document.getElementById("clearDbBtn");
const thresholdSlider = document.getElementById("threshold");
const handPickerEl = document.getElementById("handPicker");
const readoutEls = {
  Left: document.getElementById("readoutLeft"),
  Right: document.getElementById("readoutRight"),
};
const enableAudioBtn = document.getElementById("enableAudioBtn");
const nowPlayingEl = document.getElementById("nowPlaying");
const volumeSlider = document.getElementById("volume");
const filterBarsEl = document.getElementById("filterBars");
const filterValueEl = document.getElementById("filterValue");
const settingsBtn = document.getElementById("settingsBtn");
const drawerEl = document.getElementById("drawer");
const drawerOverlayEl = document.getElementById("drawerOverlay");
const closeDrawerBtn = document.getElementById("closeDrawerBtn");
const guideBtn = document.getElementById("guideBtn");
const guideOverlayEl = document.getElementById("guideOverlay");
const closeGuideBtn = document.getElementById("closeGuideBtn");
const synthPresetSelect = document.getElementById("synthPreset");

// ---------- Drawer + guide modal ----------

function setDrawerOpen(open) {
  drawerEl.classList.toggle("open", open);
  drawerOverlayEl.classList.toggle("open", open);
}
settingsBtn.addEventListener("click", () => setDrawerOpen(true));
closeDrawerBtn.addEventListener("click", () => setDrawerOpen(false));
drawerOverlayEl.addEventListener("click", () => setDrawerOpen(false));

function setGuideOpen(open) {
  guideOverlayEl.classList.toggle("open", open);
}
guideBtn.addEventListener("click", () => setGuideOpen(true));
closeGuideBtn.addEventListener("click", () => setGuideOpen(false));
guideOverlayEl.addEventListener("click", (e) => {
  if (e.target === guideOverlayEl) setGuideOpen(false);
});

let handLandmarker = null;
// Per-hand live state, keyed by corrected handedness label ("Left" | "Right").
// Each entry: { landmarks, vector }
let liveHands = {};
let lastVideoTime = -1;
let videoNativeW = 0;
let videoNativeH = 0;

// ---------- Canvas sizing ----------
// The video fills the viewport with object-fit:cover (cropped), but the
// canvases draw in the video's native pixel space. We resize canvases to the
// viewport and map normalized landmark coords through the same crop/scale
// math the browser applies to the video, so overlays stay pinned to hands
// instead of drifting on non-4:3 screens.

function resizeCanvases() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  for (const c of [canvas, waveformCanvas, visualizerCanvas]) {
    c.width = w;
    c.height = h;
  }
}
window.addEventListener("resize", resizeCanvases);

function videoToCanvasCoords(nx, ny) {
  if (!videoNativeW || !videoNativeH) return { x: nx * canvas.width, y: ny * canvas.height };
  const scale = Math.max(canvas.width / videoNativeW, canvas.height / videoNativeH);
  const drawW = videoNativeW * scale;
  const drawH = videoNativeH * scale;
  const offsetX = (canvas.width - drawW) / 2;
  const offsetY = (canvas.height - drawH) / 2;
  return { x: nx * drawW + offsetX, y: ny * drawH + offsetY };
}

// ---------- Storage ("db") ----------
// Each saved gesture is tagged with which hand it was recorded for, so a
// "fist" on the right hand never matches against the left hand's live pose.

function loadGestures() {
  try {
    return JSON.parse(localStorage.getItem(DB_KEY)) ?? [];
  } catch {
    return [];
  }
}

function saveGestures(gestures) {
  localStorage.setItem(DB_KEY, JSON.stringify(gestures));
}

let gestures = loadGestures(); // [{ id, label, hand: "Left"|"Right", vector }]

function renderGestureList() {
  const counts = new Map(); // key: "hand::label" -> count
  for (const g of gestures) {
    const key = `${g.hand}::${g.label}`;
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }

  gestureTotalEl.textContent = gestures.length;

  if (counts.size === 0) {
    gestureListEl.innerHTML = `<div class="empty-note">No gestures saved yet.</div>`;
    return;
  }

  gestureListEl.innerHTML = "";
  for (const [key, count] of counts) {
    const [hand, label] = key.split("::");
    const row = document.createElement("div");
    row.className = "gesture-row";
    row.innerHTML = `
      <span><span class="name">${escapeHtml(label)}</span><span class="count">${hand} · ${count} sample${count > 1 ? "s" : ""}</span></span>
      <button class="del" data-hand="${escapeHtml(hand)}" data-label="${escapeHtml(label)}">delete</button>
    `;
    gestureListEl.appendChild(row);
  }

  gestureListEl.querySelectorAll(".del").forEach((btn) => {
    btn.addEventListener("click", () => {
      const hand = btn.getAttribute("data-hand");
      const label = btn.getAttribute("data-label");
      gestures = gestures.filter((g) => !(g.hand === hand && g.label === label));
      saveGestures(gestures);
      renderGestureList();
    });
  });
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

clearDbBtn.addEventListener("click", () => {
  if (!confirm("Delete all saved gestures? This cannot be undone.")) return;
  gestures = [];
  saveGestures(gestures);
  renderGestureList();
});

// ---------- Feature extraction ----------
// Translation-invariant (relative to wrist) + scale-invariant (normalized by
// hand size) so the same pose matches regardless of hand position/distance
// from the camera.

function toFeatureVector(landmarks) {
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  const scale = Math.hypot(
    middleMcp.x - wrist.x,
    middleMcp.y - wrist.y,
    middleMcp.z - wrist.z
  ) || 1e-6;

  const vec = [];
  for (const lm of landmarks) {
    vec.push((lm.x - wrist.x) / scale);
    vec.push((lm.y - wrist.y) / scale);
    vec.push((lm.z - wrist.z) / scale);
  }
  return vec;
}

// ---------- Handedness correction ----------
// MediaPipe's handedness label, empirically, already matches the user's
// actual anatomical hand as seen in our mirrored display — no swap needed.

function correctedHandedness(categoryName) {
  return categoryName === "Left" ? "Left" : "Right";
}

// ---------- Tilt (hand roll, left/right) ----------
// Uses the wrist->middle-finger-MCP vector as the hand's "up" axis and
// measures its angle from vertical. The video/canvas are CSS-mirrored
// (scaleX(-1)) for a natural selfie-view, so we flip x to match what the
// user actually sees on screen.

function computeTiltDegrees(landmarks) {
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  const dx = -(middleMcp.x - wrist.x); // flipped to match mirrored display
  const dy = middleMcp.y - wrist.y; // image y grows downward
  const angleRad = Math.atan2(dx, -dy); // 0 = pointing straight up
  return angleRad * (180 / Math.PI);
}

function tiltDirection(angleDeg) {
  return angleDeg > 0 ? "Right" : "Left";
}

// Hysteresis per hand so the direction doesn't flicker when the hand sits
// close to perfectly vertical (angle near 0).
const TILT_HYSTERESIS_DEG = 6;
const lastTiltDirection = {};

function getStableTiltDirection(hand, angleDeg) {
  const prev = lastTiltDirection[hand];
  let next;
  if (prev === "Left" && angleDeg > TILT_HYSTERESIS_DEG) next = "Right";
  else if (prev === "Right" && angleDeg < -TILT_HYSTERESIS_DEG) next = "Left";
  else if (!prev) next = tiltDirection(angleDeg);
  else next = prev;
  lastTiltDirection[hand] = next;
  return next;
}

// ---------- Small math helpers ----------

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

function mapLinear(v, inLo, inHi, outLo, outHi) {
  const t = clamp((v - inLo) / (inHi - inLo), 0, 1);
  return outLo + t * (outHi - outLo);
}

function mapLog(v, inLo, inHi, outLo, outHi) {
  const t = clamp((v - inLo) / (inHi - inLo), 0, 1);
  return outLo * Math.pow(outHi / outLo, t);
}

// ---------- Left-hand continuous controls ----------
// Height -> volume, tilt -> filter brightness, thumb tuck -> octave shift.
// These read raw landmarks directly (not the gesture-matching pipeline)
// since they're always-on continuous controls, not discrete poses.

const THUMB_TUCK_LOW = 0.9;
const THUMB_TUCK_HIGH = 1.3;
const lastThumbState = {};

function getStableThumbState(hand, ratio) {
  const prev = lastThumbState[hand];
  let next;
  if (prev === "In" && ratio > THUMB_TUCK_HIGH) next = "Out";
  else if (prev === "Out" && ratio < THUMB_TUCK_LOW) next = "In";
  else if (!prev) next = ratio < (THUMB_TUCK_LOW + THUMB_TUCK_HIGH) / 2 ? "In" : "Out";
  else next = prev;
  lastThumbState[hand] = next;
  return next;
}

function computeLeftHandControls(state) {
  if (!state) return null;
  const landmarks = state.landmarks;
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  const thumbTip = landmarks[4];
  const pinkyMcp = landmarks[17];
  const handScale =
    Math.hypot(middleMcp.x - wrist.x, middleMcp.y - wrist.y, middleMcp.z - wrist.z) || 1e-6;

  // Vertical wrist position (0 = top of frame, 1 = bottom) -> volume.
  const gainValue = mapLinear(wrist.y, 0, 1, 1.0, 0.12);

  // Tilt angle -> filter cutoff, tilt left = darker/lower, tilt right = brighter/higher.
  const angleDeg = computeTiltDegrees(landmarks);
  const filterHz = mapLog(clamp(angleDeg, -70, 70), -70, 70, 300, 7000);

  // Thumb tucked across the palm vs extended out -> octave shift.
  const thumbRatio =
    Math.hypot(thumbTip.x - pinkyMcp.x, thumbTip.y - pinkyMcp.y, thumbTip.z - pinkyMcp.z) /
    handScale;
  const thumbState = getStableThumbState("Left", thumbRatio);
  const octaveShift = thumbState === "In" ? 1 : -1;

  return { gainValue, filterHz, octaveShift, thumbState, angleDeg };
}

// ---------- Chord engine ----------

const NOTE_SEMITONES = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 };
const CHORD_INTERVALS = { major: [0, 4, 7], minor: [0, 3, 7] };
const CHORD_BASE_OCTAVE = 4;

function noteHue(letter) {
  return (NOTE_SEMITONES[letter] ?? 0) * 30; // 12 semitones spread across the 360° color wheel
}

const SYNTH_PRESETS = {
  warm: {
    oscillator: { type: "sawtooth" },
    envelope: { attack: 0.04, decay: 0.25, sustain: 0.55, release: 0.6 },
  },
  bright: {
    oscillator: { type: "square" },
    envelope: { attack: 0.01, decay: 0.15, sustain: 0.4, release: 0.3 },
  },
  retro: {
    oscillator: { type: "pulse", width: 0.3 },
    envelope: { attack: 0.005, decay: 0.4, sustain: 0.25, release: 0.15 },
  },
};

let synth = null;
let filter = null; // Tone.Filter, driven by left-hand tilt
let heightGain = null; // Tone.Gain, driven by left-hand height
let waveform = null; // Tone.Waveform, tapped for the oscilloscope visual
let audioEnabled = false;
let currentChordKey = null;
let currentChordNotes = null;
let currentChordHue = null; // drives the background visualizer's color

function chordNoteNames(letter, quality, octaveShift = 0) {
  const rootMidi = 12 * (CHORD_BASE_OCTAVE + 1 + octaveShift) + NOTE_SEMITONES[letter];
  return CHORD_INTERVALS[quality].map((interval) =>
    Tone.Frequency(rootMidi + interval, "midi").toNote()
  );
}

enableAudioBtn.addEventListener("click", async () => {
  await Tone.start();
  synth = new Tone.PolySynth(Tone.Synth, SYNTH_PRESETS[synthPresetSelect.value]);
  heightGain = new Tone.Gain(1);
  filter = new Tone.Filter(2000, "lowpass");
  waveform = new Tone.Waveform(512);

  synth.chain(heightGain, filter, Tone.Destination);
  filter.connect(waveform); // parallel tap for the oscilloscope, doesn't affect the audible chain

  synth.volume.value = Tone.gainToDb(parseFloat(volumeSlider.value));
  audioEnabled = true;
  enableAudioBtn.textContent = "Sound On";
  enableAudioBtn.disabled = true;
});

synthPresetSelect.addEventListener("change", () => {
  if (synth) synth.set(SYNTH_PRESETS[synthPresetSelect.value]);
});

volumeSlider.addEventListener("input", () => {
  if (synth) synth.volume.value = Tone.gainToDb(parseFloat(volumeSlider.value));
});

function applyLeftHandControls(leftControls) {
  if (!audioEnabled || !filter || !heightGain) return;
  if (leftControls) {
    filter.frequency.rampTo(leftControls.filterHz, 0.05);
    heightGain.gain.rampTo(leftControls.gainValue, 0.05);
  } else {
    filter.frequency.rampTo(2000, 0.2);
    heightGain.gain.rampTo(1, 0.2);
  }
  filterValueEl.textContent = leftControls ? `Filter: ${Math.round(leftControls.filterHz)} Hz` : "Filter: —";
  const litCount = leftControls
    ? Math.round(mapLog(leftControls.filterHz, 300, 7000, 1, 12))
    : 0;
  filterBarsEl.querySelectorAll(".bar").forEach((bar, i) => {
    bar.classList.toggle("lit", i < litCount);
  });
}

function updateChordPlayback(rightHandInfo, octaveShift) {
  const chordLabelEl = readoutEls.Right.querySelector('[data-role="chord"]');

  if (!audioEnabled || !synth) return;

  let desiredKey = null;
  let desiredNotes = null;
  let desiredDisplay = "—";

  if (rightHandInfo && rightHandInfo.thresholdOk) {
    const letter = rightHandInfo.match.label.trim().toUpperCase();
    if (Object.prototype.hasOwnProperty.call(NOTE_SEMITONES, letter)) {
      const quality = rightHandInfo.tiltDir === "Left" ? "major" : "minor";
      desiredKey = `${letter}-${quality}-${octaveShift}`;
      desiredNotes = chordNoteNames(letter, quality, octaveShift);
      desiredDisplay = `${letter} ${quality}`;
    }
  }

  if (desiredKey !== currentChordKey) {
    if (currentChordNotes) synth.triggerRelease(currentChordNotes);
    if (desiredNotes) synth.triggerAttack(desiredNotes);
    currentChordKey = desiredKey;
    currentChordNotes = desiredNotes;
    currentChordHue = desiredKey ? noteHue(desiredKey.split("-")[0]) : null;
    nowPlayingEl.textContent = `Now playing: ${desiredDisplay}`;
    chordLabelEl.textContent = desiredDisplay;
    chordLabelEl.classList.toggle("empty", desiredKey === null);
  }
}

function distance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum) / a.length; // average per-dimension distance, easier to reason about as a threshold
}

function matchGesture(vector, hand) {
  const candidates = gestures.filter((g) => g.hand === hand);
  if (candidates.length === 0) return null;
  let best = null;
  let bestDist = Infinity;
  for (const g of candidates) {
    const d = distance(vector, g.vector);
    if (d < bestDist) {
      bestDist = d;
      best = g;
    }
  }
  return { label: best.label, distance: bestDist };
}

// ---------- Hand picker (which hand to save the current pose as) ----------

function getSelectedHandChoice() {
  return handPickerEl.querySelector('input[name="handChoice"]:checked')?.value ?? "Right";
}

function syncHandPickerAvailability() {
  const present = new Set(Object.keys(liveHands));
  for (const hand of ["Left", "Right"]) {
    const radio = document.getElementById(`handChoice${hand}`);
    const label = document.getElementById(`handChoice${hand}Label`);
    const isPresent = present.has(hand);
    label.classList.toggle("disabled", !isPresent);
    // Don't disable the radio itself if it's already checked, so the user
    // isn't stuck with no selection when a hand briefly drops out of frame.
    radio.disabled = !isPresent && !radio.checked;
  }
  // If the currently selected hand isn't visible but the other one is, and
  // nothing has been typed yet, nudge selection to the visible hand.
  const selected = getSelectedHandChoice();
  if (!present.has(selected) && present.size === 1) {
    const onlyHand = [...present][0];
    document.getElementById(`handChoice${onlyHand}`).checked = true;
  }
}

// ---------- Save button ----------

function updateSaveButtonState() {
  const hand = getSelectedHandChoice();
  const hasVector = Boolean(liveHands[hand]?.vector);
  saveBtn.disabled = !hasVector || labelInput.value.trim().length === 0;
}
labelInput.addEventListener("input", updateSaveButtonState);
handPickerEl.addEventListener("change", updateSaveButtonState);

saveBtn.addEventListener("click", () => {
  const label = labelInput.value.trim();
  const hand = getSelectedHandChoice();
  const vector = liveHands[hand]?.vector;
  if (!label || !vector) return;
  gestures.push({ id: crypto.randomUUID(), label, hand, vector });
  saveGestures(gestures);
  renderGestureList();
});

// ---------- Camera + model setup ----------

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 960, height: 720, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await new Promise((resolve) => (video.onloadedmetadata = resolve));
  videoNativeW = video.videoWidth;
  videoNativeH = video.videoHeight;
  resizeCanvases();
}

async function setupHandLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });
}

// ---------- Drawing ----------

function drawAllLandmarks(handsList) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (const { landmarks, hand } of handsList) {
    const color = hand === "Left" ? "#a78bfa" : "#5eead4";
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    for (const [a, b] of HAND_CONNECTIONS) {
      const p1 = videoToCanvasCoords(landmarks[a].x, landmarks[a].y);
      const p2 = videoToCanvasCoords(landmarks[b].x, landmarks[b].y);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }
    ctx.fillStyle = "#f5f5f5";
    for (const lm of landmarks) {
      const p = videoToCanvasCoords(lm.x, lm.y);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

// ---------- Per-hand readout UI ----------
// One shared computation per hand per frame — feeds both the on-screen
// tilt/gesture pills and the chord engine, so they never disagree.

function computeHandInfo(hand, state) {
  if (!state) return null;
  const angleDeg = computeTiltDegrees(state.landmarks);
  const tiltDir = getStableTiltDirection(hand, angleDeg);
  const threshold = parseFloat(thresholdSlider.value);
  const match = matchGesture(state.vector, hand);
  const thresholdOk = Boolean(match && match.distance <= threshold);
  return { angleDeg, tiltDir, match, thresholdOk };
}

function updateReadout(hand, info) {
  const el = readoutEls[hand];
  if (!info) {
    el.classList.add("hidden");
    return;
  }
  el.classList.remove("hidden");

  const tiltEl = el.querySelector('[data-role="tilt"]');
  tiltEl.textContent = `Tilt: ${info.tiltDir} (${info.angleDeg.toFixed(0)}°)`;
  tiltEl.classList.remove("empty");

  if (hand === "Right") {
    const gestureEl = el.querySelector('[data-role="gesture"]');
    if (info.thresholdOk) {
      gestureEl.textContent = info.match.label;
      gestureEl.classList.remove("empty");
    } else {
      gestureEl.textContent = "Unrecognized pose";
      gestureEl.classList.add("empty");
    }
  }
}

function updateLeftOctaveBadge(leftControls) {
  const octaveEl = readoutEls.Left.querySelector('[data-role="octave"]');
  if (!leftControls) return;
  const sign = leftControls.octaveShift > 0 ? "+" : "";
  octaveEl.textContent = `Octave: ${sign}${leftControls.octaveShift} (thumb ${leftControls.thumbState})`;
}

// ---------- Oscilloscope ----------

function getAmplitude() {
  if (!audioEnabled || !waveform) return 0;
  const data = waveform.getValue();
  let sumSquares = 0;
  for (let i = 0; i < data.length; i++) sumSquares += data[i] * data[i];
  return Math.sqrt(sumSquares / data.length); // RMS, roughly 0..0.5 in practice
}

function drawWaveform(hue) {
  waveformCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
  if (!audioEnabled || !waveform) return;

  const data = waveform.getValue();
  const h = waveformCanvas.height;
  const w = waveformCanvas.width;
  const midY = h * 0.9; // sit near the bottom, out of the way of hand readouts
  const color = hue === null ? "#fbbf24" : `hsl(${hue}, 85%, 65%)`;

  waveformCtx.strokeStyle = color;
  waveformCtx.lineWidth = 2;
  waveformCtx.shadowColor = color;
  waveformCtx.shadowBlur = 8;
  waveformCtx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = (i / (data.length - 1)) * w;
    const y = midY + data[i] * h * 0.14;
    if (i === 0) waveformCtx.moveTo(x, y);
    else waveformCtx.lineTo(x, y);
  }
  waveformCtx.stroke();
  waveformCtx.shadowBlur = 0;
}

// ---------- Ambient visualizer ----------
// A soft color-tinted glow plus expanding pulse rings, spawned while a chord
// is held and driven by the live audio amplitude. Purely decorative — sits
// behind the hand skeleton via draw order — but gives the instrument a sense
// of "weight" when a chord lands versus silence.

let pulses = []; // { r, alpha, hue }
let lastPulseTime = 0;
const PULSE_INTERVAL_MS = 220;

function drawVisualizer(hue, amplitude) {
  const w = visualizerCanvas.width;
  const h = visualizerCanvas.height;
  visualizerCtx.clearRect(0, 0, w, h);

  const cx = w / 2;
  const cy = h / 2;

  if (hue !== null) {
    const now = performance.now();
    if (now - lastPulseTime > PULSE_INTERVAL_MS) {
      pulses.push({ r: 0, alpha: 0.5, hue });
      lastPulseTime = now;
    }

    const radius = Math.max(w, h) * 0.55;
    const grad = visualizerCtx.createRadialGradient(cx, cy, 0, cx, cy, radius);
    grad.addColorStop(0, `hsla(${hue}, 80%, 55%, ${0.08 + amplitude * 0.35})`);
    grad.addColorStop(1, `hsla(${hue}, 80%, 55%, 0)`);
    visualizerCtx.fillStyle = grad;
    visualizerCtx.fillRect(0, 0, w, h);
  }

  const maxR = Math.max(w, h) * 0.6;
  visualizerCtx.globalCompositeOperation = "lighter";
  for (const p of pulses) {
    p.r += 3 + amplitude * 14;
    p.alpha *= 0.965;
    visualizerCtx.beginPath();
    visualizerCtx.arc(cx, cy, p.r, 0, Math.PI * 2);
    visualizerCtx.strokeStyle = `hsla(${p.hue}, 85%, 65%, ${p.alpha})`;
    visualizerCtx.lineWidth = 3;
    visualizerCtx.stroke();
  }
  visualizerCtx.globalCompositeOperation = "source-over";
  pulses = pulses.filter((p) => p.alpha > 0.02 && p.r < maxR);
}

// ---------- Main loop ----------

function predictLoop() {
  if (handLandmarker && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = handLandmarker.detectForVideo(video, performance.now());

    const newLiveHands = {};
    const drawList = [];

    if (result.landmarks && result.landmarks.length > 0) {
      result.landmarks.forEach((landmarks, i) => {
        const rawCategory = result.handedness?.[i]?.[0]?.categoryName ?? "Right";
        const hand = correctedHandedness(rawCategory);
        const vector = toFeatureVector(landmarks);
        newLiveHands[hand] = { landmarks, vector };
        drawList.push({ landmarks, hand });
      });
    }

    liveHands = newLiveHands;

    const leftInfo = computeHandInfo("Left", liveHands.Left ?? null);
    const rightInfo = computeHandInfo("Right", liveHands.Right ?? null);
    const leftControls = computeLeftHandControls(liveHands.Left ?? null);

    updateReadout("Left", leftInfo);
    updateReadout("Right", rightInfo);
    updateLeftOctaveBadge(leftControls);
    applyLeftHandControls(leftControls);
    updateChordPlayback(rightInfo, leftControls ? leftControls.octaveShift : 0);
    drawAllLandmarks(drawList);
    syncHandPickerAvailability();
    updateSaveButtonState();
  }

  // Runs every animation frame (not gated on a new video frame) so the
  // waveform/glow stay smooth even between hand-tracking updates.
  const amplitude = getAmplitude();
  drawWaveform(currentChordHue);
  drawVisualizer(currentChordHue, amplitude);

  requestAnimationFrame(predictLoop);
}

// ---------- Boot ----------

function setupFilterMeter() {
  filterBarsEl.innerHTML = "";
  for (let i = 0; i < 12; i++) {
    const bar = document.createElement("div");
    bar.className = "bar";
    filterBarsEl.appendChild(bar);
  }
}

async function main() {
  renderGestureList();
  setupFilterMeter();
  statusEl.textContent = "Requesting camera…";
  await setupCamera();
  statusEl.textContent = "Loading hand-tracking model…";
  await setupHandLandmarker();
  statusEl.textContent = "Tracking live";
  predictLoop();
}

main().catch((err) => {
  console.error(err);
  statusEl.textContent = "Error: " + err.message;
});
