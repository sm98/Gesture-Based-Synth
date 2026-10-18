# Gesture Based Synth

A browser-based chord synth played with your hands via the webcam (MediaPipe Hand Landmarker + Tone.js).

- **Right hand:** a trained A–G gesture picks the chord root; tilt left = major, tilt right = minor.
- **Left hand:** height = volume, tilt = filter, thumb tucked/out = octave up/down.
- Train your own gestures in the Settings drawer (stored in `localStorage`).

## Run

```bash
python3 -m http.server 8420
```

Then open http://localhost:8420 in Chrome/Safari and allow camera access.
