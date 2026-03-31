# 🫁 BreathClock — Real-Time Breathing Rate Monitor

A real-time breathing rate monitor that captures microphone input, detects your breathing rhythm using digital signal processing, visualizes each breath as a live waveform, and computes breaths-per-minute (BPM) with a rolling average.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 What It Does

BreathClock turns your laptop microphone into a **respiratory monitor**. Breathe near your mic and watch in real time as the app:

1. **Captures** raw audio from your microphone
2. **Extracts** the breath envelope using RMS amplitude detection
3. **Filters** the signal with a Butterworth bandpass filter (0.1–0.7 Hz)
4. **Detects** individual breath peaks
5. **Computes** your breathing rate (BPM) with a rolling average

---

## 🖥️ Live Visualization

The app displays three real-time panels:

| Panel | What It Shows |
|-------|---------------|
| **Raw Waveform** | Live audio signal from your microphone |
| **Breath Envelope** | RMS amplitude over time — each "hill" is a breath |
| **Filtered Signal** | Butterworth bandpass output with detected breath peaks (▼ markers) |

A large **BPM counter** updates live, color-coded by breathing rate:
- 🔵 **Blue** — Slow (< 12 BPM)
- 🟢 **Green** — Normal (12–20 BPM)
- 🟠 **Amber** — Fast (> 20 BPM)

---

## 🔬 DSP Pipeline

```
Microphone
    │
    ▼
┌──────────────┐
│  PyAudio     │  Raw PCM audio stream (44.1 kHz, 16-bit, mono)
│  Capture     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Windowing   │  Process audio in 2048-sample chunks (~46ms each)
│  (Chunking)  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Envelope    │  Full-wave rectification → RMS averaging
│  Detection   │  Converts fast oscillations to smooth amplitude
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Butterworth │  Bandpass filter: 0.1–0.7 Hz
│  Filter      │  Isolates breathing frequencies only
│  (Order 3)   │  Rejects noise, drift, and non-breath sounds
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Peak        │  scipy.signal.find_peaks()
│  Detection   │  Adaptive threshold + minimum distance
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Rolling BPM │  BPM = 60 / mean(last N inter-peak intervals)
│  Average     │  Smooths out jitter for stable reading
└──────────────┘
```

---

## 🧠 Key Concepts

### Microphone as Analog Sensor
Your microphone is a **pressure transducer** — it converts air pressure variations into voltage. Breathing near the mic creates low-frequency pressure changes that the mic picks up as audio.

### Envelope Detection
Raw audio oscillates thousands of times per second. We don't care about individual oscillations — we care about the **amplitude contour** (loudness over time). The RMS (Root Mean Square) of each audio chunk gives us one "loudness" value, creating a smooth envelope where each bump corresponds to a breath.

### Butterworth Bandpass Filter
The **Butterworth filter** is a classic filter design with **maximally flat frequency response** in the passband (no ripples). We use it as a bandpass filter to isolate only the breathing frequency range (0.1–0.7 Hz = 6–42 breaths/min), rejecting:
- DC offset and slow drift (below 0.1 Hz)
- Room noise, speech, typing, etc. (above 0.7 Hz)

This is the same filter type used in medical devices (ECG, pulse oximeters), audio hardware, and IoT sensors.

### Peak Detection & Rolling BPM
After filtering, each breath appears as a clear peak. `scipy.signal.find_peaks()` locates these peaks, and we compute BPM from the average time between consecutive peaks. The rolling window (last 8 intervals) prevents the BPM from jumping wildly.

---

## 📦 Requirements

- **Python** 3.8+
- **A working microphone** (built-in or external)
- **macOS / Linux / Windows** (with mic permissions granted)

### Dependencies

| Package | Purpose |
|---------|---------|
| `pyaudio` | Real-time microphone audio capture |
| `numpy` | Numerical array operations |
| `scipy` | Butterworth filter design, peak detection |
| `matplotlib` | Real-time animated visualization |

---

## 🚀 Installation & Usage

### 1. Install dependencies

```bash
pip install pyaudio numpy scipy matplotlib
```

> **Note (macOS):** If `pyaudio` fails to install, you may need PortAudio:
> ```bash
> brew install portaudio
> pip install pyaudio
> ```

> **Note (Linux):** Install the ALSA dev headers first:
> ```bash
> sudo apt-get install python3-dev portaudio19-dev
> pip install pyaudio
> ```

### 2. Run it

```bash
python breathclock.py
```

### 3. Breathe near your mic

- Position your face **15–30 cm** from the microphone
- Breathe normally — exhale is usually detected more easily
- The BPM will stabilize after **3–5 breaths**
- Close the plot window or press `Ctrl+C` to stop

---

## ⚙️ Configuration

You can tune these constants at the top of `breathclock.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RATE` | 44100 | Audio sample rate (Hz) |
| `CHUNK` | 2048 | Samples per audio frame |
| `BREATH_LOW_HZ` | 0.1 | Bandpass lower bound (Hz) |
| `BREATH_HIGH_HZ` | 0.7 | Bandpass upper bound (Hz) |
| `FILTER_ORDER` | 3 | Butterworth filter order |
| `PEAK_MIN_DISTANCE_SEC` | 1.5 | Min seconds between detected breaths |
| `ROLLING_WINDOW` | 8 | Number of intervals for BPM average |
| `DISPLAY_SECONDS` | 15 | Seconds of history shown in plots |

---

## 🔗 Hardware Concept

This project demonstrates the same **sensor → ADC → DSP → feature extraction** pipeline used in:

- **Medical devices** — Pulse oximeters, ECG monitors, respiratory monitors
- **Audio ICs** — Noise cancellation, voice activity detection
- **IoT sensors** — Acoustic event detection, vibration monitoring
- **Wearables** — Sleep tracking, stress detection

The microphone acts as the analog front-end, your computer's sound card is the ADC, and the Python code performs the same DSP that runs on dedicated silicon in production hardware.

---

## 📁 Project Structure

```
Breathclock/
├── breathclock.py    # Main application (single file)
├── README.md         # This file
└── requirements.txt  # Python dependencies
```

---

## 📝 License

MIT License — feel free to modify and use for learning.
# BreathClock
