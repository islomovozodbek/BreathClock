"""
BreathClock — Real-Time Breathing Rate Monitor
================================================
Captures microphone input, detects breathing rhythm from audio amplitude,
visualizes each breath as a live waveform/envelope, and computes
breaths-per-minute (BPM) with a rolling average.

DSP Pipeline:
  Mic → Raw Audio → Windowing → Envelope Detection → Butterworth Bandpass
  → Peak Detection → Rolling BPM

Usage:
  python breathclock.py

Dependencies:
  pip install pyaudio numpy scipy matplotlib
"""

import sys
import struct
import threading
import time
from collections import deque

import numpy as np
import pyaudio
from scipy.signal import butter, lfilter, find_peaks
import matplotlib
matplotlib.use("macosx")  # Native macOS backend (no tkinter needed)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
RATE = 44100            # Sample rate (Hz)
CHUNK = 2048            # Samples per frame (~46ms per chunk)
FORMAT = pyaudio.paInt16
CHANNELS = 1

# Envelope / display settings
ENVELOPE_WINDOW = 80          # Samples to average for envelope smoothing
DISPLAY_SECONDS = 15          # Seconds of envelope history to show
ENVELOPE_DOWNSAMPLE = 4       # Downsample factor for envelope (reduces points)

# Butterworth bandpass filter for breathing frequencies
BREATH_LOW_HZ = 0.1           # ~6 breaths/min minimum
BREATH_HIGH_HZ = 0.7          # ~42 breaths/min maximum
FILTER_ORDER = 3              # Butterworth filter order

# Peak detection
PEAK_MIN_DISTANCE_SEC = 1.5   # Minimum time between breaths (seconds)
ROLLING_WINDOW = 8            # Number of recent intervals for BPM average

# ─────────────────────────────────────────────
# DSP UTILITIES
# ─────────────────────────────────────────────

def compute_envelope(audio_chunk: np.ndarray, window_size: int = ENVELOPE_WINDOW) -> float:
    """
    Envelope detection via rectification + smoothing.
    1. Take absolute value of each sample (full-wave rectification)
    2. Compute the mean over the window (moving average)
    Returns a single envelope amplitude value for this chunk.
    """
    rectified = np.abs(audio_chunk.astype(np.float64))
    # Use RMS (root mean square) for a more accurate power envelope
    rms = np.sqrt(np.mean(rectified ** 2))
    return rms


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = FILTER_ORDER):
    """
    Design a Butterworth bandpass filter.
    Returns filter coefficients (b, a) for use with lfilter.
    
    The Butterworth filter has maximally flat magnitude response in the
    passband — no ripples, making it ideal for biosignal processing.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    # Clamp to valid range (0, 1) exclusive
    low = max(low, 0.001)
    high = min(high, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass(data: np.ndarray, lowcut: float, highcut: float, fs: float) -> np.ndarray:
    """Apply the Butterworth bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data)


def detect_breaths(envelope_signal: np.ndarray, sample_rate: float, 
                   min_distance_sec: float = PEAK_MIN_DISTANCE_SEC) -> np.ndarray:
    """
    Detect breath peaks in the filtered envelope signal.
    Uses scipy.signal.find_peaks with:
      - minimum distance between peaks (prevents double-counting)
      - minimum height (adaptive threshold based on signal statistics)
    Returns array of peak indices.
    """
    min_distance_samples = int(min_distance_sec * sample_rate)
    if min_distance_samples < 1:
        min_distance_samples = 1

    # Adaptive threshold: peaks must be above the mean
    threshold = np.mean(envelope_signal) + 0.1 * np.std(envelope_signal)
    
    peaks, _ = find_peaks(
        envelope_signal,
        distance=min_distance_samples,
        height=threshold,
        prominence=0.05 * np.max(np.abs(envelope_signal)) if np.max(np.abs(envelope_signal)) > 0 else 0
    )
    return peaks


def compute_bpm(peak_indices: np.ndarray, sample_rate: float, 
                rolling_window: int = ROLLING_WINDOW) -> float:
    """
    Compute breaths-per-minute from peak indices using a rolling average
    of inter-breath intervals.
    
    BPM = 60 / (mean interval between consecutive peaks in seconds)
    """
    if len(peak_indices) < 2:
        return 0.0
    
    # Compute intervals between consecutive peaks (in seconds)
    intervals = np.diff(peak_indices) / sample_rate
    
    # Use only the most recent intervals for rolling average
    recent = intervals[-rolling_window:]
    
    # Filter out physiologically implausible intervals
    # (less than 1 second or more than 15 seconds between breaths)
    valid = recent[(recent >= 1.0) & (recent <= 15.0)]
    
    if len(valid) == 0:
        return 0.0
    
    mean_interval = np.mean(valid)
    bpm = 60.0 / mean_interval
    return bpm


# ─────────────────────────────────────────────
# AUDIO CAPTURE THREAD
# ─────────────────────────────────────────────

class AudioCapture:
    """
    Threaded audio capture using PyAudio.
    Continuously reads microphone data and stores envelope values
    in a thread-safe deque for the visualizer to consume.
    """
    
    def __init__(self):
        # Envelope sample rate (one value per chunk)
        self.envelope_rate = RATE / CHUNK  # ~21.5 Hz
        
        # How many envelope samples we keep for display
        display_samples = int(DISPLAY_SECONDS * self.envelope_rate)
        self.envelope_buffer = deque(maxlen=display_samples)
        
        # Larger buffer for filtering & peak detection (60 seconds)
        analysis_samples = int(60 * self.envelope_rate)
        self.analysis_buffer = deque(maxlen=analysis_samples)
        
        # Current state
        self.current_bpm = 0.0
        self.breath_count = 0
        self.is_running = False
        self.lock = threading.Lock()
        
        # Raw audio buffer for waveform display
        raw_display_samples = RATE * 2  # 2 seconds of raw audio
        self.raw_buffer = deque(maxlen=raw_display_samples)
        
        # Peak times for visualization
        self.peak_times = deque(maxlen=50)
        
        # Initialize PyAudio
        self.pa = pyaudio.PyAudio()
        
    def start(self):
        """Start the audio capture thread."""
        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("🎤 Microphone capture started. Breathe near your mic!")
        
    def stop(self):
        """Stop the audio capture."""
        self.is_running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        self.pa.terminate()
        print("\n🛑 Microphone capture stopped.")
        
    def _capture_loop(self):
        """Main capture loop — runs in a background thread."""
        try:
            stream = self.pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )
        except Exception as e:
            print(f"\n❌ Could not open microphone: {e}")
            print("   Make sure your mic is connected and permissions are granted.")
            self.is_running = False
            return
        
        print(f"   Sample rate: {RATE} Hz | Chunk size: {CHUNK}")
        print(f"   Envelope rate: {self.envelope_rate:.1f} Hz")
        print(f"   Bandpass filter: {BREATH_LOW_HZ}–{BREATH_HIGH_HZ} Hz "
              f"(Butterworth order {FILTER_ORDER})")
        print(f"   Close the plot window to stop.\n")
        
        while self.is_running:
            try:
                raw_data = stream.read(CHUNK, exception_on_overflow=False)
                # Unpack raw bytes → numpy int16 array
                samples = np.array(
                    struct.unpack(f'{CHUNK}h', raw_data), 
                    dtype=np.float64
                )
                
                # Normalize to [-1, 1]
                samples /= 32768.0
                
                # Compute envelope for this chunk
                env_val = compute_envelope(samples)
                
                with self.lock:
                    self.envelope_buffer.append(env_val)
                    self.analysis_buffer.append(env_val)
                    
                    # Store raw samples for waveform display
                    self.raw_buffer.extend(samples[::ENVELOPE_DOWNSAMPLE])
                    
                    # Run breath detection on the analysis buffer
                    if len(self.analysis_buffer) > int(5 * self.envelope_rate):
                        analysis_data = np.array(self.analysis_buffer)
                        
                        # Apply Butterworth bandpass filter
                        filtered = apply_bandpass(
                            analysis_data,
                            BREATH_LOW_HZ,
                            BREATH_HIGH_HZ,
                            self.envelope_rate
                        )
                        
                        # Detect peaks (breaths)
                        peaks = detect_breaths(filtered, self.envelope_rate)
                        
                        # Compute BPM
                        bpm = compute_bpm(peaks, self.envelope_rate)
                        if bpm > 0:
                            self.current_bpm = bpm
                        self.breath_count = len(peaks)
                        
                        # Store peak times relative to current position
                        self.peak_times.clear()
                        for p in peaks:
                            self.peak_times.append(p)
                
            except IOError:
                continue
            except Exception as e:
                print(f"⚠ Audio error: {e}")
                continue
        
        stream.stop_stream()
        stream.close()


# ─────────────────────────────────────────────
# REAL-TIME VISUALIZATION
# ─────────────────────────────────────────────

class BreathVisualizer:
    """
    Real-time matplotlib visualization with:
      - Top panel: Raw audio waveform (scrolling)
      - Middle panel: Breath envelope with detected peaks
      - Bottom panel: Filtered breathing signal
      - BPM display overlay
    """
    
    def __init__(self, audio: AudioCapture):
        self.audio = audio
        self.setup_plot()
        
    def setup_plot(self):
        """Configure the matplotlib figure and axes."""
        # Dark theme
        plt.style.use('dark_background')
        
        self.fig, (self.ax_wave, self.ax_env, self.ax_filt) = plt.subplots(
            3, 1, figsize=(14, 9),
            gridspec_kw={'height_ratios': [1, 1.5, 1], 'hspace': 0.3}
        )
        
        self.fig.patch.set_facecolor('#0a0a1a')
        self.fig.suptitle('', fontsize=1)  # placeholder
        
        # ── Title ──
        self.title_text = self.fig.text(
            0.5, 0.97, '🫁  B R E A T H C L O C K',
            ha='center', va='top',
            fontsize=22, fontweight='bold',
            color='#00e5ff',
            fontfamily='monospace'
        )
        self.subtitle_text = self.fig.text(
            0.5, 0.935, 'Real-Time Breathing Rate Monitor',
            ha='center', va='top',
            fontsize=11,
            color='#667788',
            fontfamily='monospace'
        )
        
        # ── BPM Display ──
        self.bpm_text = self.fig.text(
            0.92, 0.97, '-- BPM',
            ha='right', va='top',
            fontsize=28, fontweight='bold',
            color='#00ff88',
            fontfamily='monospace'
        )
        
        self.breath_count_text = self.fig.text(
            0.92, 0.935, 'Breaths: 0',
            ha='right', va='top',
            fontsize=11,
            color='#888888',
            fontfamily='monospace'
        )
        
        self.status_text = self.fig.text(
            0.08, 0.97, '● LIVE',
            ha='left', va='top',
            fontsize=12, fontweight='bold',
            color='#ff3355',
            fontfamily='monospace'
        )
        
        # ── Panel 1: Raw Waveform ──
        self.ax_wave.set_facecolor('#0d0d22')
        self.ax_wave.set_title('Raw Audio Waveform', fontsize=10, color='#4488aa', 
                                loc='left', pad=8, fontfamily='monospace')
        self.ax_wave.set_ylim(-0.5, 0.5)
        self.ax_wave.set_xlim(0, 1000)
        self.ax_wave.set_ylabel('Amplitude', fontsize=9, color='#556677')
        self.ax_wave.tick_params(colors='#334455', labelsize=8)
        self.ax_wave.spines['bottom'].set_color('#1a1a3a')
        self.ax_wave.spines['left'].set_color('#1a1a3a')
        self.ax_wave.spines['top'].set_visible(False)
        self.ax_wave.spines['right'].set_visible(False)
        self.ax_wave.grid(True, alpha=0.1, color='#334466')
        
        self.line_wave, = self.ax_wave.plot([], [], color='#00bfff', linewidth=0.5, alpha=0.8)
        
        # ── Panel 2: Envelope ──
        self.ax_env.set_facecolor('#0d0d22')
        self.ax_env.set_title('Breath Envelope (RMS Amplitude)', fontsize=10, color='#44aa88', 
                               loc='left', pad=8, fontfamily='monospace')
        self.ax_env.set_xlim(0, DISPLAY_SECONDS)
        self.ax_env.set_ylabel('RMS', fontsize=9, color='#556677')
        self.ax_env.tick_params(colors='#334455', labelsize=8)
        self.ax_env.spines['bottom'].set_color('#1a1a3a')
        self.ax_env.spines['left'].set_color('#1a1a3a')
        self.ax_env.spines['top'].set_visible(False)
        self.ax_env.spines['right'].set_visible(False)
        self.ax_env.grid(True, alpha=0.1, color='#334466')
        
        self.line_env, = self.ax_env.plot([], [], color='#00ff88', linewidth=2, alpha=0.9)
        self.fill_env = None  # Will be created dynamically
        
        # ── Panel 3: Filtered Signal ──
        self.ax_filt.set_facecolor('#0d0d22')
        self.ax_filt.set_title('Filtered Breathing Signal (Butterworth Bandpass)', fontsize=10, 
                                color='#aa8844', loc='left', pad=8, fontfamily='monospace')
        self.ax_filt.set_xlim(0, DISPLAY_SECONDS)
        self.ax_filt.set_xlabel('Time (seconds)', fontsize=9, color='#556677')
        self.ax_filt.set_ylabel('Filtered', fontsize=9, color='#556677')
        self.ax_filt.tick_params(colors='#334455', labelsize=8)
        self.ax_filt.spines['bottom'].set_color('#1a1a3a')
        self.ax_filt.spines['left'].set_color('#1a1a3a')
        self.ax_filt.spines['top'].set_visible(False)
        self.ax_filt.spines['right'].set_visible(False)
        self.ax_filt.grid(True, alpha=0.1, color='#334466')
        
        self.line_filt, = self.ax_filt.plot([], [], color='#ffaa33', linewidth=2, alpha=0.9)
        self.scatter_peaks = self.ax_filt.scatter([], [], color='#ff3366', s=80, 
                                                    zorder=5, marker='v', edgecolors='white', linewidth=0.5)
        
        # Adjust layout
        self.fig.subplots_adjust(top=0.9, bottom=0.06, left=0.08, right=0.95)
        
    def update(self, frame):
        """Animation update function — called every frame."""
        with self.audio.lock:
            # ── Update raw waveform ──
            if len(self.audio.raw_buffer) > 0:
                raw = np.array(self.audio.raw_buffer)
                # Show last ~2000 points
                display_raw = raw[-2000:]
                self.line_wave.set_data(np.arange(len(display_raw)), display_raw)
                self.ax_wave.set_xlim(0, len(display_raw))
                
                # Auto-scale y-axis with some padding
                if len(display_raw) > 0:
                    ymax = max(np.max(np.abs(display_raw)) * 1.2, 0.01)
                    self.ax_wave.set_ylim(-ymax, ymax)
            
            # ── Update envelope ──
            if len(self.audio.envelope_buffer) > 2:
                env = np.array(self.audio.envelope_buffer)
                t = np.linspace(0, len(env) / self.audio.envelope_rate, len(env))
                
                self.line_env.set_data(t, env)
                self.ax_env.set_xlim(t[0], t[-1])
                
                # Auto-scale
                if len(env) > 0:
                    ymax = max(np.max(env) * 1.3, 0.001)
                    self.ax_env.set_ylim(0, ymax)
                
                # Update fill under curve
                if self.fill_env is not None:
                    self.fill_env.remove()
                self.fill_env = self.ax_env.fill_between(
                    t, 0, env, alpha=0.15, color='#00ff88'
                )
            
            # ── Update filtered signal + peaks ──
            if len(self.audio.analysis_buffer) > int(5 * self.audio.envelope_rate):
                analysis = np.array(self.audio.analysis_buffer)
                
                # Apply filter
                try:
                    filtered = apply_bandpass(
                        analysis, BREATH_LOW_HZ, BREATH_HIGH_HZ, self.audio.envelope_rate
                    )
                    
                    # Show only the display window portion
                    display_len = int(DISPLAY_SECONDS * self.audio.envelope_rate)
                    show_filt = filtered[-display_len:] if len(filtered) > display_len else filtered
                    t_filt = np.linspace(0, len(show_filt) / self.audio.envelope_rate, len(show_filt))
                    
                    self.line_filt.set_data(t_filt, show_filt)
                    self.ax_filt.set_xlim(t_filt[0], t_filt[-1])
                    
                    if len(show_filt) > 0:
                        ymax = max(np.max(np.abs(show_filt)) * 1.3, 0.0001)
                        self.ax_filt.set_ylim(-ymax, ymax)
                    
                    # Show peaks in the display window
                    peaks = list(self.audio.peak_times)
                    offset = len(analysis) - len(show_filt)
                    visible_peaks = [p - offset for p in peaks if offset <= p < len(analysis)]
                    
                    if visible_peaks:
                        peak_t = np.array(visible_peaks) / self.audio.envelope_rate
                        peak_v = show_filt[visible_peaks]
                        self.scatter_peaks.set_offsets(np.column_stack([peak_t, peak_v]))
                    else:
                        self.scatter_peaks.set_offsets(np.empty((0, 2)))
                        
                except Exception:
                    pass
            
            # ── Update BPM text ──
            bpm = self.audio.current_bpm
            if bpm > 0:
                self.bpm_text.set_text(f'{bpm:.1f} BPM')
                # Color based on BPM range
                if bpm < 12:
                    self.bpm_text.set_color('#00aaff')   # Low - blue
                elif bpm < 20:
                    self.bpm_text.set_color('#00ff88')   # Normal - green
                else:
                    self.bpm_text.set_color('#ffaa33')   # High - amber
            else:
                self.bpm_text.set_text('-- BPM')
                self.bpm_text.set_color('#444466')
            
            self.breath_count_text.set_text(f'Breaths: {self.audio.breath_count}')
            
            # Blinking LIVE indicator
            if int(time.time() * 2) % 2:
                self.status_text.set_color('#ff3355')
            else:
                self.status_text.set_color('#661122')
        
        return (self.line_wave, self.line_env, self.line_filt, 
                self.scatter_peaks, self.bpm_text, self.breath_count_text, self.status_text)
    
    def run(self):
        """Start the animation loop."""
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update,
            interval=80,    # ~12.5 FPS
            blit=False,      # Full redraw needed for fill_between
            cache_frame_data=False
        )
        plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  🫁  B R E A T H C L O C K")
    print("  Real-Time Breathing Rate Monitor")
    print("=" * 55)
    print()
    print("  DSP Pipeline:")
    print("  Mic → Windowing → RMS Envelope → Butterworth")
    print("  Bandpass → Peak Detection → Rolling BPM")
    print()
    
    # Create audio capture
    audio = AudioCapture()
    
    try:
        # Start capturing audio
        audio.start()
        
        # Give it a moment to initialize
        time.sleep(0.5)
        
        if not audio.is_running:
            print("❌ Failed to start audio capture. Exiting.")
            sys.exit(1)
        
        # Start visualization (this blocks until window is closed)
        viz = BreathVisualizer(audio)
        viz.run()
        
    except KeyboardInterrupt:
        print("\n\n⏹  Interrupted by user.")
    finally:
        audio.stop()
        print("  Goodbye! 🌙")


if __name__ == "__main__":
    main()
