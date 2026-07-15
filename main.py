"""
Birdy Call
---------------------
A locally-run, forest-themed GUI for recording bird calls to WAV files.
Built for fieldwork on the Indian subcontinent, but works anywhere.

Run:
    python app.py

Dependencies (see requirements.txt):
    sounddevice, numpy   (Tkinter ships with Python)
"""

import os
import io
import sys
import math
import time
import wave
import random
import queue
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
from datetime import datetime

try:
    import numpy as np
    import sounddevice as sd
except ImportError:
    print("Missing dependencies. Please run: pip install -r requirements.txt")
    sys.exit(1)


# ----------------------------------------------------------------------------
# Palette & constants
# ----------------------------------------------------------------------------
BG_TOP = (13, 40, 24)          # deep dusk green
BG_BOTTOM = (4, 15, 9)         # near-black green
TREE_BACK = "#173d24"
TREE_MID = "#0f2e1a"
TREE_FRONT = "#081c10"
PANEL_BG = "#0e2919"
PANEL_BORDER = "#2b5238"
ACCENT_GOLD = "#e0b563"
ACCENT_GOLD_DIM = "#8a6f3a"
EMBER = "#e07a4f"
EMBER_DIM = "#7a3f2b"
TEXT_CREAM = "#f0e6cf"
TEXT_MUTED = "#9fbf9f"
FIREFLY = "#f5df8e"

WINDOW_W, WINDOW_H = 980, 660
CANVAS_W, CANVAS_H = 660, 660
SIDEBAR_W = WINDOW_W - CANVAS_W

SAMPLE_RATE = 44100
CHANNELS = 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Make sure predict.py / config.py / preprocess.py (expected alongside this
# file) are importable no matter what directory the app is launched from.
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

MEDAL_COLORS = ["#e0b563", "#c9c9c9", "#b08d57"]  # gold / silver / bronze
MEDAL_ICONS = ["🥇", "🥈", "🥉"]

STATUS_PHRASES_IDLE = [
    "The canopy is quiet... press record to listen.",
    "Waiting for a call from the trees.",
    "Ready whenever the forest speaks.",
]
STATUS_PHRASES_RECORDING = [
    "Listening to the canopy...",
    "Capturing the call...",
    "Recording — hold still, let it sing.",
]


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb


def rounded_rect(canvas, x1, y1, x2, y2, r=18, **kwargs):
    points = [
        x1 + r, y1,
        x2 - r, y1,
        x2, y1,
        x2, y1 + r,
        x2, y2 - r,
        x2, y2,
        x2 - r, y2,
        x1 + r, y2,
        x1, y2,
        x1, y2 - r,
        x1, y1 + r,
        x1, y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


# ----------------------------------------------------------------------------
# Audio recorder (runs the sounddevice stream on its own thread)
# ----------------------------------------------------------------------------
class AudioRecorder:
    def __init__(self, level_queue, device=None):
        self.device = device
        self.level_queue = level_queue
        self._frames = []
        self._stream = None
        self.recording = False

    def _callback(self, indata, frames, time_info, status):
        if self.recording:
            self._frames.append(indata.copy())
            # push a rolling amplitude value (0-1) for the visualizer
            level = float(np.abs(indata).mean())
            try:
                self.level_queue.put_nowait(min(level * 12, 1.0))
            except queue.Full:
                pass

    def start(self):
        self._frames = []
        self.recording = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        self.recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if not self._frames:
            return None
        return np.concatenate(self._frames, axis=0)

    @staticmethod
    def save_wav(path, audio_data):
        with wave.open(path, "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # int16
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())


# ----------------------------------------------------------------------------
# Species identification (wraps the existing predict.py / config.py / preprocess.py)
# ----------------------------------------------------------------------------
class SpeciesIdentifier:
    """
    Lazily loads the trained TensorFlow model (predict.py + config.py) on a
    background thread the first time it's needed, then reuses it for every
    subsequent prediction so the GUI never re-loads the model per-click.
    """

    def __init__(self):
        self._model = None
        self._labels = None
        self._predict_file = None
        self._lock = threading.Lock()

    def ensure_loaded_sync(self):
        """Blocking load — must be called from a background thread."""
        with self._lock:
            if self._model is not None:
                return
            import config
            import tensorflow as tf
            from predict import load_labels, predict_file

            if not os.path.exists(config.MODEL_PATH):
                raise FileNotFoundError(
                    f"No trained model found at {config.MODEL_PATH}.\n"
                    f"Run `python train.py` first."
                )

            model = tf.keras.models.load_model(config.MODEL_PATH)
            labels = load_labels()

            self._model = model
            self._labels = labels
            self._predict_file = predict_file

    def predict(self, wav_path):
        """Blocking predict — must be called after ensure_loaded_sync()."""
        return self._predict_file(self._model, self._labels, wav_path)


# ----------------------------------------------------------------------------
# Main application
# ----------------------------------------------------------------------------
class BirdyCallApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Birdy Call")
        self.geometry(f"{WINDOW_W}x{WINDOW_H}")
        self.resizable(False, False)
        self.configure(bg=PANEL_BG)

        random.seed(7)  # deterministic, natural-looking forest each launch

        self.level_queue = queue.Queue(maxsize=64)
        self.recorder = AudioRecorder(self.level_queue)
        self.identifier = SpeciesIdentifier()
        self.is_recording = False
        self.record_start_time = None
        self.tick = 0
        self.bars = [0.0] * 40
        self.fireflies = []
        self.status_phrase_idx = 0

        self._build_layout()
        self._draw_background()
        self._init_fireflies()
        self._populate_devices()
        self._refresh_recordings_list()
        self._animate()
        self._poll_levels()

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- layout -----------------------------------------------------------
    def _build_layout(self):
        self.canvas = tk.Canvas(
            self, width=CANVAS_W, height=CANVAS_H, highlightthickness=0, bd=0
        )
        self.canvas.place(x=0, y=0)

        sidebar = tk.Frame(self, bg=PANEL_BG, width=SIDEBAR_W, height=WINDOW_H)
        sidebar.place(x=CANVAS_W, y=0)
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar, text="🌿 Field Notes", bg=PANEL_BG, fg=ACCENT_GOLD,
            font=("Georgia", 16, "bold")
        ).pack(pady=(22, 4), padx=20, anchor="w")

        tk.Label(
            sidebar, text="Input device", bg=PANEL_BG, fg=TEXT_MUTED,
            font=("Georgia", 10)
        ).pack(pady=(10, 2), padx=20, anchor="w")

        self.device_var = tk.StringVar()
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure(
            "Forest.TCombobox",
            fieldbackground="#123322",
            background="#123322",
            foreground=TEXT_CREAM,
            arrowcolor=ACCENT_GOLD,
        )
        self.device_combo = ttk.Combobox(
            sidebar, textvariable=self.device_var, state="readonly",
            style="Forest.TCombobox", width=30
        )
        self.device_combo.pack(padx=20, anchor="w")

        ttk.Separator(sidebar).pack(fill="x", padx=20, pady=18)

        tk.Label(
            sidebar, text="Past recordings", bg=PANEL_BG, fg=ACCENT_GOLD,
            font=("Georgia", 13, "bold")
        ).pack(padx=20, anchor="w")

        list_frame = tk.Frame(sidebar, bg=PANEL_BG)
        list_frame.pack(fill="both", expand=True, padx=20, pady=(8, 8))

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")

        self.rec_listbox = tk.Listbox(
            list_frame, bg="#0f2b1c", fg=TEXT_CREAM, selectbackground=EMBER_DIM,
            font=("Consolas", 10), bd=0, highlightthickness=0,
            yscrollcommand=scrollbar.set
        )
        self.rec_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.rec_listbox.yview)

        btn_row = tk.Frame(sidebar, bg=PANEL_BG)
        btn_row.pack(fill="x", padx=20, pady=(0, 16))

        self._make_small_button(btn_row, "▶ Play", self._play_selected).pack(
            side="left", expand=True, fill="x", padx=(0, 4)
        )
        self._make_small_button(btn_row, "📁 Reveal", self._reveal_selected).pack(
            side="left", expand=True, fill="x", padx=(4, 4)
        )
        self._make_small_button(btn_row, "⟳ Refresh", self._refresh_recordings_list).pack(
            side="left", expand=True, fill="x", padx=(4, 0)
        )

        self.identify_btn = tk.Button(
            sidebar, text="🔍  Identify Species", command=self._identify_selected,
            bg=EMBER, fg="#241008", activebackground=ACCENT_GOLD,
            activeforeground="#241008", font=("Georgia", 11, "bold"),
            bd=0, relief="flat", pady=10, cursor="hand2"
        )
        self.identify_btn.pack(fill="x", padx=20, pady=(0, 14))

        tk.Label(
            sidebar, text=f"Saved to:\n{RECORDINGS_DIR}", bg=PANEL_BG,
            fg=TEXT_MUTED, font=("Consolas", 8), justify="left", wraplength=SIDEBAR_W - 40
        ).pack(padx=20, pady=(0, 16), anchor="w")

    def _make_small_button(self, parent, text, command):
        b = tk.Button(
            parent, text=text, command=command, bg="#173d24", fg=TEXT_CREAM,
            activebackground=ACCENT_GOLD_DIM, activeforeground="#0d2818",
            font=("Georgia", 9), bd=0, relief="flat", padx=4, pady=6,
            cursor="hand2"
        )
        return b

    # -- background art ----------------------------------------------------
    def _draw_background(self):
        c = self.canvas
        # vertical gradient sky
        steps = 120
        for i in range(steps):
            t = i / steps
            color = rgb_to_hex(lerp_color(BG_TOP, BG_BOTTOM, t))
            y0 = int(CANVAS_H * i / steps)
            y1 = int(CANVAS_H * (i + 1) / steps)
            c.create_rectangle(0, y0, CANVAS_W, y1 + 1, fill=color, outline="")

        # soft moon
        c.create_oval(CANVAS_W - 160, 60, CANVAS_W - 80, 140, fill="#f4ecd0", outline="")
        c.create_oval(CANVAS_W - 168, 52, CANVAS_W - 72, 148, fill="", outline="#f4ecd0", width=1)

        # layered tree silhouettes (back to front)
        self._draw_tree_line(CANVAS_H * 0.55, TREE_BACK, jitter=40, count=9)
        self._draw_tree_line(CANVAS_H * 0.68, TREE_MID, jitter=55, count=8)
        self._draw_tree_line(CANVAS_H * 0.84, TREE_FRONT, jitter=70, count=7)

        c.create_text(
            CANVAS_W / 2, 46, text="Birdy Call", fill=ACCENT_GOLD,
            font=("Georgia", 22, "bold")
        )
        c.create_text(
            CANVAS_W / 2, 78, text="birds of the Indian subcontinent · local & offline",
            fill=TEXT_MUTED, font=("Georgia", 11, "italic")
        )

        # static layer tag so we can redraw dynamic bits above it
        c.tag_lower("all")

    def _draw_tree_line(self, base_y, color, jitter, count):
        c = self.canvas
        seg = CANVAS_W / count
        for i in range(count + 1):
            x = i * seg + random.uniform(-10, 10)
            height = random.uniform(60, 60 + jitter)
            width = seg * random.uniform(0.5, 0.9)
            top_y = base_y - height
            # trunk
            c.create_rectangle(
                x - 3, base_y - 10, x + 3, CANVAS_H, fill=color, outline=""
            )
            # canopy (layered triangles for a pine/deodar-ish look)
            for j in range(3):
                w = width * (1 - j * 0.22)
                y = top_y + j * height * 0.28
                c.create_polygon(
                    x, y - height * 0.32,
                    x - w / 2, y + height * 0.18,
                    x + w / 2, y + height * 0.18,
                    fill=color, outline=""
                )

    def _init_fireflies(self):
        for _ in range(14):
            self.fireflies.append({
                "x": random.uniform(20, CANVAS_W - 20),
                "y": random.uniform(CANVAS_H * 0.4, CANVAS_H - 40),
                "dx": random.uniform(-0.3, 0.3),
                "dy": random.uniform(-0.15, 0.15),
                "phase": random.uniform(0, math.tau),
                "id_glow": None,
                "id_core": None,
            })

    # -- devices -------------------------------------------------------------
    def _populate_devices(self):
        try:
            devices = sd.query_devices()
            input_devices = [
                f"{i}: {d['name']}" for i, d in enumerate(devices) if d["max_input_channels"] > 0
            ]
        except Exception:
            input_devices = []
        if not input_devices:
            input_devices = ["Default system microphone"]
        self.device_combo["values"] = input_devices
        self.device_combo.current(0)

    def _selected_device_index(self):
        val = self.device_var.get()
        if ":" in val:
            try:
                return int(val.split(":")[0])
            except ValueError:
                return None
        return None

    # -- dynamic canvas elements (redrawn each frame) -----------------------
    RECORD_CX = CANVAS_W / 2
    RECORD_CY = CANVAS_H - 150
    RECORD_R = 52

    def _animate(self):
        self.tick += 1
        c = self.canvas
        c.delete("dynamic")

        # fireflies drifting + soft pulsing glow
        for f in self.fireflies:
            f["x"] += f["dx"]
            f["y"] += f["dy"]
            if f["x"] < 10 or f["x"] > CANVAS_W - 10:
                f["dx"] *= -1
            if f["y"] < CANVAS_H * 0.35 or f["y"] > CANVAS_H - 20:
                f["dy"] *= -1
            glow = 0.5 + 0.5 * math.sin(self.tick * 0.05 + f["phase"])
            r_core = 2 + glow * 1.5
            r_glow = 6 + glow * 5
            c.create_oval(
                f["x"] - r_glow, f["y"] - r_glow, f["x"] + r_glow, f["y"] + r_glow,
                fill="", outline=FIREFLY, width=1, tags="dynamic", stipple="gray25"
            )
            c.create_oval(
                f["x"] - r_core, f["y"] - r_core, f["x"] + r_core, f["y"] + r_core,
                fill=FIREFLY, outline="", tags="dynamic"
            )

        # visualizer bars, arcing above the record button
        bar_area_w = 360
        bar_x0 = self.RECORD_CX - bar_area_w / 2
        n = len(self.bars)
        bw = bar_area_w / n
        for i, level in enumerate(self.bars):
            h = 6 + level * 70
            x = bar_x0 + i * bw
            y_base = self.RECORD_CY - self.RECORD_R - 30
            color = EMBER if self.is_recording else "#2b5238"
            c.create_rectangle(
                x, y_base - h, x + bw * 0.6, y_base, fill=color, outline="", tags="dynamic"
            )

        # decay bars toward zero when idle
        if not self.is_recording:
            self.bars = [b * 0.85 for b in self.bars]

        # record button glow (pulsing ember when active)
        if self.is_recording:
            pulse = 0.5 + 0.5 * math.sin(self.tick * 0.15)
            glow_r = self.RECORD_R + 14 + pulse * 10
            c.create_oval(
                self.RECORD_CX - glow_r, self.RECORD_CY - glow_r,
                self.RECORD_CX + glow_r, self.RECORD_CY + glow_r,
                outline=EMBER, width=2, fill="", tags="dynamic"
            )
            btn_fill = EMBER
            ring = EMBER_DIM
        else:
            btn_fill = "#173d24"
            ring = ACCENT_GOLD_DIM

        c.create_oval(
            self.RECORD_CX - self.RECORD_R, self.RECORD_CY - self.RECORD_R,
            self.RECORD_CX + self.RECORD_R, self.RECORD_CY + self.RECORD_R,
            fill=btn_fill, outline=ring, width=3, tags=("dynamic", "record_btn")
        )
        self._draw_mic_icon(self.RECORD_CX, self.RECORD_CY)

        # timer / status text
        if self.is_recording and self.record_start_time:
            elapsed = time.time() - self.record_start_time
            mins, secs = divmod(int(elapsed), 60)
            timer_text = f"{mins:02d}:{secs:02d}"
        else:
            timer_text = "00:00"
        c.create_text(
            self.RECORD_CX, self.RECORD_CY + self.RECORD_R + 34, text=timer_text,
            fill=TEXT_CREAM, font=("Consolas", 18, "bold"), tags="dynamic"
        )

        if self.tick % 90 == 0:
            phrases = STATUS_PHRASES_RECORDING if self.is_recording else STATUS_PHRASES_IDLE
            self.status_phrase_idx = (self.status_phrase_idx + 1) % len(phrases)
        phrases = STATUS_PHRASES_RECORDING if self.is_recording else STATUS_PHRASES_IDLE
        status_text = phrases[self.status_phrase_idx % len(phrases)]
        c.create_text(
            self.RECORD_CX, self.RECORD_CY + self.RECORD_R + 62, text=status_text,
            fill=TEXT_MUTED, font=("Georgia", 11, "italic"), tags="dynamic"
        )

        c.tag_bind("record_btn", "<Button-1>", lambda e: self._toggle_recording())
        c.tag_raise("dynamic")

        self.after(30, self._animate)

    def _draw_mic_icon(self, cx, cy):
        c = self.canvas
        w, h = 18, 26
        c.create_rectangle(
            cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2 - 6,
            fill=TEXT_CREAM, outline="", tags="dynamic"
        )
        c.create_arc(
            cx - w / 2 - 4, cy - h / 2, cx + w / 2 + 4, cy + h / 2 + 10,
            start=180, extent=180, style="arc", outline=TEXT_CREAM, width=2, tags="dynamic"
        )
        c.create_line(cx, cy + h / 2 + 10, cx, cy + h / 2 + 20, fill=TEXT_CREAM, width=2, tags="dynamic")
        c.create_line(
            cx - 8, cy + h / 2 + 20, cx + 8, cy + h / 2 + 20, fill=TEXT_CREAM, width=2, tags="dynamic"
        )

    # -- recording control ---------------------------------------------------
    def _toggle_recording(self):
        if not self.is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        try:
            self.recorder.device = self._selected_device_index()
            self.recorder.start()
        except Exception as e:
            messagebox.showerror("Microphone error", f"Couldn't start recording:\n{e}")
            return
        self.is_recording = True
        self.record_start_time = time.time()

    def _stop_recording(self):
        audio_data = self.recorder.stop()
        self.is_recording = False
        self.record_start_time = None
        if audio_data is None or len(audio_data) == 0:
            messagebox.showinfo("Nothing recorded", "No audio was captured.")
            return

        default_name = f"bird_call_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        name = simpledialog.askstring(
            "Save recording", "Name for this call (no extension):", initialvalue=default_name
        )
        if not name:
            name = default_name
        safe_name = "".join(ch for ch in name if ch.isalnum() or ch in ("_", "-")) or default_name
        path = os.path.join(RECORDINGS_DIR, f"{safe_name}.wav")

        try:
            AudioRecorder.save_wav(path, audio_data)
        except Exception as e:
            messagebox.showerror("Save error", f"Couldn't save WAV file:\n{e}")
            return

        self._refresh_recordings_list()
        messagebox.showinfo("Saved", f"Saved recording as:\n{os.path.basename(path)}")

    # -- level polling (visualizer) ------------------------------------------
    def _poll_levels(self):
        latest = None
        try:
            while True:
                latest = self.level_queue.get_nowait()
        except queue.Empty:
            pass
        if latest is not None:
            self.bars.pop(0)
            self.bars.append(latest)
        self.after(30, self._poll_levels)

    # -- recordings list ------------------------------------------------------
    def _refresh_recordings_list(self):
        self.rec_listbox.delete(0, tk.END)
        files = sorted(
            [f for f in os.listdir(RECORDINGS_DIR) if f.lower().endswith(".wav")],
            reverse=True,
        )
        for f in files:
            self.rec_listbox.insert(tk.END, f)

    def _selected_file_path(self):
        sel = self.rec_listbox.curselection()
        if not sel:
            return None
        return os.path.join(RECORDINGS_DIR, self.rec_listbox.get(sel[0]))

    def _play_selected(self):
        path = self._selected_file_path()
        if not path:
            messagebox.showinfo("No selection", "Select a recording from the list first.")
            return

        def _play():
            try:
                with wave.open(path, "rb") as wf:
                    fs = wf.getframerate()
                    n = wf.getnframes()
                    data = np.frombuffer(wf.readframes(n), dtype=np.int16)
                sd.play(data, fs)
                sd.wait()
            except Exception as e:
                messagebox.showerror("Playback error", str(e))

        threading.Thread(target=_play, daemon=True).start()

    def _reveal_selected(self):
        path = self._selected_file_path()
        target = path if path else RECORDINGS_DIR
        try:
            if sys.platform.startswith("win"):
                os.startfile(os.path.dirname(target) if path else target)
            elif sys.platform == "darwin":
                subprocess.run(["open", "-R", target] if path else ["open", target])
            else:
                subprocess.run(["xdg-open", os.path.dirname(target) if path else target])
        except Exception as e:
            messagebox.showerror("Couldn't open folder", str(e))

    # -- species identification ---------------------------------------------
    def _identify_selected(self):
        path = self._selected_file_path()
        if not path:
            messagebox.showinfo(
                "No selection", "Select a recording from the list first."
            )
            return
        if self.is_recording:
            messagebox.showinfo("Still recording", "Stop recording before identifying a call.")
            return

        self.identify_btn.config(state="disabled")
        overlay = LoadingOverlay(self, os.path.basename(path))

        def worker():
            try:
                self.identifier.ensure_loaded_sync()
                top3, probs = self.identifier.predict(path)
                self.after(0, lambda: self._on_identify_success(overlay, path, top3))
            except Exception as e:
                err = str(e)
                self.after(0, lambda: self._on_identify_error(overlay, err))

        threading.Thread(target=worker, daemon=True).start()

    def _on_identify_success(self, overlay, path, top3):
        overlay.close()
        self.identify_btn.config(state="normal")
        ResultsWindow(self, os.path.basename(path), top3)

    def _on_identify_error(self, overlay, error_message):
        overlay.close()
        self.identify_btn.config(state="normal")
        messagebox.showerror("Identification failed", error_message)

    def _on_close(self):
        if self.is_recording:
            self.recorder.stop()
        self.destroy()


# ----------------------------------------------------------------------------
# Forest-themed popups: loading overlay + identification results
# ----------------------------------------------------------------------------
class LoadingOverlay(tk.Toplevel):
    """A small borderless, forest-themed 'thinking' window shown while the
    model loads and runs inference. Not closable by the user."""

    def __init__(self, parent, filename):
        super().__init__(parent)
        self.overrideredirect(True)
        self.configure(bg=PANEL_BG)
        w, h = 380, 200
        px = parent.winfo_rootx() + (parent.winfo_width() - w) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - h) // 2
        self.geometry(f"{w}x{h}+{px}+{py}")
        self.attributes("-topmost", True)

        self.canvas = tk.Canvas(self, width=w, height=h, bg=PANEL_BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        rounded_rect(self.canvas, 4, 4, w - 4, h - 4, r=20, fill="#0f2b1c", outline=PANEL_BORDER, width=2)

        self.canvas.create_text(
            w / 2, 44, text="🕊  Consulting the forest...", fill=ACCENT_GOLD,
            font=("Georgia", 14, "bold")
        )
        self.canvas.create_text(
            w / 2, 70, text=filename, fill=TEXT_MUTED, font=("Consolas", 9)
        )

        self._tick = 0
        self._dots_text_id = self.canvas.create_text(
            w / 2, 110, text="", fill=TEXT_CREAM, font=("Georgia", 20)
        )
        self._ring_id = self.canvas.create_arc(
            w / 2 - 22, 130, w / 2 + 22, 174, start=0, extent=120,
            style="arc", outline=EMBER, width=4
        )
        self._closed = False
        self._animate()

    def _animate(self):
        if self._closed:
            return
        self._tick += 1
        dots = "." * ((self._tick // 8) % 4)
        self.canvas.itemconfig(self._dots_text_id, text=dots)
        start = (self._tick * 6) % 360
        self.canvas.itemconfig(self._ring_id, start=start)
        self.after(30, self._animate)

    def close(self):
        self._closed = True
        try:
            self.destroy()
        except tk.TclError:
            pass


class ResultsWindow(tk.Toplevel):
    """Displays the top-3 predicted species for a recording, forest-styled,
    with animated confidence bars."""

    def __init__(self, parent, filename, top3):
        super().__init__(parent)
        self.title("Identification Result")
        self.configure(bg=PANEL_BG)
        self.resizable(False, False)
        w, h = 460, 420
        px = parent.winfo_rootx() + (parent.winfo_width() - w) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - h) // 2
        self.geometry(f"{w}x{h}+{px}+{py}")
        self.transient(parent)
        self.grab_set()

        canvas = tk.Canvas(self, width=w, height=h, bg=PANEL_BG, highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        # background gradient, echoing the main app
        steps = 60
        for i in range(steps):
            t = i / steps
            color = rgb_to_hex(lerp_color(BG_TOP, BG_BOTTOM, t))
            y0 = int(h * i / steps)
            y1 = int(h * (i + 1) / steps)
            canvas.create_rectangle(0, y0, w, y1 + 1, fill=color, outline="")

        canvas.create_text(
            w / 2, 34, text="🌿 Most Likely Species", fill=ACCENT_GOLD,
            font=("Georgia", 18, "bold")
        )
        canvas.create_text(
            w / 2, 60, text=filename, fill=TEXT_MUTED, font=("Consolas", 9)
        )

        row_y_positions = [110, 210, 310]
        bar_x0, bar_x1 = 150, w - 40
        bar_max_w = bar_x1 - bar_x0

        for i, (species, confidence) in enumerate(top3[:3]):
            y = row_y_positions[i]
            color = MEDAL_COLORS[i]
            icon = MEDAL_ICONS[i]

            rounded_rect(
                canvas, 24, y - 34, w - 24, y + 34, r=16,
                fill="#0f2b1c", outline=color, width=2
            )
            canvas.create_text(
                52, y, text=icon, font=("Georgia", 20)
            )
            canvas.create_text(
                bar_x0, y - 16, text=species, fill=TEXT_CREAM, anchor="w",
                font=("Georgia", 13, "bold")
            )
            canvas.create_text(
                bar_x1, y - 16, text=f"{confidence:.1f}%", fill=color, anchor="e",
                font=("Consolas", 12, "bold")
            )
            # empty track
            canvas.create_rectangle(
                bar_x0, y + 2, bar_x1, y + 14, fill="#173d24", outline=""
            )
            # animated fill bar
            self._animate_bar(canvas, bar_x0, y + 2, bar_x1, y + 14, bar_max_w, confidence / 100.0, color)

        close_btn = tk.Button(
            self, text="Close", command=self.destroy, bg="#173d24", fg=TEXT_CREAM,
            activebackground=ACCENT_GOLD_DIM, activeforeground="#0d2818",
            font=("Georgia", 10), bd=0, relief="flat", padx=14, pady=6, cursor="hand2"
        )
        close_btn.place(x=w / 2 - 40, y=h - 46, width=80, height=32)

    def _animate_bar(self, canvas, x0, y0, x1, y1, max_w, target_fraction, color, step=0):
        progress = min(step / 20, 1.0)
        current_w = max_w * target_fraction * progress
        canvas.delete(f"bar_fill_{y0}")
        if current_w > 0:
            canvas.create_rectangle(
                x0, y0, x0 + current_w, y1, fill=color, outline="", tags=f"bar_fill_{y0}"
            )
        if progress < 1.0:
            self.after(15, lambda: self._animate_bar(
                canvas, x0, y0, x1, y1, max_w, target_fraction, color, step + 1
            ))


if __name__ == "__main__":
    app = BirdyCallApp()
    app.mainloop()