"""
PrintGuard — Topdon TC001 Thermal Camera Capture Module
Handles UVC device capture, raw Y16 → Celsius conversion, and base64 encoding for VLM
"""

import cv2
import numpy as np
import base64
import time
import threading
from dataclasses import dataclass
from typing import Optional


@dataclass
class ThermalFrame:
    """A single captured and processed thermal frame."""
    timestamp: float
    heatmap_bgr: np.ndarray        # False-color BGR image for display
    temp_matrix: np.ndarray        # 256×192 float32 array of °C values
    temp_min: float
    temp_max: float
    temp_mean: float
    base64_png: str                # Base64-encoded PNG for Gemma 4 VLM input


class ThermalCamera:
    """
    Interface for the Topdon TC001 thermal camera.
    
    The TC001 presents as a standard UVC device (USB Video Class).
    Raw Y16 frames: top 192 rows = thermal data, bottom rows = calibration metadata.
    Temperature conversion: T_celsius = raw_uint16 / 64.0 - 273.15
    
    On Windows: use CAP_DSHOW backend. If not detected, try device indices 0-4.
    """

    THERMAL_HEIGHT = 192
    THERMAL_WIDTH = 256
    COLORMAP = cv2.COLORMAP_INFERNO  # Scientific standard for thermal imaging

    def __init__(self, device_index: int = 0, use_dshow: bool = True):
        self.device_index = device_index
        self.use_dshow = use_dshow
        self.cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[ThermalFrame] = None
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self.is_demo_mode = False

    def find_device(self) -> int:
        """Auto-detect the TC001 by checking device indices 0-4."""
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Topdon TC001 raw frame over DSHOW/MSMF has size 196608 bytes (384 * 256 * 2)
                    if frame.size == 196608 or (frame.shape[0] >= 192 and frame.shape[1] <= 400):
                        print(f"[TC001] Found thermal camera at index {i}, size={frame.size}, shape={frame.shape}")
                        cap.release()
                        return i
                    else:
                        print(f"[TC001] Ignoring webcam at index {i} (shape={frame.shape}, size={frame.size})")
                cap.release()
        return -1

    def open(self) -> bool:
        """Open the camera. Returns True if successful."""
        self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            print(f"[TC001] Could not open device at index {self.device_index}")
            return False
        # Request raw unconverted format from the UVC device
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        ret, test = self.cap.read()
        if ret and test is not None:
            print(f"[TC001] Camera opened at index {self.device_index} | shape={test.shape}")
        return True

    def _process_raw_frame(self, raw_frame: np.ndarray) -> ThermalFrame:
        """Convert raw Y16 frame → ThermalFrame with Celsius data and heatmap."""
        if raw_frame.shape == (self.THERMAL_HEIGHT, self.THERMAL_WIDTH):
            # Simulated / Demo frame passed directly in Celsius (uint16 or float)
            temp_matrix = raw_frame.astype(np.float32)
            thermal_vis = temp_matrix
        elif raw_frame.dtype == np.uint8 and raw_frame.size >= 384 * 256 * 2:
            # Reshape 1D byte array from DSHOW into 16-bit 384x256 matrix
            y16 = raw_frame.flatten()[:384 * 256 * 2].view(np.uint16).reshape((384, 256))
            thermal_vis = y16[:self.THERMAL_HEIGHT, :].astype(np.float32)
            temp_matrix = y16[self.THERMAL_HEIGHT:self.THERMAL_HEIGHT*2, :].astype(np.float32) / 64.0 - 273.15
        elif raw_frame.dtype == np.uint16 and raw_frame.shape[0] >= 384:
            y16 = raw_frame[:384, :]
            thermal_vis = y16[:self.THERMAL_HEIGHT, :].astype(np.float32)
            temp_matrix = y16[self.THERMAL_HEIGHT:self.THERMAL_HEIGHT*2, :].astype(np.float32) / 64.0 - 273.15
        else:
            temp_matrix = np.full((self.THERMAL_HEIGHT, self.THERMAL_WIDTH), 25.0, dtype=np.float32)
            thermal_vis = temp_matrix

        # Stats
        t_min = float(temp_matrix.min())
        t_max = float(temp_matrix.max())
        t_mean = float(temp_matrix.mean())

        # Normalize for colormap (0-255)
        normalized = cv2.normalize(
            thermal_vis, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        heatmap_bgr = cv2.applyColorMap(normalized, self.COLORMAP)

        # Annotate with temperature info overlay
        heatmap_bgr = self._add_overlay(heatmap_bgr, t_min, t_max, t_mean)

        # Encode to base64 PNG for VLM
        _, buffer = cv2.imencode('.png', heatmap_bgr)
        b64 = base64.b64encode(buffer).decode('utf-8')

        return ThermalFrame(
            timestamp=time.time(),
            heatmap_bgr=heatmap_bgr,
            temp_matrix=temp_matrix,
            temp_min=t_min,
            temp_max=t_max,
            temp_mean=t_mean,
            base64_png=b64
        )

    def _add_overlay(self, img: np.ndarray, t_min: float, t_max: float, t_mean: float) -> np.ndarray:
        """Add temperature stats overlay to the heatmap image."""
        overlay = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Semi-transparent dark background for text
        cv2.rectangle(overlay, (0, 0), (130, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        cv2.putText(img, f"Min: {t_min:6.1f}C", (5, 15), font, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f"Max: {t_max:6.1f}C", (5, 30), font, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f"Avg: {t_mean:6.1f}C", (5, 45), font, 0.4, (200, 200, 200), 1)
        # Timestamp
        ts = time.strftime("%H:%M:%S")
        cv2.putText(img, ts, (self.THERMAL_WIDTH - 60, 15), font, 0.35, (150, 150, 150), 1)
        return img

    def read_frame(self) -> Optional[ThermalFrame]:
        """Read a single frame from the camera."""
        if self.is_demo_mode:
            return self._generate_demo_frame()
        if self.cap is None or not self.cap.isOpened():
            return None
        with self._lock:
            ret, raw_frame = self.cap.read()
            if not ret or raw_frame is None:
                return None
            return self._process_raw_frame(raw_frame)

    def start_capture_thread(self):
        """Start background capture thread — updates latest_frame continuously at 25fps."""
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def _capture_loop(self):
        while self._running:
            frame = self.read_frame()
            if frame:
                with self._lock:
                    self._latest_frame = frame
            time.sleep(1.0 / 25.0)  # 25 FPS

    def get_latest_frame(self) -> Optional[ThermalFrame]:
        with self._lock:
            return self._latest_frame

    def _generate_demo_frame(self) -> ThermalFrame:
        """
        Generate a realistic simulated thermal frame for demo purposes.
        Simulates a PLA print bed with a heating nozzle and optional defects.
        """
        t = time.time()
        frame = np.zeros((self.THERMAL_HEIGHT, self.THERMAL_WIDTH), dtype=np.float32)

        # Ambient background (~25°C)
        frame[:] = 25.0

        # Heated bed (bottom 2/3 of frame) — uniform nominal ~60°C
        bed_temp = 60.0
        frame[64:, :] = bed_temp + np.random.normal(0, 0.5, (128, self.THERMAL_WIDTH))

        # Print object on bed — warm region (~120-150°C for recently deposited layers)
        cx, cy = self.THERMAL_WIDTH // 2, 100
        for r in range(70, 150):
            for c in range(80, 176):
                d = np.sqrt((r - cy) ** 2 + (c - cx) ** 2)
                if d < 50:
                    frame[r, c] = 130 + (50 - d) * 0.8

        # Nozzle hotspot (~205°C) — moves slightly over time to simulate printing
        nozzle_x = int(cx + 30 * np.sin(t * 0.5))
        nozzle_y = int(cy - 10 + 5 * np.cos(t * 0.7))
        for r in range(max(0, nozzle_y - 8), min(self.THERMAL_HEIGHT, nozzle_y + 8)):
            for c in range(max(0, nozzle_x - 8), min(self.THERMAL_WIDTH, nozzle_x + 8)):
                d = np.sqrt((r - nozzle_y) ** 2 + (c - nozzle_x) ** 2)
                if d < 8:
                    frame[r, c] = 205 - d * 3

        # Clamp temperatures
        frame = np.clip(frame, 20, 250)

        return self._process_raw_frame(frame.astype(np.uint16))

    def enable_demo_mode(self):
        """Use generated demo frames instead of live camera (for testing without TC001)."""
        self.is_demo_mode = True
        print("[TC001] Demo mode enabled — using simulated thermal frames")

    def release(self):
        self._running = False
        if self._capture_thread:
            self._capture_thread.join(timeout=2)
        if self.cap:
            self.cap.release()
