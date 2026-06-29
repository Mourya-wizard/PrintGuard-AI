# -*- coding: utf-8 -*-
"""
PrintGuard — Moonraker/Klipper Client
Talks to the Creality K2 Plus via Moonraker REST API.
Falls back to simulation mode automatically if printer is unreachable.
"""

import requests
import time
import random
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PrinterState:
    extruder_temp: float = 205.0
    extruder_target: float = 205.0
    bed_temp: float = 60.0
    bed_target: float = 60.0
    fan_speed: float = 0.75       # 0.0–1.0
    print_state: str = "printing"  # printing | paused | complete | error
    filename: str = "benchy.gcode"
    progress: float = 0.42         # 0.0–1.0
    current_layer: int = 47
    total_layers: int = 120
    is_simulated: bool = False


class MoonrakerClient:
    """
    Wraps the Moonraker HTTP API for the Creality K2 Plus.
    Automatically enters simulation mode if the printer is unreachable.
    """

    def __init__(self, printer_ip: str = "192.168.1.100", port: int = 7125):
        self.base_url = f"http://{printer_ip}:{port}"
        self._sim_state = PrinterState(is_simulated=True)
        self._sim_layer = 47
        self._sim_start = time.time()
        self.available = self._test_connection()
        if not self.available:
            print(f"[Moonraker] [!] Printer unreachable at {self.base_url} - running in simulation mode")
        else:
            print(f"[Moonraker] [OK] Connected to printer at {self.base_url}")

    def _test_connection(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/server/info", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    # ── READ ─────────────────────────────────────────────────────────────────

    def get_printer_state(self) -> PrinterState:
        if not self.available:
            return self._get_simulated_state()
        try:
            endpoint = "/printer/objects/query"
            params = {"extruder": "", "heater_bed": "",
                      "print_stats": "", "virtual_sdcard": "", "fan": ""}
            r = requests.get(f"{self.base_url}{endpoint}", params=params, timeout=3)
            s = r.json()["result"]["status"]
            return PrinterState(
                extruder_temp=s["extruder"]["temperature"],
                extruder_target=s["extruder"]["target"],
                bed_temp=s["heater_bed"]["temperature"],
                bed_target=s["heater_bed"]["target"],
                fan_speed=s.get("fan", {}).get("speed", 0.75),
                print_state=s["print_stats"]["state"],
                filename=s["print_stats"].get("filename", "unknown.gcode"),
                progress=s["virtual_sdcard"].get("progress", 0.0),
                current_layer=s["print_stats"].get("info", {}).get("current_layer", 0),
                total_layers=s["print_stats"].get("info", {}).get("total_layer", 0),
                is_simulated=False
            )
        except Exception as e:
            print(f"[Moonraker] Query failed: {e} — returning simulated state")
            return self._get_simulated_state()

    def _get_simulated_state(self) -> PrinterState:
        """Generate realistic, slowly-changing simulated printer state."""
        elapsed = time.time() - self._sim_start
        # Slowly advance print progress
        self._sim_state.progress = min(0.99, 0.42 + elapsed / 3600)
        self._sim_state.current_layer = int(47 + elapsed / 30)
        # Add small realistic temp fluctuations (PID control noise)
        self._sim_state.extruder_temp = 205.0 + random.gauss(0, 0.8)
        self._sim_state.bed_temp = 60.0 + random.gauss(0, 0.4)
        self._sim_state.is_simulated = True
        return self._sim_state

    # ── WRITE ─────────────────────────────────────────────────────────────────

    def send_gcode(self, command: str) -> dict:
        """Send a G-code command. Returns status dict."""
        result = {"command": command, "sent_at": time.time(), "simulated": not self.available}
        if self.available:
            try:
                r = requests.post(
                    f"{self.base_url}/printer/gcode/script",
                    params={"script": command},
                    timeout=4
                )
                result["success"] = r.status_code == 200
                result["response"] = r.text
            except Exception as e:
                result["success"] = False
                result["error"] = str(e)
        else:
            # Update sim state to reflect the command
            self._apply_gcode_to_sim(command)
            result["success"] = True
            result["response"] = "OK (simulated)"
        return result

    def _apply_gcode_to_sim(self, command: str):
        """Apply a G-code command to the simulated printer state."""
        cmd = command.strip().upper()
        if "SET_HEATER_TEMPERATURE" in cmd and "EXTRUDER" in cmd and "TARGET=" in cmd:
            try:
                target = float(cmd.split("TARGET=")[1].split()[0])
                self._sim_state.extruder_target = target
                self._sim_state.extruder_temp = target + random.gauss(0, 0.5)
            except Exception:
                pass
        elif "SET_HEATER_TEMPERATURE" in cmd and "HEATER_BED" in cmd and "TARGET=" in cmd:
            try:
                target = float(cmd.split("TARGET=")[1].split()[0])
                self._sim_state.bed_target = target
                self._sim_state.bed_temp = target + random.gauss(0, 0.3)
            except Exception:
                pass
        elif cmd.startswith("M106"):
            try:
                s = int(cmd.split("S")[1].split()[0])
                self._sim_state.fan_speed = s / 255.0
            except Exception:
                pass
        elif cmd == "PAUSE":
            self._sim_state.print_state = "paused"
        elif cmd == "RESUME":
            self._sim_state.print_state = "printing"
        elif cmd == "M112":
            self._sim_state.print_state = "error"

    # ── CONVENIENCE METHODS ──────────────────────────────────────────────────

    def set_nozzle_temp(self, new_target: float) -> dict:
        new_target = round(max(180, min(230, new_target)))
        return self.send_gcode(f"SET_HEATER_TEMPERATURE HEATER=extruder TARGET={new_target}")

    def set_bed_temp(self, new_target: float) -> dict:
        new_target = round(max(45, min(75, new_target)))
        return self.send_gcode(f"SET_HEATER_TEMPERATURE HEATER=heater_bed TARGET={new_target}")

    def set_fan_percent(self, percent: int) -> dict:
        pwm = int(255 * max(0, min(100, percent)) / 100)
        return self.send_gcode(f"M106 S{pwm}")

    def pause_print(self) -> dict:
        if self.available:
            try:
                r = requests.post(f"{self.base_url}/printer/print/pause", timeout=3)
                return {"success": r.status_code == 200, "simulated": False}
            except Exception:
                pass
        return self.send_gcode("PAUSE")

    def emergency_stop(self) -> dict:
        if self.available:
            try:
                requests.post(f"{self.base_url}/printer/emergency_stop", timeout=3)
                return {"success": True, "simulated": False}
            except Exception:
                pass
        self._sim_state.print_state = "error"
        return {"success": True, "simulated": True, "command": "M112"}
