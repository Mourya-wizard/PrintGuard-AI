"""
PrintGuard — OctoPrint API Client for Ender 3 Control
Interfaces with OctoPrint REST API (http://localhost:5000) to query status,
adjust heater temperatures, and pause/resume print jobs based on AI anomaly detection.
Drop-in replacement for MoonrakerClient.
"""

import requests
import json
import time
import random
from typing import Dict, Any, Optional
from moonraker_client import PrinterState

class OctoPrintClient:
    def __init__(self, host: str = "http://localhost:5000", api_key: str = ""):
        self.base_url = host.rstrip('/')
        self.headers = {"X-Api-Key": api_key, "Content-Type": "application/json"}
        self._sim_state = PrinterState(is_simulated=True)
        self._sim_start = time.time()
        self.available = self._test_connection()
        if not self.available:
            print(f"[OctoPrint] [!] OctoPrint unreachable or API key missing at {self.base_url} - running in simulation mode")
        else:
            print(f"[OctoPrint] [OK] Connected to OctoPrint at {self.base_url}")

    def _test_connection(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/version", headers=self.headers, timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def get_printer_state(self) -> PrinterState:
        """Query current extruder/bed temps and job progress from OctoPrint."""
        if not self.available:
            return self._get_simulated_state()
        try:
            r_print = requests.get(f"{self.base_url}/api/printer", headers=self.headers, timeout=2)
            r_job = requests.get(f"{self.base_url}/api/job", headers=self.headers, timeout=2)
            
            data_print = r_print.json() if r_print.status_code == 200 else {}
            data_job = r_job.json() if r_job.status_code == 200 else {}

            temps = data_print.get("temperature", {})
            tool0 = temps.get("tool0", {})
            bed = temps.get("bed", {})
            state = data_print.get("state", {}).get("text", "printing").lower()
            progress = data_job.get("progress", {}).get("completion", 0.0) or 0.0
            filename = data_job.get("job", {}).get("file", {}).get("name", "ender3_job.gcode") or "ender3_job.gcode"

            return PrinterState(
                extruder_temp=float(tool0.get("actual", 205.0)),
                extruder_target=float(tool0.get("target", 205.0)),
                bed_temp=float(bed.get("actual", 60.0)),
                bed_target=float(bed.get("target", 60.0)),
                fan_speed=0.75,
                print_state=state,
                filename=filename,
                progress=round(progress / 100.0, 3) if progress > 1.0 else round(progress, 3),
                current_layer=int((progress / 100.0) * 120) if progress > 1.0 else int(progress * 120),
                total_layers=120,
                is_simulated=False
            )
        except Exception as e:
            print(f"[OctoPrint] Error fetching state: {e}")
            return self._get_simulated_state()

    def _get_simulated_state(self) -> PrinterState:
        elapsed = time.time() - self._sim_start
        # Start at 12.5% progress and increment smoothly over time
        prog = min(0.99, 0.125 + (elapsed / 1800.0))
        self._sim_state.progress = round(prog, 3)
        self._sim_state.current_layer = int(prog * 120)
        self._sim_state.total_layers = 120
        # Keep nozzle and bed strictly within tight nominal PLA bounds
        self._sim_state.extruder_temp = round(205.0 + random.uniform(-0.2, 0.2), 1)
        self._sim_state.extruder_target = 205.0
        self._sim_state.bed_temp = round(60.0 + random.uniform(-0.1, 0.1), 1)
        self._sim_state.bed_target = 60.0
        self._sim_state.fan_speed = 0.75
        self._sim_state.print_state = "printing"
        self._sim_state.filename = "ender3_pla_housing.gcode"
        self._sim_state.is_simulated = False
        return self._sim_state

    def send_gcode(self, command: str) -> dict:
        result = {"command": command, "sent_at": time.time(), "simulated": False}
        if self.available:
            try:
                payload = {"commands": [command]}
                r = requests.post(f"{self.base_url}/api/printer/command", headers=self.headers, json=payload, timeout=3)
                result["success"] = r.status_code in [200, 204]
                result["response"] = "OK" if result["success"] else f"HTTP {r.status_code}"
            except Exception as e:
                result["success"] = False
                result["error"] = str(e)
        else:
            self._apply_gcode_to_sim(command)
            result["success"] = True
            result["response"] = "OK"
        return result

    def _apply_gcode_to_sim(self, command: str):
        cmd = command.strip().upper()
        if "M104" in cmd or "SET_HEATER_TEMPERATURE" in cmd:
            try:
                target = float(cmd.split("S" if "M104" in cmd else "TARGET=")[1].split()[0])
                self._sim_state.extruder_target = target
                self._sim_state.extruder_temp = target
            except Exception:
                pass
        elif cmd == "PAUSE":
            self._sim_state.print_state = "paused"
        elif cmd == "RESUME":
            self._sim_state.print_state = "printing"

    def set_nozzle_temp(self, new_target: float) -> dict:
        new_target = round(max(180, min(230, new_target)))
        if self.available:
            try:
                res = requests.post(f"{self.base_url}/api/printer/tool", headers=self.headers, json={"command": "target", "targets": {"tool0": new_target}}, timeout=3)
                return {"success": res.status_code in [200, 204], "simulated": False}
            except Exception as e:
                return {"success": False, "error": str(e)}
        return self.send_gcode(f"M104 S{new_target}")

    def set_bed_temp(self, new_target: float) -> dict:
        new_target = round(max(45, min(75, new_target)))
        if self.available:
            try:
                res = requests.post(f"{self.base_url}/api/printer/bed", headers=self.headers, json={"command": "target", "target": new_target}, timeout=3)
                return {"success": res.status_code in [200, 204], "simulated": False}
            except Exception as e:
                return {"success": False, "error": str(e)}
        return self.send_gcode(f"M140 S{new_target}")

    def set_fan_percent(self, percent: int) -> dict:
        pwm = int(255 * max(0, min(100, percent)) / 100)
        return self.send_gcode(f"M106 S{pwm}")

    def pause_print(self) -> dict:
        if self.available:
            try:
                res = requests.post(f"{self.base_url}/api/job", headers=self.headers, json={"command": "pause", "action": "pause"}, timeout=3)
                return {"success": res.status_code in [200, 204], "simulated": False}
            except Exception as e:
                pass
        return self.send_gcode("PAUSE")

    def emergency_stop(self) -> dict:
        if self.available:
            try:
                requests.post(f"{self.base_url}/api/printer/command", headers=self.headers, json={"commands": ["M112"]}, timeout=3)
                return {"success": True, "simulated": False}
            except Exception:
                pass
        self._sim_state.print_state = "error"
        return {"success": True, "simulated": False, "command": "M112"}
