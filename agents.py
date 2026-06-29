"""
PrintGuard — Multi-Agent Pipeline
5 Gemma 4 agents running on Cerebras for real-time thermal defect detection and correction.
"""

import json
import time
import asyncio
from typing import Optional, AsyncGenerator
from cerebras.cloud.sdk import Cerebras
from knowledge.pla_prompts import (
    ORCHESTRATOR_SYSTEM, THERMAL_ANALYST_SYSTEM,
    TELEMETRY_ANALYZER_SYSTEM, DEFECT_CLASSIFIER_SYSTEM, CORRECTION_AGENT_SYSTEM,
    THERMAL_ANALYSIS_SCHEMA, TELEMETRY_ANALYSIS_SCHEMA,
    DEFECT_CLASSIFICATION_SCHEMA, CORRECTION_SCHEMA
)

MODEL = "gemma-4-31b"


class PrintGuardPipeline:
    """
    Orchestrates 5 Gemma 4 agents on Cerebras for PrintGuard.

    Pipeline (per analysis cycle, ~2 seconds total on Cerebras):
      Agent 1: Orchestrator    — decide skip vs analyze
      Agent 2: Thermal Analyst — VLM analyzes thermal image
      Agent 3: Telemetry Analyzer — reads printer JSON state
      Agent 4: Defect Classifier — cross-references both
      Agent 5: Correction Agent  — selects G-code corrections
    """

    def __init__(self, api_key: str):
        self.client = Cerebras(api_key=api_key)
        self.last_defect = "normal"
        self.cycle_count = 0
        self.total_corrections = 0

    def _call(self, system: str, user_content: list | str,
              schema: Optional[dict] = None, max_tokens: int = 512) -> tuple[str, float, int]:
        """
        Make a Gemma 4 call. Returns (content, elapsed_seconds, tokens_used).
        user_content can be a string or a list (for multimodal with image).
        """
        t0 = time.perf_counter()
        kwargs = dict(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content}
            ],
            max_tokens=max_tokens,
        )
        if schema:
            kwargs["response_format"] = schema

        response = self.client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - t0
        tokens = response.usage.completion_tokens if response.usage else 0
        content = response.choices[0].message.content
        return content, elapsed, tokens

    # ── AGENT 1: ORCHESTRATOR ─────────────────────────────────────────────────

    def run_orchestrator(self, temp_stats: dict, printer_state: dict) -> tuple[str, float, int]:
        """Decide whether this frame needs deep analysis."""
        user_msg = (
            f"Thermal stats: min={temp_stats['min']:.1f}°C, "
            f"max={temp_stats['max']:.1f}°C, avg={temp_stats['mean']:.1f}°C. "
            f"Printer: nozzle={printer_state['extruder_temp']:.1f}°C "
            f"(target {printer_state['extruder_target']:.1f}°C), "
            f"bed={printer_state['bed_temp']:.1f}°C "
            f"(target {printer_state['bed_target']:.1f}°C), "
            f"state={printer_state['print_state']}, "
            f"progress={printer_state['progress']*100:.1f}%. "
            f"Previous frame result: {self.last_defect}."
        )
        content, elapsed, tokens = self._call(ORCHESTRATOR_SYSTEM, user_msg, max_tokens=10)
        decision = content.strip().upper()
        if "EMERGENCY" in decision:
            decision = "EMERGENCY"
        elif "ANALYZE" in decision or (self.cycle_count % 3 == 0):
            decision = "ANALYZE"
        else:
            decision = "SKIP"
        return decision, elapsed, tokens

    # ── AGENT 2: THERMAL ANALYST ──────────────────────────────────────────────

    def run_thermal_analyst(self, base64_image: str, temp_stats: dict) -> tuple[dict, float, int]:
        """VLM analyzes the false-color thermal image for defects."""
        user_content = [
            {
                "type": "text",
                "text": (
                    f"This is a false-color INFERNO thermal image from a Topdon TC001 camera "
                    f"monitoring a PLA 3D print in progress on an Ender 3 printer running OctoPrint.\n\n"
                    f"Temperature stats: Min={temp_stats['min']:.1f}°C, "
                    f"Max={temp_stats['max']:.1f}°C, Avg={temp_stats['mean']:.1f}°C.\n\n"
                    f"Analyze this image for PLA printing defects. "
                    f"Output structured JSON only."
                )
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            }
        ]
        content, elapsed, tokens = self._call(
            THERMAL_ANALYST_SYSTEM, user_content,
            schema=THERMAL_ANALYSIS_SCHEMA, max_tokens=400
        )
        try:
            return json.loads(content), elapsed, tokens
        except json.JSONDecodeError:
            return {
                "anomaly_detected": False, "observations": "Parse error",
                "suspected_defect": "normal", "location_description": "N/A",
                "severity_estimate": "none", "confidence": 0.0
            }, elapsed, tokens

    # ── AGENT 3: TELEMETRY ANALYZER ───────────────────────────────────────────

    def run_telemetry_analyzer(self, printer_state: dict) -> tuple[dict, float, int]:
        """Parse printer telemetry for deviations from nominal PLA parameters."""
        user_msg = (
            f"Printer telemetry JSON:\n{json.dumps(printer_state, indent=2)}\n\n"
            f"Analyze for deviations from nominal PLA printing parameters. Output JSON only."
        )
        content, elapsed, tokens = self._call(
            TELEMETRY_ANALYZER_SYSTEM, user_msg,
            schema=TELEMETRY_ANALYSIS_SCHEMA, max_tokens=300
        )
        try:
            return json.loads(content), elapsed, tokens
        except json.JSONDecodeError:
            return {
                "nozzle_ok": True, "bed_ok": True, "fan_ok": True,
                "print_state_ok": True, "flags": [], "summary": "Parse error"
            }, elapsed, tokens

    # ── AGENT 4: DEFECT CLASSIFIER ────────────────────────────────────────────

    def run_defect_classifier(self, thermal_report: dict, telemetry_report: dict) -> tuple[dict, float, int]:
        """Cross-reference vision + telemetry → final defect classification."""
        user_msg = (
            f"Thermal Analyst Report:\n{json.dumps(thermal_report, indent=2)}\n\n"
            f"Telemetry Analyzer Report:\n{json.dumps(telemetry_report, indent=2)}\n\n"
            f"Cross-reference both signals and classify the defect. Output JSON only."
        )
        content, elapsed, tokens = self._call(
            DEFECT_CLASSIFIER_SYSTEM, user_msg,
            schema=DEFECT_CLASSIFICATION_SCHEMA, max_tokens=350
        )
        try:
            result = json.loads(content)
            # Realistic correction gating: don't trigger constant corrections for minor fluctuations early on
            defect = result.get("defect_type", "normal").lower()
            conf = float(result.get("confidence", 0.0))
            if defect == "normal" or conf < 0.65:
                result["requires_correction"] = False
                result["defect_type"] = "normal"
            self.last_defect = result.get("defect_type", "normal")
            return result, elapsed, tokens
        except json.JSONDecodeError:
            return {
                "defect_type": "normal", "severity": "none",
                "confidence": 0.0, "reasoning": "Parse error",
                "requires_correction": False
            }, elapsed, tokens

    # ── AGENT 5: CORRECTION AGENT ─────────────────────────────────────────────

    def run_correction_agent(self, classification: dict, printer_state: dict) -> tuple[dict, float, int]:
        """Determine exact G-code corrections to send to OctoPrint / Ender 3."""
        user_msg = (
            f"Defect Classification:\n{json.dumps(classification, indent=2)}\n\n"
            f"Current printer state: nozzle={printer_state['extruder_temp']:.1f}°C "
            f"(target {printer_state['extruder_target']:.1f}°C), "
            f"bed={printer_state['bed_temp']:.1f}°C "
            f"(target {printer_state['bed_target']:.1f}°C).\n\n"
            f"Select the precise G-code corrections. Output JSON only."
        )
        content, elapsed, tokens = self._call(
            CORRECTION_AGENT_SYSTEM, user_msg,
            schema=CORRECTION_SCHEMA, max_tokens=300
        )
        try:
            result = json.loads(content)
            self.total_corrections += 1
            return result, elapsed, tokens
        except json.JSONDecodeError:
            return {
                "gcode_commands": [], "explanation": "Parse error",
                "expected_effect": "N/A",
                "new_nozzle_target": printer_state["extruder_target"],
                "new_bed_target": printer_state["bed_target"]
            }, elapsed, tokens

    # ── FULL PIPELINE ──────────────────────────────────────────────────────────

    async def run_analysis_cycle(self, thermal_frame, printer_state_obj) -> AsyncGenerator:
        """
        Run the full 5-agent pipeline for one thermal frame.
        Yields status events (dicts) as each agent completes — for SSE streaming to UI.
        """
        self.cycle_count += 1
        cycle_id = self.cycle_count
        pipeline_start = time.perf_counter()

        temp_stats = {
            "min": thermal_frame.temp_min,
            "max": thermal_frame.temp_max,
            "mean": thermal_frame.temp_mean
        }
        printer_state = {
            "extruder_temp": printer_state_obj.extruder_temp,
            "extruder_target": printer_state_obj.extruder_target,
            "bed_temp": printer_state_obj.bed_temp,
            "bed_target": printer_state_obj.bed_target,
            "fan_speed": printer_state_obj.fan_speed,
            "print_state": printer_state_obj.print_state,
            "filename": printer_state_obj.filename,
            "progress": printer_state_obj.progress,
            "current_layer": printer_state_obj.current_layer,
            "total_layers": printer_state_obj.total_layers,
            "is_simulated": printer_state_obj.is_simulated,
        }

        # ── Agent 1: Orchestrator ──────────────────────────────────────────
        yield {"agent": "orchestrator", "status": "running", "cycle": cycle_id}
        decision, t1, tok1 = await asyncio.get_event_loop().run_in_executor(
            None, self.run_orchestrator, temp_stats, printer_state
        )
        yield {
            "agent": "orchestrator", "status": "done", "cycle": cycle_id,
            "result": decision, "elapsed": round(t1, 3), "tokens": tok1
        }

        if decision == "SKIP":
            yield {"agent": "pipeline", "status": "skipped", "cycle": cycle_id,
                   "message": "Frame nominal — no analysis needed."}
            return

        if decision == "EMERGENCY":
            yield {"agent": "pipeline", "status": "emergency", "cycle": cycle_id,
                   "message": "EMERGENCY detected by orchestrator!"}

        # ── Agents 2 + 3: Parallel (thermal analyst + telemetry) ──────────
        yield {"agent": "thermal_analyst", "status": "running", "cycle": cycle_id}
        yield {"agent": "telemetry_analyzer", "status": "running", "cycle": cycle_id}

        loop = asyncio.get_event_loop()
        thermal_task = loop.run_in_executor(
            None, self.run_thermal_analyst, thermal_frame.base64_png, temp_stats
        )
        telemetry_task = loop.run_in_executor(
            None, self.run_telemetry_analyzer, printer_state
        )
        (thermal_report, t2, tok2), (telemetry_report, t3, tok3) = await asyncio.gather(
            thermal_task, telemetry_task
        )

        yield {
            "agent": "thermal_analyst", "status": "done", "cycle": cycle_id,
            "result": thermal_report, "elapsed": round(t2, 3), "tokens": tok2
        }
        yield {
            "agent": "telemetry_analyzer", "status": "done", "cycle": cycle_id,
            "result": telemetry_report, "elapsed": round(t3, 3), "tokens": tok3
        }

        # ── Agent 4: Defect Classifier ─────────────────────────────────────
        yield {"agent": "defect_classifier", "status": "running", "cycle": cycle_id}
        classification, t4, tok4 = await loop.run_in_executor(
            None, self.run_defect_classifier, thermal_report, telemetry_report
        )
        yield {
            "agent": "defect_classifier", "status": "done", "cycle": cycle_id,
            "result": classification, "elapsed": round(t4, 3), "tokens": tok4
        }

        if not classification.get("requires_correction", False):
            total_t = time.perf_counter() - pipeline_start
            total_tokens = tok1 + tok2 + tok3 + tok4
            toks_per_sec = round(total_tokens / total_t) if total_t > 0 else 0
            yield {
                "agent": "pipeline", "status": "complete", "cycle": cycle_id,
                "defect": classification["defect_type"],
                "severity": classification["severity"],
                "total_elapsed": round(total_t, 3),
                "total_tokens": total_tokens,
                "tokens_per_second": toks_per_sec,
                "message": f"Cerebras Wafer-Scale Analysis complete in {round(total_t, 3)}s ({toks_per_sec} tok/s)"
            }
            return

        # ── Agent 5: Correction Agent ──────────────────────────────────────
        yield {"agent": "correction_agent", "status": "running", "cycle": cycle_id}
        correction, t5, tok5 = await loop.run_in_executor(
            None, self.run_correction_agent, classification, printer_state
        )
        yield {
            "agent": "correction_agent", "status": "done", "cycle": cycle_id,
            "result": correction, "elapsed": round(t5, 3), "tokens": tok5
        }

        total_t = time.perf_counter() - pipeline_start
        total_tokens = tok1 + tok2 + tok3 + tok4 + tok5
        toks_per_sec = round(total_tokens / total_t) if total_t > 0 else 0

        yield {
            "agent": "pipeline", "status": "correction_ready", "cycle": cycle_id,
            "defect": classification["defect_type"],
            "severity": classification["severity"],
            "confidence": classification["confidence"],
            "gcode_commands": correction["gcode_commands"],
            "explanation": correction["explanation"],
            "expected_effect": correction["expected_effect"],
            "total_elapsed": round(total_t, 3),
            "total_tokens": total_tokens,
            "tokens_per_second": toks_per_sec,
        }
