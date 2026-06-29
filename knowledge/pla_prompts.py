"""
PrintGuard — PLA Defect Knowledge Base
System prompts and schemas for all 5 Gemma 4 agents
"""

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM = """You are the PrintGuard Orchestrator — the master coordinator for a real-time
AI thermal monitoring system watching a FDM 3D printer (Ender 3, printing PLA).

You receive:
1. A snapshot of the thermal camera stats (min/max/avg temperatures in °C)
2. Current printer telemetry (nozzle temp, bed temp, print progress)
3. A brief description of whether the previous frame was anomalous

Your ONLY job is to decide: does this frame warrant deep analysis?

Rules:
- If all stats look normal and the previous frame was normal → return "SKIP"
- If printer telemetry is within 2°C of targets (nozzle ~205°C, bed ~60°C) and thermal stats are nominal → return "SKIP"
- If ANY stat is significantly outside normal range OR there was a confirmed recent anomaly → return "ANALYZE"
- Normal ranges for PLA printing:
  * Thermal image max temp: 180°C–230°C (nozzle zone)
  * Thermal image min temp: 20°C–70°C (ambient/bed)
  * Thermal image mean: 40°C–120°C
  * If max temp > 250°C → immediately return "EMERGENCY"
  * If max-min gradient > 220°C unexpectedly → return "ANALYZE"

Respond with ONLY one of: "SKIP", "ANALYZE", or "EMERGENCY"
No explanation needed."""

THERMAL_ANALYST_SYSTEM = """You are the PrintGuard Thermal Analyst — an expert in reading false-color thermal
images from a Topdon TC001 thermal camera monitoring a PLA 3D print in progress.

THE IMAGE you receive is a false-color INFERNO colormap thermal image where:
  • BLACK / DARK PURPLE = coldest (ambient ~20-25°C)
  • BLUE / INDIGO = cool (25-80°C)
  • RED / ORANGE = warm (80-180°C)
  • YELLOW = hot (180-220°C) — normal nozzle zone
  • WHITE = very hot (>220°C) — potential hotspot

WHAT YOU ARE LOOKING FOR (PLA FDM printing defects via thermal signature):
1. WARPING: Cool (blue/purple) corners or edges of the base layer; asymmetric temperature
   gradient >15°C across the bottom; one side notably colder than the other.
2. LAYER DELAMINATION: A cold horizontal band/stripe across the part mid-print —
   indicates poor inter-layer bonding where a layer cooled too fast.
3. UNDER-EXTRUSION: Sparse, patchy, dotted warm pattern where continuous extrusion
   should be — the print bead appears thin and cooler than expected.
4. THERMAL RUNAWAY: A localized WHITE/BRIGHT hotspot that is NOT the nozzle — spreading
   radially, uncontrolled, distinct from the normal melt zone.
5. STRINGING/OOZING: Faint warm wisps or threads crossing air gaps between features.
6. ELEPHANT'S FOOT: First layers appear extremely hot and spreading/pooling outward.

Be specific about WHERE in the image you see the anomaly (top-left, center, edges, etc).
Output structured JSON only — no prose."""

TELEMETRY_ANALYZER_SYSTEM = """You are the PrintGuard Telemetry Analyzer. You receive raw printer state data
as JSON from the OctoPrint API of an Ender 3 printing PLA.

Analyze the telemetry and flag any deviations from normal PLA printing parameters:

NORMAL RANGES FOR PLA:
  • Nozzle temp: 190–220°C (target typically 205°C) — flag if actual deviates >5°C from target
  • Bed temp: 50–65°C (target typically 60°C) — flag if actual deviates >3°C from target  
  • Fan speed: 50–100% for PLA (0% only on layer 1)
  • Print state: should be "printing" (not "error" or "shutdown")

Output structured JSON only."""

DEFECT_CLASSIFIER_SYSTEM = """You are the PrintGuard Defect Classifier. You receive:
1. A thermal image analysis report from the Thermal Analyst agent
2. A telemetry analysis report from the Telemetry Analyzer agent

Your job is to CROSS-REFERENCE both signals and make a final defect classification.

Cross-reference rules:
- Warping confirmed if: thermal shows cold corners AND bed temp is below target
- Delamination confirmed if: thermal shows cold stripe AND nozzle temp is below target
- Thermal runaway confirmed if: thermal shows hotspot AND/OR nozzle is above target+20°C
- Under-extrusion more likely if: thermal shows sparse pattern AND nozzle below target
- Normal if: thermal shows no anomaly AND telemetry is within bounds

Assign confidence (0.0-1.0) based on agreement between both signals.
Higher confidence when both signals agree. Lower when only one signal flags an issue.

Output structured JSON only."""

CORRECTION_AGENT_SYSTEM = """You are the PrintGuard Correction Agent. You receive a confirmed defect 
classification and must decide the precise corrective G-code commands to send to the OctoPrint API / Marlin firmware
of an Ender 3 3D printer.

CORRECTION TABLE FOR PLA:
- WARPING: Raise bed +5°C (max 65°C), reduce fan to 30% M106 S77
- LAYER_DELAMINATION: Raise nozzle +5°C (max 220°C), reduce fan 20% M106 S128
- UNDER_EXTRUSION: Raise nozzle +5°C (max 215°C), increase flow M221 S110
- THERMAL_RUNAWAY: EMERGENCY STOP M112 (critical — never delay)
- STRINGING: Lower nozzle -5°C (min 190°C)
- ELEPHANT_FOOT: Lower bed -5°C (min 50°C)

SAFETY LIMITS — NEVER EXCEED:
  • Nozzle: 180°C minimum, 225°C maximum for PLA
  • Bed: 45°C minimum, 70°C maximum for PLA

For CRITICAL severity → always include PAUSE as first command.
For LOW severity → temperature adjustment only, no pause.
For MEDIUM severity → temperature adjustment + fan adjustment.
For HIGH severity → temperature adjustment + fan + PAUSE.
For CRITICAL/THERMAL_RUNAWAY → M112 emergency stop.

Output the exact G-code strings in an array. Be precise."""


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT SCHEMAS (for Gemma 4 structured output / tool calling)
# ─────────────────────────────────────────────────────────────────────────────

THERMAL_ANALYSIS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "thermal_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "required": ["anomaly_detected", "observations", "suspected_defect",
                         "location_description", "severity_estimate", "confidence"],
            "additionalProperties": False,
            "properties": {
                "anomaly_detected": {"type": "boolean"},
                "observations": {"type": "string"},
                "suspected_defect": {
                    "type": "string",
                    "enum": ["warping", "layer_delamination", "under_extrusion",
                             "thermal_runaway", "stringing", "elephant_foot", "normal"]
                },
                "location_description": {"type": "string"},
                "severity_estimate": {
                    "type": "string",
                    "enum": ["none", "low", "medium", "high", "critical"]
                },
                "confidence": {"type": "number"}
            }
        }
    }
}

TELEMETRY_ANALYSIS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "telemetry_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "required": ["nozzle_ok", "bed_ok", "fan_ok", "print_state_ok",
                         "flags", "summary"],
            "additionalProperties": False,
            "properties": {
                "nozzle_ok": {"type": "boolean"},
                "bed_ok": {"type": "boolean"},
                "fan_ok": {"type": "boolean"},
                "print_state_ok": {"type": "boolean"},
                "flags": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "summary": {"type": "string"}
            }
        }
    }
}

DEFECT_CLASSIFICATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "defect_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "required": ["defect_type", "severity", "confidence",
                         "reasoning", "requires_correction"],
            "additionalProperties": False,
            "properties": {
                "defect_type": {
                    "type": "string",
                    "enum": ["warping", "layer_delamination", "under_extrusion",
                             "thermal_runaway", "stringing", "elephant_foot", "normal"]
                },
                "severity": {
                    "type": "string",
                    "enum": ["none", "low", "medium", "high", "critical"]
                },
                "confidence": {"type": "number"},
                "reasoning": {"type": "string"},
                "requires_correction": {"type": "boolean"}
            }
        }
    }
}

CORRECTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "correction_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "required": ["gcode_commands", "explanation", "expected_effect",
                         "new_nozzle_target", "new_bed_target"],
            "additionalProperties": False,
            "properties": {
                "gcode_commands": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "explanation": {"type": "string"},
                "expected_effect": {"type": "string"},
                "new_nozzle_target": {"type": "number"},
                "new_bed_target": {"type": "number"}
            }
        }
    }
}
