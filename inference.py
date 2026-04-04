#!/usr/bin/env python3
"""
Baseline inference script for Incident Response Triage OpenEnv.
Runs an LLM agent against all 3 tasks using the OpenAI client.
Emits structured [START] / [STEP] / [END] logs for evaluation scoring.
"""

import os
import json
import time
import httpx
from openai import OpenAI

# ── Config from environment variables ─────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", HF_TOKEN)

TASKS = ["task1_easy", "task2_medium", "task3_hard"]

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) performing incident triage.
You will be given alerts, logs, and metrics from a production system.

Your job is to respond with a JSON object containing EXACTLY these fields:
{
  "root_cause": "<string — the identified root cause service or mechanism>",
  "silenced_alert_ids": ["<list of alert IDs that are noise/unrelated>"],
  "escalate_to": "<team name to escalate to, or null if resolving autonomously>",
  "incident_summary": "<2-4 sentence concise summary of the incident>",
  "severity_assessment": "<one of: low, medium, high, critical>",
  "proposed_fix": "<short proposed remediation, or null>"
}

Rules:
- Identify the ROOT UPSTREAM cause, not downstream symptoms
- Only silence alerts that are genuinely unrelated to the incident
- Be precise about the failure mechanism (not just which service)
- Your summary should mention: what failed, why, and business impact
- Return ONLY valid JSON — no markdown, no explanation outside the JSON
"""


def call_env(method: str, endpoint: str, payload: dict = None) -> dict:
    url = f"{API_BASE_URL}{endpoint}"
    with httpx.Client(timeout=30) as http:
        if method == "POST":
            r = http.post(url, json=payload or {})
        else:
            r = http.get(url)
        r.raise_for_status()
        return r.json()


def run_task(task_id: str) -> dict:
    # ── Reset ──────────────────────────────────────────────────────────────────
    obs = call_env("POST", "/reset", {"task_id": task_id})

    print(json.dumps({
        "type": "[START]",
        "task_id": task_id,
        "step": 0,
        "observation_alerts": len(obs.get("alerts", [])),
        "observation_logs": len(obs.get("logs", [])),
        "observation_metrics": len(obs.get("metrics", [])),
        "prompt_preview": obs.get("prompt", "")[:120],
    }))

    # ── Build user message for LLM ─────────────────────────────────────────────
    user_message = f"""TASK: {task_id}

PROMPT: {obs['prompt']}

ALERTS:
{json.dumps(obs['alerts'], indent=2)}

LOGS:
{json.dumps(obs['logs'], indent=2)}

METRICS:
{json.dumps(obs['metrics'], indent=2)}

CONTEXT:
{json.dumps(obs.get('context', {}), indent=2)}

Respond with the JSON action object now.
"""

    # ── Call LLM ──────────────────────────────────────────────────────────────
    t0 = time.time()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.0,
        max_tokens=800,
    )
    latency_ms = int((time.time() - t0) * 1000)

    raw_action = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw_action.startswith("```"):
        raw_action = raw_action.split("```")[1]
        if raw_action.startswith("json"):
            raw_action = raw_action[4:]
    raw_action = raw_action.strip()

    action_dict = json.loads(raw_action)

    print(json.dumps({
        "type": "[STEP]",
        "task_id": task_id,
        "step": 1,
        "action": action_dict,
        "llm_latency_ms": latency_ms,
        "model": MODEL_NAME,
    }))

    # ── Submit action to env ───────────────────────────────────────────────────
    step_result = call_env("POST", "/step", action_dict)
    reward = step_result["reward"]
    done   = step_result["done"]
    info   = step_result["info"]

    print(json.dumps({
        "type": "[END]",
        "task_id": task_id,
        "step": 1,
        "done": done,
        "reward": reward["total"],
        "reward_breakdown": {
            "root_cause_score":     reward["root_cause_score"],
            "noise_filtering_score": reward["noise_filtering_score"],
            "escalation_score":     reward["escalation_score"],
            "summary_score":        reward["summary_score"],
            "penalty":              reward["penalty"],
        },
        "feedback": reward["feedback"],
    }))

    return {
        "task_id": task_id,
        "reward": reward["total"],
        "breakdown": reward,
        "feedback": reward["feedback"],
    }


def main():
    print(json.dumps({
        "type": "[START]",
        "event": "baseline_run_begin",
        "model": MODEL_NAME,
        "api_base": API_BASE_URL,
        "tasks": TASKS,
    }))

    results = []
    for task_id in TASKS:
        print(f"\n{'='*60}")
        print(f"Running task: {task_id}")
        print('='*60)
        try:
            result = run_task(task_id)
            results.append(result)
        except Exception as e:
            print(json.dumps({
                "type": "[END]",
                "task_id": task_id,
                "error": str(e),
                "reward": 0.0,
            }))
            results.append({"task_id": task_id, "reward": 0.0, "error": str(e)})

    # ── Final summary ─────────────────────────────────────────────────────────
    avg_reward = sum(r["reward"] for r in results) / len(results)
    print(json.dumps({
        "type": "[END]",
        "event": "baseline_run_complete",
        "model": MODEL_NAME,
        "results": [{"task_id": r["task_id"], "reward": r["reward"]} for r in results],
        "average_reward": round(avg_reward, 4),
    }))


if __name__ == "__main__":
    main()
