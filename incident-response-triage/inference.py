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
LLM_KEY = os.environ["API_KEY"]
LLM_BASE = os.environ["API_BASE_URL"].rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o")

ENV_URL = "https://KeyaChaudhary-incident-response-triage.hf.space"

TASKS = ["task1_easy", "task2_medium", "task3_hard"]

print(f"LLM_BASE={LLM_BASE}", flush=True)

client = OpenAI(
    api_key=LLM_KEY,
    base_url=LLM_BASE
)

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
def normalize_score(score: float) -> float:
    if score >= 1.0:
        return 0.999
    if score <= 0.0:
        return 0.001
    return score

def call_env(method: str, endpoint: str, payload: dict = None) -> dict:
    url = f"{ENV_URL.rstrip('/')}{endpoint}"
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

    print(
        f"[START] task={task_id} step=0 "
        f"alerts={len(obs.get('alerts', []))} "
        f"logs={len(obs.get('logs', []))} "
        f"metrics={len(obs.get('metrics', []))}",
        flush=True
    )

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

    print(
        f"[STEP] task={task_id} step=1 "
        f"latency_ms={latency_ms}",
        flush=True
    )

    # ── Submit action to env ───────────────────────────────────────────────────
    step_payload = {
        "task_id": task_id,
        **action_dict
    }

    print(action_dict, flush=True)
    step_result = call_env("POST", "/step", step_payload)
    reward = step_result["reward"]
    done   = step_result["done"]
    info   = step_result["info"]

    print(
        f"[END] task={task_id} "
        f"score={normalize_score(reward['total'])} "
        f"steps=1",
        flush=True
    )


    return {
        "task_id": task_id,
        "reward": normalize_score(reward["total"]),
        "breakdown": reward,
        "feedback": reward["feedback"],
    }


def main():
    print(
        f"[START] event=baseline_run_begin "
        f"model={MODEL_NAME} "
        f"tasks={len(TASKS)}",
        flush=True
    )

    results = []

    for task_id in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_id}", flush=True)
        print('='*60, flush=True)

        try:
            result = run_task(task_id)
            results.append(result)

        except Exception as e:
            print(
                f"[END] task={task_id} "
                f"score=0.001 "
                f"steps=0 "
                f"error={str(e)}",
                flush=True
            )

            results.append({
                "task_id": task_id,
                "reward": 0.001,
                "error": str(e)
            })

    avg_reward = sum(r["reward"] for r in results) / len(results)

    print(
        f"[END] event=baseline_run_complete "
        f"average_reward={round(avg_reward, 4)}",
        flush=True
    )


if __name__ == "__main__":
    main()
