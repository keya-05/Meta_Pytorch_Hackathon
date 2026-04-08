#!/usr/bin/env python3
"""
Baseline inference script for Incident Response Triage OpenEnv.

STDOUT FORMAT (mandatory):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import json
import textwrap
from typing import List, Optional

import httpx
from openai import OpenAI

# ── Mandatory env vars ────────────────────────────────────────────────────────
LLM_PROXY_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
API_KEY = os.environ["API_KEY"]

SPACE_URL = "https://KeyaChaudhary-incident-response-triage.hf.space"

BENCHMARK = "incident-response-triage"
TASKS = ["task1_easy", "task2_medium", "task3_hard"]
SUCCESS_SCORE_THRESHOLD = 0.5


client = OpenAI(
    api_key=API_KEY,
    base_url=LLM_PROXY_URL
)

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert Site Reliability Engineer (SRE) performing incident triage.
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
""").strip()


# ── Mandatory log helpers ─────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_oneline = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_oneline} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Env helpers ───────────────────────────────────────────────────────────────

def call_env(method: str, endpoint: str, payload: dict = None) -> dict:
    url = f"{SPACE_URL}{endpoint}"
    with httpx.Client(timeout=30) as http:
        if method == "POST":
            r = http.post(url, json=payload or {})
        else:
            r = http.get(url)
        r.raise_for_status()
        return r.json()


def get_llm_action(obs: dict) -> dict:
    user_prompt = textwrap.dedent(f"""
        TASK: {obs['task_id']}
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
    """).strip()

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=800,
    )
    raw = (response.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ── Per-task runner ───────────────────────────────────────────────────────────

def run_task(task_id: str) -> dict:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env="incident-response-triage", model=MODEL_NAME)

    action_str = "{}"
    error_msg = None
    reward_val = 0.0
    done = True

    try:
        obs = call_env("POST", "/reset", {"task_id": task_id})

        try:
            action_dict = get_llm_action(obs)
            action_str = json.dumps(action_dict)
            step_result = call_env("POST", "/step", action_dict)
            reward_val = step_result["reward"]["total"]
            done = step_result["done"]
        except Exception as e:
            error_msg = str(e)
            done = True

        steps_taken = 1
        rewards.append(reward_val)
        score = reward_val
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        error_msg = str(e)
        rewards = [0.0]
        score = 0.0
        success = False

    log_step(
        step=1,
        action=action_str,
        reward=reward_val,
        done=done,
        error=error_msg
    )
    log_end(
        success=score >= 0.5,
        steps=1,
        score=score,
        rewards=[reward_val]
    )

    return {"task_id": task_id, "score": score, "success": success}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results = []
    for task_id in TASKS:
        result = run_task(task_id)
        results.append(result)
        print(flush=True)

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"# average_score={avg_score:.3f}", flush=True)


if __name__ == "__main__":
    main()