#!/usr/bin/env python3
"""Run LAB-Bench CloningScenarios against a model via OpenRouter.

Multiple choice, so there is no LLM grader: the model picks a letter and the letter is
matched exactly. That removes the second API call per item (half the cost of the SimpleQA
harness this is adapted from) and the grader-disagreement failure mode with it.

Why this benchmark: 33 items, and it carries the highest per-cell mode disagreement in the
K=3 fit. Our 31 rows come from RAND RR-A3797-1, a published report that will never gain a
model, so every 2025-2026 test-taker is permanently missing from that column and can only
be filled by running the public set here.

Effort is a first-class flag because this repo treats effort variants as DISTINCT
test-takers (`claude-sonnet-5_max` is not `claude-sonnet-5`). `--effort` is passed through
to OpenRouter verbatim and never silently downgraded: a rejected value means that
test-taker cannot be reproduced here, which is a result, not something to paper over.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
DATA_CACHE = HERE / "data_cache"
DATA_JSON = DATA_CACHE / "cloning_scenarios.json"
DATA_URL = ("https://datasets-server.huggingface.co/rows"
            "?dataset=futurehouse%2Flab-bench&config=CloningScenarios&split=train"
            "&offset=0&length=100")
OUT_DIR = HERE / "out"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
N_ITEMS = 33
SCRIPT_VERSION = "run.py 3.2.0"
FIELDNAMES = ["id", "n_options", "correct_letter", "picked_letter", "grade_letter", "grade_str",
              "finish_reason", "raw_reply", "provider"]
GRADE_NAMES = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED", "ERROR": "ERROR"}
BACKOFFS = [1, 2, 4, 8]  # seconds, one retry per entry, exhausted -> raise
DEFAULT_TIMEOUT = 600     # per-call seconds; reasoning models on 6k-token prompts are slow
DEFAULT_MAX_TOKENS = 24000 # HARD output cap per call. Never leave this unset: thinking tokens
                          # bill as output and are not returned, so an uncapped high-effort
                          # model can burn 50k tokens per item and still emit empty content.
DEFAULT_TOKEN_BUDGET = 400_000  # abort the whole run past this many output tokens
TRUNC_ABORT = 3           # consecutive truncations that mean the config is wrong, not the item.
                          # On a thinking model a truncated reply is EMPTY (thinking comes first,
                          # the answer last), so each one is a full-price call returning nothing.
                          # Without this the run pays max_tokens x every remaining item for zero
                          # answers - the exact way one run burned 559k tokens and scored 0.

def load_dataset():
    """Fetch the 33-row public CloningScenarios set once, cache it, sanity-check its shape."""
    DATA_CACHE.mkdir(exist_ok=True)
    if not DATA_JSON.exists():
        resp = requests.get(DATA_URL, timeout=120)
        resp.raise_for_status()
        DATA_JSON.write_bytes(resp.content)
    doc = json.loads(DATA_JSON.read_text())
    rows = [r["row"] for r in doc["rows"]]
    if len(rows) != N_ITEMS:
        raise AssertionError(f"expected {N_ITEMS} rows in {DATA_JSON}, got {len(rows)}")
    for col in ("id", "question", "ideal", "distractors"):
        if any(col not in r for r in rows):
            raise AssertionError(f"dataset missing column: {col}")
    sha256 = hashlib.sha256(DATA_JSON.read_bytes()).hexdigest()
    return rows, sha256


def build_prompt(row):
    """Lettered MC prompt. Returns (prompt, correct_letter, n_options).

    Option order is shuffled per item, seeded on the item id, so the layout is identical on a
    re-run and a resumed run scores the same item the same way. The `canary` field is a
    training-contamination marker and is deliberately never sent to the model.

    No abstention option: RAND scored the 4-8 real options only, and adding an "insufficient
    information" choice measured a different task. It cost one model 10 of 33 items to
    declining rather than answering, and pushed its score 15pp below RAND's.
    """
    options = [row["ideal"]] + list(row["distractors"])
    random.Random(row["id"]).shuffle(options)
    letters = [chr(ord("A") + i) for i in range(len(options))]
    correct_letter = letters[options.index(row["ideal"])]
    block = "\n".join(f"{l}. {o}" for l, o in zip(letters, options))
    prompt = (f"{row['question']}\n\n{block}\n\n"
              "Reason as much as you need, then end your reply with a final line of exactly "
              "the form 'Answer: X' where X is the letter of the correct option.")
    return prompt, correct_letter, len(options)


def parse_letter(reply, n_options):
    """The option letter the reply CONCLUDES on, or None if it never states one.

    Two hard-won rules, both from real failures on this benchmark:

    1. An explicit "Answer: X" marker wins, and the LAST one wins. Anything else loses to the
       reasoning: a model that writes 900 tokens of analysis mentions half the options along
       the way, so the first letter it utters is not its answer.
    2. Never scan for a bare letter first. The English indefinite article "a" is a
       word-bounded single letter, so "...cloning into a given plasmid" reads as option A.
       That artifact scored one model A on 33/33 items and looked like a plausible 0.182.

    The fallback (last standalone letter) is only reached when the model ignored the required
    marker, and taking the last occurrence keeps a trailing "a" from beating a real answer in
    most replies. It is a fallback, not a protocol."""
    valid = {chr(ord("A") + i) for i in range(n_options)}
    up = reply.upper()
    marked = re.findall(r"(?:FINAL\s+)?ANSWER\s*(?:IS)?\s*[:\-]?\s*\(?\*{0,2}([A-Z])\*{0,2}\)?", up)
    for cand in reversed(marked):
        if cand in valid:
            return cand
    tail = [m.group(1) for m in re.finditer(r"\b([A-Z])\b", up) if m.group(1) in valid]
    return tail[-1] if tail else None


def store_reply(reply, head=120, tail=380):
    """What to persist in raw_reply. Keeps the TAIL, because parse_letter reads the conclusion
    and a head-only truncation makes a run un-auditable: the "Answer: X" marker of a 900-token
    reply falls outside a first-300-chars window, so a mis-parse cannot be diagnosed or
    re-scored without paying for the whole run again."""
    reply = reply or ""
    if len(reply) <= head + tail + 5:
        return reply
    return f"{reply[:head]} […] {reply[-tail:]}"


def zero_usage():
    return {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cost": 0.0}


def extract_usage(resp):
    """Read prompt/completion/cached token counts off an OpenRouter response's usage block.
    Cache-hit field names vary by upstream provider, so check the known spellings in order and
    treat a missing field as 0 rather than raising."""
    usage = resp.get("usage") or {}
    cached = usage.get("cached_tokens")
    if cached is None:
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None:
        cached = usage.get("cache_read_input_tokens")
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
        "completion_tokens": usage.get("completion_tokens", 0) or 0,
        "cached_tokens": cached or 0,
        # OpenRouter bills this call and reports what it charged. Reading it beats a local
        # price table: no per-model maintenance, and it is right for every model.
        "cost": float(usage.get("cost") or 0.0),
    }


def add_usage(totals, usage):
    for field in ("prompt_tokens", "completion_tokens", "cached_tokens", "cost"):
        totals[field] += usage.get(field, 0)


def print_usage(totals, model_id):
    print("--- token usage ---")
    print(f"model ({model_id}): prompt={totals['prompt_tokens']} "
          f"completion={totals['completion_tokens']} cached={totals['cached_tokens']}")
    print(f"spend this invocation: ${totals['cost']:.4f}   (as billed by OpenRouter)")


def call_api(model, user_content, provider, effort, api_key, timeout=DEFAULT_TIMEOUT,
             max_tokens=DEFAULT_MAX_TOKENS):
    """One chat/completions call. Retries 429/5xx with backoff 1,2,4,8s, then raises. Asks
    OpenRouter to report usage (including cache info) so cost is measured, not guessed.

    `effort` is forwarded verbatim. It is NOT normalised or downgraded: reproducing the
    `_max` / `_xhigh` test-takers depends on the upstream accepting that exact level, and a
    silent downgrade would file the result against the wrong test-taker.

    `timeout` is generous by default: a reasoning model on a 6k-token plasmid prompt can think
    for minutes, and a timeout that fires mid-thought lands as an ERROR row that looks like a
    model failure rather than a harness limit."""
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"model": model, "messages": [{"role": "user", "content": user_content}],
               "max_tokens": max_tokens, "usage": {"include": True}}
    if provider:
        payload["provider"] = {"order": [provider], "allow_fallbacks": False}
    if effort:
        payload["reasoning"] = {"effort": effort}
    for delay in BACKOFFS + [None]:
        # stream=True + an explicit clock is the only way to get a WALL-CLOCK cap here.
        # requests' `timeout` is a per-read timeout: it resets on every byte received, so a
        # gateway that sends headers early and trickles keep-alive bytes while the model
        # generates never trips it. A --timeout that cannot time out is worse than none.
        started = time.monotonic()
        resp = requests.post(API_URL, headers=headers, json=payload,
                             timeout=timeout, stream=True)
        if resp.status_code == 429 or resp.status_code >= 500:
            resp.close()
            if delay is None:
                resp.raise_for_status()
            time.sleep(delay)
            continue
        resp.raise_for_status()
        body = bytearray()
        for chunk in resp.iter_content(8192):
            body.extend(chunk)
            if time.monotonic() - started > timeout:
                resp.close()
                raise requests.exceptions.Timeout(
                    f"wall-clock timeout after {timeout}s ({len(body)} bytes received)")
        resp.close()
        return json.loads(body)


def run_item(row, model, provider, effort, api_key, timeout=DEFAULT_TIMEOUT,
             max_tokens=DEFAULT_MAX_TOKENS):
    """One completion for a dataset row. Never raises: any failure (network, HTTP, missing
    field) lands as an ERROR row so the batch keeps going.

    Grades: A correct, B wrong option, C answered without naming an option, ERROR
    harness/network failure. An unparseable reply is C rather than B because the model did not
    choose a wrong answer, it failed to answer, and scoring it wrong would flatter the task."""
    prompt, correct_letter, n_options = build_prompt(row)
    base = {"id": row["id"], "n_options": n_options, "correct_letter": correct_letter}
    try:
        completion = call_api(model, prompt, provider, effort, api_key, timeout, max_tokens)
        choice = completion["choices"][0]
        reply = choice["message"]["content"] or ""
        finish = choice.get("finish_reason") or choice.get("native_finish_reason") or ""
        usage = extract_usage(completion)
        picked = parse_letter(reply, n_options)
        # Truncated before the answer: the model never got to answer, so this is a harness
        # config failure (max_tokens too small for the chosen effort), not an abstention.
        # Grading it C would bury the cause and count it against the model's score.
        if picked is None and (finish == "length" or (not reply.strip() and
                                                     usage["completion_tokens"] > 0)):
            return {**base, "picked_letter": "", "grade_letter": "ERROR",
                    "grade_str": (f"ERROR: truncated before answering "
                                  f"(finish={finish or 'none'}, "
                                  f"{usage['completion_tokens']} completion tokens, "
                                  f"empty visible content) - raise --max-tokens or lower --effort"),
                    "finish_reason": finish, "raw_reply": store_reply(reply),
                    "provider": completion.get("provider", "unknown"), "usage": usage}
        if picked is None:
            letter = "C"          # answered, but never stated an option
        elif picked == correct_letter:
            letter = "A"
        else:
            letter = "B"
        return {**base, "picked_letter": picked or "", "grade_letter": letter,
                "grade_str": GRADE_NAMES[letter], "finish_reason": finish,
                "raw_reply": store_reply(reply),
                "provider": completion.get("provider", "unknown"), "usage": usage}
    except Exception as exc:
        return {**base, "picked_letter": "", "grade_letter": "ERROR",
                "grade_str": f"ERROR: {exc}"[:300], "finish_reason": "", "raw_reply": "",
                "provider": "unknown", "usage": zero_usage()}


def read_existing(path):
    """Rows already scored on disk, for append-and-resume."""
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plan_todo(rows, out_path):
    """(kept_rows, todo_rows) for an append-and-resume run.

    ERROR rows are NOT "done": print_metrics and the README both tell the user to re-run to
    fill them, so counting them as finished makes that instruction a lie and strands a run one
    item short forever. They are dropped from the file here and returned in `todo`, which also
    stops the retry from writing a duplicate id.
    """
    prior = read_existing(out_path)
    kept = [r for r in prior if r.get("grade_letter") != "ERROR"]
    n_err = len(prior) - len(kept)
    if n_err:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for r in kept:
                writer.writerow({k: r.get(k, "") for k in FIELDNAMES})
        print(f"dropped {n_err} ERROR row(s) so this run retries them")
    done_ids = {r["id"] for r in kept}
    return kept, [r for r in rows if r["id"] not in done_ids]


def compute_metrics(rows):
    """rows: dicts with a grade_letter (A/B/C/ERROR), from a fresh run or read back off disk.
    Documented choice: ERROR counts toward n_total (it is a real row, a real failed attempt,
    and hiding it would inflate the score) but never toward attempted/correct/incorrect."""
    n_total = len(rows)
    correct = sum(1 for r in rows if r["grade_letter"] == "A")
    incorrect = sum(1 for r in rows if r["grade_letter"] == "B")
    not_attempted = sum(1 for r in rows if r["grade_letter"] == "C")
    errors = sum(1 for r in rows if r["grade_letter"] == "ERROR")
    attempted = correct + incorrect
    correct_rate = correct / n_total if n_total else 0.0
    accuracy_given_attempted = correct / attempted if attempted else 0.0
    stderr = math.sqrt(correct_rate * (1 - correct_rate) / n_total) if n_total else 0.0
    n_graded = n_total - errors
    return {"n_total": n_total, "correct": correct, "incorrect": incorrect,
            "not_attempted": not_attempted, "errors": errors, "correct_rate": correct_rate,
            "accuracy_given_attempted": accuracy_given_attempted, "stderr": stderr,
            "n_graded": n_graded,
            "correct_rate_graded": correct / n_graded if n_graded else 0.0}


def print_metrics(m):
    print(f"correct_rate: {m['correct_rate']:.4f} (+- {m['stderr']:.4f})   <- the column's metric")
    print(f"accuracy_given_attempted: {m['accuracy_given_attempted']:.4f}")
    print(f"A/B/C/ERROR counts: {m['correct']}/{m['incorrect']}/{m['not_attempted']}/{m['errors']}"
          f"  (n_total={m['n_total']})")
    print("chance floor 0.226 (4-8 options, mean ~4.6); RAND's 31 rows span 0.121-0.636")
    if m["errors"]:
        print(f"correct_rate over graded rows only: {m['correct_rate_graded']:.4f} "
              f"(n_graded={m['n_graded']})")
        print("WARNING: errors present. Re-run the same command to fill them (resume skips "
              "finished items). Do not compare against the RAND rows until errors == 0.")
    if m["n_total"] != N_ITEMS:
        print(f"WARNING: n_total != {N_ITEMS}, this number is NOT comparable to the RAND rows.")


def main_run(args):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("set OPENROUTER_API_KEY")
    rows, sha256 = load_dataset()
    if args.limit:
        rows = rows[:args.limit]

    OUT_DIR.mkdir(exist_ok=True)
    stem = args.out_name or (args.model.replace("/", "_") +
                             (f"__{args.effort}" if args.effort else ""))
    out_path = OUT_DIR / f"{stem}.csv"
    done, todo = plan_todo(rows, out_path)
    print(f"{len(done)} already scored, {len(todo)} to run -> {out_path}")

    totals = zero_usage()
    fresh = []
    if todo:
        write_header = not out_path.exists() or out_path.stat().st_size == 0
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            consecutive_trunc, written = 0, set()

            def bank(res):
                """Write one result and count its usage. Every row that reaches here has been
                billed, so it must be persisted even during an abort."""
                add_usage(totals, res.pop("usage"))
                writer.writerow(res)
                f.flush()
                fresh.append(res)
                written.add(res["id"])

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(run_item, r, args.model, args.provider,
                                       args.effort, api_key, args.timeout,
                                       args.max_tokens) for r in todo]
                for done_n, fut in enumerate(as_completed(futures), 1):
                    res = fut.result()
                    bank(res)
                    print(f"  [{done_n}/{len(todo)}] {res['grade_letter']} "
                          f"{res['id'][:8]}  out_tok={totals['completion_tokens']} "
                          f"${totals['cost']:.3f}", flush=True)

                    consecutive_trunc = (consecutive_trunc + 1
                                         if "truncated before answering" in res["grade_str"]
                                         else 0)
                    if consecutive_trunc >= TRUNC_ABORT:
                        abort = (f"{consecutive_trunc} consecutive truncations at --max-tokens "
                                 f"{args.max_tokens}; each was a full-price call that returned "
                                 f"nothing. Lower --effort or raise --max-tokens.")
                    elif args.token_budget and totals["completion_tokens"] > args.token_budget:
                        abort = (f"{totals['completion_tokens']} output tokens exceeds "
                                 f"--token-budget {args.token_budget}.")
                    else:
                        continue

                    # cancel() only stops QUEUED work; threads already running are billed no
                    # matter what, so drain and bank them rather than paying for discarded rows.
                    cancelled = sum(1 for pend in futures if pend.cancel())
                    print(f"\nABORTED: {abort}\n  cancelled {cancelled} queued item(s); "
                          f"draining {len(futures) - cancelled - len(written)} already in "
                          f"flight so their paid-for rows are kept.", flush=True)
                    for pend in futures:
                        if pend.cancelled():
                            continue
                        try:
                            extra = pend.result()
                        except Exception:
                            continue
                        if extra["id"] not in written:
                            bank(extra)
                    print(f"  saved {len(written)} row(s), ${totals['cost']:.4f} spent. "
                          f"Re-run to continue; ERROR rows retry.", flush=True)
                    break

    all_rows = done + fresh
    metrics = compute_metrics(all_rows)
    print_metrics(metrics)
    print_usage(totals, args.model)
    meta = {"script_version": SCRIPT_VERSION, "model": args.model, "effort": args.effort,
            "provider_pin": args.provider, "dataset_sha256": sha256,
            "benchmark": "LAB-Bench Cloning", "n_items": N_ITEMS,
            "max_tokens": args.max_tokens, "token_budget": args.token_budget,
            "run_at": datetime.now(timezone.utc).isoformat(), "metrics": metrics,
            "usage_this_invocation": totals,
            "cost_usd": totals["cost"]}
    (OUT_DIR / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2))


def selftest():
    """Offline checks on the two pieces of real logic: prompt construction and reply parsing.
    No network, no API key. This is the check that fails if the grading logic breaks."""
    row = {"id": "fixed-seed-id", "question": "Q?", "ideal": "RIGHT",
           "distractors": ["W1", "W2", "W3"]}
    p1, correct1, n1 = build_prompt(row)
    p2, correct2, n2 = build_prompt(row)
    assert (p1, correct1, n1) == (p2, correct2, n2), "prompt must be deterministic per id"
    assert n1 == 4, f"4 real options, no abstention = 4, got {n1}"
    assert f"{correct1}. RIGHT" in p1, "correct_letter must point at the ideal answer"
    assert "Insufficient information" not in p1, "no abstention option may be offered"
    assert "canary" not in p1.lower(), "the contamination canary must never reach the model"
    assert p1.rstrip().endswith("correct option."), "the answer-format instruction must be last"
    assert "Answer: X" in p1, "the prompt must demand the marker parse_letter reads"

    assert parse_letter("B", 5) == "B"
    assert parse_letter("Answer: **C**", 5) == "C"
    assert parse_letter("The answer is d", 5) == "D"
    assert parse_letter("F", 5) is None, "letter outside the option range must not parse"
    assert parse_letter("", 5) is None
    # The bug that scored one model A on 33/33: prose hits the article "a" within a few words.
    assert parse_letter("## Step 1\nThe task involves cloning into a given plasmid.\n"
                        "Answer: D", 5) == "D", "the article 'a' must not beat the marker"
    assert parse_letter("Option A looks wrong, B is closer. Answer: C", 5) == "C", \
        "mid-reasoning mentions must not beat the marker"
    assert parse_letter("Answer: B\nWait, reconsidering. Answer: E", 5) == "E", \
        "the LAST marker wins"
    assert parse_letter("I cannot determine this", 5) is None, "'I' is outside A-E"
    assert parse_letter("after review the best choice is D", 5) == "D", \
        "fallback takes the last standalone letter when the marker is missing"

    long = "x" * 2000 + "\nAnswer: D"
    assert store_reply(long).endswith("Answer: D"), \
        "the stored reply must retain the marker parse_letter reads"
    assert parse_letter(store_reply(long), 5) == "D", "stored reply must re-score"
    assert store_reply("Answer: B") == "Answer: B", "short replies stored verbatim"

    import tempfile
    with tempfile.TemporaryDirectory() as _d:
        _p = Path(_d) / "r.csv"
        with open(_p, "w", newline="") as _f:
            _w = csv.DictWriter(_f, fieldnames=FIELDNAMES)
            _w.writeheader()
            _w.writerow({**{k: "" for k in FIELDNAMES}, "id": "ok", "grade_letter": "A"})
            _w.writerow({**{k: "" for k in FIELDNAMES}, "id": "bad", "grade_letter": "ERROR"})
        _kept, _todo = plan_todo([{"id": "ok"}, {"id": "bad"}, {"id": "new"}], _p)
        _ids = {r["id"] for r in _todo}
        assert "ok" not in _ids, "a completed item must not be re-run"
        assert "bad" in _ids, "an ERROR row must be retried, as print_metrics promises"
        assert "new" in _ids, "an unseen item must be run"
        assert {r["id"] for r in _kept} == {"ok"}, "ERROR rows must be dropped from the file"
        assert {r["id"] for r in read_existing(_p)} == {"ok"}, "file must be rewritten"

    m = compute_metrics([{"grade_letter": g} for g in ["A", "A", "B", "C", "ERROR"]])
    assert m["n_total"] == 5 and m["correct"] == 2 and m["errors"] == 1
    assert abs(m["correct_rate"] - 0.4) < 1e-9, "ERROR must stay in the n_total denominator"
    assert abs(m["accuracy_given_attempted"] - 2 / 3) < 1e-9, "ERROR must not count as attempted"
    print("selftest ok")


def main():
    p = argparse.ArgumentParser(
        description="Run LAB-Bench CloningScenarios (33 items, MC) against a model via OpenRouter.")
    p.add_argument("--model", help="OpenRouter model id, e.g. anthropic/claude-sonnet-4.6")
    p.add_argument("--effort", default=None,
                   help="reasoning effort forwarded verbatim to OpenRouter (low|medium|high|...). "
                        "Required to reproduce an effort-suffixed test-taker; never downgraded.")
    p.add_argument("--limit", type=int, default=None, help="only run the first N items")
    p.add_argument("--provider", default=None, help="pin the upstream OpenRouter provider")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help=f"hard output cap per call (default {DEFAULT_MAX_TOKENS}); at high "
                        "effort thinking eats this budget and an answer never appears")
    p.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET,
                   help=f"abort the run past this many output tokens (default "
                        f"{DEFAULT_TOKEN_BUDGET}); 0 disables")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help=f"per-call seconds (default {DEFAULT_TIMEOUT}); raise for slow "
                        "reasoning models rather than accepting ERROR rows")
    p.add_argument("--out-name", default=None, help="override the out/<name>.csv basename")
    p.add_argument("--selftest", action="store_true", help="offline logic checks, no API calls")
    args = p.parse_args()
    if args.selftest:
        selftest()
        return
    if not args.model:
        p.error("--model is required (or use --selftest)")
    main_run(args)


if __name__ == "__main__":
    main()
