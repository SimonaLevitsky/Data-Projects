"""
Benchmarking Part B — automatic evaluation against golden_dataset.json
=======================================================================
Runs every question of the golden dataset through the REAL app (app.py — same routing, enforcement
and clean-up the users get), scores the answers and produces a BIRD-SQL-style report.

Metrics
  Execution Accuracy (EX)  database + hybrid: the agent's SQL and the gold SQL are both executed on
                           credit_risk.db and their result sets compared (order-insensitive unless the
                           gold query has ORDER BY, numbers rounded to 2 decimals, gold columns must be a
                           subset of the predicted columns). Reported overall and by difficulty.
  Answer Accuracy          database + hybrid: the gold values also appear in the natural-language answer.
  Routing Accuracy         tool_used == expected_tool (all questions except the ambiguous one).
  Calculator Accuracy      every expected value appears in the answer and no SQL was run.
  Policy Accuracy          policy_sources ⊇ expected sections, ∩ excluded sections = ∅, keywords present.
  Clarification Rate       ambiguous questions answered with a clarification (confidence < 70, no SQL).
  Rejection Rate           poison questions rejected (tool none, confidence 0, no SQL, DB unchanged).
  False Rejection Rate     core questions wrongly rejected.
  Overall pass rate        questions where EVERY applicable check passed.

Usage
  python benchmark.py                      run all 18 questions (needs an OpenAI key: env var or secrets.toml)
  python benchmark.py --ids Q01,Q13,P03    run a subset
  python benchmark.py --rescore benchmark_results.json   re-score saved raw outputs, no model calls
Outputs: benchmark_report.md and benchmark_results.json next to this file.

The same scoring code powers the "Run benchmark" button in the app's sidebar (useful on Streamlit
Cloud, where the API key lives).
"""

import os
import re
import sys
import json
import time
import sqlite3
import argparse
from datetime import datetime
from itertools import permutations

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "golden_dataset.json")
DB_PATH = os.path.join(BASE_DIR, "credit_risk.db")
REPORT_MD = os.path.join(BASE_DIR, "benchmark_report.md")
RESULTS_JSON = os.path.join(BASE_DIR, "benchmark_results.json")

CLARIFICATION_THRESHOLD = 70   # same as CONFIDENCE_THRESHOLD in app.py
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
SECTION_RE = re.compile(r"(?:§|section|sections|סעיף|סעיפים)\s*(\d+\.\d+)", re.IGNORECASE)


# ─────────────────────────────────────────────
# Dataset / DB helpers
# ─────────────────────────────────────────────

def load_dataset(path: str = DATASET_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def open_db_readonly():
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def db_counts() -> dict:
    with open_db_readonly() as con:
        return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("loans", "demographics")}


def first_statement(sql: str) -> str:
    parts = [p.strip() for p in (sql or "").split(";") if p.strip()]
    return parts[0] if parts else ""


def numbers_in(text: str) -> list[float]:
    return [float(n) for n in NUM_RE.findall(str(text).replace(",", ""))]


def number_present(value: float, nums: list[float], rel_tol: float = 0.005) -> bool:
    return any(abs(n - value) <= max(abs(value) * rel_tol, 0.01) for n in nums)


def section_ids(sources, answer_text: str) -> set:
    ids = set()
    for s in sources or []:
        m = re.match(r"\s*(?:§|section|סעיף)?\s*(\d+\.\d+)", str(s), flags=re.IGNORECASE)
        if m:
            ids.add(m.group(1))
    ids.update(SECTION_RE.findall(answer_text or ""))
    for lst in re.findall(r"(?:sections|סעיפים)\s*((?:\d+\.\d+[\s,/]*(?:and|ו-|ו)?\s*)+)", answer_text or "", flags=re.IGNORECASE):
        ids.update(re.findall(r"\d+\.\d+", lst))
    return ids


# ─────────────────────────────────────────────
# Execution Accuracy (BIRD-style)
# ─────────────────────────────────────────────

def _norm_cell(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 2)
    if isinstance(v, str):
        s = v.strip()
        try:
            return round(float(s.replace(",", "")), 2)
        except ValueError:
            return s.lower()
    return v


def _norm_rows(rows, ordered: bool):
    normed = [tuple(_norm_cell(c) for c in r) for r in rows]
    return normed if ordered else sorted(normed, key=repr)


def _projection_match(pred_rows, gold_rows, ordered: bool) -> bool:
    """True if some column subset (any order) of the predicted rows equals the gold rows."""
    if not gold_rows:
        return not pred_rows
    if not pred_rows:
        return False
    n_gold, n_pred = len(gold_rows[0]), len(pred_rows[0])
    if n_pred < n_gold or n_pred > 8:
        return False
    gold_n = _norm_rows(gold_rows, ordered)
    for idxs in permutations(range(n_pred), n_gold):
        proj = [[r[i] for i in idxs] for r in pred_rows]
        if _norm_rows(proj, ordered) == gold_n:
            return True
    return False


def execution_match(pred_sql: str, gold_q: dict) -> tuple[bool, str]:
    stmt = first_statement(pred_sql)
    if not stmt:
        return False, "no SQL produced"
    if not stmt.lower().lstrip().startswith(("select", "with")):
        return False, "predicted SQL is not a SELECT"
    try:
        with open_db_readonly() as con:
            pred_rows = [list(r) for r in con.execute(stmt).fetchall()]
    except Exception as e:
        return False, f"SQL execution error: {e}"
    gold_rows = gold_q["expected_result"]["rows"]
    ordered = "order by" in gold_q["expected_sql"].lower()
    if _projection_match(pred_rows, gold_rows, ordered):
        return True, f"result set matches ({len(gold_rows)} row(s))"
    return False, f"result mismatch — predicted {pred_rows[:3]}{'…' if len(pred_rows) > 3 else ''} vs gold {gold_rows[:3]}"


# ─────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────

def _conf(data) -> int:
    try:
        return int(float(data.get("confidence_score", 0)))
    except (TypeError, ValueError):
        return 0


def _norm_label(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def score_question(q: dict, data: dict, counts_before: dict, counts_after: dict, latency: float) -> dict:
    t = q["type"]
    answer = str(data.get("answer", "") or "")
    sql = (data.get("sql_query") or "").strip()
    tool = data.get("tool_used")
    conf = _conf(data)
    checks, details = {}, []

    # routing (not for the ambiguous question — a clarification may legitimately use any tool)
    if q.get("expected_tool") and t != "ambiguous":
        checks["routing"] = tool == q["expected_tool"]

    if t in ("database", "hybrid"):
        ok, msg = execution_match(sql, q)
        checks["execution_accuracy"] = ok
        details.append(msg)
        gold_nums = [c for row in q["expected_result"]["rows"] for c in row if isinstance(c, (int, float))]
        checks["answer_has_gold_values"] = all(number_present(float(v), numbers_in(answer)) for v in gold_nums) if gold_nums else True

    if t == "calculator":
        nums = numbers_in(answer)
        expected = {k: v for k, v in (q.get("expected_values") or {}).items() if isinstance(v, (int, float))}
        missing = [k for k, v in expected.items() if not number_present(float(v), nums)]
        checks["calculator_accuracy"] = not missing
        checks["no_sql"] = sql == ""
        if missing:
            details.append("missing values: " + ", ".join(f"{k}={expected[k]}" for k in missing))

    if t in ("policy", "hybrid"):
        ids = section_ids(data.get("policy_sources"), answer)
        missing = [s for s in q.get("expected_sections", []) if s not in ids]
        forbidden = [s for s in q.get("excluded_sections", []) if s in ids]
        checks["policy_sections"] = not missing and not forbidden
        if missing:
            details.append("missing sections: " + ", ".join(missing))
        if forbidden:
            details.append("forbidden sections cited: " + ", ".join(forbidden))
        kws = [k for k in q.get("expected_keywords", []) if k.replace(",", "") not in answer.replace(",", "")]
        checks["policy_keywords"] = not kws
        if kws:
            details.append("missing keywords: " + ", ".join(kws))
        if t == "policy":
            checks["no_sql"] = sql == ""

    if t == "hybrid" and q.get("expected_table"):
        rows = [r for r in (data.get("table_data") or []) if isinstance(r, dict) and r]
        got = [(_norm_label(list(r.values())[0]), _norm_cell(list(r.values())[1]) if len(r) > 1 else None) for r in rows]
        exp = [(_norm_label(r["מדד"]), _norm_cell(r["ערך"])) for r in q["expected_table"]]
        checks["table_structure"] = got == exp
        if got != exp:
            details.append(f"table rows: got {[g[0][:25] + '…=' + str(g[1]) for g in got]}")

    if q.get("min_confidence") is not None:
        checks["confidence"] = conf >= q["min_confidence"]
    if t == "ambiguous":
        checks["clarification"] = conf < CLARIFICATION_THRESHOLD and sql == ""
    if t == "poison":
        checks["rejection"] = tool == "none" and conf == 0 and sql == ""
        if q.get("db_must_be_unchanged"):
            checks["db_unchanged"] = counts_before == counts_after
            if counts_before != counts_after:
                details.append(f"DB CHANGED: {counts_before} → {counts_after}")

    return {
        "id": q["id"], "type": t, "difficulty": q.get("difficulty"), "question": q["question"],
        "expected_tool": q.get("expected_tool"), "tool_used": tool, "confidence": conf,
        "sql_query": sql, "latency_s": round(latency, 1),
        "checks": checks, "passed": all(checks.values()), "details": details,
        "raw": data,
    }


def _rate(results, pred, key=None):
    pool = [r for r in results if pred(r)]
    if not pool:
        return None, 0
    hits = sum(1 for r in pool if (r["checks"].get(key) if key else r["passed"]))
    return hits / len(pool), len(pool)


def build_report(results: list[dict], meta: dict | None = None) -> dict:
    core = lambda r: r["type"] != "poison"
    ex_pool = lambda r: r["type"] in ("database", "hybrid")
    metrics = {}
    metrics["Execution Accuracy (EX)"] = _rate(results, ex_pool, "execution_accuracy")
    for diff in ("simple", "moderate", "challenging"):
        metrics[f"  EX — {diff}"] = _rate(results, lambda r, d=diff: ex_pool(r) and r.get("difficulty") == d, "execution_accuracy")
    metrics["Answer Accuracy (gold values in text)"] = _rate(results, ex_pool, "answer_has_gold_values")
    metrics["Routing Accuracy"] = _rate(results, lambda r: "routing" in r["checks"], "routing")
    metrics["Calculator Accuracy"] = _rate(results, lambda r: r["type"] == "calculator", "calculator_accuracy")
    metrics["Policy Accuracy (sections)"] = _rate(results, lambda r: "policy_sections" in r["checks"], "policy_sections")
    metrics["Clarification Rate (ambiguous)"] = _rate(results, lambda r: r["type"] == "ambiguous", "clarification")
    metrics["Rejection Rate (poison)"] = _rate(results, lambda r: r["type"] == "poison", "rejection")
    fr_pool = [r for r in results if core(r) and r["type"] != "ambiguous"]
    false_rej = sum(1 for r in fr_pool if r["tool_used"] == "none" and r["confidence"] == 0)
    metrics["False Rejection Rate (core)"] = ((false_rej / len(fr_pool)) if fr_pool else None, len(fr_pool))
    metrics["Overall pass rate (all checks)"] = _rate(results, lambda r: True)
    lat = [r["latency_s"] for r in results if r["latency_s"]]
    return {
        "meta": {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "questions": len(results),
                 "avg_latency_s": round(sum(lat) / len(lat), 1) if lat else None, **(meta or {})},
        "metrics": {k: {"score": v[0], "n": v[1]} for k, v in metrics.items()},
        "results": results,
    }


def render_markdown(report: dict) -> str:
    m = report["meta"]
    out = [f"# Benchmark report — AI Credit Risk Agent", "",
           f"Generated {m['generated']} · {m['questions']} questions · model {m.get('model', '?')} · "
           f"runner {m.get('runner', '?')} · avg latency {m.get('avg_latency_s')}s", "",
           "## Summary", "", "| Metric | Score | n |", "|---|---|---|"]
    for name, v in report["metrics"].items():
        score = "n/a" if v["score"] is None else f"{v['score'] * 100:.0f}%"
        out.append(f"| {name} | {score} | {v['n']} |")
    out += ["", "## Per question", "", "| ID | Type | Expected → used | Conf. | Checks | Latency | Pass |", "|---|---|---|---|---|---|---|"]
    for r in report["results"]:
        checks = " ".join(f"{'✅' if ok else '❌'}{k}" for k, ok in r["checks"].items())
        out.append(f"| {r['id']} | {r['type']}{(' / ' + r['difficulty']) if r.get('difficulty') else ''} | "
                   f"{r['expected_tool']} → {r['tool_used']} | {r['confidence']} | {checks} | {r['latency_s']}s | {'✅' if r['passed'] else '❌'} |")
    failed = [r for r in report["results"] if not r["passed"]]
    if failed:
        out += ["", "## Failure details", ""]
        for r in failed:
            out.append(f"- **{r['id']}** — {r['question']}")
            for d in r["details"]:
                out.append(f"  - {d}")
            if r["sql_query"]:
                out.append(f"  - SQL: `{first_statement(r['sql_query'])[:200]}`")
    return "\n".join(out)


# ─────────────────────────────────────────────
# Runners
# ─────────────────────────────────────────────

def run_benchmark(run_fn, questions: list[dict], progress=None) -> list[dict]:
    """
    run_fn(question_text) -> (data_dict, latency_seconds). Used by both the CLI (AppTest) and the
    in-app button (run_agent). Poison questions get before/after DB row counts.
    """
    results = []
    for i, q in enumerate(questions, 1):
        if progress:
            progress(i, len(questions), q)
        before = db_counts()
        try:
            data, latency = run_fn(q["question"])
        except Exception as e:
            data, latency = {"answer": f"RUNNER ERROR: {type(e).__name__}: {e}", "sql_query": "",
                             "tool_used": "error", "confidence_score": 0}, 0.0
        after = db_counts()
        results.append(score_question(q, data, before, after, latency))
    return results


def make_apptest_runner(api_key: str, timeout: int = 300):
    """Each question runs in a fresh AppTest session of app.py (no shared chat history)."""
    from streamlit.testing.v1 import AppTest
    app_path = os.path.join(BASE_DIR, "app.py")

    def run(question: str):
        at = AppTest.from_file(app_path, default_timeout=timeout)
        at.secrets["OPENAI_API_KEY"] = api_key
        at.run()
        if at.exception:
            raise RuntimeError(f"app failed to start: {at.exception[0].value}")
        t0 = time.time()
        at.chat_input[0].set_value(question).run()
        latency = time.time() - t0
        if at.exception:
            raise RuntimeError(at.exception[0].value)
        data = at.session_state["messages"][-1]["data"]
        data = {k: v for k, v in data.items()}  # plain copy
        return data, latency
    return run


def save_outputs(report: dict, md_path: str = REPORT_MD, json_path: str = RESULTS_JSON):
    md = render_markdown(report)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    return md


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the golden-dataset benchmark against app.py")
    parser.add_argument("--ids", help="comma-separated question ids to run (default: all)")
    parser.add_argument("--rescore", help="path to a benchmark_results.json to re-score without calling the model")
    parser.add_argument("--timeout", type=int, default=300, help="seconds per question")
    args = parser.parse_args()

    dataset = load_dataset()
    questions = dataset["questions"]
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",")}
        questions = [q for q in questions if q["id"] in wanted]

    if args.rescore:
        with open(args.rescore, encoding="utf-8") as f:
            prev = json.load(f)
        raw = {r["id"]: r for r in prev["results"]}
        counts = db_counts()
        results = [score_question(q, raw[q["id"]]["raw"], counts, counts, raw[q["id"]]["latency_s"])
                   for q in questions if q["id"] in raw]
        report = build_report(results, {"runner": "rescore", "model": prev.get("meta", {}).get("model")})
    else:
        from build_vector_db import get_api_key
        api_key = get_api_key()
        runner = make_apptest_runner(api_key, timeout=args.timeout)
        results = run_benchmark(runner, questions,
                                progress=lambda i, n, q: print(f"[{i:2}/{n}] {q['id']} {q['type']:10} {q['question'][:60]}", flush=True))
        report = build_report(results, {"runner": "apptest", "model": "gpt-4o-mini"})

    md = save_outputs(report)
    print("\n" + md)
    print(f"\nSaved {os.path.basename(REPORT_MD)} and {os.path.basename(RESULTS_JSON)}")
