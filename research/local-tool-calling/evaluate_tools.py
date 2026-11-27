"""Evaluate bounded local function calls and final-answer correctness."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "local-sentiment-evaluation"))
from evaluate import model_identity, request, sha256


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1e9


def dispatch(name, args, inventory):
    if not isinstance(args, dict):
        raise ValueError("Arguments must be an object")
    if name == "stock" and set(args) == {"sku"}:
        if not isinstance(args["sku"], str) or len(args["sku"]) > 100:
            raise ValueError("Invalid SKU")
        if args["sku"] not in inventory:
            return {"error": "unknown SKU"}
        return {"quantity": inventory[args["sku"]]}
    if name == "arithmetic" and set(args) == {"operation", "left", "right"}:
        left, right = args["left"], args["right"]
        if not numeric(left) or not numeric(right):
            raise ValueError("Invalid numeric operands")
        operations = {"add": lambda: left + right, "subtract": lambda: left - right,
                      "multiply": lambda: left * right, "divide": lambda: left / right}
        if args["operation"] not in operations:
            raise ValueError("Unsupported operation")
        if args["operation"] == "divide" and right == 0:
            return {"error": "division by zero"}
        return {"value": operations[args["operation"]]()}
    if name == "distance" and set(args) == {"value", "from_unit", "to_unit"}:
        if not numeric(args["value"]) or args["from_unit"] not in ("m", "km") or args["to_unit"] not in ("m", "km"):
            raise ValueError("Invalid distance arguments")
        meters = args["value"] * (1000 if args["from_unit"] == "km" else 1)
        return {"value": meters / (1000 if args["to_unit"] == "km" else 1)}
    raise ValueError("Unknown tool or invalid argument keys")


def correct(task, answer, calls, rejected):
    if type(answer) in (int, float) and type(task["expected"]) in (int, float):
        matches = math.isclose(answer, task["expected"], rel_tol=1e-9, abs_tol=1e-9)
    else:
        matches = type(answer) is type(task["expected"]) and answer == task["expected"]
    return matches and Counter(calls) == Counter(task["required_tools"]) and rejected == 0


def run_task(model, task, config):
    messages = [{"role": "system", "content": config["system"]},
                {"role": "user", "content": task["prompt"]}]
    trace, calls = [], []
    rejected = 0
    answer = None
    outcome = "turn_limit"
    start = time.perf_counter()
    for turn in range(config["max_turns"]):
        body = {"model": model, "messages": messages, "tools": config["tools"],
                "stream": False, "options": config["options"], "keep_alive": "5m"}
        if model.startswith("qwen3:"):
            body["think"] = config["think_qwen3"]
        try:
            response = request("/api/chat", 120, body)
        except (OSError, ValueError) as error:
            outcome = "request_failure"
            trace.append({"error": str(error)})
            break
        message = response.get("message", {})
        tool_calls = message.get("tool_calls") or []
        assistant = {"role": "assistant", "content": message.get("content", "")}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        messages.append(assistant)
        trace.append({"turn": turn + 1, "assistant": assistant, "done": response.get("done")})
        if response.get("done") is not True:
            outcome = "incomplete_response"
            break
        if not tool_calls:
            try:
                final = json.loads(message.get("content", ""))
                if not isinstance(final, dict) or set(final) != {"answer"}:
                    raise ValueError("Invalid answer object")
                answer = final["answer"]
                outcome = "completed"
            except (ValueError, TypeError):
                outcome = "invalid_final"
            break
        for call in tool_calls:
            if len(calls) >= config["max_calls"]:
                outcome = "call_limit"
                return {"id": task["id"], "answer": answer, "outcome": outcome,
                        "calls": calls, "rejected": rejected, "trace": trace,
                        "seconds": time.perf_counter()-start, "correct": False}
            function = call.get("function", {})
            name, arguments = function.get("name"), function.get("arguments")
            calls.append(name)
            try:
                result = dispatch(name, arguments, config["inventory"])
            except (ValueError, TypeError) as error:
                rejected += 1
                result = {"error": str(error)}
            trace.append({"tool": name, "arguments": arguments, "result": result})
            messages.append({"role": "tool", "tool_name": name, "content": json.dumps(result)})
    return {"id": task["id"], "answer": answer, "outcome": outcome,
            "calls": calls, "rejected": rejected, "trace": trace,
            "seconds": time.perf_counter()-start,
            "correct": outcome == "completed" and correct(task, answer, calls, rejected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    raw = {name: (HERE / (name+".json")).read_bytes() for name in ("tasks", "protocol")}
    manifest = {name: sha256(data) for name, data in raw.items()}
    if args.manifest_only:
        print(json.dumps(manifest, indent=2))
        return
    if manifest != json.loads((HERE / "manifest.json").read_bytes()):
        raise ValueError("Inputs changed")
    tasks, config = json.loads(raw["tasks"]), json.loads(raw["protocol"])
    rounds, identities = [], {}
    for order in (config["models"], list(reversed(config["models"]))):
        runs = {}
        for model in order:
            identity = model_identity({"model": model})
            info = request("/api/show", 15, {"model": model})
            if "tools" not in info.get("capabilities", []):
                raise ValueError("Tool support unavailable")
            identities.setdefault(model, identity)
            if identities[model] != identity:
                raise ValueError("Model identity changed")
            rows = []
            try:
                for task in tasks:
                    print(f"{model}: {task['id']}", file=sys.stderr, flush=True)
                    rows.append(run_task(model, task, config))
                if model_identity({"model": model}) != identity:
                    raise ValueError("Model identity changed during run")
            finally:
                request("/api/generate", 30, {"model": model, "keep_alive": 0, "stream": False})
            runs[model] = {"identity": identity, "correct": sum(row["correct"] for row in rows),
                           "n": len(rows), "rows": rows}
        rounds.append({"order": order, "runs": runs})
    changed = {
        model: [a["id"] for a, b in zip(run["rows"], rounds[1]["runs"][model]["rows"])
                if (a["answer"], a["correct"], a["calls"], a["outcome"]) !=
                   (b["answer"], b["correct"], b["calls"], b["outcome"])]
        for model, run in rounds[0]["runs"].items()
    }
    print(json.dumps({"runtime": {"python": platform.python_version(), "architecture": platform.machine()},
                      "manifest": manifest, "rounds": rounds, "changed_results": changed}, indent=2))


if __name__ == "__main__":
    main()
