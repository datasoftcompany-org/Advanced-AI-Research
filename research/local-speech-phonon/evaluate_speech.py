"""Generate local PCM speech, save proper WAV files and measure delivery."""

import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import urllib.request
import wave

HERE = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:11434"


def local_json(path, body=None):
    payload = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=payload, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)


def identity(name):
    for model in local_json("/api/tags")["models"]:
        if model["name"] == name and not model.get("remote_host") and "cloud" not in name:
            return {"name":name, "digest":model["digest"],
                    "ollama":local_json("/api/version")["version"]}
    raise ValueError("The requested speech model must already be installed locally")


def pcm_stats(raw, rate):
    if not raw or len(raw) % 2:
        raise ValueError("Empty PCM or incomplete 16-bit sample")
    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    rms = math.sqrt(sum(value * value for value in samples) / len(samples))
    return {"samples":len(samples), "duration_seconds":len(samples)/rate,
            "rms":rms, "peak_absolute":max(abs(value) for value in samples),
            "full_scale_samples":sum(value in (-32768, 32767) for value in samples),
            "non_silent":rms > 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    raw = {name:(HERE / (name+".json")).read_bytes() for name in ("cases","protocol")}
    hashes = {key:hashlib.sha256(value).hexdigest() for key,value in raw.items()}
    if args.manifest_only:
        print(json.dumps(hashes,indent=2))
        return
    if not args.output:
        parser.error("--output is required")
    if hashes != json.loads((HERE/"manifest.json").read_bytes()):
        raise ValueError("Inputs differ from the frozen protocol")
    cases, config = json.loads(raw["cases"]), json.loads(raw["protocol"])
    before = identity(config["model"])
    info = local_json("/api/show", {"model":config["model"]})
    if "speech" not in info.get("capabilities", []):
        raise ValueError("Model does not advertise speech capability")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"manifest":hashes,"identity":before,
              "model_details":info.get("details"), "voices":info.get("voices"),
              "runtime":{"python":platform.python_version(),"architecture":platform.machine()},
              "output_directory":str(args.output.resolve()),"runs":[]}
    for round_ in range(1,config["rounds"]+1):
        for case in cases:
            print(f"Round {round_}: {case['id']}", file=sys.stderr, flush=True)
            start = time.perf_counter()
            first = None
            chunks = []
            req = urllib.request.Request(BASE+"/v1/audio/speech",
                data=json.dumps({"model":config["model"],"voice":config["voice"],
                                 "input":case["text"],"response_format":config["response_format"]}).encode(),
                headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=config["timeout_seconds"]) as response:
                content_type = response.headers.get("Content-Type")
                while chunk := response.read(config["read_chunk_bytes"]):
                    if first is None:
                        first = time.perf_counter()-start
                    chunks.append(chunk)
            elapsed = time.perf_counter()-start
            pcm = b"".join(chunks)
            stats = pcm_stats(pcm,config["sample_rate"])
            name = f"round-{round_}-{case['id']}.wav"
            with wave.open(str(args.output/name),"wb") as wav:
                wav.setnchannels(config["channels"])
                wav.setsampwidth(config["sample_width"])
                wav.setframerate(config["sample_rate"])
                wav.writeframes(pcm)
            report["runs"].append({"round":round_,"id":case["id"],"wav":name,
                "pcm_sha256":hashlib.sha256(pcm).hexdigest(),"content_type":content_type,
                "first_chunk_seconds":first,"generation_seconds":elapsed,
                "real_time_factor":elapsed/stats["duration_seconds"],**stats})
            (args.output/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    if identity(config["model"]) != before:
        raise ValueError("Model or service identity changed")
    report["identity_verified_after_run"] = True
    (args.output/"results.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
