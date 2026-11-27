"""Evaluate small local vision models on fixed images and spatial questions."""

import argparse
import base64
from collections import Counter
import json
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent / "local-sentiment-evaluation"))
from evaluate import model_identity, request, sha256


def parse_answer(response):
    if response.get("done") is not True or not isinstance(response.get("message",{}).get("content"),str):
        return "failure"
    text = response["message"]["content"].strip()
    return text if text in ("yes","no") else "invalid"


def score(rows):
    return {"n":len(rows),"correct":sum(row["prediction"]==row["reference"] for row in rows),
            "counts":dict(Counter(row["prediction"] for row in rows))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-only",action="store_true")
    args=parser.parse_args()
    raw={key:(HERE/(key+".json")).read_bytes() for key in ("sources","cases","protocol")}
    sources,cases,config=(json.loads(raw[key]) for key in ("sources","cases","protocol"))
    images={}
    for source in sources:
        data=(HERE/source["file"]).read_bytes()
        if sha256(data)!=source["sha256"]:
            raise ValueError("Image hash mismatch")
        images[source["id"]]=base64.b64encode(data).decode()
    manifest={key:sha256(value) for key,value in raw.items()}
    if args.manifest_only:
        print(json.dumps(manifest,indent=2))
        return
    if manifest!=json.loads((HERE/"manifest.json").read_bytes()):
        raise ValueError("Inputs changed")
    def chat(model,image,prompt):
        return request("/api/chat",180,{"model":model,"stream":False,"keep_alive":"5m",
            "options":config["options"],
            "messages":[{"role":"user","content":prompt,"images":[images[image]]}]})
    rounds=[];identities={}
    for order in (config["models"],list(reversed(config["models"]))):
        runs={}
        for model in order:
            print(f"Evaluating {model}",file=sys.stderr,flush=True)
            identity=model_identity({"model":model})
            info=request("/api/show",15,{"model":model})
            if "vision" not in info.get("capabilities",[]):
                raise ValueError("Local model does not advertise vision")
            identities.setdefault(model,identity)
            if identities[model]!=identity:
                raise ValueError("Model identity changed")
            rows=[];descriptions=[]
            try:
                for case in cases:
                    started=time.perf_counter()
                    try:
                        response=chat(model,case["image"],config["question_prefix"]+case["question"])
                        prediction=parse_answer(response)
                    except (OSError,ValueError):
                        response={};prediction="failure"
                    rows.append({"id":case["id"],"reference":case["reference"],"prediction":prediction,
                                 "raw_response":response.get("message",{}).get("content"),
                                 "seconds":time.perf_counter()-started})
                for source in sources:
                    response=chat(model,source["id"],config["description_prompt"])
                    descriptions.append({"image":source["id"],"done":response.get("done"),
                                         "description":response.get("message",{}).get("content")})
                if model_identity({"model":model})!=identity:
                    raise ValueError("Identity changed during evaluation")
            finally:
                request("/api/generate",30,{"model":model,"keep_alive":0,"stream":False})
            runs[model]={"identity":identity,"scores":score(rows),"rows":rows,"descriptions":descriptions}
        rounds.append({"order":order,"runs":runs})
    changes={model:[a["id"] for a,b in zip(run["rows"],rounds[1]["runs"][model]["rows"])
                    if a["id"]!=b["id"] or a["prediction"]!=b["prediction"]]
             for model,run in rounds[0]["runs"].items()}
    print(json.dumps({"runtime":{"python":platform.python_version(),"architecture":platform.machine()},
                      "manifest":manifest,"rounds":rounds,"changed_predictions":changes},indent=2))


if __name__=="__main__":
    main()
