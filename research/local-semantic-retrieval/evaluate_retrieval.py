"""Evaluate local cosine retrieval against a BM25 lexical baseline."""

import argparse
from array import array
from collections import Counter
import json
import math
from pathlib import Path
import platform
import re
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "local-sentiment-evaluation"))
from evaluate import model_identity, request, sha256


def normalize(vector, dimensions):
    if len(vector) != dimensions or any(type(x) not in (float, int) or not math.isfinite(x) for x in vector):
        raise ValueError("Invalid embedding dimensions or values")
    length = math.sqrt(sum(x*x for x in vector))
    if not length:
        raise ValueError("Zero-length embedding")
    return list(array("f", (x/length for x in vector)))


def rank_vectors(ids, vectors, query):
    if len(ids) != len(vectors) or any(len(v) != len(query) for v in vectors):
        raise ValueError("Index dimensions differ")
    scores = [(id, sum(a*b for a,b in zip(vector,query))) for id,vector in zip(ids,vectors)]
    return sorted(scores, key=lambda pair: (-pair[1], pair[0]))


def bm25(documents, query, k1, b):
    tokens = [re.findall(r"[a-z0-9]+", doc["title"].lower()+" "+doc["text"].lower()) for doc in documents]
    average = sum(map(len,tokens))/len(tokens)
    frequency = Counter(word for words in tokens for word in set(words))
    terms = set(re.findall(r"[a-z0-9]+",query.lower()))
    scores = []
    for doc, words in zip(documents,tokens):
        counts = Counter(words)
        total = 0
        for term in terms:
            freq = counts[term]
            idf = math.log(1+(len(tokens)-frequency[term]+0.5)/(frequency[term]+0.5))
            total += idf*freq*(k1+1)/(freq+k1*(1-b+b*len(words)/average))
        scores.append((doc["id"],total))
    return sorted(scores,key=lambda pair:(-pair[1],pair[0]))


def metrics(rows):
    if not rows:
        raise ValueError("No queries to score")
    ranks = [next(i for i,id in enumerate(row["ranking"],1) if id in row["relevant"]) for row in rows]
    return {"n":len(rows),"hit_at_1":sum(rank==1 for rank in ranks)/len(rows),
            "hit_at_3":sum(rank<=3 for rank in ranks)/len(rows),
            "mrr":sum(1/rank for rank in ranks)/len(rows)}


def embed(texts,config):
    response = request("/api/embed",120,{"model":config["model"],"input":texts,
                                      "truncate":False,"options":{"num_ctx":2048},"keep_alive":"5m"})
    vectors = response.get("embeddings",[])
    if len(vectors)!=len(texts):
        raise ValueError("Embedding response count differs")
    return [normalize(vector,config["dimensions"]) for vector in vectors]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--manifest-only",action="store_true")
    args=parser.parse_args()
    raw={name:(HERE/(name+".json")).read_bytes() for name in ("documents","queries","protocol")}
    manifest={name:sha256(value) for name,value in raw.items()}
    if args.manifest_only:
        print(json.dumps(manifest,indent=2));return
    if not args.output:
        parser.error("--output is required")
    if manifest!=json.loads((HERE/"manifest.json").read_bytes()):
        raise ValueError("Inputs changed")
    docs,queries,config=(json.loads(raw[key]) for key in ("documents","queries","protocol"))
    ids=[doc["id"] for doc in docs]
    if len(set(ids))!=len(ids) or any(not set(q["relevant"])<=set(ids) for q in queries):
        raise ValueError("Invalid document references")
    before=model_identity(config)
    if "embedding" not in request("/api/show",10,{"model":config["model"]}).get("capabilities",[]):
        raise ValueError("Model lacks embedding capability")
    args.output.mkdir(parents=True,exist_ok=False)
    rounds=[]
    for round_ in range(1,config["rounds"]+1):
        print(f"Building round {round_} index",file=sys.stderr,flush=True)
        started=time.perf_counter()
        vectors=embed([config["document_prefix"]+d["title"]+"\n"+d["text"] for d in docs],config)
        build_seconds=time.perf_counter()-started
        packed=array("f",[value for vector in vectors for value in vector])
        if sys.byteorder!="little":
            packed.byteswap()
        data=packed.tobytes()
        index_path=args.output/f"index-{round_}.f32"
        index_path.write_bytes(data)
        metadata={"ids":ids,"dimensions":config["dimensions"],"format":"float32-little-endian",
                  "identity":before,"manifest":manifest,"vector_sha256":sha256(data)}
        meta=(json.dumps(metadata,indent=2)+"\n").encode()
        (args.output/f"index-{round_}.json").write_bytes(meta)
        rows=[]
        query_vectors=[]
        for query in queries:
            started=time.perf_counter()
            vector=embed([config["query_prefix"]+query["text"]],config)[0]
            embed_seconds=time.perf_counter()-started
            started=time.perf_counter()
            ranking=rank_vectors(ids,vectors,vector)
            search_seconds=time.perf_counter()-started
            started=time.perf_counter()
            lexical=bm25(docs,query["text"],config["bm25_k1"],config["bm25_b"])
            lexical_seconds=time.perf_counter()-started
            query_vectors.append(vector)
            rows.append({"id":query["id"],"group":query["group"],"relevant":query["relevant"],
                         "ranking":[id for id,_ in ranking],"scores":[score for _,score in ranking],
                         "lexical_ranking":[id for id,_ in lexical],"embedding_seconds":embed_seconds,
                         "search_seconds":search_seconds,"lexical_seconds":lexical_seconds})
        (args.output/f"queries-{round_}.json").write_text(json.dumps(query_vectors))
        rounds.append({"build_seconds":build_seconds,"vector_bytes":len(data),"metadata_bytes":len(meta),
                       "index_sha256":sha256(data),"embedding_metrics":metrics(rows),
                       "lexical_metrics":metrics([row|{"ranking":row["lexical_ranking"]} for row in rows]),
                       "group_metrics":{group:metrics([row for row in rows if row["group"]==group])
                                        for group in ("literal","paraphrase")},"rows":rows})
    if model_identity(config)!=before:
        raise ValueError("Model identity changed")
    print(json.dumps({"runtime":{"python":platform.python_version(),"architecture":platform.machine()},
                      "identity":before,"manifest":manifest,"rounds":rounds,
                      "changed_rankings":[a["id"] for a,b in zip(rounds[0]["rows"],rounds[1]["rows"])
                                          if a["ranking"]!=b["ranking"]],
                      "output_directory":str(args.output.resolve())},indent=2))


if __name__=="__main__":
    main()
