"""Resumable downloader: python -m tidal.public_data.download [name ...]. Skips files already present."""
import argparse, fnmatch, json, time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download
from tidal.public_data.manifest import DATA, DATASETS

def files_for(name, d, api, info=None):
    if d.get("files"): return d["files"]
    info = info or api.dataset_info(d["repo"], revision=d.get("revision"), files_metadata=True)
    fs = [s.rfilename for s in info.siblings if not s.rfilename.startswith(".")]
    if d.get("files_glob"): fs = [f for f in fs if any(fnmatch.fnmatch(f.rsplit("/", 1)[-1], g) for g in d["files_glob"])]
    return fs

def main(names, workers=1):
    if workers < 1: raise ValueError("workers must be positive")
    api = HfApi()
    for name in names or list(DATASETS):
        d = DATASETS[name]; out = DATA / name; out.mkdir(parents=True, exist_ok=True); t = time.time()
        info = api.dataset_info(d["repo"], revision=d.get("revision"), files_metadata=True)
        fs = files_for(name, d, api, info); n = 0
        def fetch(f):
            for attempt in range(4):
                try:
                    hf_hub_download(d["repo"], f, repo_type="dataset", revision=info.sha, local_dir=out)
                    print(f"[{name}] downloaded {f}", flush=True)
                    return 1
                except Exception as e:
                    print(f"[{name}] retry {attempt} {f}: {type(e).__name__}", flush=True)
                    time.sleep(5 * (attempt + 1))
            return 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            n = sum(pool.map(fetch, fs))
        (out / "_done.json").write_text(json.dumps(dict(repo=d["repo"], revision=info.sha, license=d["license"], files=len(fs), ok=n)))
        print(f"[{name}] {n}/{len(fs)} files in {time.time() - t:.0f}s", flush=True)
        if n != len(fs): raise RuntimeError(f"Incomplete dataset {name}: {n}/{len(fs)} files")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*")
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()
    main(args.names, args.workers)
