"""Resumable downloader: python -m tidal.public_data.download [name ...]. Skips files already present."""
import fnmatch, json, sys, time
from huggingface_hub import HfApi, hf_hub_download
from tidal.public_data.manifest import DATA, DATASETS

def files_for(name, d, api):
    if d.get("files"): return d["files"]
    info = api.dataset_info(d["repo"], files_metadata=True)
    fs = [s.rfilename for s in info.siblings if not s.rfilename.startswith(".")]
    if d.get("files_glob"): fs = [f for f in fs if any(fnmatch.fnmatch(f.rsplit("/", 1)[-1], g) for g in d["files_glob"])]
    return fs

def main(names):
    api = HfApi()
    for name in names or list(DATASETS):
        d = DATASETS[name]; out = DATA / name; out.mkdir(parents=True, exist_ok=True); t = time.time()
        fs = files_for(name, d, api); n = 0
        for f in fs:
            for attempt in range(4):
                try: hf_hub_download(d["repo"], f, repo_type="dataset", local_dir=out); n += 1; break
                except Exception as e: print(f"[{name}] retry {attempt} {f}: {type(e).__name__}", flush=True); time.sleep(5 * (attempt + 1))
        (out / "_done.json").write_text(json.dumps(dict(repo=d["repo"], license=d["license"], files=len(fs), ok=n)))
        print(f"[{name}] {n}/{len(fs)} files in {time.time() - t:.0f}s", flush=True)

if __name__ == "__main__": main(sys.argv[1:])
