"""Fit the residual inject MLP on a frozen causal LM.

Value = answer-span pool at the inject layer (not L2-normalized).
Phone-like answers pool the last four answer tokens. Birthday answers
concatenate field-token slices. Other answers pool the whole span.
``fit_phone_readout`` and ``fit_date_readout`` fit the linear spellings
and do not touch the MLP.
``add = scale(v) + MLP(scale(v))`` with the MLP's last layer starting at zero.
Teacher-forced next-token cross-entropy on the answer, given the prompt only
(the fact text is not in the prompt). A fraction of steps adds the v21 hinge
on a wrong value.

CPU smoke, no download::

    python -m tidal.llm_memory_train --tiny --steps 1 --n-cache 4 --neg-p 0

The original SmolLM run used 1500 steps, batch 8, 2000 cached facts, lr 1e-3,
and a 20% wrong-value hinge. This script's defaults are that step count and
hinge, with a smaller cache so a CPU box can finish a real checkpoint later.
``--model`` defaults to SmolLM2-135M and will download it. ``--tiny`` does not.
That default checkpoint and its default layer already hold the shipped trained
inject (``tidal/llm_inject_smollm2_135m.pt``), so a non-tiny run continues from
those weights. ``--tiny`` and any other model or layer still start from a zero
last layer.

Retrieval is the episodic store. This script does not train a second q/k
adapter; contrastive questions for tidal features stay in ``tidal.question_head``.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

_NAMES = ("Ada", "Bea", "Cal", "Dee", "Ezra", "Finn")
_TABLE = {
    "phone": ("555-0100", "555-0199", "555-0134", "555-0118"),
    "city": ("Lisbon", "Milan", "Oslo", "Bern"),
    "color": ("green", "amber", "cobalt", "ivory"),
    "team": ("north", "harbor", "cedar", "lantern"),
}


def sample_facts(n: int, seed: int = 0) -> list:
    """Synthetic ``(fact, prompt, answer)`` rows. ASCII, so the byte tokenizer fits."""
    rng = random.Random(int(seed))
    rels = list(_TABLE)
    facts = []
    for i in range(int(n)):
        name = _NAMES[i % len(_NAMES)]
        rel = rels[i % len(rels)]
        # Rotate the answer with i so a cache of one per name/rel still varies.
        choices = _TABLE[rel]
        answer = choices[(i // max(1, len(_NAMES))) % len(choices)]
        if rng.random() < 0.5:
            answer = rng.choice(choices)
        prompt = f"{name}'s {rel} is "
        facts.append(
            {
                "text": prompt + answer,
                "a": answer,
                "q": prompt,
                "n": name,
                "rel": rel,
            }
        )
    return facts


def _cache(mem, facts: list) -> list:
    rows = []
    for fact in facts:
        value = mem.encode_value(fact["text"], fact["a"])
        rows.append(
            {
                "value": value,
                "prompt": fact["q"],
                "answer": fact["a"],
                "text": fact["text"],
            }
        )
    return rows


def fit_inject(
    mem,
    facts: list,
    *,
    steps: int = 1500,
    bs: int = 8,
    lr: float = 1e-3,
    neg_p: float = 0.2,
    seed: int = 0,
    weight_decay: float = 1e-4,
) -> list:
    """Teacher-forced CE on ``facts``. Returns the logged step records.

    ``facts`` items are mappings with ``text``, ``a`` (answer), and ``q``
    (prompt). The prompt should not contain the answer span.
    """
    import torch

    if steps < 1:
        raise ValueError("steps must be >= 1")
    if bs < 1:
        raise ValueError("bs must be >= 1")
    if not facts:
        raise ValueError("facts is empty")
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    cache = _cache(mem, facts)
    opt = torch.optim.AdamW(mem.inject.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    mem.inject.train()
    log = []
    try:
        for step in range(1, int(steps) + 1):
            losses = []
            for _ in range(int(bs)):
                ex = cache[random.randrange(len(cache))]
                loss = mem.teacher_force_loss(ex["prompt"], ex["answer"], ex["value"])
                if neg_p and random.random() < float(neg_p) and len(cache) > 1:
                    other = ex
                    for _try in range(8):
                        other = cache[random.randrange(len(cache))]
                        if other["answer"] != ex["answer"]:
                            break
                    if other["answer"] != ex["answer"]:
                        loss_neg = mem.teacher_force_loss(ex["prompt"], ex["answer"], other["value"])
                        loss = loss + torch.relu(loss.new_tensor(2.0) - loss_neg) * 0.3
                losses.append(loss)
            loss = torch.stack(losses).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            mem._delta = None
            if step == 1 or step % 100 == 0 or step == int(steps):
                row = {"step": step, "loss": float(loss.detach())}
                log.append(row)
                print(row, flush=True)
    finally:
        mem.inject.eval()
        mem._delta = None
    return log


def fit_phone_readout(mem, *, n_numbers: int = 120, alpha: float = 100.0, seed: int = 0) -> dict:
    """Ridge the four variable phone digits from the phone value pool.

    Train-split names only. Writes the inject buffers ``phone_on``,
    ``phone_mu``, ``phone_sd``, ``phone_w``, and ``phone_b``. The MLP weights
    stay where they are. ``alpha`` is the ridge penalty on standardized
    features. Returns the train exact-match count.
    """
    import numpy as np

    from tidal.llm_memory_data import _TEXT, _answer, names

    if int(n_numbers) < 10:
        raise ValueError("n_numbers must be >= 10")
    if float(alpha) < 0:
        raise ValueError("alpha must be >= 0")
    rng = __import__("random").Random(int(seed))
    phone_ids = list(range(8000))
    rng.shuffle(phone_ids)
    chosen = phone_ids[: int(n_numbers)]
    train_names = list(names("train"))
    values = []
    digits = []
    for name in train_names:
        for i in chosen:
            answer = _answer("phone", i)
            text = _TEXT["phone"].format(name=name, a=answer)
            values.append(mem.encode_value(text, answer))
            digits.append([int(c) for c in answer[-4:]])
    x = np.stack(values).astype(np.float64)
    y = np.asarray(digits, dtype=np.int64)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-6] = 1.0
    z = (x - mu) / sd
    eye = np.eye(z.shape[1])
    gram = z.T @ z + float(alpha) * eye
    weight = np.zeros((4, 10, z.shape[1]), dtype=np.float64)
    bias = np.full((4, 10), -1.0e4, dtype=np.float64)
    for pos in range(4):
        classes = np.unique(y[:, pos])
        targets = np.eye(len(classes))[np.searchsorted(classes, y[:, pos])]
        solved = np.linalg.solve(gram, z.T @ targets)  # [d, n_class]
        for col, digit in enumerate(classes):
            weight[pos, int(digit)] = solved[:, col]
            bias[pos, int(digit)] = 0.0
    import torch

    mem.inject.phone_mu.copy_(torch.tensor(mu, dtype=torch.float32))
    mem.inject.phone_sd.copy_(torch.tensor(sd, dtype=torch.float32))
    mem.inject.phone_w.copy_(torch.tensor(weight, dtype=torch.float32))
    mem.inject.phone_b.copy_(torch.tensor(bias, dtype=torch.float32))
    mem.inject.phone_on.fill_(1.0)
    pred = []
    for pos in range(4):
        logits = z @ weight[pos].T + bias[pos]
        pred.append(logits.argmax(axis=1))
    hat = np.stack(pred, axis=1)
    hit = int((hat == y).all(axis=1).sum())
    return {"train_exact": hit, "n": int(y.shape[0]), "n_numbers": int(n_numbers), "n_names": len(train_names), "alpha": float(alpha)}


def _date_block_ok(value, day: int, hidden: int) -> bool:
    """One-digit days store an exact zero in the day-tens block."""
    from tidal.llm_memory import date_block_sizes

    if int(day) > 9:
        return True
    sizes = date_block_sizes(hidden)
    tens = np.asarray(value, dtype=np.float32)[sizes[0] : sizes[0] + sizes[1]]
    return bool(np.all(tens == 0))


def _solve_class(z, y, n_class: int, gram):
    """Ridge one-hot weights. Unseen classes keep a large negative bias."""
    y = np.asarray(y, dtype=np.int64)
    classes = np.unique(y)
    if classes.min() < 0 or classes.max() >= int(n_class):
        raise ValueError(f"class {int(classes.min())}..{int(classes.max())} outside 0..{int(n_class) - 1}")
    targets = np.eye(len(classes))[np.searchsorted(classes, y)]
    solved = np.linalg.solve(gram, z.T @ targets)
    weight = np.zeros((int(n_class), z.shape[1]), dtype=np.float64)
    bias = np.full(int(n_class), -1.0e4, dtype=np.float64)
    for col, label in enumerate(classes):
        weight[int(label)] = solved[:, col]
        bias[int(label)] = 0.0
    return weight, bias


def _standardize(x, alpha: float):
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-6] = 1.0
    z = (x - mu) / sd
    gram = z.T @ z + float(alpha) * np.eye(z.shape[1])
    return mu, sd, z, gram


def _write_date_readout(mem, x, months, days, years, alpha: float) -> None:
    import torch

    mu, sd, z, gram = _standardize(x, alpha)
    month_w, month_b = _solve_class(z, months, 12, gram)
    digit_w = np.zeros((6, 10, z.shape[1]), dtype=np.float64)
    digit_b = np.zeros((6, 10), dtype=np.float64)
    columns = (
        days // 10,
        days % 10,
        years // 1000,
        (years // 100) % 10,
        (years // 10) % 10,
        years % 10,
    )
    for pos, column in enumerate(columns):
        digit_w[pos], digit_b[pos] = _solve_class(z, column, 10, gram)
    mem.inject.date_mu.copy_(torch.tensor(mu, dtype=torch.float32))
    mem.inject.date_sd.copy_(torch.tensor(sd, dtype=torch.float32))
    mem.inject.date_month_w.copy_(torch.tensor(month_w, dtype=torch.float32))
    mem.inject.date_month_b.copy_(torch.tensor(month_b, dtype=torch.float32))
    mem.inject.date_digit_w.copy_(torch.tensor(digit_w, dtype=torch.float32))
    mem.inject.date_digit_b.copy_(torch.tensor(digit_b, dtype=torch.float32))
    mem.inject.date_on.fill_(1.0)


def _score_date_rows(mem, rows) -> dict:
    """Float32 module exact spelling. ``rows`` is ``(value, answer)``."""
    import torch

    hit = 0
    misses = []
    for value, answer in rows:
        spelled = mem.inject.date_spelling(torch.tensor(np.asarray(value, dtype=np.float32)))
        if spelled == answer:
            hit += 1
        elif len(misses) < 8:
            misses.append({"gold": answer, "spelled": spelled})
    n = len(rows)
    return {"hits": hit, "n": n, "acc": (hit / n) if n else None, "misses": misses}


def _encode_dates(mem, name_list, dates) -> list:
    import time

    from tidal.llm_memory_data import parse_date

    rows = []
    t0 = time.time()
    for i, name in enumerate(name_list):
        for answer in dates:
            text = f"{name}'s birthday is {answer}"
            value = mem.encode_value(text, answer)
            parsed = parse_date(answer)
            month, day, year, _spans = parsed
            if not _date_block_ok(value, day, mem.hidden):
                raise RuntimeError(f"date value for {answer!r} is not the field-slice pack")
            rows.append({"value": value, "a": answer, "month": month, "day": day, "year": year, "name": name})
        print(f"date encode {name} {i + 1}/{len(name_list)} n={len(rows)} {time.time() - t0:.0f}s", flush=True)
    return rows


def _sample_dates(n: int, seed: int, *, ban_years=(), ban_days=(), exclude=()) -> list:
    """Dates in the generator's range. Covers every allowed month, day, and year it can."""
    from tidal.llm_memory_data import _MONTHS

    rng = __import__("random").Random(int(seed))
    ban_years = set(ban_years)
    ban_days = set(ban_days)
    exclude = set(exclude)
    months = list(_MONTHS)
    days = [day for day in range(1, 28) if day not in ban_days]
    years = [year for year in range(1970, 2020) if year not in ban_years]
    if not days or not years:
        raise ValueError("date sample removed every day or year")
    chosen = []
    seen = set()

    def add(month, day, year):
        answer = f"{month} {day}, {year}"
        if answer in seen or answer in exclude:
            return False
        seen.add(answer)
        chosen.append(answer)
        return True

    for i, year in enumerate(years):
        add(months[i % 12], days[i % len(days)], year)
    for i, day in enumerate(days):
        add(months[(i * 3) % 12], day, years[(i * 5) % len(years)])
    for i, month in enumerate(months):
        add(month, days[(i * 5) % len(days)], years[(i * 7) % len(years)])
    bag = [f"{month} {day}, {year}" for month in months for day in days for year in years]
    rng.shuffle(bag)
    for answer in bag:
        if len(chosen) >= int(n):
            break
        if answer in seen or answer in exclude:
            continue
        seen.add(answer)
        chosen.append(answer)
    if len(chosen) < int(n):
        raise RuntimeError(f"only {len(chosen)} dates available, wanted {n}")
    return chosen[: int(n)]


def _dates_matching(n: int, seed: int, *, years=(), days=(), skip_years=(), skip_days=()) -> list:
    """Random generator dates whose year or day is restricted to the given sets."""
    from tidal.llm_memory_data import _MONTHS

    rng = __import__("random").Random(int(seed))
    year_set = set(years) if years else set(range(1970, 2020))
    day_set = set(days) if days else set(range(1, 28))
    year_set -= set(skip_years)
    day_set -= set(skip_days)
    bag = [
        f"{month} {day}, {year}"
        for month in _MONTHS
        for day in sorted(day_set)
        for year in sorted(year_set)
    ]
    rng.shuffle(bag)
    if len(bag) < int(n):
        raise RuntimeError(f"only {len(bag)} dates match, wanted {n}")
    return bag[: int(n)]


def _arrays(rows):
    x = np.stack([row["value"] for row in rows]).astype(np.float64)
    months = np.array([row["month"] for row in rows], dtype=np.int64)
    days = np.array([row["day"] for row in rows], dtype=np.int64)
    years = np.array([row["year"] for row in rows], dtype=np.int64)
    return x, months, days, years


def fit_date_readout(mem, *, n_dates: int = 120, alpha: float = 100.0, seed: int = 0) -> dict:
    """Ridge month, day digits, and year digits from the date value pack.

    Train-split names only. Writes ``date_on``, ``date_mu``, ``date_sd``,
    ``date_month_w``, ``date_month_b``, ``date_digit_w``, and ``date_digit_b``.
    The MLP and the phone readout stay where they are. A one-digit day must
    land on an exact zero tens block, which is how this checks that
    ``encode_value`` used the field slices.

    Before that fit, a smaller ridge is trained with six years and four days
    removed, and scored on those held-out fields. Its weights are not the
    ones saved. The saved fit covers the generator's full day and year range.
    """
    from tidal.llm_memory_data import names, session

    if int(n_dates) < 50:
        raise ValueError("n_dates must be >= 50")
    if float(alpha) < 0:
        raise ValueError("alpha must be >= 0")
    held_years = (1971, 1984, 1993, 2002, 2008, 2016)
    held_days = (7, 13, 16, 24)
    train_names = list(names("train"))
    # Recombination check: same pool, fewer names, held-out years and days.
    recomb_dates = _sample_dates(96, seed + 1, ban_years=held_years, ban_days=held_days)
    recomb_rows = _encode_dates(mem, train_names[:8], recomb_dates)
    rx, rm, rd, ry = _arrays(recomb_rows)
    _write_date_readout(mem, rx, rm, rd, ry, alpha)
    recomb_train = _score_date_rows(mem, [(row["value"], row["a"]) for row in recomb_rows])
    year_dates = _dates_matching(24, seed + 2, years=held_years, skip_days=held_days)
    day_dates = _dates_matching(16, seed + 4, days=held_days, skip_years=held_years)
    test_names = list(names("test"))[:4]
    year_rows = _encode_dates(mem, test_names, year_dates)
    year_seen_rows = _encode_dates(mem, train_names[:4], year_dates)
    day_rows = _encode_dates(mem, test_names, day_dates)
    recomb = {
        "train": {"hits": recomb_train["hits"], "n": recomb_train["n"]},
        "held_year_testname": _score_date_rows(mem, [(r["value"], r["a"]) for r in year_rows]),
        "held_year_seenname": _score_date_rows(mem, [(r["value"], r["a"]) for r in year_seen_rows]),
        "held_day": _score_date_rows(mem, [(r["value"], r["a"]) for r in day_rows]),
        "n_year_dates": len(year_dates),
        "n_day_dates": len(day_dates),
        "held_years": list(held_years),
        "held_days": list(held_days),
    }
    # Saved fit: every train name, full day and year range.
    full_dates = _sample_dates(int(n_dates), seed)
    full_rows = _encode_dates(mem, train_names, full_dates)
    x, months, days, years = _arrays(full_rows)
    _write_date_readout(mem, x, months, days, years, alpha)
    train = _score_date_rows(mem, [(row["value"], row["a"]) for row in full_rows])
    held_dates = _sample_dates(40, seed + 5, exclude=set(full_dates))
    held_rows = _encode_dates(mem, list(names("test"))[:8], held_dates)
    held = _score_date_rows(mem, [(row["value"], row["a"]) for row in held_rows])
    grid_rows = []
    for distract in (0, 4, 16, 32):
        rng = __import__("random").Random(5000 + distract)
        for _ in range(48):
            turns = session(rng, "test", distract)
            facts = [turn for turn in turns if turn["kind"] == "fact"]
            target = rng.choice(facts)
            if target["rel"] != "birthday":
                continue
            value = mem.encode_value(target["text"], target["a"])
            grid_rows.append((value, target["a"]))
    grid = _score_date_rows(mem, grid_rows)
    return {
        "train_exact": train["hits"],
        "n": train["n"],
        "n_dates": int(n_dates),
        "n_names": len(train_names),
        "alpha": float(alpha),
        "train_misses": train["misses"],
        "heldout_names": held,
        "grid": {"hits": grid["hits"], "n": grid["n"], "misses": grid["misses"]},
        "recombination": recomb,
    }


def main(argv=None) -> None:
    import torch

    from tidal.llm_memory import DEFAULT_MODEL, LLMMemory, build_tiny_lm

    p = argparse.ArgumentParser(description="Fit residual inject on a frozen causal LM (CPU ok).")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face causal LM. Ignored with --tiny.")
    p.add_argument("--tiny", action="store_true", help="Random 4-layer Llama and a byte tokenizer. No download.")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--neg-p", type=float, default=0.2, dest="neg_p")
    p.add_argument("--n-cache", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--hid", type=int, default=None, help="MLP width. Default: the LM hidden size.")
    p.add_argument("--out", default="models/llm_inject.pt")
    p.add_argument(
        "--phone-readout",
        action="store_true",
        help="Fit the linear phone readout on the train split and save it. Does not train the MLP.",
    )
    p.add_argument("--phone-numbers", type=int, default=120, dest="phone_numbers")
    p.add_argument("--phone-alpha", type=float, default=100.0, dest="phone_alpha")
    p.add_argument(
        "--date-readout",
        action="store_true",
        help="Fit the linear birthday readout on the train split and save it. Does not train the MLP.",
    )
    p.add_argument("--date-samples", type=int, default=120, dest="date_samples")
    p.add_argument("--date-alpha", type=float, default=100.0, dest="date_alpha")
    args = p.parse_args(argv)

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    if args.phone_readout and args.date_readout:
        raise SystemExit("pass only one of --phone-readout and --date-readout")
    if args.date_readout:
        if args.tiny:
            raise SystemExit("--date-readout does not run on --tiny")
        mem = LLMMemory(model_name=args.model, layer=args.layer, inject_hidden=args.hid, dtype=torch.float32)
        frozen = {
            key: tensor.detach().cpu().clone()
            for key, tensor in mem.inject.state_dict().items()
            if key not in {
                "date_on",
                "date_mu",
                "date_sd",
                "date_month_w",
                "date_month_b",
                "date_digit_w",
                "date_digit_b",
            }
        }
        info = fit_date_readout(mem, n_dates=args.date_samples, alpha=args.date_alpha, seed=args.seed)
        for key, tensor in frozen.items():
            if not torch.equal(tensor, mem.inject.state_dict()[key].detach().cpu()):
                raise SystemExit(f"date readout changed {key}")
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        mem.save_inject(out)
        print({"saved": str(out), **info}, flush=True)
        mem.close()
        return
    if args.phone_readout:
        if args.tiny:
            raise SystemExit("--phone-readout does not run on --tiny")
        mem = LLMMemory(model_name=args.model, layer=args.layer, inject_hidden=args.hid, dtype=torch.float32)
        info = fit_phone_readout(mem, n_numbers=args.phone_numbers, alpha=args.phone_alpha, seed=args.seed)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        mem.save_inject(out)
        print({"saved": str(out), **info}, flush=True)
        mem.close()
        return
    if args.tiny:
        model, tok = build_tiny_lm(seed=args.seed)
        mem = LLMMemory(model, tok, layer=args.layer, inject_hidden=args.hid)
    else:
        mem = LLMMemory(model_name=args.model, layer=args.layer, inject_hidden=args.hid)
    facts = sample_facts(args.n_cache, seed=args.seed)
    log = fit_inject(
        mem,
        facts,
        steps=args.steps,
        bs=args.bs,
        lr=args.lr,
        neg_p=args.neg_p,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "inject": mem.inject.state_dict(),
            "layer": mem.layer,
            "hidden": mem.hidden,
            "log": log,
            "args": vars(args),
        },
        out,
    )
    print("saved", out, flush=True)
    mem.close()


if __name__ == "__main__":
    main()
