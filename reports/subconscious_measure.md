# Subconscious measurement

This pass does not change retrieval, injection, surprise, valence, retrieval-induced forgetting, polarity, or spelling. The locked banks are untouched. The new banks are an additional measurement, 50 items in each family. Bank sha256 `0f5cbd1c3787f7cf1de93be89ee7b2b9b17a3ad5f5dc430e3d0b735d5fb97228`. Protocol `subconscious-measure-v1`. Model `HuggingFaceTB/SmolLM2-135M`. Commit `e537b5ace7f3077ac6890040f7706ea4a60d33a3`.

## How retrieval feeds injection

Readout on this path is hard top-1. `LLMMemory._rank` calls `EpisodicStore.read` with `mode="top1"`. For that mode `_mix_visible` returns `s0.v.copy()`, the entire stored value of the winning slot, not a mixture. `EpisodicStore.read` still defaults to `softtopk`; that default is the continuum path, and `generate` does not read `soft_k` or `soft_temp`.

The vector that is added is not the raw value. `_set_delta` passes the value and the winning key through `InjectMLP.forward`: `scaled = value * scale`, `residual = scaled + net(scaled)`, then the polarity gain when the polarity vector is nonzero. One hidden-sized residual comes out. It is multiplied by `confidence * strength * valence_gain`. `retrieval_confidence` returns 1 when the store has fewer than two slots, and 0 on a tie.

`_on_layer` adds that one residual through `_broadcast`. The broadcast reshape is `[1] * (hidden.dim() - 1) + [hidden]`, so the same full vector is added at every position of the layer output. It is not a per-position retrieval and it is not a slice of the vector.

### LLMMemory._rank

```python
    def _rank(self, prompt: str) -> dict:
        """One query. Hard top-1, plus the gap to the runner-up."""
        info = {
            "n": len(self.store),
            "value": None,
            "key": None,
            "top": None,
            "maxcos": None,
            "gap": None,
            "confidence": 0.0,
        }
        if len(self.store) == 0:
            return info
        query = self._query_vector(prompt)
        # Hard top-1. Soft top-4 at T=0.05 matched this on prefix free-gen and
        # trailed it on natural questions; T=0.20 mixed neighbors and dropped
        # N32 questions from 36/48 to 19/48. Measured in docs/llm_memory.md.
        value, read_info = self.store.read(query, mode="top1")
        slots = self.store.slots
        keys = np.stack([np.asarray(slot.k, dtype=np.float64) for slot in slots], 0)
        scores = cosine_scores(keys, np.asarray(query, dtype=np.float64))
        top_i = int(np.argmax(scores))
        gap = None
        if scores.size >= 2:
            rest = float(np.max(np.delete(scores, top_i)))
            gap = float(scores[top_i] - rest)
        top = read_info.get("top") if isinstance(read_info, dict) else None
        n = len(slots)
        winner = int(read_info.get("top_i", top_i)) if isinstance(read_info, dict) else top_i
        info.update(
            {
                "n": n,
                "value": value,
                "key": np.asarray(slots[winner].k, dtype=np.float32).copy(),
                "top": top,
                "maxcos": float(scores[top_i]),
                "gap": gap,
                "confidence": retrieval_confidence(n, gap, DEFAULT_GAP_TAU),
                "strength": float(slots[winner].strength),
                "answer": slots[winner].answer,
                "valence": float(slots[winner].valence),
            }
        )
        # The strength above is the one this retrieval uses. Forgetting runs
        # after that read. A tie adds nothing and forgets nothing. The key
        # cosines are the slot keys already stored; there is no extra forward.
        forgotten = 0
        if gap is not None and float(gap) > 0.0 and n >= 2:
            forgotten = forget_rivals(slots, winner)
        self.last_rif = {"n": int(forgotten)}
        return info
```

### EpisodicStore._mix_visible

```python
    def _mix_visible(self, visible: list, c: np.ndarray, mode: str) -> Tuple[Optional[np.ndarray], dict]:
        """Soft top-k or top-1 over ``c``, one score per visible slot, in that order.

        ``read`` passes raw cosine. ``read_scored`` passes caller scores. The
        mix itself is the same convex combination of stored values.
        """
        info: dict[str, Any] = {"n": len(visible), "maxcos": -1.0, "top": None, "weights": None}
        if not visible:
            return None, info
        info["maxcos"] = float(c.max())
        top_local = int(np.argmax(c))
        top_i, s0 = visible[top_local]
        info["top"] = (s0.name, s0.rel)
        info["top_i"] = top_i
        info["group"] = s0.group
        if mode == "top1":
            return s0.v.copy(), info
        if mode not in ("softtopk", "softk4", "soft"):
            raise ValueError(f"unknown read mode {mode!r}")
        kk = min(self.soft_k, c.size)
        local = np.argpartition(-c, kk - 1)[:kk]
        local = local[np.argsort(-c[local])]
        tv = c[local]
        temp = max(self.soft_temp, 1e-8)
        w = np.exp((tv - tv.max()) / temp)
        w = w / w.sum()
        info["weights"] = w
        info["idx"] = np.array([visible[int(j)][0] for j in local], dtype=np.int64)
        out = np.zeros_like(visible[0][1].v)
        for j, loc in enumerate(local):
            out = out + w[j] * visible[int(loc)][1].v
        return out, info
```

### EpisodicStore.read

```python
    def read(
        self,
        q: ArrayLike,
        *,
        mode: str = "softtopk",
        group: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], dict]:
        """Value for query ``q``. ``top1`` is argmax raw cosine; ``softtopk`` mixes the top keys.

        Scores are raw cosine (``kn * qn + 1e-12``). The question head is not
        applied. ``group is None`` searches every slot. A group sees its own
        slots plus global slots (``group is None`` on the slot). Returns
        ``(None, info)`` when nothing is visible.
        """
        visible = self._visible(group)
        if not visible:
            return None, {"n": 0, "maxcos": -1.0, "top": None, "weights": None}
        qq = _as1d(q)
        if self.dk is not None and qq.size != self.dk:
            raise ValueError(f"query dim {qq.size} != store dk {self.dk}")
        K = np.stack([s.k for _, s in visible], 0)
        return self._mix_visible(visible, cosine_scores(K, qq), mode)
```

### InjectMLP.forward

```python
    def forward(self, value: torch.Tensor, key: Optional[torch.Tensor] = None) -> torch.Tensor:
        scaled = value * self.scale
        residual = scaled + self.net(scaled)
        if key is None:
            return residual
        # w = 0 is exactly the residual. Skip the multiply on the inference
        # path so a zero vector stays bitwise identical. Training keeps the
        # graph, including at the zero initialization.
        inactive = (not torch.is_grad_enabled()) or (not self.polarity.requires_grad)
        if inactive and float(self.polarity.detach().abs().sum()) == 0.0:
            return residual
        return self.polarity_gain(key) * residual
```

### retrieval_confidence

```python
def retrieval_confidence(n: int, gap: Optional[float], tau: float) -> float:
    """Scale for the residual. One slot is unscaled. A tie adds nothing."""
    if int(n) < 2:
        return 1.0
    if float(tau) <= 0.0:
        raise ValueError("gap_tau must be positive")
    if gap is None:
        return 0.0
    return float(min(1.0, max(0.0, float(gap) / float(tau))))
```

### LLMMemory._set_delta

```python
    def _set_delta(self, rank: dict) -> None:
        value = rank.get("value")
        conf = float(rank.get("confidence") or 0.0)
        strength = float(rank.get("strength", 1.0))
        if value is None or conf <= 0.0 or strength <= 0.0:
            self._delta = None
            return
        composed = self._compose(value, key=rank.get("key"), grad=False)
        # Valence 0 is gain 1. Full confidence and strength 1 then leave the
        # composed residual unmultiplied, matching the replay inject.
        scale = conf * strength * valence_gain_of(float(rank.get("valence", 0.0)))
        if scale == 1.0:
            self._delta = composed
            return
        if scale == 0.0:
            self._delta = None
            return
        self._delta = composed * scale
```

### LLMMemory._on_layer

```python
    def _on_layer(self, _module, _args, output):
        delta = self._delta
        if delta is None:
            self.last_added = None
            return None
        if (not delta.requires_grad) and torch.count_nonzero(delta) == 0:
            self.last_added = None
            return None
        self.last_added = delta.detach()
        return _add_delta(output, delta)
```

### _broadcast

```python
def _broadcast(delta: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    add = delta.to(device=hidden.device, dtype=hidden.dtype).reshape(-1)
    if add.numel() != hidden.shape[-1]:
        raise ValueError(f"inject dim {add.numel()} != hidden {hidden.shape[-1]}")
    shape = [1] * (hidden.dim() - 1) + [add.numel()]
    return add.view(*shape)
```

## Stored key cosines

The question is whether stored keys are still all above 0.99. The encoder distribution is pairwise cosine of `encode_key` on the new prefer and avoid items plus the locked replay held prefer/avoid items and the locked valence held toward/away items. `remember` is then run on the new prefer and avoid items only. `overwrite_only` writes those same encoder keys through `store.write`, so cosine overwrite at 0.9 is visible without the surprise gate.

Encoder vectors: 132.

| min | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.4681 | 0.6177 | 0.6761 | 0.8328 | 0.9160 | 0.9414 | 0.9492 | 0.9600 | 0.9763 | 0.7914 |

Pairs 8646. Fraction > 0.99: 0.0 (0). Fraction > 0.9: 0.27735368956743 (2398). All pairs > 0.99: False.

Histogram: [-1.00, 0.00] 0, [0.00, 0.25] 0, [0.25, 0.50] 15, [0.50, 0.75] 3467, [0.75, 0.90] 2766, [0.90, 0.99] 2398, [0.99, 1.00] 0. Overflow outside [-1, 1]: 0.

After `remember`, attempted 100, `remember` returned true 95 times, slots left 2.

| min | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8774 | 0.8774 | 0.8774 | 0.8774 | 0.8774 | 0.8774 | 0.8774 | 0.8774 | 0.8774 | 0.8774 |

Pairs 1. Fraction > 0.99: 0.0 (0). Fraction > 0.9: 0.0 (0). All pairs > 0.99: False.

Histogram: [-1.00, 0.00] 0, [0.00, 0.25] 0, [0.25, 0.50] 0, [0.50, 0.75] 0, [0.75, 0.90] 1, [0.90, 0.99] 0, [0.99, 1.00] 0. Overflow outside [-1, 1]: 0.

Overwrite-only write, attempted 100, accepted 100, slots left 2.

| min | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8774 | 0.8774 | 0.8774 | 0.8774 | 0.8774 | 0.8774 | 0.8774 | 0.8774 | 0.8774 | 0.8774 |

Pairs 1. Fraction > 0.99: 0.0 (0). Fraction > 0.9: 0.0 (0). All pairs > 0.99: False.

Histogram: [-1.00, 0.00] 0, [0.00, 0.25] 0, [0.25, 0.50] 0, [0.50, 0.75] 0, [0.75, 0.90] 1, [0.90, 0.99] 0, [0.99, 1.00] 0. Overflow outside [-1, 1]: 0.

Encoder pairs within each source:

| source | vectors | pairs | min | median | max | frac > 0.99 |
| --- | --- | --- | --- | --- | --- | --- |
| measure-avoid | 50 | 1225 | 0.8806 | 0.9401 | 0.9763 | 0.0000 |
| measure-prefer | 50 | 1225 | 0.8798 | 0.9311 | 0.9703 | 0.0000 |
| replay-held-avoid | 8 | 28 | 0.5921 | 0.7461 | 0.8340 | 0.0000 |
| replay-held-prefer | 8 | 28 | 0.6512 | 0.6977 | 0.7806 | 0.0000 |
| valence-held-away | 8 | 28 | 0.6479 | 0.7516 | 0.8190 | 0.0000 |
| valence-held-toward | 8 | 28 | 0.6328 | 0.7693 | 0.8610 | 0.0000 |

## Expanded banks

Each family has 50 items, disjoint from the existing data-module surfaces at token Jaccard under 0.50. The locked n=8 banks were not edited and were not the rows scored here. Rates use the existing scorers. A 95% Wilson interval is on the rate, not on the mean margin.

### Prefer and avoid

Prefer stores “favors ANSWER” and scores good minus bad against an empty store. Avoid stores “dodges ANSWER”; the good continuation is the other word and the bad continuation is the stored word. `stored_down` counts items whose stored-side log-probability fell.

Prefer margin up: 50/50 = 1.0000 (95% Wilson 0.9287–1.0000). Mean delta 10.2672.

Avoid margin up: 39/50 = 0.7800 (95% Wilson 0.6476–0.8725). Mean delta 0.2907.

Avoid stored-side down: 42/50 = 0.8400 (95% Wilson 0.7149–0.9166). Mean stored lift -0.4041.

### Valence

`score_signed` repeats the shipped step 3 times. Toward wants a positive margin change. Away wants a negative one.

Toward positive: 45/50 = 0.9000 (95% Wilson 0.7864–0.9565). Mean delta 1.9167.

Away negative: 48/50 = 0.9600 (95% Wilson 0.8654–0.9890). Mean delta -5.0418.

### Surprise

Thirty-two seed notes fill the window. Quantile 0 sets the bar at the minimum seed NLL. A low is rejected only when its answer-token NLL is strictly below every seed. Highs are the same frames with the answer replaced.

Seeds written 32/32. Bar 2.8281. Low mean surprise 2.8766. High mean surprise 25.0159.

Lows not written: 27/50 = 0.5400 (95% Wilson 0.4040–0.6703).

Highs written: 50/50 = 1.0000 (95% Wilson 0.9287–1.0000).

### Retrieval-induced forgetting

Fifty quartets, each a target, three same-relation competitors, and one outsider. `score_quartets` compares one top-1 with a strength-1 control. `geometry_ok` requires five slots, the target first, three competitors, one outsider, a strict win, a hit on every row, winner and outsider strength 1, every competitor below 1, and `forgotten` equal to 3.

`score_quartets` zips the written facts onto the surviving slots. When cosine overwrite leaves fewer than five slots, later roles drop out of that zip. The rows that remain are not a competitor or outsider sample. Drop rates are reported only when `geometry_ok` is nonzero.

Geometry ok: 0/50 = 0.0000 (95% Wilson 0.0000–0.0713).

| slots left | quartets |
| --- | --- |
| 1 | 27 |
| 2 | 18 |
| 3 | 5 |
| 4 | 0 |
| 5 | 0 |

Quartets with at least two slots: max key cosine n=23, min 0.8274, median 0.8923, max 0.9276. A one-slot quartet records max key cosine 0. That 0 is the scorer’s n<2 sentinel.

The zip emitted target 50, competitor 28, outsider 0. Those counts are not a rate denominator.

Forgotten counter by quartet: 0: 46, 1: 4.

Competitor and outsider rates are not estimated. No quartet kept the five-slot geometry the scorer requires.

## Implicit influence

The subconscious score is the unweighted mean of three paired Cohen’s d_z values: tone, choice, and preference, inject minus no-memory. d_z is the mean paired difference divided by its sample standard deviation. Explicit phone and birthday exact-match rates are not in that mean.

The prompt never contains the name, the relation, the stored word, or the foil. Inject writes one slot, so confidence is 1 even if the query cosine is low; the mean query cosine is reported. No-memory is an empty store. In-context prepends the fact text and a newline. Simple RAG is BGE CLS top-1 over that family’s 50 fact lines. The query is the uninformative prompt, so a hit is not guaranteed.

Subconscious score (mean d_z, inject versus no-memory): 1.3424.

| family | d_z inject−none | d_z context−none | d_z RAG−none | d_z inject−context | d_z inject−RAG | inject Δ>0 | RAG hit | mean maxcos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tone | 1.1024 | 0.0907 | 0.1881 | 0.8585 | 0.9434 | 43/50 = 0.8600 (95% Wilson 0.7381–0.9305) | 8/50 = 0.1600 (95% Wilson 0.0834–0.2851) | 0.4908 |
| choice | 1.5510 | 0.2728 | 0.0884 | 1.6353 | 1.7932 | 50/50 = 1.0000 (95% Wilson 0.9287–1.0000) | 4/50 = 0.0800 (95% Wilson 0.0315–0.1884) | 0.5112 |
| preference | 1.3738 | 2.3705 | 0.2200 | 0.3603 | 1.3495 | 48/50 = 0.9600 (95% Wilson 0.8654–0.9890) | 3/50 = 0.0600 (95% Wilson 0.0206–0.1622) | 0.5111 |

tone: n=50, inject−none mean 1.2035, sd 1.0918, d_z 1.1024, rate 43/50 = 0.8600 (95% Wilson 0.7381–0.9305). Mean margins nomem -0.0109, inject 1.1927, context 0.0730, RAG 0.0487. Inject confidence mean 1.0000. RAG unique lines 8.
choice: n=50, inject−none mean 8.6682, sd 5.5888, d_z 1.5510, rate 50/50 = 1.0000 (95% Wilson 0.9287–1.0000). Mean margins nomem 0.0498, inject 8.7179, context 0.6653, RAG 0.2012. Inject confidence mean 1.0000. RAG unique lines 4.
preference: n=50, inject−none mean 5.7841, sd 4.2101, d_z 1.3738, rate 48/50 = 0.9600 (95% Wilson 0.8654–0.9890). Mean margins nomem -0.0833, inject 5.7007, context 4.0424, RAG 0.2027. Inject confidence mean 1.0000. RAG unique lines 3.

## Added compute per token

Wall clock on this CPU, median over the repeats below. An armed forward minus an empty forward at lengths 16, 32, and 64 changes sign and stays under a millisecond. That difference is timer noise, not a measured speedup and not a measured residual cost. The structural cost inside the noise is one 576-d add at every position (576 scalar adds) and one InjectMLP (665280 parameters) once per prompt. Retrieval is not inside that forward.

The added time that clears the noise is one `_rank` (the query forward and the top-1 read) plus one `InjectMLP` evaluation. Those seconds are also divided by the continuation-token count of one implicit pair (good plus bad). The inject total uses that amortized retrieval. It does not add the noisy residual mean.

In-context is the longer forward minus the short forward. The table below keeps the undivided extra seconds, the added prompt tokens, and the continuation-token count. The summary row averages the per-continuation-token extras of one tone, one choice, and one preference item. RAG adds one uncached BGE top-1 over the 50 tone lines on top of that longer-prompt cost.

Repeats 7, warmup 2. Signed mean residual extra per token across 16/32/64: -0.000003 s. The sign flips with length.

| piece | seconds | per continuation token |
| --- | --- | --- |
| residual add | within timer noise | within timer noise |
| `_rank` | 0.027927 | 0.013964 |
| InjectMLP | 0.000077 | 0.000039 |
| inject total (`_rank` + InjectMLP) |  | 0.014002 |
| in-context extra forward |  | 0.001779 |
| BGE corpus + query | 0.061686 | 0.030843 |
| RAG total (uncached BGE + in-context extra) |  | 0.032622 |

The JSON field `inject_added_per_continuation_token_s` (0.014000) adds the signed residual mean onto the inject total. The table leaves the residual out.

| length | empty s | armed s | extra s | extra per token s |
| --- | --- | --- | --- | --- |
| 16 | 0.026105 | 0.026637 | 0.000532 | 0.000033 |
| 32 | 0.034060 | 0.033062 | -0.000998 | -0.000031 |
| 64 | 0.047331 | 0.046719 | -0.000612 | -0.000010 |

| family | short tokens | long tokens | added prompt tokens | continuation tokens | extra s | extra per added prompt token s | extra per continuation token s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tone | 16 | 27 | 11 | 2 | 0.004854 | 0.000441 | 0.002427 |
| choice | 17 | 34 | 17 | 5 | 0.006952 | 0.000409 | 0.001390 |
| preference | 16 | 30 | 14 | 4 | 0.006072 | 0.000434 | 0.001518 |

## Explicit recall

Phone and birthday exact string match. These rates are not in the subconscious score. `generate` copies the stored answer when the prompt is completing that relation and the winning slot stores it, so a rate of 1.0 on a one-slot store is the spelling shortcut, not evidence that the residual spelled the string.

phone: 50/50 = 1.0000 (95% Wilson 0.9287–1.0000).
birthday: 50/50 = 1.0000 (95% Wilson 0.9287–1.0000).

## Citations

Choice is the associative-priming analogue. Meyer and Schvaneveldt (1971), Experiment II, found that “same” responses were 117 ± 18 ms faster for associated word pairs than for unassociated pairs, F(1, 11) = 42.6, p < .001. The margin here is a mean log-probability, not a reaction time, and the associate is a residual rather than a string on the screen.

Tulving and Schacter (1990) describe priming as a nonconscious form of human memory, separate from other memory systems. Perceptual priming, in their account, is pre-semantic. Conceptual priming seems based on semantic memory. The implicit prompts do not ask for the stored fact.

Preference is the mere-exposure analogue. Zajonc (1968) hypothesized that mere repeated exposure of a person to a stimulus enhances attitude toward it, where “mere” means the stimulus is accessible to perception. This probe stores the word once and does not show it in the prompt. That is one encoding, not Zajonc’s exposure-frequency schedule. No F statistic from that monograph is used here.

Tone is the evaluative-priming analogue. Fazio, Sanbonmatsu, Powell, and Kardes (1986) reported that an attitude with a strong object–evaluation association is activated automatically on mere presentation of the object, and that a following adjective is facilitated when its valence matches, provided the association is strong. The event word in this probe is not presented. Only the residual is. This is an analogue, not a replication.

Meyer, D. E., & Schvaneveldt, R. W. (1971). Facilitation in recognizing pairs of words: Evidence of a dependence between retrieval operations. Journal of Experimental Psychology, 90(2), 227–234. https://doi.org/10.1037/h0031564

Tulving, E., & Schacter, D. L. (1990). Priming and human memory systems. Science, 247(4940), 301–306. https://doi.org/10.1126/science.2296719

Zajonc, R. B. (1968). Attitudinal effects of mere exposure. Journal of Personality and Social Psychology, 9(2, Pt. 2), 1–27. https://doi.org/10.1037/h0025848

Fazio, R. H., Sanbonmatsu, D. M., Powell, M. C., & Kardes, F. R. (1986). On the automatic activation of attitudes. Journal of Personality and Social Psychology, 50(2), 229–238. https://doi.org/10.1037/0022-3514.50.2.229

The retrieval-induced-forgetting rates use the same sources already cited for that mechanism in `tidal/subconscious_rif_retry_data.py`: Anderson, Bjork, and Bjork (1994), https://doi.org/10.1037/0278-7393.20.5.1063, and Anderson (2003), https://doi.org/10.1016/j.jml.2003.08.006. Those papers measure category-cued recall after retrieval practice. The rate in this report is a log-probability margin.


Wall clock 206.7 s. Load 2.9 s.

