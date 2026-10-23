"""Phase 4C (shadow only): multi-timescale online adaptation of a frozen event encoder ("continuum memory").

Inspired by Nested Learning / Hope's Continuum Memory System (Behrouz et al., NeurIPS 2025, arXiv:2512.24695) and
Titans' surprise-driven test-time memory (Behrouz et al., arXiv:2501.00663). Levels, fastest -> slowest:
  fast    the backbone's recurrent / KV state (per event, unchanged)
  medium  per-session fast weights: low-rank residual adapter on the encoder output h, updated online after every
          matured target with Titans-style momentum + surprise-scaled step + forgetting (weight decay toward 0)
  slow    per-group adapter (same form), updated every C_slow matured targets with the accumulated gradient
          (CMS eq. 71: chunked update), persists across that group's sessions
  global  per-deployment (one "streamer"/persona) adapter, updated every C_glob targets, tiny step
  frozen  base model weights (all heads included)
h_adapt = h + sum_levels B_l (A_l h) + b_l ; h' = h_adapt + episodic_readout(h_adapt) ; heads(h') -> logits. B_l, b_l start at 0.
Episodic readout is always on (tidal/episodic.py): one soft top-k over fact slots, ranked by raw cosine, added in full.
An empty store adds zeros, so a fresh Continuum matches the static model. There is no enable flag.
Facts are explicit (identity + key/value from this frozen encoder, or a positional byte code when the value is a string or token list). Replay does not store raw events.
Natural-language questions use ask / query. A flat row keeps the 512-d sentence vector and is ranked by sentence cosine plus a relation logit (tidal/question_surface.py). When that surface string has selected people, ranking uses only their facts. The cue-nearest narrowing stays on the missing-relation check. Two matched names both stay unless one is only in a dismissed clause, only one possessively governs the live cue, or a focus restrictor governs a proper subset. That path returns no fact when the person who remains lacks the asked relation, or the string names nobody and is not second person and not an indirect hearer description with one first-person fact in the store. A same-script edit of one character is that stored person only when the dialogue log names them and nobody else at that distance (tidal/dialogue_state.py). A fact write is logged and does not license the edit. An empty log leaves name resolution unchanged. After an utterance, you / 你 / 您 and a hearer-role cue name the hearer note_turn stored, and he / she / they / 他 / 她 / ta name the one person on the newest conversational turn who is neither speaker nor hearer. No utterance and no hearer leaves that pronoun on the ranker. A sparse name/relation row has no sentence vector and is ranked by the contrastive head (tidal/question_head.py), which does not abstain. Neither ranker is added into heads, and replay does not call them.
Learning signal: the self-supervised VAP targets only (no labels). Strictly causal: target bin k of event i is used only
once the stream clock has passed ts_i + hi_k (bin end); each (event, bin) is used exactly once.
No live decisions: this module only replays logged streams and reports metrics."""
import heapq, json, math, os, sys, time, numpy as np, pandas as pd, torch, torch.nn.functional as F
from tidal import vap_targets as VT
from tidal.dialogue_state import DialogueState
from tidal.episodic import EpisodicStore, cosine_scores, decode_span, decode_tokens, is_structured_value
from tidal.question_head import QuestionHead, install_matching, question_scores
from tidal.question_surface import ask_people, limit_scores, matched_fact_rows, surface_abstain, surface_scores
BIN_END = np.repeat(np.array([hi for _, hi in VT.BINS], float)[None], len(VT.CHANNELS), 0).reshape(-1)  # [NV] channel-major
_GROUP_OMIT = object()

class Level:
    def __init__(self, d, rank, lr, mom, forget, every, surprise=True, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.A = torch.randn(rank, d, generator=g) / math.sqrt(d)   # fixed random down-projection (key map)
        self.B = torch.zeros(d, rank); self.b = torch.zeros(d)
        self.SB = torch.zeros_like(self.B); self.Sb = torch.zeros_like(self.b)
        self.gB = torch.zeros_like(self.B); self.gb = torch.zeros_like(self.b); self.n = 0
        self.lr, self.mom, self.forget, self.every, self.surprise = lr, mom, forget, every, surprise
    def delta(self, h): return (h @ self.A.T) @ self.B.T + self.b
    def accumulate(self, gB, gb, w):
        self.gB += w * gB; self.gb += w * gb; self.n += 1
        if self.n >= self.every:
            # Titans: S_t = mom*S_{t-1} - lr*grad ; M_t = (1-forget)*M_{t-1} + S_t   (grad = chunk mean, CMS eq. 71)
            self.SB = self.mom * self.SB - self.lr * self.gB / self.n; self.Sb = self.mom * self.Sb - self.lr * self.gb / self.n
            self.B = (1 - self.forget) * self.B + self.SB; self.b = (1 - self.forget) * self.b + self.Sb
            self.gB.zero_(); self.gb.zero_(); self.n = 0
    def state(self): return [t.clone() for t in (self.B, self.b, self.SB, self.Sb)]
    def load(self, s): self.B, self.b, self.SB, self.Sb = [t.clone() for t in s]; self.gB.zero_(); self.gb.zero_(); self.n = 0

CFG_DEFAULT = dict(rank=8, med=dict(lr=3e-3, mom=0.9, forget=0.01, every=1), slow=dict(lr=1e-3, mom=0.9, forget=0.0, every=32),
                   glob=dict(lr=3e-4, mom=0.9, forget=0.0, every=256), surprise=True, levels=("med", "slow", "glob"), session_gap=1800.0,
                   # slot hyperparameters (capacity, retrieval). Not an on/off switch.
                   episodic_cfg=dict(capacity=256, soft_k=4, soft_temp=0.05, overwrite_cos=0.9))

# A feature row whose constant direction holds more than half its energy has no
# coordinate contrast. ``fproj`` of that row, bias included, is then one shared
# vector several times larger than ``tproj`` of a unit sentence, and raw cosine
# treats unrelated sentences as the same fact. Half the energy is the cut:
# sparse name/relation rows sit below it, flat rows sit above it. Not a switch.
_DC_ENERGY = 0.5
_TEXT_AXES = {}


def _text_axes(width, ref):
    """Fixed hyperplanes for a sign code of a 512-d text vector. Not a parameter.

    Kept off the encoder modules on purpose: the question-head fingerprint is a
    hash of those weights, and this matrix is not part of that fit.
    """
    key = (int(width), ref.device.type, ref.device.index, str(ref.dtype))
    found = _TEXT_AXES.get(key)
    if found is None:
        raw = np.random.default_rng(0).normal(size=(int(width), 512)).astype(np.float32)
        found = torch.as_tensor(raw, device=ref.device, dtype=ref.dtype)
        _TEXT_AXES[key] = found
    return found


def row_has_no_contrast(feats) -> bool:
    """True when a feature row is constant-heavy and the sentence has to carry the fact.

    Half the energy in the constant direction is the cut. Sparse name/relation
    rows sit below it. Flat rows, and an all-zero row, sit above it. Not a switch.
    """
    row = feats.reshape(-1)
    if not torch.is_tensor(row):
        row = torch.as_tensor(np.asarray(feats, dtype=np.float32)).reshape(-1)
    row = row.detach().float()
    nrm = float(row.norm())
    width = int(row.numel())
    if nrm < 1e-8 or width < 1:
        return True
    align2 = (float(row.sum()) / (nrm * math.sqrt(width))) ** 2
    return align2 > _DC_ENERGY


def _combine_text(model, feats, text):
    """Pre-norm sum of the feature projection and the text projection.

    ``feats`` is ``[1, F]``, ``text`` is ``[1, 512]``. A row with contrast
    (the question head's sparse name/relation codes) returns ``fproj + tproj``
    unchanged. A row with no contrast drops the constant feature offset and
    the feature bias, and adds a fixed sign code of the text vector scaled to
    the larger of the leftover feature ripple and ``tproj``. The code is what
    keeps two flat rows from agreeing after the constant is gone.
    """
    if not row_has_no_contrast(feats):
        return model.fproj(feats) + model.tproj(text)
    row = feats.reshape(-1)
    nrm = float(row.detach().float().norm()) if torch.is_tensor(row) else float(np.linalg.norm(np.asarray(row)))
    if nrm < 1e-8:
        centered = torch.zeros_like(row)
    else:
        centered = row - row.mean()
    fp = F.linear(centered.reshape(1, -1), model.fproj.weight, bias=None)
    tp = model.tproj(text)
    code = torch.sign(text @ _text_axes(fp.shape[-1], text).T)
    fn = fp.norm(dim=-1, keepdim=True)
    tn = tp.norm(dim=-1, keepdim=True)
    cn = code.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    anchor = torch.maximum(fn, tn)
    return fp + tp + code * (anchor / cn)


class Continuum:
    def __init__(self, model, cfg):
        self.m = model; self.cfg = cfg; d = model.vap[0].in_features; self.d = d
        mk = lambda name, seed: Level(d, cfg["rank"], seed=seed, surprise=cfg["surprise"], **cfg[name])
        self.glob = mk("glob", 3); self.slow_by_group = {}; self.med_by_group = {}; self.mk = mk; self.ema = None
        ec = dict(cfg.get("episodic_cfg") or {})
        self.episodic = EpisodicStore(
            capacity=int(ec.get("capacity", 256)), dk=d, dv=d,
            overwrite_cos=float(ec.get("overwrite_cos", 0.9)),
            soft_k=int(ec.get("soft_k", 4)), soft_temp=float(ec.get("soft_temp", 0.05)),
            surprise_thr=ec.get("surprise_thr"),
        )
        self._mem_group = None   # set by replay; None reads every slot (global + grouped)
        # Question head is a separate ranker for ask/query. Construction must
        # not advance the global RNG (tests seed the model, then sample h).
        # Zero last layer until a fit for this encoder is installed. heads()
        # never reads it. There is no enable flag.
        rng_state = torch.random.get_rng_state()
        try:
            self.question_head = QuestionHead(d)
        finally:
            torch.random.set_rng_state(rng_state)
        self.question_head.eval()
        self.question_head.requires_grad_(False)
        install_matching(self)
        # Sentence vectors for flat rows, keyed by the encoder-state bytes.
        # ask uses them. recall does not. Not a flag.
        self._sentence_by_state = {}
        self._surface_by_state = {}
        self._sentence_conflicts = 0
        # Per-group dialogue log. Facts stay in the store. clear() drops both.
        self._dialogues = {}
        _store_clear = self.episodic.clear

        def _clear_store_and_dialogue():
            _store_clear()
            self._dialogues.clear()

        self.episodic.clear = _clear_store_and_dialogue
        for p in model.parameters(): p.requires_grad_(False)
    def active(self, group):
        lv = []
        if "glob" in self.cfg["levels"]: lv.append(self.glob)
        if "slow" in self.cfg["levels"]:
            if group not in self.slow_by_group: self.slow_by_group[group] = self.mk("slow", 2)
            lv.append(self.slow_by_group[group])
        if "med" in self.cfg["levels"]: lv.append(self.med_by_group[group] if group in self.med_by_group else self.med)
        return lv
    def new_session(self, group=None):
        # medium adapter resets; episodic facts stay until overwrite, eviction, or clear()
        # Mentions for this group drop. Facts stay.
        self.med = self.mk("med", 1)
        if group is not None: self.med_by_group[group] = self.med
        self._dialogues.pop(group, None)
    def _mix(self, h, lv):
        """Adapter residual plus episodic readout. Empty store adds exact zeros."""
        hp = h + sum(l.delta(h) for l in lv) if lv else h
        return hp + self.episodic.readout_torch(hp, group=self._mem_group)
    def heads(self, h, lv):
        hp = self._mix(h, lv)
        o = {k: m(hp) for k, m in self.m.heads.items()}; o["vap"] = self.m.vap(hp); return o
    def embed_event(self, feats, text_emb=None, *, surface=None):
        """Encoder state h for one fact, from this frozen model. No extra LM.

        ``feats`` is one event-feature row (``features.py`` / ``features_g.py``,
        including emotion columns when the model was built with them).
        ``text_emb`` is a 512-d sentence vector from
        ``question_surface.embed_sentences`` and is added through ``model.tproj``
        when that projection exists. ``surface`` is that sentence's text. A flat
        row records it so ``ask`` can abstain. The VAP frontend ``tidal.embed`` is a
        different model and is not this slot.
        A feature row whose energy is mostly constant is not added as that
        constant: the offset would otherwise pin every sentence to one cosine.
        Sparse rows are left on ``fproj + tproj``. No switch.
        """
        m = self.m
        x = torch.as_tensor(np.asarray(feats, dtype=np.float32)).reshape(1, -1)
        if x.shape[-1] != m.fproj.in_features:
            raise ValueError(
                f"event features are {x.shape[-1]}-d, model.fproj expects {m.fproj.in_features}. "
                "Pass a features.py / features_g.py row (plus emotion columns if the model has them)."
            )
        training = m.training
        m.eval()
        note = None
        surface_note = None
        try:
            with torch.no_grad():
                if text_emb is None:
                    h = m.step(x)[0]["h"].reshape(-1)
                else:
                    if not hasattr(m, "tproj"):
                        raise ValueError(
                            "text_emb is a 512-d sentence vector and needs the frozen model's tproj "
                            "(the in-repo text pathway). This model has no text projection; "
                            "put the sentence vector into the event features upstream, or pass encoder state h to remember()."
                        )
                    te = torch.as_tensor(np.asarray(text_emb, dtype=np.float32)).reshape(1, -1)
                    if te.shape[-1] != 512:
                        raise ValueError(f"sentence embedding must be 512-d, got {te.shape[-1]}")
                    # Flat rows keep the sentence vector for ask. Sparse rows do not:
                    # those are the synthetic name/relation codes, ranked by the contrastive head.
                    note = np.asarray(text_emb, dtype=np.float64).reshape(-1) if row_has_no_contrast(x) else None
                    surface_note = surface if note is not None else None
                    tok = m.inorm(_combine_text(m, x, te))
                    if m.kind == "gru":
                        y, _ = m.body(tok.unsqueeze(1)); h = y[0, 0]
                    else:
                        h, _ = m.body.step(tok, None); h = h.reshape(-1)
                h = h.detach().cpu().numpy().astype(np.float64, copy=False)
        finally:
            m.train(training)
        if int(h.size) != int(self.d):
            raise RuntimeError(f"encoder state dim {h.size} != continuum d {self.d}")
        if text_emb is not None and note is not None:
            self._note_sentence(h, note, surface=surface_note)
        return h
    def remember(self, key, value, name=None, rel=None, *, kind="fact", surprise=None, group=None):
        """Write one fact. ``value`` is an encoder state, a string, or a token list.

        A string or token list is stored as a positional byte code of width ``d``
        (phones, dates, multi-token spans). Numeric values must already be dim ``d``.
        """
        k = np.asarray(key, np.float64).reshape(-1)
        if is_structured_value(value):
            if k.size != self.d:
                raise ValueError(
                    f"episodic key must be an encoder state of dim {self.d} (got {k.size}). "
                    "Episodic memory does not load another language model."
                )
            kept = self.episodic.write(
                k, value, name, rel, kind=kind, surprise=surprise, group=group, sentence=self._sentence_for(k),
            )
        else:
            v = np.asarray(value, np.float64).reshape(-1)
            if k.size != self.d or v.size != self.d:
                raise ValueError(
                    f"episodic key/value must be encoder states of dim {self.d} (got {k.size}, {v.size}). "
                    "Use embed_event on event features and, when the model has tproj, a 512-d sentence vector. "
                    "Pass a string or a list of strings for a phone, date, or multi-token span. "
                    "Episodic memory does not load another language model."
                )
            kept = self.episodic.write(
                k, v, name, rel, kind=kind, surprise=surprise, group=group, sentence=self._sentence_for(k),
            )
        if kept and isinstance(name, str) and name:
            self._dialogue_for(group).note_write(name)
        return kept
    def remember_event(self, key_feats, value_feats, name=None, rel=None, *, key_text=None, value_text=None,
                       kind="fact", surprise=None, group=None):
        """Encode two feature rows (optional sentence vectors) and write the resulting states."""
        k = self.embed_event(key_feats, key_text); v = self.embed_event(value_feats, value_text)
        return self.remember(k, v, name, rel, kind=kind, surprise=surprise, group=group)
    def _recall_query(self, query, group):
        g = self._mem_group if group is _GROUP_OMIT else group
        q = np.asarray(query, np.float64).reshape(-1)
        if q.size != self.d:
            raise ValueError(f"episodic query must be dim {self.d} (got {q.size})")
        return q, g
    def recall(self, query, *, group=_GROUP_OMIT):
        """String in the winning slot under the current group scope, or None.

        Decodes the slot's positional code. Token lists are ``recall_tokens``.
        Omit ``group`` to use the replay scope (``_mem_group``); pass ``None``
        to search every slot.
        """
        q, g = self._recall_query(query, group)
        return self.episodic.read_span(q, group=g)
    def recall_tokens(self, query, *, group=_GROUP_OMIT):
        """Token list in the winning slot, or None. Same scope rule as ``recall``."""
        q, g = self._recall_query(query, group)
        return self.episodic.read_tokens(q, group=g)
    def _question_visible(self, question, group):
        """Encoder-state question, group scope, visible slots, raw keys. Empty → no keys."""
        q, g = self._recall_query(question, group)
        visible = self.episodic._visible(g)
        if not visible:
            return q, g, visible, None
        keys = np.stack([s.k for _, s in visible], 0)
        return q, g, visible, keys
    def _state_token(self, vec) -> bytes:
        return np.ascontiguousarray(np.asarray(vec, dtype=np.float64).reshape(-1)).tobytes()
    def _note_sentence(self, state, text_emb, surface=None) -> None:
        """Remember the sentence vector that this flat-row state was built from."""
        vec = np.asarray(text_emb, dtype=np.float64).reshape(-1)
        nrm = float(np.linalg.norm(vec))
        if vec.size != 512 or nrm < 1e-12:
            return
        vec = vec / nrm
        key = self._state_token(state)
        prev = self._sentence_by_state.get(key)
        if prev is not None and not np.array_equal(prev, vec):
            self._sentence_conflicts += 1
            return
        self._sentence_by_state[key] = vec
        if isinstance(surface, str) and surface != "":
            self._surface_by_state[key] = surface
    def _sentence_lookup(self, state):
        return self._sentence_by_state.get(self._state_token(state))
    def _surface_lookup(self, state):
        return self._surface_by_state.get(self._state_token(state))
    def _sentence_for(self, state):
        found = self._sentence_lookup(state)
        return None if found is None else found.copy()
    def _slot_surfaces(self, visible):
        """Surface text recorded for each visible slot, or None when the write had none."""
        return [self._surface_lookup(slot.k) for _, slot in visible]
    def _dialogue_key(self, group):
        return self._mem_group if group is _GROUP_OMIT else group
    def _dialogue_for(self, group) -> DialogueState:
        key = self._dialogue_key(group)
        state = self._dialogues.get(key)
        if state is None:
            state = DialogueState()
            self._dialogues[key] = state
        return state
    def _active_names(self, group, surface):
        """Conversational names, excluding an ask recorded from this same string.

        No recorded surface means no license. A missing log is the empty list.
        """
        if not isinstance(surface, str) or surface == "":
            return ()
        state = self._dialogues.get(self._dialogue_key(group))
        if state is None:
            return ()
        return state.conversational(ignore_surface=surface)
    def _record_ask(self, group, surface, names, surfaces) -> None:
        if not isinstance(surface, str) or surface == "":
            return
        state = self._dialogue_for(group)
        active = state.conversational(ignore_surface=surface)
        mentioned = ask_people(surface, names, surfaces, active)
        state.note_ask(mentioned, surface)
    def note_turn(self, text, *, speaker=None, hearer=None, group=_GROUP_OMIT):
        """Record one utterance. Returns the stored people it named.

        Resolves names with the active list as it stands, then opens a turn.
        Speaker and hearer are stored when passed. They are not inferred.
        An empty string records nothing.
        """
        if not isinstance(text, str) or text == "":
            return ()
        g = self._dialogue_key(group)
        visible = self.episodic._visible(g)
        names = [slot.name for _, slot in visible]
        surfaces = [self._surface_lookup(slot.k) for _, slot in visible]
        state = self._dialogue_for(group)
        mentioned = ask_people(text, names, surfaces, state.conversational())
        state.note_utterance(mentioned, speaker=speaker, hearer=hearer, surface=text)
        return tuple(mentioned)
    def rank_question_scores(self, question, *, group=_GROUP_OMIT):
        """Scores ``ask`` / ``query`` use, one per visible slot, or None when empty.

        Flat-row slots carry the sentence vector and are ranked by sentence
        cosine plus the relation logit. People the surface string selected
        as the ask target limit those scores to their facts; other slots
        stay in the vector and cannot win. A dismissed name, a name that
        does not possess the live cue when the other name does, or a name
        outside a focus restrictor is not a target. Cue-nearest is not
        applied here.
        Anything else, including the synthetic name/relation codes, is ranked
        by the contrastive head. ``recall`` does not call this. Abstention is
        applied in ``query``, not here, so a caller can still read the scores
        of a question that returns no fact.
        """
        q, g, visible, keys = self._question_visible(question, group)
        if keys is None:
            return None
        active = self._active_names(g, self._surface_lookup(q))
        logged = self._dialogues.get(self._dialogue_key(g))
        scores, _rank = self._scores_for(q, keys, visible, active, logged)
        return scores
    def _scores_for(self, q, keys, visible, active=(), dialogue=None):
        sent_q = self._sentence_lookup(q)
        if sent_q is not None and all(slot.sentence is not None for _, slot in visible):
            stacked = np.stack([slot.sentence for _, slot in visible], 0)
            rels = [slot.rel for _, slot in visible]
            scores = surface_scores(sent_q, stacked, rels)
            rows = matched_fact_rows(
                self._surface_lookup(q),
                [slot.name for _, slot in visible],
                self._slot_surfaces(visible),
                active,
                dialogue,
            )
            return limit_scores(scores, rows), "sentence"
        # question_scores is the contrastive head. Sparse rows land here.
        return question_scores(self.question_head, q, keys), "question"
    def query(self, question, *, group=_GROUP_OMIT, mode="softtopk"):
        """Soft top-k value for a question encoder-state, or ``(None, info)``.

        Flat-row questions rank by sentence cosine plus a relation logit,
        limited to the ask targets. A stored name, including a nickname or
        pinyin, selects those people. An indirect hearer description selects
        the unique first-person person when the store has exactly one. Two
        stored names both stay unless one is only in a dismissed clause,
        only one possessively governs the live cue, or a focus restrictor
        governs a proper subset. Cue-nearest stays on abstain, and only
        among the names that remain.
        Sparse name/relation questions rank by the contrastive question head
        and are not limited.
        An edit of one character selects that stored person only when the
        dialogue log names them and nobody else at that distance. The log
        is empty unless ``note_turn`` or an earlier ask recorded them. A
        fact write does not. After an utterance, a second-person or hearer
        cue selects the hearer ``note_turn`` stored, and a third-person
        pronoun selects the one person on the newest turn in the window who
        is neither speaker nor hearer. Unset, ambiguous, tied, aged, or
        write-only, that pronoun abstains. No utterance and no hearer leaves
        the pronoun on the ranker. The people appended after the decision
        are still the names ``ask_people`` matched, not the pronoun.
        Either way the value mix is ``EpisodicStore.read`` when a fact is
        returned. A flat-row question whose recorded surface string names a
        person without the asked relation returns ``(None, info)`` with
        ``info["abstain"] == "missing_rel"``. The relation check and the
        exact-sentence cosine use only the selected person's facts. A string
        that names nobody and is not second person, and is not an indirect
        hearer description with exactly one first-person fact, returns
        ``info["abstain"] == "absent_person"``.
        ``info["top"]`` is None either way: no blend of other people's values.
        Sparse rows do not abstain (``info["abstain"] is None``). A question
        with no recorded surface string does not abstain. This does not run
        inside ``heads`` or ``replay``, and it does not write slots. Omit
        ``group`` to use the replay scope; pass ``None`` to search every slot.
        An empty scope returns ``(None, info)`` with ``n == 0`` and
        ``abstain is None``.
        """
        q, g, visible, keys = self._question_visible(question, group)
        if keys is None:
            return None, {
                "n": 0, "maxcos": -1.0, "top": None, "weights": None,
                "rank": "question", "abstain": None,
            }
        surface = self._surface_lookup(q)
        names = [slot.name for _, slot in visible]
        surfaces = self._slot_surfaces(visible)
        active = self._active_names(g, surface)
        logged = self._dialogues.get(self._dialogue_key(g))
        scores, rank = self._scores_for(q, keys, visible, active, logged)
        abstain = None
        if rank == "sentence":
            abstain = surface_abstain(
                self._sentence_lookup(q),
                np.stack([slot.sentence for _, slot in visible], 0),
                [slot.rel for _, slot in visible],
                names,
                query_text=surface,
                surfaces=surfaces,
                active=active,
                dialogue=logged,
            )
        raw_max = float(np.max(cosine_scores(keys, q)))
        if rank == "sentence":
            self._record_ask(g, surface, names, surfaces)
        if abstain:
            return None, {
                "n": len(visible),
                "maxcos": float(np.max(scores)),
                "top": None,
                "weights": None,
                "rank": rank,
                "abstain": abstain,
                "raw_maxcos": raw_max,
            }
        value, info = self.episodic.read_scored(scores, mode=mode, group=g)
        info["rank"] = rank
        info["abstain"] = None
        info["raw_maxcos"] = raw_max
        return value, info
    def ask(self, question, *, group=_GROUP_OMIT):
        """String in the slot ranked first, or None when there is no fact.

        None is an empty scope, or a flat-row question the sentence ranker
        abstains (``query``'s ``info["abstain"]`` is ``"missing_rel"`` or
        ``"absent_person"``). Decodes that slot's positional byte code when a
        fact is returned, not the soft top-k blend (``query`` returns the
        blend). Same group scope as ``recall``. The stored-key path stays
        ``recall``, which is raw cosine and does not abstain. A flat-row
        question is ranked in sentence space, and only among the matched
        people's facts when the surface string selected an ask target. A
        same-script edit is included only when the dialogue log licenses it,
        the same rule as ``query``. A sparse name/relation question is
        ranked by the contrastive head and does not abstain.
        """
        value, _info = self.query(question, group=group, mode="top1")
        if value is None:
            return None
        return decode_span(value)
    def ask_tokens(self, question, *, group=_GROUP_OMIT):
        """Token list in the slot ranked first, or None. Same ranker and abstention as ``ask``."""
        value, _info = self.query(question, group=group, mode="top1")
        if value is None:
            return None
        return decode_tokens(value)
    def update(self, h, v, lv):
        """h [n, d] for matured events, v [n, NV] target with NaN except the newly matured bins.

        The VAP loss sees the same episodic residual as ``heads`` (zeros when the store is empty).
        """
        params = []
        for l in lv: l.B.requires_grad_(True); l.b.requires_grad_(True); params += [l.B, l.b]
        hp = self._mix(h, lv); lo = self.m.vap(hp); mk = ~torch.isnan(v)
        per = F.binary_cross_entropy_with_logits(lo[mk], v[mk], reduction="none")
        loss = per.mean(); grads = torch.autograd.grad(loss, params)
        for l in lv: l.B.requires_grad_(False); l.b.requires_grad_(False)
        s = float(loss.detach()); self.ema = s if self.ema is None else 0.99 * self.ema + 0.01 * s
        w = min(3.0, s / max(self.ema, 1e-6)) if self.cfg["surprise"] else 1.0     # surprise-scaled step (Titans)
        for i, l in enumerate(lv): l.accumulate(grads[2 * i].detach(), grads[2 * i + 1].detach(), w)
        return s

def replay(cont, H, V, ts, conv, rows_eval, log_every=None):
    """Replay in global timestamp order; return outputs/costs in the original row order.

    Each group has its own session adapter. Matured bins update the originating session,
    group and shared deployment adapter before the next prediction, even if that group
    has gone quiet. No group's future targets can influence another group's past.
    Episodic readout runs on every step (an empty store adds zero). This loop does not
    write stream events into the slot store; facts enter through ``Continuum.remember``.
    Grouped slots are visible only while that group (or a global slot) is the query scope.
    """
    n = len(H); out_vap = torch.zeros(n, VT.NV)
    Ht = torch.from_numpy(H); Vt = torch.from_numpy(V); cost = np.zeros(n)
    head_out = {k: torch.zeros(n, m[-1].out_features) for k, m in cont.m.heads.items()}
    order = np.argsort(ts, kind="stable")
    pending = []; row_state = {}; sessions = {}; last_ts = {}
    nb = len(VT.BINS)
    for i in order:
        t0 = time.perf_counter(); now = ts[i]
        # Apply updates at their actual maturity times, rather than batching by the
        # next arriving event. Another group's traffic must not change local cadence.
        while pending and pending[0][0] <= now:
            due = pending[0][0]; matured = {}
            while pending and pending[0][0] == due:
                _, r, k = heapq.heappop(pending)
                session, lv, remaining = row_state[r]
                batch = matured.setdefault(session, (lv, {}))[1]
                cols = np.arange(k, VT.NV, nb)
                batch.setdefault(r, []).extend(cols[np.isfinite(V[r, cols])].tolist())
                remaining -= 1
                if remaining: row_state[r] = (session, lv, remaining)
                else: del row_state[r]
            for session, (lv, batch) in matured.items():
                cont._mem_group = session[0]
                idx = list(batch); vv = torch.full((len(idx), VT.NV), float("nan"))
                for j, r in enumerate(idx): vv[j, batch[r]] = Vt[r, batch[r]]
                cont.update(Ht[idx], vv, lv)
        group = conv[i]
        if group not in last_ts or now - last_ts[group] > cont.cfg["session_gap"]:
            cont.new_session(group); sessions[group] = sessions.get(group, -1) + 1
        last_ts[group] = now; lv = cont.active(group); cont._mem_group = group
        with torch.no_grad():
            o = cont.heads(Ht[i:i + 1], lv); out_vap[i] = o["vap"][0]
            for k in head_out: head_out[k][i] = o[k][0]
        bins = [k for k in range(nb) if np.isfinite(V[i, k::nb]).any()]
        if bins and lv:
            row_state[i] = ((group, sessions[group]), lv, len(bins))
            for k in bins: heapq.heappush(pending, (now + VT.BINS[k][1], i, k))
        cost[i] = time.perf_counter() - t0
    return out_vap.numpy(), {k: v.numpy() for k, v in head_out.items()}, cost


def encode_rows(model, D, rows, conv_mask):
    """frozen base: encoder output h at each row (causal <=64-event window, same as evaluation in phase 3)."""
    from tidal.vap import conv_rows, batch
    from tidal.seqdata import eval_windows
    convs = conv_rows(D["meta"], conv_mask); W = eval_windows(convs, rows); out = []
    with torch.no_grad():
        for b in range(0, len(W), 512):
            Fm, _, _, valid = batch(D, W[b:b + 512]); out.append(model(torch.from_numpy(Fm), torch.from_numpy(valid))["h"][:, -1].numpy())
    return np.concatenate(out)

# ----------------------------------------------------------------------------------------------------------------------
GRID = [dict(levels=lv, surprise=sp, med=dict(lr=lr, mom=0.9, forget=0.01, every=1))
        for lv in (("med",), ("med", "slow"), ("med", "slow", "glob")) for sp in (True, False) for lr in (1e-3, 3e-3, 1e-2)]

def _cfg(over):
    c = json.loads(json.dumps(CFG_DEFAULT)); c.update({k: v for k, v in over.items() if k != "med"})
    if "med" in over: c["med"] = over["med"]
    c["levels"] = tuple(c["levels"]); return c

def _rowloss(vl, V):
    m = ~np.isnan(V); l = np.where(m, np.logaddexp(0, vl) - np.nan_to_num(V) * vl, 0.0); return l.sum(1) / np.maximum(m.sum(1), 1), m.any(1)

def _bdelta(la, lb, blocks, n_boot=1000):
    from tidal.metrics import ci
    ub, inv = np.unique(blocks, return_inverse=True); rng = np.random.default_rng(0)
    sa, sb, n = np.bincount(inv, la), np.bincount(inv, lb), np.bincount(inv); out = []
    for _ in range(n_boot):
        k = np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub)); out.append(((k * sa).sum() - (k * sb).sum()) / (k * n).sum())
    return dict(adapted=float(la.mean()), static=float(lb.mean()), d=float(la.mean() - lb.mean()), ci=ci(np.array(out)), n=int(len(la)))

def stream_set(D, mask, max_events=None, seed=0):
    """all rows of the selected conversations, in (conv, ts) order (the meta table is already sorted that way)."""
    meta = D["meta"]; rows = np.flatnonzero(mask)
    if max_events and len(rows) > max_events:
        convs = meta.conv.to_numpy()[rows]; u = np.random.default_rng(seed).permutation(np.unique(convs)); keep = []; tot = 0
        cnt = pd.Series(convs).value_counts()
        for c in u:
            if tot >= max_events: break
            keep.append(c); tot += cnt[c]
        rows = rows[np.isin(convs, keep)]
    return rows


def _test(model, cfg, R, mode, split, conv, ts, D, boot):
    from tidal.eval3 import probs, prim
    from tidal.dataset import HEADS
    from tidal.model import HEAD_DIMS
    from tidal.metrics import paired_delta, best_threshold
    rows, H, V, sv, sh = R["test"]; c = Continuum(model, _cfg(cfg)); av, ah, cost = replay(c, H, V, ts[rows], conv[rows], None)
    la, ok = _rowloss(av, V); ls, _ = _rowloss(sv, V)
    blocks = np.array([f"{a}:{int(b // 3600)}" for a, b in zip(conv[rows], ts[rows])])
    pos = np.zeros(len(rows), int); starts = np.r_[0, np.flatnonzero(conv[rows][1:] != conv[rows][:-1]) + 1]
    for a, b in zip(starts, list(starts[1:]) + [len(rows)]): pos[a:b] = np.arange(b - a)
    T = dict(selected=cfg, cost_ms=dict(p50=float(np.percentile(cost, 50) * 1e3), p95=float(np.percentile(cost, 95) * 1e3), mean=float(cost.mean() * 1e3)))
    if mode == "real":
        ev = {"unseen_groups(test_group)": split[rows] == "test_group", "seen_groups(test_time)": split[rows] == "test_time"}
    else:
        ev = {"unseen_streams(pub_test)": np.ones(len(rows), bool)}
    for nm, em in ev.items():
        s = em & ok; T[nm] = _bdelta(la[s], ls[s], blocks[s], boot)
        for N in (16, 64):
            s2 = s & (pos < N); T[f"{nm}_first{N}"] = _bdelta(la[s2], ls[s2], blocks[s2], boot) if s2.sum() > 20 else None
    # drift / stability: Δ by position bucket in long streams + adapter norm
    edges = [0, 64, 256, 1024, 4096, 16384, 10 ** 9]; T["by_position"] = []
    evr = np.isin(split[rows], ["test_time", "test_group"]) if mode == "real" else np.ones(len(rows), bool)
    for a_, b_ in zip(edges[:-1], edges[1:]):
        s = ok & evr & (pos >= a_) & (pos < b_)
        if s.sum() > 50: T["by_position"].append(dict(pos=[a_, b_], **_bdelta(la[s], ls[s], blocks[s], 300)))
    T["adapter_norms"] = dict(glob=float(c.glob.B.norm()), slow_max=float(max([l.B.norm() for l in c.slow_by_group.values()], default=torch.zeros(())).item() if c.slow_by_group else 0.0))
    # forgetting: re-score the evaluation rows with the FINAL adapter states (after the whole replay), no updates
    if mode == "real":
        fin = []
        with torch.no_grad():
            Ht = torch.from_numpy(H)
            for i in range(len(rows)):
                lv = []
                if "glob" in cfg["levels"]: lv.append(c.glob)
                if "slow" in cfg["levels"] and conv[rows][i] in c.slow_by_group: lv.append(c.slow_by_group[conv[rows][i]])
                c._mem_group = conv[rows][i]
                fin.append(c.heads(Ht[i:i + 1], lv)["vap"][0].numpy())
        lf, _ = _rowloss(np.array(fin), V); s = ok & (split[rows] == "test_time")
        T["forgetting_seen_final_state_vs_static"] = _bdelta(lf[s], ls[s], blocks[s], boot)
        # heads on real held-out rows (labels never used for adaptation)
        Y = D["Y"][rows]; T["heads"] = {}
        for hi, h in enumerate(HEADS):
            b = HEAD_DIMS[h] == 1; y = Y[:, hi]; lab = ~np.isnan(y)
            pa = probs({h: ah[h]}, h); pb = probs({h: sh[h]}, h); va = lab & (split[rows] == "val")
            for spn in ("test_time", "test_group"):
                te = lab & (split[rows] == spn)
                if te.sum() < 5 or (b and not (0 < y[te].sum() < te.sum())): continue
                thr_a = best_threshold(y[va], pa[va]) if b else .5; thr_b = best_threshold(y[va], pb[va]) if b else .5
                T["heads"][f"{h}|{spn}"] = dict(adapted=prim(y[te], pa[te], b), static=prim(y[te], pb[te], b),
                                                delta=paired_delta(y[te], pa[te], pb[te], thr_a, thr_b, blocks[te], "binary" if b else "multi", n_boot=boot))
    return T

def run_eval(mode="real", base_tags=None, boot=1000, out_path=None, log=print):
    from tidal import vap as VP
    from tidal.eval3 import probs, prim
    from tidal.dataset import HEADS
    from tidal.model import HEAD_DIMS
    from tidal.metrics import paired_delta, best_threshold
    torch.set_num_threads(int(os.environ.get("TIDAL_THREADS", 2)))
    D = VP.load("p3"); meta = D["meta"]; src = meta.source.to_numpy(); split = meta.split.to_numpy(); conv = meta.conv.to_numpy(); ts = meta.ts.to_numpy(float)
    if mode == "real":
        allm = src == "real"; sets = {"val": allm, "test": allm}
    else:
        pubs = ["pub_tg", "pub_irc", "pub_twitch", "pub_candor", "pub_aishell4"]
        sets = {"val": np.isin(src, pubs) & (split == "pub_val"), "test": np.isin(src, pubs) & (split == "pub_test")}
    res = dict(mode=mode, base=base_tags, grid=[], note="shadow replay only; VAP loss per event predicted before any of its targets matured")
    for tag in base_tags:
        model, ck = VP.load_model(tag)
        R = {}
        for sname, sm in sets.items():
            rows = stream_set(D, sm, max_events=None if mode == "real" else 60000, seed=0)
            H = encode_rows(model, D, rows, sm); V = D["V"][rows]
            static = Continuum(model, _cfg(dict(levels=()))); sv, sh, _ = replay(static, H, V, ts[rows], conv[rows], None)
            R[sname] = (rows, H, V, sv, sh)
        # ---- pre-registered selection on the validation streams: lowest mean VAP loss (val rows only for real)
        rows, H, V, sv, _ = R["val"]; vm = (split[rows] == "val") if mode == "real" else np.ones(len(rows), bool)
        ls, ok = _rowloss(sv, V); best = None
        for g in GRID:
            c = Continuum(model, _cfg(g)); av, _, cost = replay(c, H, V, ts[rows], conv[rows], None); la, _ = _rowloss(av, V)
            s = vm & ok; d = float(la[s].mean() - ls[s].mean()); res["grid"].append(dict(tag=tag, cfg=g, val_d_vap=d)); log(f"[{tag}] grid {g} val Δvap={d:+.5f}")
            if best is None or d < best[0]: best = (d, g)
        full = min([g for g in res["grid"] if g["tag"] == tag and len(g["cfg"]["levels"]) == 3], key=lambda g: g["val_d_vap"])["cfg"]
        TT = {}
        for cname, cfg in (("selected", best[1]), ("full_cms", full)):
            log(f"[{tag}] {cname} {cfg}")
            TT[cname] = _test(model, cfg, R, mode, split, conv, ts, D, boot)
        T = TT
        res[tag] = T
        for cname in T: log(cname + " " + json.dumps({k: v for k, v in T[cname].items() if k not in ("by_position", "heads")}, default=float)[:1500])
    json.dump(res, open(out_path or f"reports/continuum_{mode}_phase4.json", "w"), indent=1, default=float)
    return res

if __name__ == "__main__":
    mode = sys.argv[1]; tags = sys.argv[2].split(",")
    run_eval(mode, tags, out_path=sys.argv[3] if len(sys.argv) > 3 else None)
