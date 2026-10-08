"""全双工 / 非单线程 control plane (phase-2 scaffold).

tidal is the always-on CONTROL model: every tick (default 100 ms) it reads every input stream in parallel and emits a
control decision. CONTENT (what to say) comes from an external generator (LLM/TTS) running in parallel; tidal never
waits for it. The two sides talk only through the small message types below.

  inputs per tick:  text/time events (real), her OWN output stream (speaking state + her own events, fed back),
                    audio frame features (stub interface), vision frame features (stub interface), content events
  outputs per tick: action in {silent, start, keep, yield, continue, backchannel, initiate}, which buffered event to
                    address, and content requests / TTS stop signals for the content side.

Implemented: incremental event featurizer (identical to tidal.features_g), streaming event encoder (phase-2 VAP GRU,
O(1)/event), tick feature builder, tick GRU controller, addressing pointer, message types. Audio/vision front ends are
Phase 3: a real streaming audio front end (tidal.audio_fe, AudioFrontEnd below) turns raw 2-channel PCM into AudioFrame
(VAD per channel, energy, P(shift), P(backchannel)); vision is still an interface only. The tick policy itself is still
trained on the script simulator (tidal/duplex_sim.py); audio shift/bc scores are passed through plus a simple threshold hint."""
from __future__ import annotations
import math, collections, numpy as np, torch, torch.nn as nn
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from tidal import features_g as FG, vap_targets as VT
from tidal.features import TZ
from tidal.dataset import HEADS

ACTIONS = ["silent", "start", "keep", "yield", "continue", "backchannel", "initiate"]
TICK_S = 0.1

# ------------------------------------------------------------------ message types (control <-> content interface)
@dataclass
class Event:                 # a discrete event on any channel (chat message, danmaku, gift, finished ASR utterance, her own line)
    id: str; ts: float; role: str            # role: 'self' | 'current' | 'other'  (relative, scenario-general)
    speaker: Optional[str] = None; modality: Optional[str] = None
@dataclass
class AudioFrame:            # produced by AudioFrontEnd (tidal.audio_fe) from raw PCM, or by any external VAD
    vad_other: float = 0.0; vad_self: float = 0.0; music: float = 0.0; energy: float = 0.0
    shift: Optional[float] = None; bc: Optional[float] = None   # P(self should take the floor), P(backchannel slot) from audio
@dataclass
class VisionFrame:           # stub interface: produced by a future screen/video salience front end
    salience: float = 0.0; scene_change: float = 0.0
@dataclass
class SelfState:             # her own output stream, fed back as an input (from the TTS/content side)
    speaking: bool = False; elapsed_s: float = 0.0; remaining_s: float = 0.0
    content_ready: bool = False; pending_request: bool = False; interrupted_remaining_s: float = 0.0
@dataclass
class TickInput:
    t: float; events: List[Event] = field(default_factory=list); audio: Optional[AudioFrame] = None
    vision: Optional[VisionFrame] = None; self_state: SelfState = field(default_factory=SelfState)
    n_participants: Optional[float] = None
    pcm: Optional[tuple] = None   # (other_pcm, self_pcm): 16 kHz float32 audio of this tick; used if the controller has an audio front end
@dataclass
class ControlOut:            # what the content side receives every tick (non-blocking)
    t: float; action: str; probs: dict; address_event: Optional[str] = None
    request_content_for: Optional[str] = None   # ask the generator to prepare a reply to this event (async)
    stop_tts: bool = False                       # yield: stop speaking now, keep the remainder for 'continue'
    audio_shift: Optional[float] = None; audio_bc: Optional[float] = None   # raw audio front-end scores (pass-through)
    hint: Optional[str] = None                   # audio-only hint: 'take_turn' / 'backchannel' (simple thresholds, see AUDIO_THR)

# ------------------------------------------------------------------ incremental event featurizer (== features_g.compute)
class EventFeaturizer:
    def __init__(self, mu, sd):
        self.mu, self.sd = np.asarray(mu, np.float32), np.asarray(sd, np.float32); self.reset()
    def reset(self):
        self.ts = collections.deque(); self.last_t = None; self.last_bot = None; self.prev_self = False; self.n = 0
    def __call__(self, e: Event, n_participants=None):
        h = datetime.fromtimestamp(e.ts, TZ); hr = h.hour + h.minute / 60
        while self.ts and self.ts[0] < e.ts - 600: self.ts.popleft()
        a60 = sum(1 for x in self.ts if x >= e.ts - 60); a600 = len(self.ts)
        base = [float(e.role == "self"), math.log1p(e.ts - self.last_t) if self.last_t is not None else 0.0, float(self.n == 0),
                math.log1p(e.ts - self.last_bot) if self.last_bot is not None else 0.0, float(self.last_bot is None),
                math.log1p(a60), math.log1p(a600), math.sin(2 * math.pi * hr / 24), math.cos(2 * math.pi * hr / 24),
                float(self.n > 0 and self.prev_self)]
        g = np.zeros(len(FG.FEAT_G), np.float32); g[:FG.NB] = (np.array(base, np.float32) - self.mu) / self.sd
        if n_participants is not None: g[FG.NB] = math.log1p(n_participants) / 4.0; g[FG.NB + 1] = 1.0
        col = FG._MODMAP.get(e.modality) if e.modality else None
        if col: g[FG.FEAT_G.index("mod_known")] = 1.0; g[FG.FEAT_G.index(col)] = 1.0
        self.ts.append(e.ts); self.last_t = e.ts; self.n += 1; self.prev_self = e.role == "self"
        if e.role == "self": self.last_bot = e.ts
        return g

# ------------------------------------------------------------------ streaming event encoder + addressing buffer
class EventEncoder:
    """Wraps a phase-2 VAPModel; one GRU step per event. Keeps the last K events with their head outputs so the
    controller can point at 'which event to respond to'."""
    def __init__(self, model, mu, sd, K=16):
        self.m = model.eval(); self.f = EventFeaturizer(mu, sd); self.K = K; self.reset()
    def reset(self):
        self.f.reset(); self.state = None; self.h = torch.zeros(1, self.m.body.hidden_size); self.buf = collections.deque(maxlen=self.K)
        self.last = None
    @torch.no_grad()
    def push(self, e: Event, n_participants=None):
        x = torch.from_numpy(self.f(e, n_participants))[None]
        o, self.state = self.m.step(x, self.state); self.h = self.state[-1]
        ps = torch.softmax(o["y_act"], -1)[0, 0].item(); pa = torch.sigmoid(o["y_addr"])[0, 0].item()
        pe = torch.sigmoid(o["y_eot"])[0, 0].item(); vp = torch.sigmoid(o["vap"])[0].numpy()
        self.last = dict(id=e.id, ts=e.ts, role=e.role, p_speak=ps, p_addr=pa, p_eot=pe,
                         vap_self_0_2=float(vp[VT.idx("self", 0)]), vap_self_2_5=float(vp[VT.idx("self", 1)]))
        self.buf.append(self.last)
    def address(self, now, tau=20.0, window=30.0):
        """which buffered (non-self) event to respond to: argmax p_speak * recency decay."""
        c = [b for b in self.buf if b["role"] != "self" and now - b["ts"] <= window]
        if not c: return None, {}
        sc = {b["id"]: b["p_speak"] * math.exp(-(now - b["ts"]) / tau) for b in c}
        return max(sc, key=sc.get), sc

# ------------------------------------------------------------------ tick features + tick model
TICK_FEATS = (["ev_rate_1s", "ev_rate_5s", "log_since_event", "log_since_self_event", "ev_p_speak", "ev_p_addr", "ev_p_eot",
               "ev_vap_self_0_2", "ev_vap_self_2_5", "ev_last_is_self"]
              + ["audio_avail", "vad_other", "vad_self", "music", "energy", "log_other_run", "log_other_quiet", "log_all_quiet"]
              + ["vision_avail", "salience", "scene_change"]
              + ["speaking", "log_elapsed", "log_remaining", "progress", "content_ready", "pending", "log_interrupted_rem"]
              + ["log_participants", "participants_known"])
N_TICK = len(TICK_FEATS)

class TickFeaturizer:
    def __init__(self): self.reset()
    def reset(self):
        self.ev_t = collections.deque(); self.last_ev = None; self.last_self_ev = None
        self.other_run = 0.0; self.other_quiet = 60.0; self.all_quiet = 60.0
    def __call__(self, ti: TickInput, enc_last: Optional[dict]):
        for e in ti.events:
            self.ev_t.append(e.ts); self.last_ev = e.ts
            if e.role == "self": self.last_self_ev = e.ts
        while self.ev_t and self.ev_t[0] < ti.t - 5: self.ev_t.popleft()
        r1 = sum(1 for x in self.ev_t if x >= ti.t - 1)
        a = ti.audio; s = ti.self_state
        if a is not None:
            on = a.vad_other >= 0.5
            self.other_run = self.other_run + TICK_S if on else 0.0
            self.other_quiet = 0.0 if on else min(60.0, self.other_quiet + TICK_S)
            self.all_quiet = 0.0 if (on or s.speaking) else min(60.0, self.all_quiet + TICK_S)
        elif ti.events or s.speaking: self.all_quiet = 0.0
        else: self.all_quiet = min(60.0, self.all_quiet + TICK_S)
        L = enc_last or {}
        f = [math.log1p(r1), math.log1p(len(self.ev_t)), math.log1p(ti.t - self.last_ev) if self.last_ev is not None else 5.0,
             math.log1p(ti.t - self.last_self_ev) if self.last_self_ev is not None else 5.0,
             L.get("p_speak", 0.0), L.get("p_addr", 0.0), L.get("p_eot", 0.0), L.get("vap_self_0_2", 0.0), L.get("vap_self_2_5", 0.0),
             float(L.get("role") == "self")]
        f += ([1.0, a.vad_other, a.vad_self, a.music, a.energy, math.log1p(self.other_run), math.log1p(self.other_quiet), math.log1p(self.all_quiet)]
              if a is not None else [0.0] * 7 + [math.log1p(self.all_quiet)])
        v = ti.vision; f += [1.0, v.salience, v.scene_change] if v is not None else [0.0, 0.0, 0.0]
        tot = s.elapsed_s + s.remaining_s
        f += [float(s.speaking), math.log1p(s.elapsed_s), math.log1p(s.remaining_s), s.elapsed_s / tot if tot > 0 else 0.0,
              float(s.content_ready), float(s.pending_request), math.log1p(s.interrupted_remaining_s)]
        f += [math.log1p(ti.n_participants) / 4.0, 1.0] if ti.n_participants is not None else [0.0, 0.0]
        return np.asarray(f, np.float32)

class TickModel(nn.Module):
    """[tick features ; projected event-encoder state] -> causal GRU -> action logits. Streaming via step()."""
    def __init__(self, n_tick=N_TICK, enc_dim=192, d=96):
        super().__init__()
        self.ep = nn.Linear(enc_dim, 32); self.inp = nn.Sequential(nn.Linear(n_tick + 32, d), nn.LayerNorm(d), nn.GELU())
        self.gru = nn.GRU(d, d, batch_first=True); self.head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, len(ACTIONS)))
    def forward(self, x, e, state=None):
        h, state = self.gru(self.inp(torch.cat([x, self.ep(e)], -1)), state); return self.head(h), state

class AudioFrontEnd:
    """raw 2-channel PCM (other, self) per tick -> AudioFrame, via the streaming audio encoder (tidal.audio_fe)."""
    def __init__(self, path):
        from tidal.audio_fe import AudioEncoder, AudioStream
        z = torch.load(path, weights_only=False); m = AudioEncoder(); m.load_state_dict(z["state"])
        self.s = AudioStream(m, z["mu"], z["sd"])
    def reset(self): self.s.reset()
    def __call__(self, other_pcm, self_pcm) -> Optional[AudioFrame]:
        o = self.s.push(other_pcm, self_pcm)
        return None if o is None else AudioFrame(o["vad_other"], o["vad_self"], 0.0, o["energy"], o["shift"], o["bc"])

AUDIO_THR = dict(take_turn=0.6, backchannel=0.5)

class DuplexController:
    """Always-on control loop. step() must return well inside one tick; it never blocks on content generation."""
    def __init__(self, tick_model: TickModel, encoder: EventEncoder, audio_fe: Optional[AudioFrontEnd] = None):
        self.tm = tick_model.eval(); self.enc = encoder; self.tf = TickFeaturizer(); self.state = None; self.afe = audio_fe
    def reset(self):
        self.enc.reset(); self.tf.reset(); self.state = None
        if self.afe is not None: self.afe.reset()
    @torch.no_grad()
    def step(self, ti: TickInput) -> ControlOut:
        if self.afe is not None and ti.pcm is not None:
            af = self.afe(*ti.pcm)
            if af is not None: ti.audio = af
        for e in ti.events: self.enc.push(e, ti.n_participants)
        x = torch.from_numpy(self.tf(ti, self.enc.last))[None, None]
        lg, self.state = self.tm(x, self.enc.h[None], self.state)
        p = torch.softmax(lg[0, 0], -1).numpy(); act = ACTIONS[int(p.argmax())]
        addr, _ = self.enc.address(ti.t)
        req = addr if (not ti.self_state.speaking and not ti.self_state.pending_request and not ti.self_state.content_ready
                       and self.enc.last is not None and self.enc.last["role"] != "self" and self.enc.last["p_eot"] > 0.5) else None
        a = ti.audio; hint = None
        if a is not None and a.shift is not None and not ti.self_state.speaking:
            hint = ("take_turn" if a.shift >= AUDIO_THR["take_turn"] and a.vad_other < 0.5 else
                    "backchannel" if a.bc is not None and a.bc >= AUDIO_THR["backchannel"] and a.vad_other >= 0.5 else None)
        return ControlOut(ti.t, act, dict(zip(ACTIONS, p.round(4).tolist())), addr, req, stop_tts=act == "yield",
                          audio_shift=None if a is None else a.shift, audio_bc=None if a is None else a.bc, hint=hint)
