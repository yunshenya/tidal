"""Optional controller adapter; all new heads are logged independently of actions."""
from collections import deque
import math

from tidal.task_heads import HeadEvent, TaskHeadShadow, OverlapStream, MAX_CONTEXT


class ControllerTaskShadow:
    def __init__(self, directory, reply_shadow=None, self_speaker=''):
        if not isinstance(self_speaker, str): raise ValueError('Expected bot speaker ID')
        self.reply_shadow = reply_shadow; self.self_speaker = self_speaker
        self.model = TaskHeadShadow(directory); self.stream = OverlapStream(self.model); self.reset()

    def reset(self):
        self.history = deque(maxlen=MAX_CONTEXT); self.stream.reset(); self.last_tick = None; self.last_overlap_onset = None

    def observe(self, ti):
        if not math.isfinite(ti.t) or ti.t < 0: raise ValueError('Invalid tick time')
        if self.last_tick is not None and ti.t <= self.last_tick: raise ValueError('Ticks must increase')
        for text, available in ((ti.draft_text, ti.draft_available_at), (ti.incoming_prefix, ti.prefix_available_at)):
            if text is not None and (not isinstance(text, str) or available is None or not math.isfinite(available) or available < 0 or available > ti.t):
                raise ValueError('Text must be available by this tick')
        if ti.incoming_onset is not None and (not math.isfinite(ti.incoming_onset) or ti.incoming_onset < 0 or ti.incoming_onset > ti.t):
            raise ValueError('Incoming onset must already be observed')
        events = [HeadEvent(e.id, e.ts, e.text or '', e.speaker or '', e.role) for e in ti.events]
        if any(e.ts > ti.t for e in events): raise ValueError('Future event')
        from tidal.task_heads import context_at
        context_at(list(self.history)+events, ti.t)  # validate before mutating history
        if ti.pcm is not None:
            from tidal.audio_spec import SR
            durations = [len(c)/SR for c in ti.pcm]
            if len(durations) != 2 or abs(durations[0]-durations[1]) > 1e-9:
                raise ValueError('Controller adapter requires synchronized PCM chunks')
            if self.last_tick is None or abs(ti.t-self.last_tick-durations[0]) > 1e-6:
                self.stream.reset(max(0., ti.t-durations[0]))
            self.stream.push(*ti.pcm)
        elif self.last_tick is not None:
            self.stream.reset(ti.t)  # missing PCM cannot be bridged as continuous audio
        self.history.extend(events); self.last_tick = ti.t
        s = ti.self_state
        incoming_elapsed = max(0., ti.t-ti.incoming_onset) if ti.incoming_onset is not None else 0.
        out = dict(shadow_only=True, automatic_action_enabled=False)
        out['policy'] = self.model.policy(self.history, ti.t, prefix=ti.incoming_prefix or '', speaking=s.speaking,
                                          elapsed_s=s.elapsed_s, remaining_s=s.remaining_s, incoming_elapsed_s=incoming_elapsed)
        out['reply_to'] = (self.reply_shadow or self.model).reply(self.history, ti.draft_text, ti.t, speaker=self.self_speaker)
        out['overlap_outcome'] = None
        if ti.pcm is not None and ti.incoming_onset != self.last_overlap_onset:
            score = self.stream.score_due(ti.t, ti.incoming_onset, ti.t-s.elapsed_s, s.speaking)
            out['overlap_outcome'] = score
            if score is not None: self.last_overlap_onset = ti.incoming_onset
        return out
