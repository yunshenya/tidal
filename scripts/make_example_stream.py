"""Generate a small, fully synthetic event stream for the shadow-mode demo (no LLM, no real data).
usage: python scripts/make_example_stream.py [N_MSGS] > examples/synthetic_stream.jsonl"""
import json, random, sys, time
rng = random.Random(7)
PEOPLE = ["甲", "乙", "丙", "丁"]
LINES = ["今晚谁上号", "我刚到家", "等下", "然后那个", "哈哈哈哈", "[图片]", "[语音]", "[表情包]", "这个版本好难",
         "有人吃了吗", "我跟你说", "明天早八", "笑死", "[视频]", "小潮你觉得呢？", "小潮在吗", "好的", "收到"]
def main(n=200):
    t = time.time() - 6 * 3600
    for i in range(n):
        t += rng.choice([2, 3, 5, 8, 15, 30, 90, 400])
        if rng.random() < 0.08:
            e = dict(msg_id=f"ex{i}", conv="demo-group", conv_type="group", ts=t, role="self", speaker=None, text="嗯嗯，我在～")
        else:
            txt = rng.choice(LINES)
            e = dict(msg_id=f"ex{i}", conv="demo-group", conv_type="group", ts=t, role="other", speaker=rng.choice(PEOPLE), text=txt,
                     addressed=("小潮" in txt), bot_action=rng.choice(["speak"] if "小潮" in txt else ["silent"] * 9 + ["speak"]))
        print(json.dumps(e, ensure_ascii=False))
if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
