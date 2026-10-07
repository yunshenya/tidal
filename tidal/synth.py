"""Synthetic multi-party Chinese group chats via DeepSeek (training-only data).

Prompts contain ONLY abstract scene/style descriptions sampled from hand-written lists below --
never any real chat content. Hard budget caps: MAX_CALLS calls or MAX_CNY estimated spend
(estimated at the most expensive published PEAK rates), whichever comes first.
Key is read from .secrets/deepseek.env and never printed/logged.
"""
import os, json, random, time, hashlib, pathlib, sys, threading, concurrent.futures as cf
import httpx

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data/synth"; OUT.mkdir(parents=True, exist_ok=True)
RAWF = OUT / "raw_generations.jsonl"; USAGE = OUT / "usage.json"
MAX_CALLS = int(os.environ.get("SYNTH_MAX_CALLS", 3000)); MAX_CNY = float(os.environ.get("SYNTH_MAX_CNY", 29.0))
# Conservative (peak, cache-miss) CNY per 1M tokens from api-docs.deepseek.com pricing pages (2026-10).
PRICE_IN, PRICE_IN_HIT, PRICE_OUT = 3.0, 0.10, 9.0

def load_env():
    for line in (ROOT / ".secrets/deepseek.env").read_text().splitlines():
        k, v = line.split("=", 1); os.environ.setdefault(k, v)

TOPICS = ["周末开黑打游戏", "期末考试复习", "追新番讨论剧情", "养猫养狗日常", "做饭翻车", "加班吐槽", "突然降温", "看球赛",
          "二次元手游抽卡", "宿舍生活", "旅行攻略", "新手机选购", "健身减肥", "外卖点什么", "租房找室友", "考研上岸",
          "社团活动筹备", "生日惊喜策划", "电影观后感", "学车考驾照", "游戏版本更新", "画画约稿", "音乐节抢票", "早八起不来",
          "钓鱼", "摄影修图", "装电脑", "小说推荐", "股票基金亏了", "宠物生病", "失眠熬夜", "实习面试", "节日回家", "露营",
          "桌游狼人杀", "做饭食谱分享", "天气暴雨", "猫咪表情包大战", "群友晒图", "群里接龙", "学英语", "相亲吐槽"]
STYLES = ["大家都爱用缩写和网络梗", "消息很短，一句话拆成好几条发", "表情包和[图片]很多", "有人特别话痨、有人只潜水偶尔冒泡",
          "语气轻松爱开玩笑", "偶尔有人发长段文字", "很多“哈哈哈”和语气词", "有人打字慢、经常发半句话后补充", "节奏很快、经常有人插话",
          "节奏慢、消息之间常隔几分钟", "两三个小圈子各聊各的", "有人经常@别人", "有人用引用回复", "夹杂错别字和口语"]
BOT_ROLES = ["机器人很少说话，只在被叫到或被@时回应", "机器人偶尔主动接话，但大部分时候沉默", "机器人比较活跃，但不会每句都接",
             "机器人被叫到时回应，有时回应后再补一句", "机器人在冷场时偶尔开个话题"]
from tidal.config import SYNTH_BOT_NAMES as BOT_NAMES
TIMES = ["深夜", "早上", "中午", "下午", "晚上"]

SCHEMA = """输出一个 JSON 对象，格式：
{"participants": ["A","B",...], "messages": [ {"s": "A" 或 "BOT", "dt": 距上一条消息的秒数(数字, 第一条为0), "text": "消息内容", "tags": [...]} , ...]}
tags 可选值（可多个，也可为空列表）：
- "addr_bot": 这条消息是在跟机器人说话（叫它名字、@它、回复它、或明显在问它）
- "reply_to:N": 引用回复第 N 条消息（N 从 0 开始）
- "unfinished": 这句话没说完，同一个人后面会接着补
- "self_cont": 同一个人在停顿或被别人打断之后，又补发了一条接着自己前面的话
- "interrupt": 别人话还没说完时插话
- "side": 和机器人无关的支线对话（两个人私下聊）
- "eot": 这个人这一轮话说完了（接下来要么别人说，要么隔很久）
要求：
1. 发言者用 A、B、C… 表示，机器人用 BOT；BOT 的消息也要写出来（它说话时机要自然，不要每句都接）。
2. dt 要贴近真实 QQ 群节奏：连发一般 1-8 秒，接话 2-30 秒，冷场可以几分钟甚至十几分钟。
3. 真实群聊风格：短句、拆分发送、口语、表情、[图片]/[表情] 占位、偶尔错别字；不要写成剧本或旁白。
4. 不要出现任何真实人名、手机号、QQ号、网址；人物都是虚构的。
5. 只输出 JSON。"""

def make_prompt(rng):
    n_people = rng.randint(3, 8); n_msgs = rng.randint(45, 75)
    spec = dict(topic=rng.choice(TOPICS), topic2=rng.choice(TOPICS), styles=rng.sample(STYLES, 3), bot=rng.choice(BOT_ROLES),
                bot_name=rng.choice(BOT_NAMES), time=rng.choice(TIMES), n_people=n_people, n_msgs=n_msgs,
                p_addr=rng.choice(["很少有人叫机器人", "偶尔有人叫机器人", "好几个人会和机器人聊"]))
    user = (f"请虚构一段 QQ 群聊天记录。群里有 {n_people} 个真人（虚构）和一个叫“{spec['bot_name']}”的聊天机器人。\n"
            f"时间：{spec['time']}。主要话题：{spec['topic']}，中途可能岔到：{spec['topic2']}。\n"
            f"群聊风格：{'；'.join(spec['styles'])}。\n机器人行为：{spec['bot']}；{spec['p_addr']}。\n"
            f"要包含：被打断后又补充的情况、没说完分几条发的情况、和机器人无关的支线对话、至少一处冷场。\n"
            f"总共约 {n_msgs} 条消息。\n\n{SCHEMA}")
    return spec, user

lock = threading.Lock()
def usage_state():
    if USAGE.exists(): return json.loads(USAGE.read_text())
    return dict(calls=0, ok=0, failed=0, prompt_tokens=0, cache_hit_tokens=0, completion_tokens=0, est_cny=0.0)

def call(client, spec, user, st):
    with lock:
        if st["calls"] >= MAX_CALLS or st["est_cny"] >= MAX_CNY: return None
        st["calls"] += 1
    body = {"model": os.environ["DEEPSEEK_MODEL"], "messages": [
                {"role": "system", "content": "你是一个擅长还原真实中文网络群聊的数据生成器，只输出合法 JSON。"},
                {"role": "user", "content": user}],
            "response_format": {"type": "json_object"}, "thinking": {"type": "disabled"},
            "temperature": 1.1, "max_tokens": 6000}
    try:
        r = client.post("/chat/completions", json=body, timeout=180)
        r.raise_for_status(); j = r.json()
    except Exception as e:
        with lock: st["failed"] += 1
        return {"error": type(e).__name__ + ":" + str(getattr(e, "response", None) and e.response.status_code)}
    u = j.get("usage", {}); hit = u.get("prompt_cache_hit_tokens", 0) or 0
    cost = ((u.get("prompt_tokens", 0) - hit) * PRICE_IN + hit * PRICE_IN_HIT + u.get("completion_tokens", 0) * PRICE_OUT) / 1e6
    with lock:
        st["ok"] += 1; st["prompt_tokens"] += u.get("prompt_tokens", 0); st["cache_hit_tokens"] += hit
        st["completion_tokens"] += u.get("completion_tokens", 0); st["est_cny"] = round(st["est_cny"] + cost, 4)
        USAGE.write_text(json.dumps(st, indent=1))
    return {"spec": spec, "content": j["choices"][0]["message"]["content"], "usage": u, "model": j.get("model")}

def main(n_target, workers=8, seed=None):
    load_env(); st = usage_state()
    rng = random.Random(seed if seed is not None else time.time())
    headers = {"Authorization": "Bearer " + os.environ["DEEPSEEK_API_KEY"]}
    with httpx.Client(base_url=os.environ["DEEPSEEK_BASE_URL"], headers=headers) as client, open(RAWF, "a") as f, \
         cf.ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(call, client, *make_prompt(rng), st) for _ in range(n_target)]
        done = 0
        for fu in cf.as_completed(futs):
            res = fu.result(); done += 1
            if res is None: continue
            with lock:
                f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
            if done % 25 == 0:
                print(time.strftime("%H:%M:%S"), f"done={done} calls={st['calls']} ok={st['ok']} failed={st['failed']} est_cny={st['est_cny']:.2f}", flush=True)
    print("FINAL", json.dumps(st))

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 8)
