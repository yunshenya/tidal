"""Unattended task-head pipeline: real data, freeze, evaluation, ONNX and full tests."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from tidal.config import ROOT
from tidal.public_data.manifest import DATA, DATASETS
from tidal.head_data import irc_data, overlap_data, human_data, read_annotations
from tidal.head_train import digest, fit_task
from tidal.task_heads import CLASSES

DIRECTORY = ROOT/'models'/'task_heads'
REPORT = ROOT/'reports'/'task_heads.json'


def signature(annotations):
    sources = ['task_heads.py', 'head_data.py', 'head_train.py', 'head_shadow.py', 'head_optimize.py', 'head_annotations.py', 'duplex.py', 'labels.py', 'audio_spec.py', 'audio_train.py']
    hashes = {p: digest(ROOT/'tidal'/p) for p in sources}
    hashes['protocol'] = digest(ROOT/'reports'/'task_heads_protocol.md')
    for name in ('irc_dis', 'magicdata_ms', 'magicdata_en'):
        manifest = DATA/name/'_done.json'
        done = json.loads(manifest.read_text())
        if done['ok'] != done['files'] or done.get('revision') != DATASETS[name]['revision']:
            raise ValueError(f'Incomplete or mismatched dataset {name}')
        # Hash source bytes, including test bytes without parsing test labels.
        for p in sorted((DATA/name).rglob('*')):
            if p.is_file() and '.cache' not in p.parts and p.suffix in ('.parquet', '.txt', '.wav'):
                hashes[str(p.relative_to(DATA))] = digest(p)
    for name in ('audio_md', 'audio_en'):
        for p in sorted((DATA/'proc'/name).glob('*.npz')):
            hashes[str(p.relative_to(DATA))] = digest(p)
    if annotations.exists(): hashes['annotations'] = digest(annotations)
    hashes['python'] = sys.version
    import numpy, torch, onnxruntime
    hashes['libraries'] = dict(numpy=numpy.__version__, torch=torch.__version__, onnxruntime=onnxruntime.__version__)
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def can_resume(report, sig):
    if report.get('signature') != sig or not report.get('artifacts'): return False
    for name, sha in report['artifacts'].items():
        path = DIRECTORY/name
        if not path.is_file() or digest(path) != sha: return False
    return True


def markdown(result):
    lines = ['# 任务头落地报告', '', '**代码、训练和影子接口已落地；策略适当性和语义意图仍受人工标注数量约束。所有新输出只供影子评估。**', '',
             '| 任务 | 状态 | 训练 / 验证 / 测试 | 选型 | 测试准确率 |', '|---|---|---:|---|---:|']
    for task, r in result['tasks'].items():
        sizes = '/'.join(str(r.get(k, 0)) for k in ('train_n', 'val_n', 'test_n'))
        acc = f"{r['selected']['accuracy']:.4f}" if 'selected' in r else '未测'
        lines.append(f"| {task} | {r['status']} | {sizes} | {r.get('champion', '未训练')} | {acc} |")
    lines += ['', '## 监督与能力边界', '',
              '- 主模型仍为六个头。`y_act_behavior` 保留历史行为，`y_act_policy` 只接受有来源的人工标签；`y_next_gap` 明确现有重检头实际预测的是下一事件时间。',
              '- `action_policy`、`eot_semantic`、`recheck_policy`、`overlap_intent` 均有人工JSONL校验和训练入口；缺标注时不创建随机权重模型，不用自然接话或合成数据冒充助手适当性。',
              '- `reply_to` 以已经可用的草稿为条件，输出最近32条消息或null。用IRC人工图训练的是自然人回复匹配代理；不证明机器人草稿或中文群聊迁移。无链接行屏蔽；超出缓冲目标明确映射为“无可见目标”。',
              '- `overlap_outcome` 在自己仍讲话、对方开口200ms时，使用已经完整收到的1s双通道音频；预测后续短附和或话轮接管的弱行为标签。它没有 stop_request / correction 真值，也不代表应该停TTS。',
              '- 所有选型、归一化、温度校准在训练/验证完成；冻结权重SHA后才构建测试。结果无论正负均报告。',
              '- 音频说话人隔离，但沿用了前轮其他任务已查看的外部英文集合；不是全新未暴露的数据。按对话的区间仅有五个测试块、两名说话人，证据有限。',
              '- 新策略与目标输出接入 `ControlOut.task_shadow`；与原动作、回复目标、内容请求和TTS停止信号分开。异常写入 `task_shadow_error`，不改变原控制结果。',
              '- 特征/推理只依赖NumPy和ONNX Runtime；哈希文本是轻量词面特征，不应视作成熟语义理解。', '', '## 实际训练结果', '']
    for task, r in result['tasks'].items():
        if r['status'] != 'trained_evaluated': continue
        lines += [f"### {task}", '', f"验证选择：{r['champion']}；神经模型通过预定门槛：{r['neural_passed_gate']}。", '',
                  f"选中模型：`{json.dumps(r['selected'], ensure_ascii=False)}`", '',
                  f"基线：`{json.dumps(r['baselines'], ensure_ascii=False)}`", '',
                  f"按块改善区间：`{json.dumps(r['deltas'], ensure_ascii=False)}`", '',
                  f"ONNX最大概率差：{r['export']['onnx_parity_max_error']:.2e}；参数：{r['export']['parameters']}。", '']
    checks = result['runtime_checks']; stats = checks['runtime']
    lines += ['## 独立影子运行时实测', '',
              f"回复特征构建+ONNX的p95 {stats['reply_full_p95_ms']:.3f}ms，新进程峰值RSS {stats['peak_rss_mb']:.1f}MiB。每次迭代更换一个消息和草稿；32个合成候选，不包含VAD/ASR、生成模型或完整控制器。", '',
              f"音频整块/100ms/不规则块最大概率差 {checks['max_chunk_probability_error']:.1e}；未来文本及未来PCM不改变过去评分；音频缓存有界；新进程不导入Torch/pandas/sklearn。", '']
    lines += ['## 使用', '', '```bash', '.venv/bin/python -m tidal.head_optimize',
              '# 有真实人工标注后：', '.venv/bin/python -m tidal.head_optimize --annotations data/head_annotations/judgements.jsonl',
              '```', '', '人工格式、采集命令和影子接入示例见 [任务头使用说明](../docs/task_heads.md)。模型、原文及逐条预测保存在本地忽略目录。', '',
              '数据：[IRC作者卡](https://huggingface.co/datasets/jkkummerfeld/irc_disentangle)、[MUIR访问说明](https://github.com/Eliot-Shen/GroupGPT#MUIR-Dataset)。IRC为CC-BY-4.0；MagicData/MagicHub仅学术研究不可商用；本轮未使用认证token。', '',
              f"验证：完整pytest退出码 {result['checks']['pytest_exit_code']}；模型恢复依赖源码、协议、数据内容和产物校验。", '']
    return '\n'.join(lines)


def run(annotations):
    import torch
    torch.set_num_threads(4)
    from tidal.public_data.download import main as download
    from tidal.audio_train import prep
    download(['irc_dis'], workers=3)
    for name, cache in [('magicdata_ms', 'audio_md'), ('magicdata_en', 'audio_en')]:
        if not (DATA/name/'_done.json').exists(): download([name], workers=4)
        prep(DATA/name, DATA/'proc'/cache)
    sig = signature(annotations)
    result = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    if can_resume(result, sig): print('Resuming validated task-head artifacts', flush=True)
    else:
        DIRECTORY.mkdir(parents=True, exist_ok=True)
        # Remove stale bundles so an untrained/insufficient task cannot serve an old model.
        for path in DIRECTORY.glob('*_bundle.json'): path.unlink()
        tasks = {}
        for task, loader in [('reply_to', irc_data), ('overlap_outcome', overlap_data)]:
            print(f'Training {task}', flush=True)
            tasks[task] = fit_task(task, loader, DIRECTORY, sig)
        annotations_info = {'status': 'data_missing'}
        for task in ('action_policy', 'eot_semantic', 'recheck_policy', 'overlap_intent'):
            if not annotations.exists(): tasks[task] = dict(task=task, status='human_annotations_missing')
            else: tasks[task] = fit_task(task, lambda split, task=task: human_data(annotations, task, split), DIRECTORY, sig)
        if annotations.exists(): _, annotations_info = read_annotations(annotations)
        result = dict(signature=sig, tasks=tasks, annotations=annotations_info, shadow_only=True, automatic_action_enabled=False,
                      artifacts={p.name: digest(p) for p in DIRECTORY.iterdir() if p.is_file() and p.suffix in ('.pt', '.onnx', '.json', '.npz')})
        REPORT.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    from tidal.head_verify import verify
    result['runtime_checks'] = verify(DIRECTORY)
    completed = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT, text=True, capture_output=True)
    (DIRECTORY/'pytest.log').write_text(completed.stdout+completed.stderr)
    if completed.returncode: raise RuntimeError(f'pytest failed; see {DIRECTORY}/pytest.log')
    result['checks'] = dict(pytest_exit_code=0, summary=completed.stdout.splitlines()[-1])
    REPORT.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    path = ROOT/'reports'/'task_heads.md'; path.write_text(markdown(result))
    print(result['checks']['summary'], flush=True); print(path, flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--annotations', type=Path, default=ROOT/'data'/'head_annotations'/'judgements.jsonl')
    args = p.parse_args(); run(args.annotations)
