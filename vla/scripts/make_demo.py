"""生成 VLA 演示视频:按指令跑闭环,把相机视角 + 动作 + 成功标记拼成网格 MP4。

用法:
    .venv/bin/python scripts/make_demo.py --ckpt runs/bc_v1/best.pt \
        --instrs 0-11 --episodes-per-instr 1 --seed 5000 --out media/vla_demo.mp4

依赖 sim/env_vla.py(VLAEnv) 与 vla/infer.py(VLAPolicy),接口见 specs/VLA-TASK-SPEC.md。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_range(text):
    out = []
    for part in str(text).split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


FONT_CANDIDATES = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/arphic/uming.ttc',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
]


def _font(size):
    from PIL import ImageFont
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_tile(img, title, sub, ok=None):
    """把 128x128 相机图放大到 256x256 并叠字（中文用系统 CJK 字体）。"""
    from PIL import Image, ImageDraw
    im = Image.fromarray(np.asarray(img, dtype=np.uint8)).resize((256, 256), Image.BILINEAR)
    canvas = Image.new('RGB', (256, 304), (16, 16, 18))
    canvas.paste(im, (0, 34))
    dr = ImageDraw.Draw(canvas)
    colour = (120, 255, 140) if ok is True else (255, 120, 120) if ok is False else (230, 230, 230)
    dr.text((6, 6), title, fill=colour, font=_font(17))
    dr.text((6, 288), sub, fill=(200, 200, 200), font=_font(14))
    return canvas


def make_grid(tiles, cols):
    from PIL import Image
    rows = (len(tiles) + cols - 1) // cols
    w, h = tiles[0].size
    grid = Image.new('RGB', (cols * w, rows * h), (10, 10, 12))
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % cols) * w, int(i / cols) * h))
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--instrs', default='0-11', help='例如 0-11 或 1,3,8')
    ap.add_argument('--episodes-per-instr', type=int, default=1)
    ap.add_argument('--seed', type=int, default=5000)
    ap.add_argument('--out', default='media/vla_demo.mp4')
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--image-size', type=int, default=None)
    ap.add_argument('--frames', type=int, default=None, choices=[1, 2],
                    help='每步输入帧数:None=按 ckpt 的 cfg.frames(单帧/两帧自动);2=(t-1,t)')
    ap.add_argument('--max-episodes', type=int, default=6)
    args = ap.parse_args()

    import imageio.v2 as imageio
    from sim.env_vla import VLAEnv
    from vla.infer import VLAPolicy

    _load_kwargs = {}
    if args.frames is not None:
        _load_kwargs['frames'] = args.frames
        if args.frames > 1:
            # 两帧先走 eager,避免单图快速路径误用
            _load_kwargs.update(merge_lora=False, fast_proc=False, cuda_graph=False)
    policy = VLAPolicy.load(args.ckpt, device=args.device, image_size=args.image_size,
                            verbose=True, **_load_kwargs)
    if hasattr(policy, 'frames'):
        print('[demo] frames=%d' % int(policy.frames))
    env = VLAEnv()
    instrs = parse_range(args.instrs)
    texts = getattr(env, 'INSTRUCTIONS', None)

    # 逐回合跑,记录每帧
    episodes = []
    for iid in instrs:
        for k in range(args.episodes_per_instr):
            if len(episodes) >= args.max_episodes:
                break
            seed = args.seed + iid * 100 + k
            try:
                img = env.reset(instr_id=iid, seed=seed)
            except TypeError:
                env.reset(seed=seed)
                if hasattr(env, 'set_instruction'):
                    env.set_instruction(iid)
                img = env.capture() if hasattr(env, 'capture') else env.image
            if hasattr(policy, 'reset_history'):
                policy.reset_history()   # 新回合:清空历史帧队列
            frames, goal, done, info = [], [], False, {}
            t0 = time.time()
            for step in range(50):
                text = texts[iid] if texts else str(iid)
                v, w = policy.act(img, text)
                nxt, done, info = env.step(np.array([v, w], dtype=np.float32))
                frames.append((np.asarray(img), float(v), float(w)))
                img = nxt
                if done:
                    break
            ok = info.get('success')
            episodes.append(dict(instr_id=iid, frames=frames, success=ok, info=info,
                                 latency=(time.time() - t0) / max(1, len(frames))))
            print('instr %2d | %-14s | steps %2d | success=%s | pos=%s' % (
                iid, (texts[iid] if texts else '?'), len(frames), ok, info.get('pos')))

    # 拼网格并写视频
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=args.fps, codec='libx264', quality=8)
    n_frames = max(len(e['frames']) for e in episodes)
    for step in range(n_frames):
        tiles = []
        for ep in episodes:
            if step < len(ep['frames']):
                img, v, w = ep['frames'][step]
                title = (texts[ep['instr_id']] if texts else str(ep['instr_id']))
                sub = 'step %02d  v=%+.2f  w=%+.2f' % (step, v, w)
                ok = None
                if step == n_frames - 1:
                    ok = ep['success']
                    sub += '  ' + ('SUCCESS' if ok else 'FAIL')
                tiles.append(draw_tile(img, title, sub, ok))
            else:
                from PIL import Image
                tiles.append(Image.new('RGB', (256, 300), (10, 10, 12)))
        writer.append_data(np.asarray(make_grid(tiles, args.cols)))
    writer.close()

    summary = {
        'ckpt': args.ckpt,
        'out': str(out),
        'episodes': [{'instr_id': e['instr_id'], 'success': e['success'],
                      'steps': len(e['frames']),
                      'mean_latency_ms': round(e['latency'] * 1000, 1)} for e in episodes],
        'success_rate': float(np.mean([bool(e['success']) for e in episodes])) if episodes else 0.0,
    }
    (out.parent / (out.stem + '.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print('saved', out, 'success_rate', summary['success_rate'])


if __name__ == '__main__':
    main()
