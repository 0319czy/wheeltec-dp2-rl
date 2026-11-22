"""环境自检：GPU / MuJoCo EGL / 依赖 / HF 缓存。在 /home/czy/倒立摆/wheeltec_vla 下运行。

    .venv/bin/python scripts/check_env.py
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ok = True


def line(name, value, good=None):
    global ok
    flag = ''
    if good is not None:
        if good:
            flag = ' OK'
        else:
            flag = ' FAIL'
            ok = False
    print(f'[{name:24s}] {value}{flag}')


print('=' * 64)
print('DP2 VLA 环境自检  根目录:', ROOT)
print('=' * 64)

# python
line('python', sys.version.split()[0])

# torch
try:
    import torch
    line('torch', f'{torch.__version__}  cuda_build={torch.version.cuda}')
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        line('cuda', f'{torch.cuda.get_device_name(0)}  free={free/2**30:.1f}G/{total/2**30:.1f}G', True)
        x = torch.randn(256, 256, device='cuda')
        line('cuda matmul', f'{(x @ x).sum().item():.1f}', True)
    else:
        line('cuda', '不可用（需要 cu128 版 torch）', False)
except Exception as e:  # noqa: BLE001
    line('torch', f'导入失败: {e}', False)

# mujoco + EGL 渲染
try:
    import mujoco
    line('mujoco', mujoco.__version__)
    os.environ.setdefault('MUJOCO_GL', 'egl')
    m = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/lowlevel/dp2_nopole3d.xml'))
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, 128, 128)
    mujoco.mj_step(m, d)
    r.update_scene(d)
    img = r.render()
    line('mujoco EGL render', f'{img.shape} {img.dtype}', img.shape == (128, 128, 3))
except Exception as e:  # noqa: BLE001
    line('mujoco EGL render', f'失败: {e}', False)

# 依赖版本
for mod, attr in [('transformers', '__version__'), ('stable_baselines3', '__version__'),
                  ('accelerate', '__version__'), ('peft', '__version__'),
                  ('PIL', '__version__'), ('numpy', '__version__')]:
    try:
        m = __import__(mod)
        line(mod, getattr(m, attr, '?'))
    except Exception as e:  # noqa: BLE001
        line(mod, f'缺失: {e}', False)

# faster-whisper
try:
    import faster_whisper
    line('faster_whisper', getattr(faster_whisper, '__version__', '?'))
except Exception as e:  # noqa: BLE001
    line('faster_whisper', f'缺失: {e}', False)

# HF 缓存
for name in ['SmolVLM-256M-Instruct', 'whisper']:
    p = ROOT / 'assets/hf' / name
    if p.exists():
        files = list(p.rglob('*'))
        size = sum(f.stat().st_size for f in files if f.is_file())
        line(f'hf/{name}', f'{len(files)} 个文件, {size/2**20:.0f} MiB')
    else:
        line(f'hf/{name}', '未下载')

# 磁盘 / 内存
du = shutil.disk_usage(ROOT)
line('disk free (/home)', f'{du.free/2**30:.0f} GiB')
try:
    import psutil  # type: ignore
    vm = psutil.virtual_memory()
    line('ram', f'可用 {vm.available/2**30:.1f} GiB / 总 {vm.total/2**30:.1f} GiB')
except Exception:  # noqa: BLE001
    with open('/proc/meminfo') as f:
        info = dict(l.split(':', 1) for l in f if ':' in l)
    line('ram', f"MemAvailable {int(info['MemAvailable'].split()[0])/2**20:.1f} GiB")

print('=' * 64)
print('结果:', 'ALL OK' if ok else '存在问题（见 FAIL 行）')
sys.exit(0 if ok else 1)
