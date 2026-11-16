"""带夹爪阈值补丁启动 examples/teleavatar_v2/main.py（仅运行时替换，不改任何文件）。

模型在"张开"阶段给出的夹爪力矩贴近 0（示范为 +2.0 Nm），而夹爪是力控的，任何负力矩空载时都会合到底。
这里对每个夹爪加带回差的阈值：输出低于 CLOSE_BELOW 才闭合（沿用模型给的力矩），高于 OPEN_ABOVE 才重新张开，
张开时一律发 +2.0 Nm（trigger 0.0）。
"""
import logging
import sys

sys.argv = [a for a in sys.argv if not a.startswith("--wrapped=")]
sys.path.insert(0, "/app")

import tyro  # noqa: E402

from examples.teleavatar_v2 import main as _main  # noqa: E402
from examples.teleavatar_v2 import ros2_interface as _ri  # noqa: E402

CLOSE_BELOW = -0.8  # Nm
OPEN_ABOVE = -0.3   # Nm
_closed = [False, False]  # [left, right]
_names = ("left", "right")
_orig_publish_cmd = _ri.TeleavatarROS2Interface._publish_cmd


LATCH = True  # 一旦闭合就锁住到回合结束（示范里夹爪合上后一直保持）
_last_log = [0.0]


def _publish_cmd(self, arm14, grip2):
    import time
    grip2 = grip2.copy()
    now = time.monotonic()
    if now - _last_log[0] >= 0.5:  # 记录模型原始的右夹爪输出，便于排查
        _last_log[0] = now
        logging.info(f"RAW right gripper effort {float(grip2[1]):+.2f} Nm, sent {'CLOSE' if _closed[1] else 'OPEN'}")
    for i in range(2):
        raw = float(grip2[i])
        was = _closed[i]
        if raw < CLOSE_BELOW:
            _closed[i] = True
        elif raw > OPEN_ABOVE and not LATCH:
            _closed[i] = False
        if _closed[i] != was:
            logging.info(f"GRIPPER {_names[i]} -> {'CLOSE' if _closed[i] else 'OPEN'} (model effort {raw:.2f} Nm)")
            logging.info(f"GRIPPER {_names[i]} arm target at switch: {[round(float(x), 3) for x in arm14[7 * i:7 * i + 7]]}")
        if _closed[i]:
            grip2[i] = min(raw, -1.6) if LATCH else raw  # 锁住后保持全力闭合
        else:
            grip2[i] = 2.0
    return _orig_publish_cmd(self, arm14, grip2)


_ri.TeleavatarROS2Interface._publish_cmd = _publish_cmd

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S", force=True)
    logging.info(f"Gripper threshold patch ON: close below {CLOSE_BELOW} Nm, reopen above {OPEN_ABOVE} Nm")
    sys.exit(_main.main(tyro.cli(_main.Args)))
