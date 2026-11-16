"""只读空跑：取一帧观测，请求策略推理，对比动作与当前关节角和限位。不创建任何发布者，不会驱动机器人。"""
import sys

import numpy as np
import yaml

sys.path.insert(0, '/app/examples/teleavatar_v2')
import env
import ros2_interface
from openpi_client.websocket_client_policy import WebsocketClientPolicy

ros2_interface.TeleavatarROS2Interface._setup_publishers = lambda self: None  # 不创建任何发布者
cfg = yaml.safe_load(open('/app/arm_config.yml'))['arms']
lower = np.array(cfg['left_arm']['lower'] + cfg['right_arm']['lower'])
upper = np.array(cfg['left_arm']['upper'] + cfg['right_arm']['upper'])
np.set_printoptions(precision=3, suppress=True, linewidth=200)
robot = env.TeleavatarEnvironment(prompt='Pick up the sandbag from the table.', video_decode='va',
                                  interpolate=False, enable_chassis=False)
try:
    obs = robot.get_observation()
    print('obs keys', {k: (v if isinstance(v, str) else getattr(v, 'shape', type(v))) for k, v in obs.items()})
    full = np.asarray(obs['observation/state'])
    idx = [*range(7), *range(8, 15)]
    state = full[idx]
    policy = WebsocketClientPolicy(host='127.0.0.1', port=8000)
    for trial in range(2):
        a = np.asarray(policy.infer(obs)['actions'])
        assert a.ndim == 2 and a.shape[1] == 16 and np.isfinite(a).all(), a.shape
        t = a[:, idx]
        viol = np.count_nonzero((t < lower - 1e-6) | (t > upper + 1e-6), axis=0)
        print(f'--- trial {trial} shape {a.shape}')
        print('state L', state[:7], 'grip', full[7].round(3))
        print('state R', state[7:], 'grip', full[15].round(3))
        print('act0  L', t[0, :7], 'grip', a[0, 7].round(3))
        print('act0  R', t[0, 7:], 'grip', a[0, 15].round(3))
        print('act15 L', t[15, :7], 'grip', a[15, 7].round(3))
        print('act15 R', t[15, 7:], 'grip', a[15, 15].round(3))
        print('act29 L', t[-1, :7], 'grip', a[-1, 7].round(3))
        print('act29 R', t[-1, 7:], 'grip', a[-1, 15].round(3))
        print('|act0-state| per joint', np.abs(t[0] - state), 'max', np.abs(t[0] - state).max().round(3))
        print('max step between consecutive actions', np.abs(np.diff(t, axis=0)).max().round(4))
        print('limit violations per joint (of %d steps)' % len(t), viol.tolist())
finally:
    robot.close()
