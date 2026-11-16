import sys, io, base64
import numpy as np
sys.path.insert(0, '/app/examples/teleavatar_v2')
import env, ros2_interface
ros2_interface.TeleavatarROS2Interface._setup_publishers = lambda self: None  # 只读，不创建发布者
robot = env.TeleavatarEnvironment(prompt='x', video_decode='va', interpolate=False, enable_chassis=False)
try:
    obs = robot.get_observation()
    full = np.asarray(obs['observation/state'])
    print('STATE', np.round(full[:16], 3).tolist(), file=sys.stderr)
    h = np.asarray(obs['observation/images/head_camera'])[::2, ::2]
    l = np.asarray(obs['observation/images/left_color']); r = np.asarray(obs['observation/images/right_color'])
    wrist = np.concatenate([l, r], axis=1)[:, ::2][::1]
    wrist = wrist[:, :h.shape[1]] if wrist.shape[1] >= h.shape[1] else np.pad(wrist, ((0,0),(0,h.shape[1]-wrist.shape[1]),(0,0)))
    img = np.concatenate([h, wrist], axis=0)
    try:
        from PIL import Image
        b = io.BytesIO(); Image.fromarray(img).save(b, format='JPEG', quality=85); data = b.getvalue()
    except ImportError:
        import cv2
        data = cv2.imencode('.jpg', img[:, :, ::-1])[1].tobytes()
    sys.stdout.write('JPGB64:' + base64.b64encode(data).decode() + '\n')
finally:
    robot.close()
