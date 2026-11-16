"""手势启动门：用模板匹配在头部相机画面里识别 OK 手势（模板是 /home/jhz/workspace/label/ 下的 8 张图）。

做法（全部是传统图像处理，不用神经网络）：
  1. 肤色分割（YCrCb 的 Cr/Cb 范围）得到画面里的皮肤区域，按连通域切出候选块（手、脸）。
  2. 每个候选块按外接框归一化成 SIZE x SIZE 的小图：背景为 0，手的区域为 0.6 + 0.4 x 灰度，保留轮廓和指间的明暗。
  3. 模板同样处理，并预先生成旋转（-40..40 度）和左右镜像的变体。
  4. 候选小图与每个模板变体比轮廓：重合度 IoU 和倒角距离（chamfer，两边轮廓点到对方轮廓的平均距离）。
     先不滑动粗筛出最像的 REFINE 个变体，再用 cv2.matchTemplate 在 +-SLIDE 像素范围内滑动求最佳对位。
     分数 = 0.5 x IoU + 0.5 x (1 - 倒角距离 / 3)，取最高分。分数 >= 阈值算这一帧识别到手势。
  只在画面的 ROI（桌子后方站人的区域）里找，地面的暖色反光不参与。

本文件既被任务脚本 import（放在 --gesture-dir 目录里），也可以单独在本机离线测试：
  /usr/bin/python3 -I gesture_gate.py <模板目录> <图片或 .npy ...>
"""
import glob
import os

import cv2
import numpy as np

SIZE = 64          # 归一化小图边长
INNER = 56         # 手的外接框长边缩放到这么多像素
ANGLES = (-40, -30, -20, -10, 0, 10, 20, 30, 40)
SCALES = (0.88, 1.0, 1.12)  # 模板缩放变体
SLIDE = 6                   # 精匹配时的平移搜索范围（像素）
REFINE = 48                 # 粗筛后做滑动精匹配的变体个数
DT_CAP = 8.0                # 距离图截断值
SKIN_LO = (40, 136, 85)    # Y, Cr, Cb 下限
SKIN_HI = (255, 176, 130)  # 上限（红色沙包的 Cr 高于 176，不会被当成皮肤）
ROI = (0.12, 0.04, 0.88, 0.58)  # 找手势的画面区域 (x0, y0, x1, y1)，占画面宽高的比例：桌子后方、人上半身所在范围
DEFAULT_THRESHOLD = 0.73   # 2026-11-15 现场标定：见 README“手势启动”一节


def skin_mask(rgb):
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    m = cv2.inRange(ycc, SKIN_LO, SKIN_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return m


def _normalize(mask, gray, size=SIZE, inner=INNER):
    """把一个二值块（mask>0）连同灰度裁到外接框，长边缩放到 inner，居中放进 size x size，返回 float32 小图。"""
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    m = (mask[y0:y1, x0:x1] > 0).astype(np.float32)
    g = gray[y0:y1, x0:x1].astype(np.float32)
    inside = m > 0
    lo, hi = np.percentile(g[inside], (5, 95))
    g = np.clip((g - lo) / max(hi - lo, 1.0), 0, 1)
    patch = m * (0.6 + 0.4 * g)
    h, w = patch.shape
    s = inner / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    patch = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = np.zeros((size, size), np.float32)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = patch
    return out


def _largest_component(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == k).astype(np.uint8) * 255


def _features(patch):
    """小图 -> (轮廓内部, 轮廓线, 到轮廓线的距离图, 轮廓点数, 面积)，都是 float32。"""
    m = patch > 0.3
    e = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    dt = np.minimum(cv2.distanceTransform((~e).astype(np.uint8), cv2.DIST_L2, 3), DT_CAP)
    m, e = m.astype(np.float32), e.astype(np.float32)
    return m, e, dt.astype(np.float32), float(e.sum()), float(m.sum())


def _scaled(t, s):
    if s == 1.0:
        return t
    n = int(round(SIZE * s))
    r = cv2.resize(t, (n, n), interpolation=cv2.INTER_AREA)
    if n <= SIZE:
        o = np.zeros((SIZE, SIZE), np.float32)
        a = (SIZE - n) // 2
        o[a:a + n, a:a + n] = r
        return o
    a = (n - SIZE) // 2
    return np.ascontiguousarray(r[a:a + SIZE, a:a + SIZE])


def load_templates(template_dir):
    """读模板目录下的图片，返回 [(名字, 角度, 是否镜像, SIZE x SIZE 小图, 轮廓特征), ...]。"""
    out = []
    files = sorted(f for d in str(template_dir).split(",") for f in glob.glob(os.path.join(d, "*"))
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    for f in files:
        im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        alpha = im[:, :, 3] if im.ndim == 3 and im.shape[2] == 4 else None
        rgb = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2RGB)
        s = 240.0 / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        m = skin_mask(rgb)
        if alpha is not None:
            m[cv2.resize(alpha, (rgb.shape[1], rgb.shape[0])) < 128] = 0
        m = _largest_component(m)
        if m is None:
            continue
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        h, w = m.shape
        # 四周补边，旋转时不裁掉手
        b = max(h, w) // 2
        m = cv2.copyMakeBorder(m, b, b, b, b, cv2.BORDER_CONSTANT, value=0)
        gray = cv2.copyMakeBorder(gray, b, b, b, b, cv2.BORDER_CONSTANT, value=0)
        c = (m.shape[1] / 2.0, m.shape[0] / 2.0)
        for ang in ANGLES:
            R = cv2.getRotationMatrix2D(c, ang, 1.0)
            mr = cv2.warpAffine(m, R, (m.shape[1], m.shape[0]), flags=cv2.INTER_NEAREST)
            gr = cv2.warpAffine(gray, R, (m.shape[1], m.shape[0]), flags=cv2.INTER_LINEAR)
            t = _normalize(mr, gr)
            if t is None:
                continue
            name = os.path.basename(f)
            for sc in SCALES:
                ts = _scaled(t, sc)
                if not (ts > 0.3).any():
                    continue
                out.append((name, ang, False, ts, _features(ts)))
                tm = np.ascontiguousarray(ts[:, ::-1])
                out.append((name, ang, True, tm, _features(tm)))
    return out


def candidates(rgb, min_side=36, max_frac=0.6, roi=ROI):
    """ROI 里的皮肤连通域 -> [(x, y, w, h, 归一化小图), ...]。min_side：外接框长边的最小像素数。"""
    H, W = rgb.shape[:2]
    rx0, ry0, rx1, ry1 = int(roi[0] * W), int(roi[1] * H), int(roi[2] * W), int(roi[3] * H)
    sub_rgb = np.ascontiguousarray(rgb[ry0:ry1, rx0:rx1])
    m = skin_mask(sub_rgb)
    gray = cv2.cvtColor(sub_rgb, cv2.COLOR_RGB2GRAY)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if max(w, h) < min_side or max(w, h) > max_frac * max(H, W):
            continue
        if area < 0.2 * w * h or min(w, h) < 0.3 * max(w, h):
            continue
        sub = (lab[y:y + h, x:x + w] == k).astype(np.uint8)
        patch = _normalize(sub, gray[y:y + h, x:x + w])
        if patch is not None:
            out.append((int(x) + rx0, int(y) + ry0, int(w), int(h), patch))
    return out


def score_patch(patch, templates, exclude=None):
    """候选小图与全部模板变体比轮廓，返回 (最高分, 模板名, 角度, 是否镜像, IoU, 倒角距离)。"""
    m, e, dt, ne, nm = _features(patch)
    best = (-1.0, "", 0, False, 0.0, 99.0)
    if ne < 1:
        return best
    eb = e > 0
    coarse = []
    for k, (name, _ang, _mir, _t, (tm, te, tdt, nte, ntm)) in enumerate(templates):
        if name == exclude:
            continue
        ch = 0.5 * (float(tdt[eb].mean()) + float((dt * te).sum()) / nte)
        coarse.append((ch, k))
    coarse.sort()
    P = SLIDE
    M = cv2.copyMakeBorder(m, P, P, P, P, cv2.BORDER_CONSTANT, value=0)
    E = cv2.copyMakeBorder(e, P, P, P, P, cv2.BORDER_CONSTANT, value=0)
    DT = cv2.copyMakeBorder(dt, P, P, P, P, cv2.BORDER_CONSTANT, value=DT_CAP)
    for _ch, k in coarse[:REFINE]:
        name, ang, mir, _t, (tm, te, tdt, nte, ntm) = templates[k]
        a = cv2.matchTemplate(DT, te, cv2.TM_CCORR) / nte   # 模板轮廓点到候选轮廓的平均距离，各平移位置
        b = cv2.matchTemplate(E, tdt, cv2.TM_CCORR) / ne    # 候选轮廓点到模板轮廓的平均距离
        inter = cv2.matchTemplate(M, tm, cv2.TM_CCORR)
        iou = inter / (nm + ntm - inter)
        sc = 0.5 * iou + 0.5 * np.maximum(0.0, 1.0 - 0.5 * (a + b) / 3.0)
        idx = np.unravel_index(int(np.argmax(sc)), sc.shape)
        if float(sc[idx]) > best[0]:
            best = (float(sc[idx]), name, ang, mir, float(iou[idx]), float(0.5 * (a[idx] + b[idx])))
    return best


def detect(rgb, templates, min_side=36, roi=ROI):
    """返回这一帧里分数最高的候选：dict(score, box=(x,y,w,h), template, angle, mirror, iou, chamfer, n)；没有候选时 score = 0。"""
    best = {"score": 0.0, "box": None, "template": "", "angle": 0, "mirror": False, "iou": 0.0, "chamfer": 99.0, "n": 0}
    cands = candidates(rgb, min_side=min_side, roi=roi)
    best["n"] = len(cands)
    for x, y, w, h, patch in cands:
        s, name, ang, mir, iou, ch = score_patch(patch, templates)
        if s > best["score"]:
            best.update(score=s, box=(x, y, w, h), template=name, angle=ang, mirror=mir, iou=iou, chamfer=ch)
    return best


def draw(rgb, det, ok):
    out = np.ascontiguousarray(rgb).copy()
    H, W = out.shape[:2]
    cv2.rectangle(out, (int(ROI[0] * W), int(ROI[1] * H)), (int(ROI[2] * W), int(ROI[3] * H)), (255, 255, 0), 1)
    if det["box"] is not None:
        x, y, w, h = det["box"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0) if ok else (255, 160, 0), 3)
        cv2.putText(out, "%.2f %s" % (det["score"], det["template"]), (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if ok else (255, 160, 0), 2)
    return out


if __name__ == "__main__":
    import sys
    tpl = load_templates(sys.argv[1])
    print("templates:", len(tpl), "variants")
    for f in sys.argv[2:]:
        img = np.load(f) if f.endswith(".npy") else cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        d = detect(img, tpl, min_side=18 if max(img.shape[:2]) <= 480 else 36)
        print("%-60s score %.3f box %s tpl %s ang %+d mir %d cands %d" % (f[-60:], d["score"], d["box"], d["template"], d["angle"], d["mirror"], d["n"]))
