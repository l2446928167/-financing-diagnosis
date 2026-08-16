"""
utils/svg_charts.py — 纯 Python SVG 图表生成（零第三方依赖）

为什么用 SVG 而非 matplotlib：目标沙箱无法安装 matplotlib/numpy（musl libc 与 manylinux
轮子不兼容）。SVG 由纯 Python 拼接字符串生成，浏览器打开即渲染，中文用系统字体，清晰可缩放，
非常适合创新大赛成果展示。所有函数返回 SVG 字符串，由调用方写盘。

配色与产品风格一致：primary #2F54EB / green #2BA471 / amber #D98B1F / red #E5484D。
"""


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_FONT = 'font-family="HarmonyOS Sans SC, PingFang SC, Microsoft YaHei, Noto Sans CJK SC, sans-serif"'

C_PRIMARY = "#2F54EB"
C_GREEN = "#2BA471"
C_AMBER = "#D98B1F"
C_RED = "#E5484D"
C_INK = "#1F2733"
C_GRID = "#E6EAF2"
C_GREY = "#8A94A6"
C_BG = "#FFFFFF"


def _header(w, h, title=None):
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
         f'viewBox="0 0 {w} {h}" font-size="12" fill="{C_INK}">']
    s.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="{C_BG}"/>')
    if title:
        s.append(f'<text x="{w/2}" y="22" text-anchor="middle" font-size="15" '
                 f'font-weight="bold" {_FONT}>{_esc(title)}</text>')
    return s


def _axis(s, x0, y0, x1, y1, xticks, yticks, xlabel, ylabel):
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{C_INK}" stroke-width="1"/>')
    s.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{C_INK}" stroke-width="1"/>')
    for tx, tl in xticks:
        s.append(f'<line x1="{tx}" y1="{y0}" x2="{tx}" y2="{y0+4}" stroke="{C_INK}"/>')
        s.append(f'<text x="{tx}" y="{y0+18}" text-anchor="middle" font-size="10" {_FONT}>{_esc(tl)}</text>')
    for ty, tl in yticks:
        s.append(f'<line x1="{x0-4}" y1="{ty}" x2="{x0}" y2="{ty}" stroke="{C_INK}"/>')
        s.append(f'<text x="{x0-7}" y="{ty+3}" text-anchor="end" font-size="10" {_FONT}>{_esc(tl)}</text>')
    if xlabel:
        s.append(f'<text x="{(x0+x1)/2}" y="{y0+38}" text-anchor="middle" font-size="11" {_FONT}>{_esc(xlabel)}</text>')
    if ylabel:
        s.append(f'<text x="14" y="{(y0+y1)/2}" text-anchor="middle" font-size="11" '
                 f'{_FONT} transform="rotate(-90 14 {(y0+y1)/2})">{_esc(ylabel)}</text>')


def _legend(s, items, x, y):
    yy = y
    for color, label in items:
        s.append(f'<rect x="{x}" y="{yy-9}" width="12" height="12" fill="{color}" rx="2"/>')
        s.append(f'<text x="{x+18}" y="{yy+1}" font-size="11" {_FONT}>{_esc(label)}</text>')
        yy += 18


def line_chart(title, xs, series, xlabel="", ylabel="", ylim=(0, 1)):
    """series: list of (name, color, ys)。"""
    w, h = 720, 420
    left, right, top, bottom = 60, 660, 50, 350
    s = _header(w, h, title)
    if not xs:
        s.append('</svg>'); return "".join(s)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = ylim
    def X(v):
        return left + (v - xmin) / (xmax - xmin or 1) * (right - left)
    def Y(v):
        return bottom - (v - ymin) / (ymax - ymin or 1) * (bottom - top)
    # grid + ticks
    yt = [(Y(ylim[0] + (ylim[1]-ylim[0])*i/5), f"{ylim[0]+(ylim[1]-ylim[0])*i/5:.2f}") for i in range(6)]
    xt = [(X(xs[i]), f"{xs[i]:g}") for i in range(0, len(xs), max(1, len(xs)//6))]
    _axis(s, left, bottom, right, top, xt, yt, xlabel, ylabel)
    for name, color, ys in series:
        pts = " ".join(f"{X(xs[i]):.1f},{Y(ys[i]):.1f}" for i in range(len(xs)))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for i in range(len(xs)):
            s.append(f'<circle cx="{X(xs[i]):.1f}" cy="{Y(ys[i]):.1f}" r="2.5" fill="{color}"/>')
    _legend(s, [(c, n) for n, c, _ in series], left + 20, top + 10)
    s.append('</svg>')
    return "".join(s)


def hbar_chart(title, names, vals, color=C_PRIMARY, unit=""):
    w, h = 760, max(320, 40 * len(names) + 70)
    left, right, top, bottom = 160, 700, 50, h - 60
    s = _header(w, h, title)
    vmax = max(vals) or 1
    n = len(names)
    step = (bottom - top) / n
    for i, (nm, v) in enumerate(zip(names, vals)):
        y = top + i * step + step / 2
        bw = (v / vmax) * (right - left)
        s.append(f'<rect x="{left}" y="{y-12}" width="{bw:.1f}" height="22" fill="{color}" rx="3"/>')
        s.append(f'<text x="{left-8}" y="{y+4}" text-anchor="end" font-size="11" {_FONT}>{_esc(nm)}</text>')
        s.append(f'<text x="{left+bw+6}" y="{y+4}" font-size="10" fill="{C_GREY}" {_FONT}>{v:.3f}{unit}</text>')
    s.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="{C_INK}"/>')
    s.append('</svg>')
    return "".join(s)


def heatmap(title, matrix, feat_names, vmax=None):
    """matrix: list of lists (n_rows x d)。红=正，蓝=负（RdBu 近似）。"""
    n = len(matrix); d = len(matrix[0]) if n else 0
    cell = 34
    left, top = 150, 60
    w = left + d * cell + 90
    h = top + n * cell + 50
    s = _header(w, h, title)
    if vmax is None:
        import math
        flat = [abs(v) for row in matrix for v in row]
        vmax = sorted(flat)[min(len(flat)-1, int(len(flat)*0.98))] or 1.0

    def color(v):
        t = max(-1.0, min(1.0, v / vmax))  # -1..1
        if t >= 0:
            # white -> red
            r = 255; g = int(255 - 150 * t); b = int(255 - 200 * t)
        else:
            # white -> blue
            t = -t
            r = int(255 - 200 * t); g = int(255 - 120 * t); b = int(255 - 60 * t)
        return f"rgb({r},{g},{b})"
    for j in range(d):
        s.append(f'<text x="{left+j*cell+cell/2}" y="{top-8}" text-anchor="middle" '
                 f'font-size="10" {_FONT}>{_esc(feat_names[j])}</text>')
    for i in range(n):
        for j in range(d):
            v = matrix[i][j]
            s.append(f'<rect x="{left+j*cell}" y="{top+i*cell}" width="{cell}" height="{cell}" '
                     f'fill="{color(v)}" stroke="#fff" stroke-width="0.5"/>')
    # colorbar
    cb_x = left + d * cell + 25
    for k in range(50):
        t = -1 + 2 * k / 49
        s.append(f'<rect x="{cb_x}" y="{top+k*(n*cell)/50}" width="14" height="{(n*cell)/50+1}" fill="{color(t)}"/>')
    s.append(f'<text x="{cb_x+18}" y="{top+6}" font-size="9" {_FONT}>+{vmax:.2f}</text>')
    s.append(f'<text x="{cb_x+18}" y="{top+n*cell}" font-size="9" {_FONT}>-{vmax:.2f}</text>')
    s.append(f'<text x="{cb_x+10}" y="{top+n*cell/2}" font-size="9" transform="rotate(90 {cb_x+10} {top+n*cell/2})" {_FONT}>贡献</text>')
    s.append('</svg>')
    return "".join(s)


def grouped_bar(title, groups, series, ylabel="Score"):
    """groups: x 标签; series: list of (name,color,vals)。"""
    w, h = 760, 430
    left, right, top, bottom = 70, 700, 50, 360
    s = _header(w, h, title)
    n = len(groups); m = len(series)
    import math
    ymax = max(max(se[2]) for se in series) * 1.15
    def X(i):
        return left + (i + 0.5) / n * (right - left)
    def Y(v):
        return bottom - v / ymax * (bottom - top)
    yt = [(Y(ymax*i/5), f"{ymax*i/5:.2f}") for i in range(6)]
    xt = [(X(i), groups[i]) for i in range(n)]
    _axis(s, left, bottom, right, top, xt, yt, "", ylabel)
    bw = (right - left) / n / m * 0.8
    for si, (name, color, vals) in enumerate(series):
        for i in range(n):
            x = X(i) - (m*bw)/2 + si*bw
            v = vals[i]
            s.append(f'<rect x="{x:.1f}" y="{Y(v):.1f}" width="{bw:.1f}" height="{bottom-Y(v):.1f}" '
                     f'fill="{color}" rx="2"/>')
            s.append(f'<text x="{x+bw/2:.1f}" y="{Y(v)-4:.1f}" text-anchor="middle" font-size="9" {_FONT}>{v:.3f}</text>')
    _legend(s, [(c, n) for n, c, _ in series], left + 20, top + 6)
    s.append('</svg>')
    return "".join(s)


def calibration_curve(title, mean_pred, frac_pos):
    w, h = 520, 480
    left, right, top, bottom = 60, 470, 50, 420
    s = _header(w, h, title)
    def X(v):
        return left + v * (right - left)
    def Y(v):
        return bottom - v * (bottom - top)
    yt = [(Y(i/5), f"{i/5:.1f}") for i in range(6)]
    xt = [(X(i/5), f"{i/5:.1f}") for i in range(6)]
    _axis(s, left, bottom, right, top, xt, yt, "预测违约概率", "实际违约比例")
    s.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{top}" stroke="{C_GREY}" '
             f'stroke-dasharray="4 3"/>')
    pts = " ".join(f"{X(mean_pred[i]):.1f},{Y(frac_pos[i]):.1f}" for i in range(len(mean_pred)))
    s.append(f'<polyline points="{pts}" fill="none" stroke="{C_PRIMARY}" stroke-width="2.5"/>')
    for i in range(len(mean_pred)):
        s.append(f'<circle cx="{X(mean_pred[i]):.1f}" cy="{Y(frac_pos[i]):.1f}" r="3" fill="{C_PRIMARY}"/>')
    _legend(s, [(C_GREY, "理想校准"), (C_PRIMARY, "模型")], left + 20, top + 6)
    s.append('</svg>')
    return "".join(s)


def confusion_matrix(title, cm, labels):
    w, h = 460, 420
    left, top = 90, 70
    cell = 130
    s = _header(w, h, title)
    mx = max(max(r) for r in cm) or 1
    for i in range(2):
        for j in range(2):
            v = cm[i][j]
            inten = v / mx
            fill = f"rgb({int(255-150*inten)},{int(255-180*inten)},{int(255-200*inten)})" if inten else "#fff"
            s.append(f'<rect x="{left+j*cell}" y="{top+i*cell}" width="{cell}" height="{cell}" '
                     f'fill="{fill}" stroke="{C_INK}"/>')
            s.append(f'<text x="{left+j*cell+cell/2}" y="{top+i*cell+cell/2-6}" text-anchor="middle" '
                     f'font-size="22" font-weight="bold" {_FONT}>{v}</text>')
            s.append(f'<text x="{left+j*cell+cell/2}" y="{top+i*cell+cell/2+16}" text-anchor="middle" '
                     f'font-size="10" fill="{C_GREY}" {_FONT}>{v/mx:.2f}</text>')
    s.append(f'<text x="{left+cell}" y="{top-12}" text-anchor="middle" font-size="12" {_FONT}>预测</text>')
    s.append(f'<text x="{left-30}" y="{top+cell}" text-anchor="middle" font-size="12" '
             f'{_FONT} transform="rotate(-90 {left-30} {top+cell})">实际</text>')
    for j in range(2):
        s.append(f'<text x="{left+j*cell+cell/2}" y="{top+2*cell+20}" text-anchor="middle" font-size="11" {_FONT}>{labels[j]}</text>')
    for i in range(2):
        s.append(f'<text x="{left-50}" y="{top+i*cell+cell/2}" text-anchor="middle" font-size="11" '
                 f'{_FONT} transform="rotate(-90 {left-50} {top+i*cell+cell/2})">{labels[i]}</text>')
    s.append('</svg>')
    return "".join(s)
