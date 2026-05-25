import math
import heapq
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

EPS = 1e-9
TARGET_ACCURACY = 99.9
GRID_CANDIDATES = [220, 300, 420, 560, 720, 900]
random.seed(42)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def orient(a, b, c):
    v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(v) < EPS:
        return 0
    return 1 if v > 0 else -1


def on_segment(a, b, c):
    return (
        min(a[0], c[0]) - EPS <= b[0] <= max(a[0], c[0]) + EPS
        and min(a[1], c[1]) - EPS <= b[1] <= max(a[1], c[1]) + EPS
        and orient(a, b, c) == 0
    )


def segments_intersect(a, b, c, d):
    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(a, c, b):
        return True
    if o2 == 0 and on_segment(a, d, b):
        return True
    if o3 == 0 and on_segment(c, a, d):
        return True
    if o4 == 0 and on_segment(c, b, d):
        return True

    return False


def is_simple_polygon(poly):
    n = len(poly)
    for i in range(n):
        a1, a2 = poly[i], poly[(i + 1) % n]
        for j in range(i + 1, n):
            b1, b2 = poly[j], poly[(j + 1) % n]
            if i == j:
                continue
            if (i + 1) % n == j or (j + 1) % n == i:
                continue
            if segments_intersect(a1, a2, b1, b2):
                return False
    return True


def point_inside(p, poly):
    x, y = p
    inside = False
    n = len(poly)

    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]

        if on_segment((x1, y1), p, (x2, y2)):
            return True

        if (y1 > y) != (y2 > y):
            xx = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xx:
                inside = not inside

    return inside


def segment_inside_polygon(a, b, poly):
    n = len(poly)
    for i in range(n):
        c = poly[i]
        d = poly[(i + 1) % n]
        if a == c or a == d or b == c or b == d:
            continue
        if segments_intersect(a, b, c, d):
            return False
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    return point_inside(mid, poly)


def random_simple_polygon(n, center=(5.0, 5.0), r_min=2.5, r_max=4.5, max_tries=500):
    if n < 3:
        raise ValueError("Кількість вершин має бути >= 3")

    cx, cy = center
    for _ in range(max_tries):
        angles = sorted(random.uniform(0, 2 * math.pi) for _ in range(n))
        points = []
        for ang in angles:
            r = random.uniform(r_min, r_max)
            x = cx + r * math.cos(ang)
            y = cy + r * math.sin(ang)
            points.append((x, y))

        if is_simple_polygon(points):
            return points

    raise RuntimeError("Не вдалося згенерувати простий многокутник")


def random_point_inside_polygon(poly, max_tries=5000):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    for _ in range(max_tries):
        p = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if point_inside(p, poly):
            return p
    raise RuntimeError("Не вдалося знайти точку всередині")


def build_grid_mask(poly, resolution=220, padding=0.4):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    min_x, max_x = min(xs) - padding, max(xs) + padding
    min_y, max_y = min(ys) - padding, max(ys) + padding

    x_grid = np.linspace(min_x, max_x, resolution)
    y_grid = np.linspace(min_y, max_y, resolution)

    X, Y = np.meshgrid(x_grid, y_grid)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    poly_path = Path(poly)
    inside = poly_path.contains_points(pts, radius=1e-12).reshape((resolution, resolution))

    hx = (max_x - min_x) / (resolution - 1)
    hy = (max_y - min_y) / (resolution - 1)
    return x_grid, y_grid, inside, hx, hy


def nearest_inside_cell(point, x_grid, y_grid, inside):
    px, py = point
    X, Y = np.meshgrid(x_grid, y_grid)
    dist2 = (X - px) ** 2 + (Y - py) ** 2
    dist2 = np.where(inside, dist2, np.inf)
    iy, ix = np.unravel_index(np.argmin(dist2), dist2.shape)
    if not np.isfinite(dist2[iy, ix]):
        raise RuntimeError("Не знайдено прохідної клітини для точки")
    return (iy, ix)


def continuous_dijkstra_grid(inside, start, goal, hx, hy):
    h, w = inside.shape
    D = np.full((h, w), np.inf, dtype=float)
    parent_y = np.full((h, w), -1, dtype=int)
    parent_x = np.full((h, w), -1, dtype=int)

    moves = [
        (-1, 0, hy),
        (1, 0, hy),
        (0, -1, hx),
        (0, 1, hx),
        (-1, -1, math.hypot(hx, hy)),
        (-1, 1, math.hypot(hx, hy)),
        (1, -1, math.hypot(hx, hy)),
        (1, 1, math.hypot(hx, hy)),
    ]

    sy, sx = start
    gy, gx = goal
    D[sy, sx] = 0.0
    pq = [(0.0, sy, sx)]

    while pq:
        cur_d, y, x = heapq.heappop(pq)
        if cur_d > D[y, x]:
            continue
        if (y, x) == (gy, gx):
            break

        for dy, dx, cost in moves:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if not inside[ny, nx]:
                continue

            nd = D[y, x] + cost
            if nd < D[ny, nx]:
                D[ny, nx] = nd
                parent_y[ny, nx] = y
                parent_x[ny, nx] = x
                heapq.heappush(pq, (nd, ny, nx))

    if not np.isfinite(D[gy, gx]):
        return [], np.inf

    path_cells = []
    cy, cx = gy, gx
    while cy != -1 and cx != -1:
        path_cells.append((cy, cx))
        py = parent_y[cy, cx]
        px = parent_x[cy, cx]
        cy, cx = py, px
    path_cells.reverse()

    return path_cells, D[gy, gx]


def path_length(path):
    if len(path) < 2:
        return 0.0
    return sum(dist(path[i], path[i + 1]) for i in range(len(path) - 1))


def smooth_path(path, poly):
    if not path:
        return []
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if segment_inside_polygon(path[i], path[j], poly):
                break
            j -= 1
        smoothed.append(path[j])
        i = j
    return smoothed


def build_visibility_graph(poly, A, B):
    points = [A, B] + poly
    n = len(points)
    g = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if segment_inside_polygon(points[i], points[j], poly):
                w = dist(points[i], points[j])
                g[i].append((j, w))
                g[j].append((i, w))
    return points, g


def dijkstra_graph(g, s, t):
    n = len(g)
    d = [float("inf")] * n
    par = [-1] * n
    d[s] = 0.0
    pq = [(0.0, s)]

    while pq:
        cur_d, v = heapq.heappop(pq)
        if cur_d > d[v]:
            continue
        if v == t:
            break
        for to, w in g[v]:
            nd = d[v] + w
            if nd < d[to]:
                d[to] = nd
                par[to] = v
                heapq.heappush(pq, (nd, to))

    if not math.isfinite(d[t]):
        return [], float("inf")

    path = []
    v = t
    while v != -1:
        path.append(v)
        v = par[v]
    path.reverse()
    return path, d[t]


def continuous_dijkstra_with_target_accuracy(poly, A, B, target_accuracy=99.9):
    points_ref, g_ref = build_visibility_graph(poly, A, B)
    idx_path_ref, exact_len = dijkstra_graph(g_ref, 0, 1)
    exact_path = [points_ref[i] for i in idx_path_ref]

    best = None
    history = []
    for res in GRID_CANDIDATES:
        xg, yg, inside, hx, hy = build_grid_mask(poly, resolution=res)
        start_cell = nearest_inside_cell(A, xg, yg, inside)
        goal_cell = nearest_inside_cell(B, xg, yg, inside)
        path_cells, _ = continuous_dijkstra_grid(inside, start_cell, goal_cell, hx, hy)
        path_xy = [(xg[x], yg[y]) for (y, x) in path_cells]

        # Додаємо реальні A/B і згладжуємо шлях
        if path_xy:
            path_candidate = [A] + path_xy + [B]
            path_candidate = smooth_path(path_candidate, poly)
            approx_len = path_length(path_candidate)
        else:
            path_candidate = []
            approx_len = float("inf")

        if math.isfinite(exact_len) and math.isfinite(approx_len):
            rel_err = abs(approx_len - exact_len) / exact_len
            accuracy = (1.0 - rel_err) * 100.0
        else:
            rel_err = float("inf")
            accuracy = 0.0

        history.append((res, approx_len, accuracy))
        if best is None or accuracy > best["accuracy"]:
            best = {
                "res": res,
                "path": path_candidate,
                "approx_len": approx_len,
                "accuracy": accuracy,
                "inside": inside,
            }

        if accuracy >= target_accuracy:
            break

    return {
        "exact_len": exact_len,
        "exact_path": exact_path,
        "best": best,
        "history": history,
    }
def point_near_border(poly, offset=0.15):
    n = len(poly)

    while True:
        i = random.randint(0, n - 1)

        p1 = poly[i]
        p2 = poly[(i + 1) % n]

        t = random.uniform(0.15, 0.85)

        x = p1[0] + t * (p2[0] - p1[0])
        y = p1[1] + t * (p2[1] - p1[1])

        # трохи зміщуємо всередину
        dx = p2[1] - p1[1]
        dy = -(p2[0] - p1[0])

        length = math.hypot(dx, dy)

        if length < EPS:
            continue

        dx /= length
        dy /= length

        cand = (
            x + dx * offset,
            y + dy * offset
        )

        if point_inside(cand, poly):
            return cand

try:
    n_vertices = int(input("Введіть кількість вершин (>=3): "))
except Exception:
    n_vertices = 8
    print("Використано значення за замовчуванням: 8")

poly = random_simple_polygon(n_vertices)
A = (
    poly[0][0] * 0.9 + poly[1][0] * 0.1,
    poly[0][1] * 0.9 + poly[1][1] * 0.1
)

# B біля протилежної сторони
B = (
    poly[len(poly)//3][0] * 0.9 + poly[len(poly)//4 + 1][0] * 0.1,
    poly[len(poly)//3][1] * 0.9 + poly[len(poly)//4 + 1][1] * 0.1
)

result = continuous_dijkstra_with_target_accuracy(poly, A, B, target_accuracy=TARGET_ACCURACY)

best = result["best"]
exact_len = result["exact_len"]

print("Кількість вершин:", n_vertices)
print("A:", A)
print("B:", B)
print("Точна довжина (еталон):", exact_len)
print("Апроксимація (краща):", best["approx_len"])
print("Точність, %:", best["accuracy"])
print("Роздільність, що обрана:", best["res"])
print("Перебір роздільностей:")
for res, approx_len, acc in result["history"]:
    print(f"  res={res:4d}  len={approx_len:.6f}  acc={acc:.4f}%")

# Візуалізація: многокутник + шлях
px = [p[0] for p in poly] + [poly[0][0]]
py = [p[1] for p in poly] + [poly[0][1]]

plt.figure(figsize=(9, 7))
plt.plot(px, py, "k-", linewidth=1.8)

if best["path"]:
    rx = [p[0] for p in best["path"]]
    ry = [p[1] for p in best["path"]]
    plt.plot(rx, ry, "r-", linewidth=2.4, label="Апрокс. шлях")

if result["exact_path"]:
    ex = [p[0] for p in result["exact_path"]]
    ey = [p[1] for p in result["exact_path"]]
    plt.plot(ex, ey, "b--", linewidth=1.8, alpha=0.8, label="Еталон (точний)")

plt.scatter(A[0], A[1], c="lime", s=90, label="A")
plt.scatter(B[0], B[1], c="cyan", s=90, label="B")
plt.text(A[0] + 0.08, A[1] + 0.08, "A")
plt.text(B[0] + 0.08, B[1] + 0.08, "B")

plt.title("Continuous Dijkstra — апроксимація до 99.9%")
plt.grid(True, alpha=0.3)
plt.axis("equal")
plt.legend()
plt.show()