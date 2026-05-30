import math
import heapq
import random
import time
import statistics
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import simpledialog, messagebox

try:
    from sortedcontainers import SortedList
    HAS_SORTED = True
except ImportError:
    HAS_SORTED = False


EPS = 1e-9
TWO_PI = 2 * math.pi
random.seed(42)

CURRENT_ORIGIN = (0.0, 0.0)
CURRENT_ANGLE = 0.0


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def same_point(a, b):
    return abs(a[0] - b[0]) < EPS and abs(a[1] - b[1]) < EPS


def orient(a, b, c):
    v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    if abs(v) < EPS:
        return 0

    if v > 0:
        return 1

    return -1


def on_segment(a, p, b):
    return (
        min(a[0], b[0]) - EPS <= p[0] <= max(a[0], b[0]) + EPS
        and min(a[1], b[1]) - EPS <= p[1] <= max(a[1], b[1]) + EPS
        and orient(a, p, b) == 0
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
        a1 = poly[i]
        a2 = poly[(i + 1) % n]

        for j in range(i + 1, n):
            b1 = poly[j]
            b2 = poly[(j + 1) % n]

            if i == j:
                continue

            if (i + 1) % n == j:
                continue

            if i == (j + 1) % n:
                continue

            if segments_intersect(a1, a2, b1, b2):
                return False

    return True


def point_inside_polygon(p, poly):
    x, y = p
    inside = False
    n = len(poly)

    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]

        if on_segment(a, p, b):
            return True

        x1, y1 = a
        x2, y2 = b

        if (y1 > y) != (y2 > y):
            x_cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1

            if x < x_cross:
                inside = not inside

    return inside


def normalize_angle(a):
    a = a % TWO_PI

    if a < 0:
        a += TWO_PI

    return a


def angle_between(a, b):
    return normalize_angle(math.atan2(b[1] - a[1], b[0] - a[0]))


def ray_segment_distance(origin, angle, a, b):
    ox, oy = origin
    rx = math.cos(angle)
    ry = math.sin(angle)

    ax, ay = a
    bx, by = b

    sx = bx - ax
    sy = by - ay

    den = rx * sy - ry * sx

    if abs(den) < EPS:
        return float("inf")

    qx = ax - ox
    qy = ay - oy

    t = (qx * sy - qy * sx) / den
    u = (qx * ry - qy * rx) / den

    if t >= -EPS and -EPS <= u <= 1 + EPS:
        return max(0.0, t)

    return float("inf")


class ActiveEdge:
    def __init__(self, idx, a, b):
        self.idx = idx
        self.a = a
        self.b = b

    def current_distance(self):
        return ray_segment_distance(CURRENT_ORIGIN, CURRENT_ANGLE, self.a, self.b)

    def __lt__(self, other):
        d1 = self.current_distance()
        d2 = other.current_distance()

        if abs(d1 - d2) > EPS:
            return d1 < d2

        return self.idx < other.idx

    def __eq__(self, other):
        return isinstance(other, ActiveEdge) and self.idx == other.idx


class SimpleSortedList:
    def __init__(self):
        self.data = []

    def add(self, x):
        self.data.append(x)
        self.data.sort()

    def remove(self, x):
        for i, item in enumerate(self.data):
            if item.idx == x.idx:
                self.data.pop(i)
                return

    def discard(self, x):
        for i, item in enumerate(self.data):
            if item.idx == x.idx:
                self.data.pop(i)
                return

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


def edge_incident_to_point(edge, p):
    return same_point(edge.a, p) or same_point(edge.b, p)


def angular_interval(origin, a, b):
    a1 = angle_between(origin, a)
    a2 = angle_between(origin, b)

    diff = (a2 - a1) % TWO_PI

    if diff <= math.pi:
        start = a1
        end = a2
    else:
        start = a2
        end = a1

    if start <= end:
        return [(start, end)]

    return [(start, TWO_PI), (0.0, end)]


def visible_by_sweep(source, target, active_edges):
    target_distance = dist(source, target)

    for edge in active_edges:
        if edge_incident_to_point(edge, source):
            continue

        if edge_incident_to_point(edge, target):
            continue

        d_edge = edge.current_distance()

        if math.isfinite(d_edge) and d_edge < target_distance - EPS:
            return False

        return True

    return True


def visible_by_sweep_with_blocker(source, target, active_edges):
    target_distance = dist(source, target)

    for edge in active_edges:
        if edge_incident_to_point(edge, source):
            continue

        if edge_incident_to_point(edge, target):
            continue

        d_edge = edge.current_distance()

        if math.isfinite(d_edge) and d_edge < target_distance - EPS:
            return False, edge

        return True, None

    return True, None


def build_visibility_from_source(source_index, points, poly):
    global CURRENT_ORIGIN, CURRENT_ANGLE

    source = points[source_index]
    n_poly = len(poly)

    edge_objects = []

    for i in range(n_poly):
        a = poly[i]
        b = poly[(i + 1) % n_poly]
        edge_objects.append(ActiveEdge(i, a, b))

    events = []

    for edge in edge_objects:
        if edge_incident_to_point(edge, source):
            continue

        intervals = angular_interval(source, edge.a, edge.b)

        for start, end in intervals:
            events.append((start, 0, edge))
            events.append((end, 2, edge))

    for j, p in enumerate(points):
        if j == source_index:
            continue

        ang = angle_between(source, p)
        events.append((ang, 1, j))

    events.sort(key=lambda x: (x[0], x[1]))

    if HAS_SORTED:
        active = SortedList()
    else:
        active = SimpleSortedList()

    visible = []
    CURRENT_ORIGIN = source

    for angle, typ, obj in events:
        CURRENT_ANGLE = angle

        if typ == 0:
            active.add(obj)

        elif typ == 2:
            try:
                active.remove(obj)
            except Exception:
                try:
                    active.discard(obj)
                except Exception:
                    pass

        else:
            target_index = obj
            target = points[target_index]

            if visible_by_sweep(source, target, active):
                visible.append(target_index)

    return visible


def build_visibility_from_source_with_steps(source_index, points, poly, steps):
    global CURRENT_ORIGIN, CURRENT_ANGLE

    source = points[source_index]
    n_poly = len(poly)

    edge_objects = []

    for i in range(n_poly):
        a = poly[i]
        b = poly[(i + 1) % n_poly]
        edge_objects.append(ActiveEdge(i, a, b))

    events = []

    for edge in edge_objects:
        if edge_incident_to_point(edge, source):
            continue

        intervals = angular_interval(source, edge.a, edge.b)

        for start, end in intervals:
            events.append((start, 0, edge))
            events.append((end, 2, edge))

    for j, p in enumerate(points):
        if j == source_index:
            continue

        ang = angle_between(source, p)
        events.append((ang, 1, j))

    events.sort(key=lambda x: (x[0], x[1]))

    if HAS_SORTED:
        active = SortedList()
    else:
        active = SimpleSortedList()

    visible = []
    active_ids = set()
    CURRENT_ORIGIN = source

    steps.append({
        "type": "source",
        "source": source_index,
        "text": f"Обрана точка {source_index}. Починаємо кутове сканування."
    })

    for angle, typ, obj in events:
        CURRENT_ANGLE = angle

        if typ == 0:
            active.add(obj)
            active_ids.add(obj.idx)

            steps.append({
                "type": "activate_edge",
                "source": source_index,
                "edge": obj.idx,
                "active_edges": list(active_ids),
                "angle": angle,
                "text": f"Сторона P{obj.idx + 1} стала активною."
            })

        elif typ == 2:
            try:
                active.remove(obj)
            except Exception:
                try:
                    active.discard(obj)
                except Exception:
                    pass

            if obj.idx in active_ids:
                active_ids.remove(obj.idx)

            steps.append({
                "type": "deactivate_edge",
                "source": source_index,
                "edge": obj.idx,
                "active_edges": list(active_ids),
                "angle": angle,
                "text": f"Сторона P{obj.idx + 1} більше не активна."
            })

        else:
            target_index = obj
            target = points[target_index]

            is_visible, blocker = visible_by_sweep_with_blocker(source, target, active)

            if is_visible:
                visible.append(target_index)

                steps.append({
                    "type": "add_visibility_edge",
                    "source": source_index,
                    "target": target_index,
                    "active_edges": list(active_ids),
                    "angle": angle,
                    "text": f"Точка {target_index} видима з точки {source_index}."
                })
            else:
                steps.append({
                    "type": "blocked_edge",
                    "source": source_index,
                    "target": target_index,
                    "blocker": blocker.idx if blocker else None,
                    "active_edges": list(active_ids),
                    "angle": angle,
                    "text": f"Точка {target_index} не видима з точки {source_index}."
                })

    return visible


def build_visibility_graph_sweep(points, poly):
    n = len(points)
    graph = {i: [] for i in range(n)}
    added = set()

    for i in range(n):
        visible_points = build_visibility_from_source(i, points, poly)

        for j in visible_points:
            if i == j:
                continue

            a = min(i, j)
            b = max(i, j)

            if (a, b) in added:
                continue

            w = dist(points[i], points[j])
            graph[i].append((j, w))
            graph[j].append((i, w))
            added.add((a, b))

    return graph


def build_visibility_graph_sweep_with_steps(points, poly):
    n = len(points)
    graph = {i: [] for i in range(n)}
    added = set()
    steps = []

    for i in range(n):
        visible_points = build_visibility_from_source_with_steps(i, points, poly, steps)

        for j in visible_points:
            if i == j:
                continue

            a = min(i, j)
            b = max(i, j)

            if (a, b) in added:
                continue

            w = dist(points[i], points[j])
            graph[i].append((j, w))
            graph[j].append((i, w))
            added.add((a, b))

    return graph, steps


def dijkstra(graph, start, finish):
    n = len(graph)
    d = [float("inf")] * n
    parent = [-1] * n

    d[start] = 0.0
    pq = [(0.0, start)]

    while pq:
        cur_dist, v = heapq.heappop(pq)

        if cur_dist > d[v]:
            continue

        if v == finish:
            break

        for to, w in graph[v]:
            new_dist = cur_dist + w

            if new_dist < d[to]:
                d[to] = new_dist
                parent[to] = v
                heapq.heappush(pq, (new_dist, to))

    if not math.isfinite(d[finish]):
        return [], float("inf")

    path = []
    cur = finish

    while cur != -1:
        path.append(cur)
        cur = parent[cur]

    path.reverse()

    return path, d[finish]


def shortest_path_visibility(poly, A, B):
    points = [A, B] + poly

    graph = build_visibility_graph_sweep(points, poly)

    path_indexes, path_len = dijkstra(graph, 0, 1)

    if not path_indexes:
        return {
            "points": points,
            "graph": graph,
            "path_indexes": [],
            "path": [],
            "length": float("inf")
        }

    path = [points[i] for i in path_indexes]

    return {
        "points": points,
        "graph": graph,
        "path_indexes": path_indexes,
        "path": path,
        "length": path_len
    }


def shortest_path_visibility_with_steps(poly, A, B):
    points = [A, B] + poly

    graph, steps = build_visibility_graph_sweep_with_steps(points, poly)

    path_indexes, path_len = dijkstra(graph, 0, 1)

    if not path_indexes:
        return {
            "points": points,
            "graph": graph,
            "steps": steps,
            "path_indexes": [],
            "path": [],
            "length": float("inf")
        }

    path = [points[i] for i in path_indexes]

    return {
        "points": points,
        "graph": graph,
        "steps": steps,
        "path_indexes": path_indexes,
        "path": path,
        "length": path_len
    }


def random_simple_polygon(n, center=(5.0, 5.0), r_min=2.0, r_max=4.5, max_tries=1000):
    if n < 3:
        raise ValueError("Кількість вершин має бути >= 3")

    cx, cy = center

    for _ in range(max_tries):
        angles = sorted(random.uniform(0, TWO_PI) for _ in range(n))
        poly = []

        for angle in angles:
            r = random.uniform(r_min, r_max)
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            poly.append((x, y))

        if is_simple_polygon(poly):
            return poly

    raise RuntimeError("Не вдалося згенерувати простий многокутник")


def random_point_inside_polygon(poly, max_tries=5000):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    for _ in range(max_tries):
        p = (random.uniform(min_x, max_x), random.uniform(min_y, max_y))

        if point_inside_polygon(p, poly):
            return p

    raise RuntimeError("Не вдалося знайти точку всередині многокутника")


def segment_crosses_polygon_boundary(a, b, poly):
    n = len(poly)

    for i in range(n):
        c = poly[i]
        d = poly[(i + 1) % n]

        if segments_intersect(a, b, c, d):
            return True

    return False


def random_points_with_crossing_segment(poly, max_tries=10000):
    for _ in range(max_tries):
        a = random_point_inside_polygon(poly)
        b = random_point_inside_polygon(poly)

        if same_point(a, b):
            continue

        if segment_crosses_polygon_boundary(a, b, poly):
            return a, b

    a = random_point_inside_polygon(poly)
    b = random_point_inside_polygon(poly)

    return a, b


def random_case_with_crossing_ab(n, max_poly_tries=500):
    for _ in range(max_poly_tries):
        poly = random_simple_polygon(n)

        try:
            a, b = random_points_with_crossing_segment(poly)
            return poly, a, b
        except RuntimeError:
            continue

    poly = random_simple_polygon(n)
    a = random_point_inside_polygon(poly)
    b = random_point_inside_polygon(poly)

    return poly, a, b


def draw_result(poly, A, B, result):
    points = result["points"]
    graph = result["graph"]
    path = result["path"]

    px = [p[0] for p in poly] + [poly[0][0]]
    py = [p[1] for p in poly] + [poly[0][1]]

    plt.figure(figsize=(9, 7))
    plt.plot(px, py, "k-", linewidth=2, label="Многокутник")

    for i in graph:
        for j, w in graph[i]:
            if i < j:
                x = [points[i][0], points[j][0]]
                y = [points[i][1], points[j][1]]
                plt.plot(x, y, color="gray", linewidth=0.5, alpha=0.25)

    vx = [p[0] for p in poly]
    vy = [p[1] for p in poly]

    plt.scatter(vx, vy, s=45, label="Вершини")
    plt.scatter(A[0], A[1], c="lime", s=100, label="A")
    plt.scatter(B[0], B[1], c="cyan", s=100, label="B")

    if path:
        rx = [p[0] for p in path]
        ry = [p[1] for p in path]

        plt.plot(rx, ry, "r-", linewidth=3, label="Найкоротший шлях")
        plt.scatter(rx, ry, c="red", s=55)

    plt.text(A[0] + 0.08, A[1] + 0.08, "A")
    plt.text(B[0] + 0.08, B[1] + 0.08, "B")

    for k, p in enumerate(poly, start=1):
        plt.text(p[0] + 0.05, p[1] + 0.05, f"P{k}", fontsize=9)

    plt.title("Граф видимості та найкоротший шлях")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.show()


def save_animation_gif(anim, base_name, interval_ms):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        script_dir = os.getcwd()

    filename = f"{base_name}_{time.strftime('%Y%m%d_%H%M%S')}.gif"
    out_path = os.path.join(script_dir, filename)

    fps = 2
    if interval_ms and interval_ms > 0:
        fps = max(1, int(round(1000.0 / interval_ms)))

    try:
        anim.save(out_path, writer="pillow", fps=fps)
        print(f"GIF збережено: {out_path}")
    except Exception as e:
        messagebox.showwarning("Попередження", f"Не вдалося зберегти GIF: {e}")


def draw_result_animated(poly, A, B, result):
    if not result["path"]:
        messagebox.showerror("Помилка", "Шлях не знайдено")
        return

    points = result["points"]
    graph = result["graph"]
    path = result["path"]

    px = [p[0] for p in poly] + [poly[0][0]]
    py = [p[1] for p in poly] + [poly[0][1]]

    rx = [p[0] for p in path]
    ry = [p[1] for p in path]

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.plot(px, py, "k-", linewidth=2, label="Многокутник")

    for i in graph:
        for j, w in graph[i]:
            if i < j:
                x = [points[i][0], points[j][0]]
                y = [points[i][1], points[j][1]]
                ax.plot(x, y, color="gray", linewidth=0.5, alpha=0.2)

    vx = [p[0] for p in poly]
    vy = [p[1] for p in poly]

    ax.scatter(vx, vy, s=45, label="Вершини")
    ax.scatter(A[0], A[1], c="lime", s=100, label="A")
    ax.scatter(B[0], B[1], c="cyan", s=100, label="B")

    ax.text(A[0] + 0.08, A[1] + 0.08, "A")
    ax.text(B[0] + 0.08, B[1] + 0.08, "B")

    for k, p in enumerate(poly, start=1):
        ax.text(p[0] + 0.05, p[1] + 0.05, f"P{k}", fontsize=9)

    line, = ax.plot([], [], "r-", linewidth=3, label="Найкоротший шлях")
    point, = ax.plot([], [], "ro", markersize=6)

    ax.set_title("Анімація побудови найкоротшого шляху")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend()

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def update(frame):
        line.set_data(rx[:frame + 1], ry[:frame + 1])
        point.set_data([rx[frame]], [ry[frame]])
        return line, point

    anim = FuncAnimation(
        fig,
        update,
        frames=len(path),
        init_func=init,
        interval=500,
        blit=False,
        repeat=False
    )

    fig.anim = anim
    save_animation_gif(anim, "shortest_path_animation", 500)

    plt.show()


def draw_algorithm_steps_animated(poly, A, B, result):
    points = result["points"]
    steps = result.get("steps", [])
    path = result["path"]

    if not steps:
        messagebox.showerror("Помилка", "Немає кроків для анімації")
        return

    px = [p[0] for p in poly] + [poly[0][0]]
    py = [p[1] for p in poly] + [poly[0][1]]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(px, py, "k-", linewidth=2, label="Многокутник")

    vx = [p[0] for p in poly]
    vy = [p[1] for p in poly]

    ax.scatter(vx, vy, s=45, label="Вершини")
    ax.scatter(A[0], A[1], c="lime", s=100, label="A")
    ax.scatter(B[0], B[1], c="cyan", s=100, label="B")

    ax.text(A[0] + 0.08, A[1] + 0.08, "A")
    ax.text(B[0] + 0.08, B[1] + 0.08, "B")

    for k, p in enumerate(poly, start=1):
        ax.text(p[0] + 0.05, p[1] + 0.05, f"P{k}", fontsize=9)

    text_obj = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.85)
    )

    temp_lines = []

    ax.set_title("Покрокова анімація побудови графа видимості")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    ax.legend()

    def clear_lines(lines):
        for line in lines:
            try:
                line.remove()
            except Exception:
                pass
        lines.clear()

    def update(frame):
        clear_lines(temp_lines)
        step = steps[frame]

        if step["type"] in ("add_visibility_edge", "blocked_edge"):
            p1 = points[step["source"]]
            p2 = points[step["target"]]
            color = "green" if step["type"] == "add_visibility_edge" else "red"
            style = "-" if step["type"] == "add_visibility_edge" else "--"

            line, = ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=color,
                linestyle=style,
                linewidth=2.2
            )
            temp_lines.append(line)

        if frame == len(steps) - 1 and path:
            rx = [p[0] for p in path]
            ry = [p[1] for p in path]
            final_line, = ax.plot(rx, ry, "r-", linewidth=3.5)
            temp_lines.append(final_line)

        text_obj.set_text(f"Крок {frame + 1}/{len(steps)}\n{step.get('text', '')}")

        return temp_lines + [text_obj]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(steps),
        interval=350,
        blit=False,
        repeat=False
    )

    fig.anim = anim
    save_animation_gif(anim, "search_steps_animation", 350)
    plt.show()


def run_algorithm(poly, A, B):
    if not HAS_SORTED:
        print("Увага: бібліотека sortedcontainers не встановлена.")
        print("Код працюватиме, але для теоретичної складності O(n^2 log n) потрібно:")
        print("pip install sortedcontainers")

    result = shortest_path_visibility(poly, A, B)

    print("\nРезультат роботи алгоритму:")
    print("Кількість вершин многокутника:", len(poly))
    print("Точка A:", A)
    print("Точка B:", B)

    if not result["path"]:
        print("Шлях не знайдено")
        draw_result(poly, A, B, result)
        return

    print("Довжина найкоротшого шляху:", round(result["length"], 6))

    print("\nТочки найкоротшого шляху:")
    for p in result["path"]:
        print(p)

    edge_count = sum(len(v) for v in result["graph"].values()) // 2

    print("\nКількість усіх точок графа:", len(result["points"]))
    print("Кількість ребер графа видимості:", edge_count)

    print("\nСкладність алгоритму:")
    print("Кутове сканування для однієї точки: O(n log n)")
    print("Кутове сканування для всіх точок: O(n^2 log n)")
    print("Дейкстра на графі видимості: O(n^2 log n)")
    print("Загальна складність: O(n^2 log n)")

    draw_result_animated(poly, A, B, result)


def draw_polygon_with_mouse():
    print("Відкриється полотно.")
    print("Клацайте мишкою, щоб поставити вершини многокутника.")
    print("Після завершення натисніть Enter.")

    plt.figure(figsize=(8, 6))
    plt.title("Клацніть вершини многокутника. Enter - завершити")
    plt.grid(True)
    plt.axis("equal")

    selected = plt.ginput(n=-1, timeout=0)
    plt.close()

    poly = [(float(x), float(y)) for x, y in selected]

    if len(poly) < 3:
        raise ValueError("Потрібно задати мінімум 3 вершини")

    if not is_simple_polygon(poly):
        raise ValueError("Многокутник некоректний: сторони перетинаються")

    plt.figure(figsize=(8, 6))

    px = [p[0] for p in poly] + [poly[0][0]]
    py = [p[1] for p in poly] + [poly[0][1]]

    plt.plot(px, py, "k-", linewidth=2)
    plt.title("Клацніть точку A, потім точку B")
    plt.grid(True)
    plt.axis("equal")

    selected_points = plt.ginput(n=2, timeout=0)
    plt.close()

    if len(selected_points) < 2:
        raise ValueError("Потрібно вибрати дві точки: A і B")

    A = (float(selected_points[0][0]), float(selected_points[0][1]))
    B = (float(selected_points[1][0]), float(selected_points[1][1]))

    if not point_inside_polygon(A, poly):
        raise ValueError("Точка A не лежить всередині многокутника")

    if not point_inside_polygon(B, poly):
        raise ValueError("Точка B не лежить всередині многокутника")

    return poly, A, B


def run_random_mode():
    try:
        n = simpledialog.askinteger(
            "Автоматична генерація",
            "Введіть кількість вершин многокутника:",
            minvalue=3
        )

        if n is None:
            return

        poly, A, B = random_case_with_crossing_ab(n)
        run_algorithm(poly, A, B)

    except Exception as e:
        messagebox.showerror("Помилка", str(e))


def run_mouse_mode():
    try:
        poly, A, B = draw_polygon_with_mouse()
        run_algorithm(poly, A, B)

    except Exception as e:
        messagebox.showerror("Помилка", str(e))


def run_search_animation_mode():
    try:
        n = simpledialog.askinteger(
            "Анімація пошуку",
            "Введіть кількість вершин многокутника:",
            minvalue=3
        )

        if n is None:
            return

        poly, A, B = random_case_with_crossing_ab(n)
        result = shortest_path_visibility_with_steps(poly, A, B)

        if not result["path"]:
            messagebox.showerror("Помилка", "Шлях не знайдено")
            return

        draw_algorithm_steps_animated(poly, A, B, result)

    except Exception as e:
        messagebox.showerror("Помилка", str(e))


def normalize_curve(values, target_max):
    max_value = max(values)

    if max_value == 0:
        return values

    return [(v / max_value) * target_max for v in values]


def rmse(real, model):
    if not real:
        return 0

    s = 0

    for a, b in zip(real, model):
        s += (a - b) ** 2

    return math.sqrt(s / len(real))


def run_analytics_mode():
    try:
        max_n = simpledialog.askinteger(
            "Аналітика",
            "Введіть верхню межу кількості вершин (>= 10):",
            minvalue=10
        )

        if max_n is None:
            return

        trials = simpledialog.askinteger(
            "Аналітика",
            "Скільки запусків робити для кожного n? Рекомендовано 3-5:",
            minvalue=1,
            initialvalue=3
        )

        if trials is None:
            return

        ns = []
        mean_times = []
        min_times = []
        max_times = []

        for n in range(3, max_n + 1):
            samples = []

            for _ in range(trials):
                poly = random_simple_polygon(n)
                A = random_point_inside_polygon(poly)
                B = random_point_inside_polygon(poly)

                while same_point(A, B):
                    B = random_point_inside_polygon(poly)

                t0 = time.perf_counter()
                _ = shortest_path_visibility(poly, A, B)
                t1 = time.perf_counter()

                samples.append(t1 - t0)

            ns.append(n)
            mean_times.append(statistics.mean(samples))
            min_times.append(min(samples))
            max_times.append(max(samples))

        max_real_time = max(mean_times)

        nlogn_raw = [n * math.log2(n) for n in ns]
        n2_raw = [n ** 2 for n in ns]
        n2logn_raw = [n ** 2 * math.log2(n) for n in ns]
        n3_raw = [n ** 3 for n in ns]

        nlogn_scaled = normalize_curve(nlogn_raw, max_real_time)
        n2_scaled = normalize_curve(n2_raw, max_real_time)
        n2logn_scaled = normalize_curve(n2logn_raw, max_real_time)
        n3_scaled = normalize_curve(n3_raw, max_real_time)

        err_nlogn = rmse(mean_times, nlogn_scaled)
        err_n2 = rmse(mean_times, n2_scaled)
        err_n2logn = rmse(mean_times, n2logn_scaled)
        err_n3 = rmse(mean_times, n3_scaled)

        errors = {
            "O(n log n)": err_nlogn,
            "O(n²)": err_n2,
            "O(n² log n)": err_n2logn,
            "O(n³)": err_n3
        }

        best_model = min(errors, key=errors.get)

        print("\nАналітика складності:")
        print("Кількість запусків для кожного n:", trials)
        print("Максимальна кількість вершин:", max_n)
        print("Похибка O(n log n):", round(err_nlogn, 8))
        print("Похибка O(n²):", round(err_n2, 8))
        print("Похибка O(n² log n):", round(err_n2logn, 8))
        print("Похибка O(n³):", round(err_n3, 8))
        print("Найближча крива за RMSE:", best_model)

        plt.figure(figsize=(12, 7))

        plt.plot(
            ns,
            mean_times,
            marker="o",
            linewidth=3,
            color="black",
            label="Реальний середній час"
        )

        plt.fill_between(
            ns,
            min_times,
            max_times,
            alpha=0.15,
            label="Мінімум-максимум часу"
        )

        plt.plot(
            ns,
            nlogn_scaled,
            linestyle="--",
            linewidth=2.5,
            color="#1f77b4",
            label="O(n log n)"
        )

        plt.plot(
            ns,
            n2_scaled,
            linestyle="--",
            linewidth=2.5,
            color="#2ca02c",
            label="O(n²)"
        )

        plt.plot(
            ns,
            n2logn_scaled,
            linestyle="--",
            linewidth=2.5,
            color="#ff7f0e",
            label="O(n² log n)"
        )

        plt.plot(
            ns,
            n3_scaled,
            linestyle="--",
            linewidth=2.5,
            color="#d62728",
            label="O(n³)"
        )

        plt.title("Порівняння реального часу з теоретичними складностями")
        plt.xlabel("Кількість вершин n")
        plt.ylabel("Час, секунди")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Додатковий нормалізований графік для наочного порівняння форм кривих.
        measured_norm = normalize_curve(mean_times, 1.0)
        nlogn_norm = normalize_curve(nlogn_raw, 1.0)
        n2_norm = normalize_curve(n2_raw, 1.0)
        n2logn_norm = normalize_curve(n2logn_raw, 1.0)
        n3_norm = normalize_curve(n3_raw, 1.0)

        plt.figure(figsize=(12, 7))
        plt.plot(ns, measured_norm, marker="o", linewidth=3, color="black", label="Реальний час (норм.)")
        plt.plot(ns, nlogn_norm, linestyle="--", linewidth=2.2, color="#1f77b4", label="O(n log n) (норм.)")
        plt.plot(ns, n2_norm, linestyle="--", linewidth=2.2, color="#2ca02c", label="O(n²) (норм.)")
        plt.plot(ns, n2logn_norm, linestyle="--", linewidth=2.2, color="#ff7f0e", label="O(n² log n) (норм.)")
        plt.plot(ns, n3_norm, linestyle="--", linewidth=2.2, color="#d62728", label="O(n³) (норм.)")
        plt.title("Порівняння форми росту (нормалізовано)")
        plt.xlabel("Кількість вершин n")
        plt.ylabel("Нормалізоване значення")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))

        names = list(errors.keys())
        values = list(errors.values())

        plt.bar(names, values)
        plt.title("Похибка наближення до теоретичних складностей")
        plt.xlabel("Теоретична складність")
        plt.ylabel("RMSE, менше — краще")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

        messagebox.showinfo(
            "Аналітика",
            "Найближча теоретична крива за RMSE: "
            + best_model
            + "\n\nПобудовано порівняння з O(n log n), O(n²), O(n² log n), O(n³)."
        )

    except Exception as e:
        messagebox.showerror("Помилка", str(e))


def clear_workspace():
    plt.close("all")
    messagebox.showinfo("Очищення", "Робочу область очищено.")


def exit_program():
    root.destroy()


root = tk.Tk()
root.title("Пошук найкоротшого шляху")
root.geometry("430x540")
root.resizable(False, False)

title_label = tk.Label(
    root,
    text="Пошук найкоротшого шляху\nв многокутнику",
    font=("Arial", 16, "bold"),
    justify="center"
)
title_label.pack(pady=25)

btn_random = tk.Button(
    root,
    text="Автоматична генерація даних",
    font=("Arial", 12),
    width=34,
    height=2,
    command=run_random_mode
)
btn_random.pack(pady=7)

btn_mouse = tk.Button(
    root,
    text="Намалювати многокутник мишкою",
    font=("Arial", 12),
    width=34,
    height=2,
    command=run_mouse_mode
)
btn_mouse.pack(pady=7)

btn_analytics = tk.Button(
    root,
    text="Аналітика",
    font=("Arial", 12),
    width=34,
    height=2,
    command=run_analytics_mode
)
btn_analytics.pack(pady=7)

btn_search_anim = tk.Button(
    root,
    text="Анімація пошуку шляху",
    font=("Arial", 12),
    width=34,
    height=2,
    command=run_search_animation_mode
)
btn_search_anim.pack(pady=7)

btn_clear = tk.Button(
    root,
    text="Очистити робочу область",
    font=("Arial", 12),
    width=34,
    height=2,
    command=clear_workspace
)
btn_clear.pack(pady=7)

btn_exit = tk.Button(
    root,
    text="Вихід",
    font=("Arial", 12),
    width=34,
    height=2,
    command=exit_program
)
btn_exit.pack(pady=7)

root.mainloop()