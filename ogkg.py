import math
import heapq
import matplotlib.pyplot as plt

polygon = [
    (1, 1),
    (12, 1),
    (12, 3),
    (9, 3),
    (9, 5),
    (11, 5),
    (11, 7),
    (7, 7),
    (7, 4),
    (5, 4),
    (5, 9),
    (12, 9),
    (12, 12),
    (1, 12),
    (1, 8),
    (3, 8),
    (3, 6),
    (1, 6)
]

S = (8, 2)
T = (10, 11)


def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


def orientation(a, b, c):
    """
    Орієнтація трьох точок.
    
    > 0  -> проти годинникової стрілки
    < 0  -> за годинниковою
    = 0  -> колінеарні
    """
    
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def on_segment(a, b, c):
    """
    Перевірка чи точка c лежить на відрізку ab
    """
    
    return (
        min(a[0], b[0]) <= c[0] <= max(a[0], b[0])
        and
        min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
    )


def segments_intersect(p1, q1, p2, q2):
    """
    Перевірка перетину двох відрізків
    """
    
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    if o1 == 0 and on_segment(p1, q1, p2):
        return True

    if o2 == 0 and on_segment(p1, q1, q2):
        return True

    if o3 == 0 and on_segment(p2, q2, p1):
        return True

    if o4 == 0 and on_segment(p2, q2, q1):
        return True

    return False


def point_in_polygon(point, polygon):
    """
    Ray Casting Algorithm
    """
    
    x, y = point
    inside = False

    n = len(polygon)

    p1x, p1y = polygon[0]

    for i in range(n + 1):

        p2x, p2y = polygon[i % n]

        if y > min(p1y, p2y):

            if y <= max(p1y, p2y):

                if x <= max(p1x, p2x):

                    if p1y != p2y:

                        xinters = (
                            (y - p1y) * (p2x - p1x)
                            / (p2y - p1y)
                            + p1x
                        )

                    if p1x == p2x or x <= xinters:
                        inside = not inside

        p1x, p1y = p2x, p2y

    return inside


def is_visible(a, b, polygon):

    n = len(polygon)

    
    for i in range(n):

        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]

        if p1 in [a, b] or p2 in [a, b]:
            continue

        if segments_intersect(a, b, p1, p2):
            return False

    
    mid = (
        (a[0] + b[0]) / 2,
        (a[1] + b[1]) / 2
    )

    return point_in_polygon(mid, polygon)



vertices = polygon + [S, T]

graph = {v: [] for v in vertices}

for i in range(len(vertices)):
    for j in range(i + 1, len(vertices)):

        a = vertices[i]
        b = vertices[j]

        if is_visible(a, b, polygon):

            d = distance(a, b)

            graph[a].append((b, d))
            graph[b].append((a, d))


def dijkstra(graph, start, end):

    pq = [(0, start)]

    distances = {v: float('inf') for v in graph}
    distances[start] = 0

    parent = {}

    while pq:

        current_dist, current = heapq.heappop(pq)

        if current == end:
            break

        for neighbor, weight in graph[current]:

            new_dist = current_dist + weight

            if new_dist < distances[neighbor]:

                distances[neighbor] = new_dist
                parent[neighbor] = current

                heapq.heappush(pq, (new_dist, neighbor))

    path = []
    current = end

    while current != start:
        path.append(current)
        current = parent[current]

    path.append(start)

    path.reverse()

    return path


path = dijkstra(graph, S, T)

px = [p[0] for p in polygon] + [polygon[0][0]]
py = [p[1] for p in polygon] + [polygon[0][1]]

plt.figure(figsize=(8, 8))
plt.plot(px, py, 'black', linewidth=2)

for p in polygon:
    plt.scatter(p[0], p[1], color='blue')

plt.scatter(S[0], S[1], color='green', s=100, label='Start')
plt.scatter(T[0], T[1], color='red', s=100, label='Target')

for v in graph:
    for neighbor, _ in graph[v]:

        plt.plot(
            [v[0], neighbor[0]],
            [v[1], neighbor[1]],
            color='gray',
            alpha=0.3
        )

for i in range(len(path)-1):

    a = path[i]
    b = path[i+1]

    plt.plot(
        [a[0], b[0]],
        [a[1], b[1]],
        color='orange',
        linewidth=4
    )

plt.legend()
plt.grid(True)
plt.axis('equal')
plt.title("Найкоротший шлях у многокутнику")

plt.show()


print("Найкоротший шлях:")
for p in path:
    print(p)