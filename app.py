from flask import Flask, render_template, request, flash, redirect
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import random
import os

matplotlib.use('Agg')  # Используем неблокирующий бэкенд

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Необходим для использования flash сообщений

STATIC_DIR = r'C:\Users\admin\PycharmProjects\Bellman-Ford LAB\static'

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.edges = []

    def add_edge(self, u, v, w):
        self.edges.append((u, v, w))

    def bellman_ford(self, src):
        distances = [float('inf')] * self.V
        distances[src] = 0
        predecessors = [-1] * self.V
        steps = []

        for _ in range(self.V - 1):
            for u, v, w in self.edges:
                if distances[u] != float('inf') and distances[u] + w < distances[v]:
                    distances[v] = distances[u] + w
                    predecessors[v] = u
                    steps.append((distances.copy(), u, v))

        for u, v, w in self.edges:
            if distances[u] != float('inf') and distances[u] + w < distances[v]:
                return None, steps

        return distances, steps, predecessors


def get_shortest_path(predecessors, target):
    path = []
    while target != -1:
        path.append(target)
        target = predecessors[target]
    return path[::-1]


def check_write_permission():
    test_file_path = os.path.join(STATIC_DIR, 'test_file.txt')
    try:
        with open(test_file_path, 'w') as f:
            f.write("Test")
        os.remove(test_file_path)  # Удаляем тестовый файл
        return True
    except Exception:
        return False


def generate_random_tree(vertices):
    edges = []
    for i in range(1, vertices):
        u = random.randint(0, i - 1)
        v = i
        w = random.randint(1, 10)
        edges.append((u, v, w))
    return edges


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'generate_random_tree' in request.form:
            # Обработка генерации случайного дерева
            try:
                vertices = int(request.form['vertices'])
                if vertices <= 0:
                    flash("Количество вершин должно быть положительным.", "error")
                    return redirect('/')

                edges = generate_random_tree(vertices)
                graph = Graph(vertices)
                for u, v, w in edges:
                    graph.add_edge(u, v, w)

                distances, steps, predecessors = graph.bellman_ford(0)
                if distances is None:
                    flash("Граф содержит отрицательный цикл.", "error")
                    return redirect('/')
                else:
                    result = {i: dist for i, dist in enumerate(distances)}
                    shortest_path = get_shortest_path(predecessors, vertices - 1)
                    visualize(graph, steps, shortest_path)
                    steps_count = len(steps)

                return render_template('index.html', result=result, steps_count=steps_count)

            except ValueError:
                flash("Ошибка при генерации дерева.", "error")
                return redirect('/')

        try:
            # Обычная обработка ввода пользователя
            vertices = int(request.form['vertices'])
            start_vertices = request.form.getlist('start_vertices')
            end_vertices = request.form.getlist('end_vertices')
            weights = request.form.getlist('weights')

            if vertices <= 0:
                flash("Количество вершин должно быть положительным.", "error")
                return redirect('/')

            graph = Graph(vertices)

            for u, v, w in zip(start_vertices, end_vertices, weights):
                u, v, w = int(u), int(v), int(w)

                if u < 0 or u >= vertices or v < 0 or v >= vertices:
                    flash(
                        f"Ошибка: Вершина {u} или {v} вне диапазона. Вершины должны быть в диапазоне 0 до {vertices - 1}.",
                        "error")
                    return redirect('/')

                graph.add_edge(u, v, w)

            distances, steps, predecessors = graph.bellman_ford(0)
            if distances is None:
                flash("Граф содержит отрицательный цикл.", "error")
                return redirect('/')
            else:
                result = {i: dist for i, dist in enumerate(distances)}
                shortest_path = get_shortest_path(predecessors, vertices - 1)
                visualize(graph, steps, shortest_path)
                steps_count = len(steps)

            return render_template('index.html', result=result, steps_count=steps_count)

        except ValueError:
            flash("Пожалуйста, убедитесь, что вы ввели числа правильно.", "error")
            return redirect('/')

    return render_template('index.html')


def visualize(graph, steps, shortest_path):
    G = nx.DiGraph()
    for u, v, w in graph.edges:
        G.add_edge(u, v, weight=w)

    pos = nx.spring_layout(G)

    # Сохраняем изображение начального состояния графа
    plt.figure()
    nx.draw(G, pos, with_labels=True, node_color='lightblue', arrows=True)
    edge_labels = {(u, v): f"{w}" for u, v, w in graph.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Исходный граф")
    plt.savefig(os.path.join(STATIC_DIR, 'original_graph.png'))
    plt.close()

    # Визуализация шагов
    for i, step in enumerate(steps):
        distances, u, v = step
        plt.figure()
        nx.draw(G, pos, with_labels=True, node_color='lightblue', arrows=True)
        edge_labels = {(u, v): f"{w}" for u, v, w in graph.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        # Отображаем расстояния
        distance_labels = {i: str(dist) for i, dist in enumerate(distances)}
        for node, label in distance_labels.items():
            plt.text(pos[node][0], pos[node][1] + 0.05, f"dist: {label}", fontsize=10, ha='center')

        plt.title(f"Шаг: Обновление расстояний от {u} до {v}")
        plt.savefig(os.path.join(STATIC_DIR, f'step_{i + 1}.png'))
        plt.close()

    # Визуализируем кратчайший путь
    plt.figure()
    edge_colors = ['red' if (u in shortest_path and v in shortest_path and abs(
        shortest_path.index(v) - shortest_path.index(u)) == 1) else 'black' for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, edge_color=edge_colors, node_color='lightblue', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Кратчайший путь")
    plt.savefig(os.path.join(STATIC_DIR, 'shortest_path.png'))
    plt.close()


if __name__ == '__main__':
    if not os.path.exists(STATIC_DIR):
        os.makedirs(STATIC_DIR)
    if check_write_permission():
        print("Разрешение на запись в папку 'static' доступно.")
    else:
        print("Нет разрешения на запись в папку 'static'.")
    app.run(debug=True)

def table():
    pass
table()