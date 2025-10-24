import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import time

# ##################################################################
# ---              ЛОГІКА АЛГОРИТМУ МУРАШИНОЇ КОЛОНІЇ (ACO)      ---
# ##################################################################
# (Клас ACO залишається без змін)
class ACO:
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.best_path = None
        self.best_distance = float('inf')

    def run(self):
        history = [] # Для можливого графіка збіжності
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheromone(all_paths)
            best_path_current_iter, best_distance_current_iter = self.get_best_path(all_paths)

            if best_distance_current_iter < self.best_distance:
                self.best_distance = best_distance_current_iter
                self.best_path = best_path_current_iter

            self.pheromone *= (1 - self.decay)
            history.append(self.best_distance)
            # print(f"Iter {i+1}/{self.n_iterations}, Best: {self.best_distance:.2f}")

        return self.best_path, self.best_distance, history # Повертаємо історію

    def spread_pheromone(self, all_paths):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:self.n_best]:
             # Перевірка на валідність відстані перед діленням
            if dist > 1e-9: # Уникаємо ділення на нуль або дуже малі числа
                pheromone_to_add = 1.0 / dist # Q=1 для простоти
                for move in path:
                    # Перевіряємо індекси перед доступом
                    if 0 <= move[0] < self.pheromone.shape[0] and 0 <= move[1] < self.pheromone.shape[1]:
                        self.pheromone[move] += pheromone_to_add
                    #else: print(f"Warning: Invalid move indices {move}") # Для дебагу


    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.get_distance(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = {start} # Використовуємо set для швидкої перевірки
        prev = start
        num_cities = len(self.distances)

        while len(visited) < num_cities:
            next_city = self.pick_next_city(prev, visited)
            # Перевірка, чи не повертаємось ми в те саме місто (може статися при помилках)
            if next_city == prev or next_city is None:
                # print(f"Warning: Stuck at city {prev}, visited {len(visited)}/{num_cities}")
                # Якщо застрягли, вибираємо випадкове невідвідане
                unvisited = [c for c in self.all_inds if c not in visited]
                if not unvisited: break # Всі відвідали
                next_city = random.choice(unvisited)

            path.append((prev, next_city))
            prev = next_city
            visited.add(next_city)

        path.append((prev, start)) # Повернення до старту
        return path


    def pick_next_city(self, prev, visited):
        pheromone = np.copy(self.pheromone[prev])
        # Зануляємо феромон для вже відвіданих міст
        for city in visited:
            if 0 <= city < len(pheromone): pheromone[city] = 0

        heuristic = 1.0 / (self.distances[prev] + 1e-10)

        row = pheromone ** self.alpha * heuristic ** self.beta

        # Зануляємо ймовірності для відвіданих (ще раз для надійності)
        for city in visited:
             if 0 <= city < len(row): row[city] = 0

        total = np.sum(row)
        if total == 0:
            unvisited_cities = [c for c in self.all_inds if c not in visited]
            return random.choice(unvisited_cities) if unvisited_cities else None # Повертаємо None якщо немає куди йти

        probabilities = row / total

        # Захист від NaN у ймовірностях
        if np.isnan(probabilities).any():
            # print(f"Warning: NaN detected in probabilities from city {prev}. Choosing randomly.")
            unvisited_cities = [c for c in self.all_inds if c not in visited]
            return random.choice(unvisited_cities) if unvisited_cities else None

        try:
            next_city = np.random.choice(self.all_inds, 1, p=probabilities)[0]
            return next_city
        except ValueError as e:
            # print(f"Error choosing next city from {prev}: {e}, probs sum: {np.sum(probabilities)}")
            # print(f"Probabilities: {probabilities}")
            # print(f"Row: {row}")
            # print(f"Pheromone: {pheromone}")
            # print(f"Heuristic: {heuristic}")
            # print(f"Visited: {visited}")
            unvisited_cities = [c for c in self.all_inds if c not in visited]
            return random.choice(unvisited_cities) if unvisited_cities else None


    def get_distance(self, path):
        total_distance = 0
        for ele in path:
            # Перевірка індексів
            if 0 <= ele[0] < self.distances.shape[0] and 0 <= ele[1] < self.distances.shape[1]:
                total_distance += self.distances[ele]
            #else: print(f"Warning: Invalid distance indices {ele}") # Для дебагу
        return total_distance

    def get_best_path(self, all_paths):
        best_dist = float('inf')
        best_path = None
        for path, dist in all_paths:
            if dist < best_dist:
                best_dist = dist
                best_path = path
        return best_path, best_dist


def calculate_distances(coords):
    num_nodes = len(coords)
    distances = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes): # Оптимізація
            dist = np.linalg.norm(coords[i] - coords[j])
            distances[i, j] = distances[j, i] = dist
    return distances

def generate_random_coords(num_nodes=30, x_range=(0, 100), y_range=(0, 100)):
    coords = np.random.rand(num_nodes, 2)
    coords[:, 0] = coords[:, 0] * (x_range[1] - x_range[0]) + x_range[0]
    coords[:, 1] = coords[:, 1] * (y_range[1] - y_range[0]) + y_range[0]
    return coords

# ##################################################################
# ---                   2-OPT ОПТИМІЗАЦІЯ (ПОКРАЩЕНО)            ---
# ##################################################################

def two_opt_swap(path_list, i, k):
    """Виконує 2-opt обмін на шляху (список міст)."""
    # [0, 1, 2, 3, 4, 5] i=1, k=3 -> [0, 3, 2, 1, 4, 5]
    new_path = path_list[:i] + path_list[i:k+1][::-1] + path_list[k+1:]
    return new_path

def calculate_path_distance_from_list(path_list, distances_matrix):
    """Обчислює відстань шляху, представленого списком міст."""
    dist = 0
    num_cities = len(path_list)
    for i in range(num_cities):
        dist += distances_matrix[path_list[i]][path_list[(i + 1) % num_cities]] # Циклічний шлях
    return dist

def two_opt_optimize(initial_path_aco_format, distances_matrix, max_iterations=10000): # Збільшено ітерації
    """
    Застосовує 2-opt оптимізацію до шляху, щоб прибрати перетини.
    """
    if not initial_path_aco_format:
        return initial_path_aco_format, float('inf')

    # Перетворюємо ACO-формат шляху на список міст [0, 1, 2, ...]
    current_path_list = [initial_path_aco_format[0][0]]
    node_set = {initial_path_aco_format[0][0]}
    last_node = initial_path_aco_format[0][0]
    # Надійніший спосіб побудови списку, якщо ACO повернув неповний шлях
    while len(current_path_list) < len(distances_matrix):
         found_next = False
         for u, v in initial_path_aco_format:
             if u == last_node and v not in node_set:
                 current_path_list.append(v)
                 node_set.add(v)
                 last_node = v
                 found_next = True
                 break
             # Можливо, шлях був записаний у зворотньому порядку
             elif v == last_node and u not in node_set:
                  current_path_list.append(u)
                  node_set.add(u)
                  last_node = u
                  found_next = True
                  break
         if not found_next:
             # print("Warning: Could not reconstruct full path list from ACO format.")
             # Додаємо вузли, яких не вистачає, у випадковому порядку (якщо таке можливо)
             remaining_nodes = [n for n in range(len(distances_matrix)) if n not in node_set]
             random.shuffle(remaining_nodes)
             current_path_list.extend(remaining_nodes)
             break # Завершуємо реконструкцію

    best_path_list = list(current_path_list)
    best_distance = calculate_path_distance_from_list(best_path_list, distances_matrix)

    improved = True
    iterations = 0

    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        for i in range(len(best_path_list) - 1): # Починаємо з 0
            for k in range(i + 1, len(best_path_list)):
                # Перевіряємо, чи обмін покращить шлях
                # Замість повного обміну і розрахунку, перевіряємо лише зміну відстані
                # delta = dist(i,k) + dist(i+1,k+1) - dist(i,i+1) - dist(k,k+1)
                # (Використовуємо % для циклічного шляху)
                n = len(best_path_list)
                node_i, node_i1 = best_path_list[i], best_path_list[(i + 1) % n]
                node_k, node_k1 = best_path_list[k], best_path_list[(k + 1) % n]

                current_dist = distances_matrix[node_i, node_i1] + distances_matrix[node_k, node_k1]
                new_dist = distances_matrix[node_i, node_k] + distances_matrix[node_i1, node_k1]

                if new_dist < current_dist:
                    # Якщо покращення є, виконуємо обмін
                    best_path_list = two_opt_swap(best_path_list, (i + 1) % n, k) # Обмін сегменту між i+1 та k
                    best_distance = calculate_path_distance_from_list(best_path_list, distances_matrix) # Перераховуємо повну відстань
                    improved = True
                    # Перезапускаємо внутрішній цикл k після покращення
                    break # Переходимо до наступного i
            if improved:
                 continue # Перезапускаємо зовнішній цикл i, щоб перевірити всі можливості знову

    # Перетворюємо оптимізований список міст назад в ACO-формат
    optimized_path_aco_format = []
    n = len(best_path_list)
    for i in range(n):
        start_node = best_path_list[i]
        end_node = best_path_list[(i + 1) % n] # Циклічний шлях
        optimized_path_aco_format.append((start_node, end_node))

    return optimized_path_aco_format, best_distance


# ##################################################################
# ---               ГОЛОВНА ФУНКЦІЯ ДЛЯ СТВОРЕННЯ ВКЛАДКИ         ---
# ##################################################################

def create_tab(tab_control):
    """
    Створює вміст для третьої вкладки (Оптимізація потоків у мережі)
    """

    tab3 = ttk.Frame(tab_control, padding=(10, 10))
    tab_control.add(tab3, text='Завдання 3: Оптимізація Потоків (ACO)')

    # --- 1. Створення фреймів ---
    main_frame = ttk.Frame(tab3)
    main_frame.pack(fill="both", expand=True)

    # Ліва колонка (Налаштування)
    left_frame = ttk.Frame(main_frame, width=400)
    left_frame.pack(side="left", fill="y", padx=10, expand=False)
    left_frame.pack_propagate(False)

    # Права колонка (Графік)
    right_frame = ttk.LabelFrame(main_frame, text="Візуалізація Оптимізації Шляху (ACO + 2-opt)", padding=10)
    right_frame.pack(side="right", fill="both", expand=True)

    # --- 2. Ліва колонка (Налаштування ACO) ---
    settings_frame = ttk.LabelFrame(left_frame, text="Налаштування ACO", padding=10)
    settings_frame.pack(fill="x", pady=5)

    ttk.Label(settings_frame, text="Кількість підстанцій (N):").grid(row=0, column=0, sticky='w', pady=2)
    num_nodes_var = tk.IntVar(value=30)
    ttk.Entry(settings_frame, textvariable=num_nodes_var, width=10).grid(row=0, column=1, sticky='ew', padx=5, pady=2)

    ttk.Label(settings_frame, text="Кількість мурах:").grid(row=1, column=0, sticky='w', pady=2)
    n_ants_var = tk.IntVar(value=50)
    ttk.Entry(settings_frame, textvariable=n_ants_var, width=10).grid(row=1, column=1, sticky='ew', padx=5, pady=2)

    ttk.Label(settings_frame, text="Кількість найкращих мурах:").grid(row=2, column=0, sticky='w', pady=2)
    n_best_var = tk.IntVar(value=10)
    ttk.Entry(settings_frame, textvariable=n_best_var, width=10).grid(row=2, column=1, sticky='ew', padx=5, pady=2)

    ttk.Label(settings_frame, text="Кількість ітерацій ACO:").grid(row=3, column=0, sticky='w', pady=2)
    n_iterations_var = tk.IntVar(value=200) # Зменшено для швидкості, 2-opt виправить
    ttk.Entry(settings_frame, textvariable=n_iterations_var, width=10).grid(row=3, column=1, sticky='ew', padx=5, pady=2)

    ttk.Label(settings_frame, text="Випаровування феромону (decay):").grid(row=4, column=0, sticky='w', pady=2)
    decay_var = tk.DoubleVar(value=0.1)
    ttk.Entry(settings_frame, textvariable=decay_var, width=10).grid(row=4, column=1, sticky='ew', padx=5, pady=2)

    ttk.Label(settings_frame, text="Важливість феромону (alpha):").grid(row=5, column=0, sticky='w', pady=2)
    alpha_var = tk.DoubleVar(value=1.0)
    ttk.Entry(settings_frame, textvariable=alpha_var, width=10).grid(row=5, column=1, sticky='ew', padx=5, pady=2)

    ttk.Label(settings_frame, text="Важливість відстані (beta):").grid(row=6, column=0, sticky='w', pady=2)
    beta_var = tk.DoubleVar(value=5.0)
    ttk.Entry(settings_frame, textvariable=beta_var, width=10).grid(row=6, column=1, sticky='ew', padx=5, pady=2)

    settings_frame.grid_columnconfigure(1, weight=1)

    # --- 3. Права колонка (Графік) ---
    fig = Figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    result_label = ttk.Label(right_frame, text="Найкращий шлях: N/A\nВтрати: N/A",
                             font=("Helvetica", 10), wraplength=500, justify="left", padding=(5,5))
    result_label.pack(side=tk.BOTTOM, fill="x")

    # --- 4. Функція запуску ACO та 2-opt ---
    def run_aco_optimization_fixed():
        print("Запуск ACO оптимізації (Вкладка 3)...")
        num_nodes = num_nodes_var.get()
        n_ants = n_ants_var.get()
        n_best = n_best_var.get()
        n_iterations = n_iterations_var.get()
        decay = decay_var.get()
        alpha = alpha_var.get()
        beta = beta_var.get()

        coords = generate_random_coords(num_nodes)
        distances = calculate_distances(coords)

        # Перевірка на валідність distances
        if np.any(distances < 0):
             print("Error: Negative distances detected.")
             result_label.config(text="Помилка: Від'ємні відстані!")
             return
        if np.isnan(distances).any():
             print("Error: NaN distances detected.")
             result_label.config(text="Помилка: NaN відстані!")
             return


        start_time = time.time()
        aco_solver = ACO(distances, n_ants, n_best, n_iterations, decay, alpha, beta)
        best_path_aco, best_distance_aco, history = aco_solver.run()
        aco_time = time.time() - start_time
        print(f"ACO finished in {aco_time:.2f}s. Initial dist: {best_distance_aco:.2f}")

        # Перевірка чи ACO повернув шлях
        if best_path_aco is None:
            print("Error: ACO did not return a valid path.")
            result_label.config(text="Помилка: ACO не знайшов шлях!")
            return

        start_time_2opt = time.time()
        # --- ВИПРАВЛЕНО: Передаємо правильний формат шляху ---
        optimized_path_aco_format, optimized_distance = two_opt_optimize(best_path_aco, distances)
        two_opt_time = time.time() - start_time_2opt
        print(f"2-opt finished in {two_opt_time:.2f}s. Optimized dist: {optimized_distance:.2f}")

        ax.clear()
        ax.plot(coords[:, 0], coords[:, 1], 'o', markersize=5, color='blue', label='Підстанції')
        for i, (x, y) in enumerate(coords):
            ax.text(x + 1.5, y + 1.5, str(i), fontsize=9, color='black')

        # --- ВИПРАВЛЕНО: Правильна побудова шляху з ACO-формату ---
        path_coords_list = []
        if optimized_path_aco_format:
            start_node = optimized_path_aco_format[0][0]
            path_coords_list.append(coords[start_node])
            current_node = start_node
            # Реконструюємо шлях, слідуючи ребрам
            visited_edges = set()
            for _ in range(len(optimized_path_aco_format)):
                 found_edge = False
                 for u, v in optimized_path_aco_format:
                     edge = tuple(sorted((u,v)))
                     if edge not in visited_edges:
                         if u == current_node:
                             path_coords_list.append(coords[v])
                             visited_edges.add(edge)
                             current_node = v
                             found_edge = True
                             break
                         elif v == current_node: # На випадок зворотнього запису
                              path_coords_list.append(coords[u])
                              visited_edges.add(edge)
                              current_node = u
                              found_edge = True
                              break
                 if not found_edge: break # Якщо шлях обірвався
            # Замикаємо цикл, якщо потрібно
            if len(path_coords_list) == num_nodes:
                 path_coords_list.append(path_coords_list[0])

        if path_coords_list:
             path_plot = np.array(path_coords_list)
             ax.plot(path_plot[:, 0], path_plot[:, 1], 'r-', linewidth=1.5, label=f'Найкращий шлях (Втрати: {optimized_distance:.2f})')
        else:
             print("Warning: Could not plot optimized path.")


        ax.set_title("Оптимізація потоків у мережі (ACO + 2-opt)")
        ax.set_xlabel("Координата X")
        ax.set_ylabel("Координата Y")
        ax.legend(fontsize='small')
        ax.grid(True)
        canvas.draw()

        result_label.config(text=f"Найкращий шлях (після 2-opt):\nВтрати: {optimized_distance:.2f}")
        print("ACO оптимізація та 2-opt завершено.")


    # --- 5. Кнопка запуску ---
    run_button = ttk.Button(left_frame, text="Запустити Оптимізацію", command=run_aco_optimization_fixed)
    run_button.pack(side="bottom", fill="x", pady=20)

    # Запускаємо 1 раз при старті
    run_aco_optimization_fixed()