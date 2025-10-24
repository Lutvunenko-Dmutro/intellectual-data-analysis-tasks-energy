import tkinter as tk
from tkinter import ttk
import numpy as np
import random
import collections
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def create_tab(tab_control):
    """
    Створює вміст для другої вкладки (Оператори відбору)
    і прикріплює його до 'tab_control'.
    """
    
    GENERATORS = [
        {'name': 'АЕС-1 (База)', 'capacity': 1000, 'cost': 15},
        {'name': 'ТЕС-1 (Вугілля)', 'capacity': 300, 'cost': 50},
        {'name': 'ТЕС-2 (Газ)', 'capacity': 250, 'cost': 75},
        {'name': 'ГЕС-1 (Пік)', 'capacity': 150, 'cost': 30},
        {'name': 'ТЕС-3 (Вугілля)', 'capacity': 350, 'cost': 55},
        {'name': 'ГЕС-2 (Маневр)', 'capacity': 100, 'cost': 35},
        {'name': 'СЕС (Сонце)', 'capacity': 200, 'cost': 20},
        {'name': 'ВЕС (Вітер)', 'capacity': 150, 'cost': 25}
    ]
    N_GENERATORS = len(GENERATORS)
    POP_SIZE = 20
    
    current_population_fitness = []

    tab2 = ttk.Frame(tab_control, padding=(10, 10))
    tab_control.add(tab2, text='Завдання 2: Оптимізація (ГА)')

    main_frame = ttk.Frame(tab2)
    main_frame.pack(fill="both", expand=True)

    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side="left", fill="y", padx=10, expand=True)
    
    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side="right", fill="both", expand=True)

    sim_control_frame = ttk.LabelFrame(left_frame, text="Параметри симуляції (Unit Commitment)", padding=10)
    sim_control_frame.pack(fill="x", pady=(0, 10))

    ttk.Label(sim_control_frame, text="Цільове навантаження (МВт):").grid(row=0, column=0, sticky="w", padx=5)
    target_load_var = tk.IntVar(value=1500)
    ttk.Entry(sim_control_frame, textvariable=target_load_var, width=10).grid(row=0, column=1, sticky="w", padx=5)
    
    pop_frame = ttk.LabelFrame(left_frame, text=f"Популяція (Графіки роботи, N={POP_SIZE})", padding=10)
    pop_frame.pack(fill="both", expand=True)
    
    pop_cols = ('id', 'chrom', 'gen', 'cost', 'fitness')
    tree_pop = ttk.Treeview(pop_frame, columns=pop_cols, show='headings', height=10)
    
    tree_pop.column('id', width=30, anchor='center'); tree_pop.heading('id', text='ID')
    tree_pop.column('chrom', width=150, anchor='w'); tree_pop.heading('chrom', text='Графік (Хромосома)')
    tree_pop.column('gen', width=110, anchor='e'); tree_pop.heading('gen', text='Генерація (МВт)')
    tree_pop.column('cost', width=100, anchor='e'); tree_pop.heading('cost', text='Вартість ($)')
    tree_pop.column('fitness', width=120, anchor='e'); tree_pop.heading('fitness', text='Фітнес (1/Варт)')
    
    pop_scrollbar = ttk.Scrollbar(pop_frame, orient="vertical", command=tree_pop.yview)
    tree_pop.configure(yscrollcommand=pop_scrollbar.set)
    pop_scrollbar.pack(side="right", fill="y")
    tree_pop.pack(fill="both", expand=True)

    theory_frame = ttk.LabelFrame(left_frame, text="Порівняльний аналіз (Теоретична відповідь)", padding=10)
    theory_frame.pack(fill="x", pady=10)

    table_headers = ['Критерій', 'Пропорційний', 'Ранжирування', 'Турнірний', 'Відсікання']
    table_data = [
        ['Принцип', 'Ймовірність = fitness / Σ(fitness)', 'Ймовірність = f(ранг)', 'Переможець з T випадкових', 'Вибираються k найкращих'],
        ['Швидкодія', 'O(N) або O(N log N)', 'O(N log N) (сортування)', 'O(k*T) (Дуже швидко)', 'O(N log N) (сортування)'],
        ['Селективний тиск', 'Високий (залежить від розкиду)', 'Контрольований (фіксований)', 'Налаштовуваний (через T)', 'Максимальний'],
        ['Різноманітність', 'Низька (домінування)', 'Вища (нівелює супер-осіб)', 'Контрольована', 'Дуже низька'],
        ['Проблеми', 'Чутливий до "супер-осіб"', '-', '-', 'Швидка втрата різноманітності']
    ]
    
    cols = ('col1', 'col2', 'col3', 'col4', 'col5')
    tree_theory = ttk.Treeview(theory_frame, columns=cols, show='headings', height=6)
    
    tree_theory.column('col1', width=110, anchor='w'); tree_theory.heading('col1', text=table_headers[0])
    tree_theory.column('col2', width=180, anchor='w'); tree_theory.heading('col2', text=table_headers[1])
    tree_theory.column('col3', width=180, anchor='w'); tree_theory.heading('col3', text=table_headers[2])
    tree_theory.column('col4', width=180, anchor='w'); tree_theory.heading('col4', text=table_headers[3])
    tree_theory.column('col5', width=180, anchor='w'); tree_theory.heading('col5', text=table_headers[4])
    
    tree_theory.tag_configure('oddrow', background='#f0f0f0')
    tree_theory.tag_configure('evenrow', background='white')

    for i, row in enumerate(table_data):
        tag = 'evenrow' if i % 2 == 0 else 'oddrow'
        tree_theory.insert('', 'end', values=row, tags=(tag,))
    
    tree_theory.pack(fill="x", expand=True)
    
    footnote = ttk.Label(theory_frame, 
                         text="*Селективний тиск - наскільки сильно алгоритм надає перевагу найкращим особам.",
                         font=("Helvetica", 9, "italic"),
                         wraplength=700)
    footnote.pack(side="left", fill="x", pady=(5, 0))
    
    control_frame = ttk.Frame(right_frame)
    control_frame.pack(fill="x", pady=5)
    
    ttk.Label(control_frame, text="Кількість батьків (k):").pack(side="left", padx=(0, 5))
    k_var = tk.IntVar(value=10)
    ttk.Entry(control_frame, textvariable=k_var, width=5).pack(side="left", padx=5)

    ttk.Label(control_frame, text="Розмір турніру (T):").pack(side="left", padx=(10, 5))
    t_var = tk.IntVar(value=3)
    ttk.Entry(control_frame, textvariable=t_var, width=5).pack(side="left", padx=5)

    ttk.Label(control_frame, text="К-ть симуляцій:").pack(side="left", padx=(10, 5))
    n_sim_var = tk.IntVar(value=1000)
    ttk.Entry(control_frame, textvariable=n_sim_var, width=7).pack(side="left", padx=5)
    
    fig = Figure(figsize=(8, 8), dpi=100)
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def proportional_selection(pop, k):
        total_fitness = sum(f for _, f in pop)
        if total_fitness == 0:
            indices = np.random.choice(len(pop), size=k)
        else:
            probs = [f / total_fitness for _, f in pop]
            indices = np.random.choice(len(pop), size=k, p=probs)
        return [pop[i] for i in indices]

    def ranking_selection(pop, k):
        sorted_pop = sorted(pop, key=lambda x: x[1])
        ranks = list(range(1, len(pop) + 1))
        total_ranks = sum(ranks)
        if total_ranks == 0:
            indices = np.random.choice(len(pop), size=k)
        else:
            probs = [r / total_ranks for r in ranks]
            indices = np.random.choice(len(pop), size=k, p=probs)
        return [sorted_pop[i] for i in indices]

    def tournament_selection(pop, k, t_size):
        selected = []
        for _ in range(k):
            tournament_group = random.sample(pop, t_size)
            winner = max(tournament_group, key=lambda x: x[1])
            selected.append(winner)
        return selected

    def threshold_truncation_selection(pop, k):
        sorted_pop = sorted(pop, key=lambda x: x[1], reverse=True)
        return sorted_pop[:k]

    def calculate_fitness(chromosome, target_load):
        total_gen = 0
        total_cost = 0
        for i, gene in enumerate(chromosome):
            if gene == 1:
                total_gen += GENERATORS[i]['capacity']
                total_cost += GENERATORS[i]['cost'] * GENERATORS[i]['capacity']
        
        deficit = max(0, target_load - total_gen)
        penalty = deficit * 10000 
        final_cost = total_cost + penalty
        fitness = 1.0 / (final_cost + 1e-6)
        
        return fitness, total_gen, final_cost

    def generate_and_run_simulation():
        """Головна функція: генерує популяцію ТА запускає симуляцію"""
        nonlocal current_population_fitness
        print("Запуск симуляції для Вкладки 2 (Енерго)...")
        
        TARGET_LOAD = target_load_var.get()
        K_PARENTS = k_var.get()
        T_SIZE = t_var.get()
        N_SIMULATIONS = n_sim_var.get()

        pop_data_list = []
        for i in range(POP_SIZE):
            chrom = np.random.randint(0, 2, N_GENERATORS)
            fitness, total_gen, final_cost = calculate_fitness(chrom, TARGET_LOAD)
            pop_data_list.append( (i, str(chrom), total_gen, f"${final_cost:,.0f}", round(fitness, 8)) )
        
        tree_pop.delete(*tree_pop.get_children())
        for row in pop_data_list:
            tree_pop.insert('', 'end', values=row)
        
        current_population_fitness = [(row[0], row[4]) for row in pop_data_list]
        
        prop_counts = collections.Counter()
        rank_counts = collections.Counter()
        tourn_counts = collections.Counter()
        trunc_counts = collections.Counter()

        for _ in range(N_SIMULATIONS):
            for p in proportional_selection(current_population_fitness, K_PARENTS):
                prop_counts[p[0]] += 1
            for p in ranking_selection(current_population_fitness, K_PARENTS):
                rank_counts[p[0]] += 1
            for p in tournament_selection(current_population_fitness, K_PARENTS, T_SIZE):
                tourn_counts[p[0]] += 1
            for p in threshold_truncation_selection(current_population_fitness, K_PARENTS):
                trunc_counts[p[0]] += 1

        all_axes = [ax1, ax2, ax3, ax4]
        titles = ['1. Пропорційний (Рулетка)', '2. Ранжирування', 
                  f'3. Турнірний (T={T_SIZE})', f'4. Відсікання (Top-K={K_PARENTS})']
        data_counts = [prop_counts, rank_counts, tourn_counts, trunc_counts]
        ids = [p[0] for p in current_population_fitness]
        
        for ax, title, counts in zip(all_axes, titles, data_counts):
            ax.clear()
            plot_data = [counts[i] for i in ids]
            ax.bar(ids, plot_data, color='blue', alpha=0.7)
            ax.set_title(title, fontsize=10)
            ax.set_ylabel("К-ть разів обрано", fontsize=8)
            ax.set_xlabel("ID Особини", fontsize=8)
            ax.set_xticks(ids[::2])
        
        canvas.draw()
        print("Симуляція Вкладки 2 (Енерго) завершена.")

    run_button = ttk.Button(control_frame, text="Запустити симуляцію", command=generate_and_run_simulation)
    run_button.pack(side="left", padx=10)
    
    gen_button = ttk.Button(sim_control_frame, text="Згенерувати нову популяцію", command=generate_and_run_simulation)
    gen_button.grid(row=0, column=2, sticky="w", padx=10)
    
    generate_and_run_simulation()
