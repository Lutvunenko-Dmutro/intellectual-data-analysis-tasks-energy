import tkinter as tk
from tkinter import ttk, scrolledtext
import random

class BaseAgent:
    """Базовий клас для всіх агентів."""
    def __init__(self, name):
        self.name = name
        self.log_widget = None 

    def log(self, message, tag='agent_msg'): 
        """Допоміжна функція для виводу в лог симуляції."""
        if self.log_widget:
            self.log_widget.config(state='normal')
            self.log_widget.insert(tk.END, f"[{self.name}]: {message}\n", tag)
            self.log_widget.see(tk.END) # Автопрокрутка
            self.log_widget.config(state='disabled')

class GeneratorAgent(BaseAgent):
    """
    Агент-Генератор. 
    Функція: Спеціалізація (тільки генерує).
    """
    def __init__(self, name, capacity_mw, cost_per_mw, min_output=0.3):
        super().__init__(name)
        self.capacity = capacity_mw
        self.cost = cost_per_mw
        self.min_output = min_output * capacity_mw
        self.is_on = False
        self.output = 0

    def get_bid(self):
        """Для Колективного Рішення (Аукціон). Повертає свою ціну."""
        self.log(f"Подаю заявку на аукціон. Моя ціна: {self.cost}$/МВт")
        return self.cost

    def coordinate(self, command, target_output=0):
        """Для Координації (Наказ). Виконує наказ оператора."""
        if command == "ON":
            self.is_on = True
            self.output = max(self.min_output, target_output) 
            self.log(f"Виконую наказ 'ON'. Запускаюсь на {self.output:.0f} МВт.")
        elif command == "OFF":
            self.is_on = False
            self.output = 0
            self.log(f"Виконую наказ 'OFF'. Зупиняюсь.")
        return self.output

class ConsumerAgent(BaseAgent):
    """
    Агент-Споживач.
    Функція: Спеціалізація (тільки споживає).
    """
    def __init__(self, name, base_demand_mw, price_sensitivity=0.1):
        super().__init__(name)
        self.base_demand = base_demand_mw
        self.price_sensitivity = price_sensitivity 

    def get_demand(self, current_price):
        """Розраховує попит, враховуючи ціну."""
        if current_price < 80:
            demand = self.base_demand
        else:
            demand = self.base_demand * (1 - (current_price / (100 / self.price_sensitivity)))
            demand = max(0, demand) 
        
        self.log(f"Поточна ціна {current_price:.0f}$. Мій попит: {demand:.0f} МВт.")
        return demand

class SystemOperatorAgent(BaseAgent):
    """
    Агент-Оператор (Наша Система Моніторингу).
    Функції: Координація, Колективне Рішення.
    """
    def __init__(self, name, generators, consumers):
        super().__init__(name)
        self.generators = generators
        self.consumers = consumers
        for agent_list in [generators, consumers]:
            for agent in agent_list:
                agent.log_widget = self.log_widget

    def log(self, message, tag='operator_msg'):
        """Перевизначаємо log, щоб змінити тег за замовчуванням."""
        super().log(message, tag)

    def run_coordination(self, total_demand):
        """
        Симуляція КООРДИНАЦІЇ (Централізований наказ).
        """
        self.log(f"--- РЕЖИМ: 1. КООРДИНАЦІЯ (НАКАЗ) ---", 'header_msg')
        self.log(f"Цільовий попит: {total_demand:.0f} МВт.")
        
        sorted_gens = sorted(self.generators, key=lambda g: g.cost)
        covered_demand = 0
        total_cost = 0
        
        for gen in sorted_gens:
            if covered_demand < total_demand:
                needed = total_demand - covered_demand
                output = gen.coordinate("ON", needed)
                covered_demand += output
                total_cost += output * gen.cost
            else:
                gen.coordinate("OFF")

        self.log(f"--- Результат Координації ---", 'header_msg')
        self.log(f"Покрито: {covered_demand:.0f} / {total_demand:.0f} МВт.", 'result_msg')
        self.log(f"Загальна вартість (погодинна): ${total_cost:,.0f}", 'result_msg')
        return total_cost

    def run_collective_decision(self, total_demand):
        """
        Симуляція КОЛЕКТИВНОГО ПРИЙНЯТТЯ РІШЕНЬ (Аукціон).
        """
        self.log(f"--- РЕЖИМ: 2. КОЛЕКТИВНЕ РІШЕННЯ (АУКЦІОН) ---", 'header_msg')
        self.log(f"Оголошую аукціон на {total_demand:.0f} МВт.")
        
        bids = []
        for gen in self.generators:
            bids.append((gen.name, gen.get_bid(), gen.capacity, gen))
            
        sorted_bids = sorted(bids, key=lambda b: b[1])
        covered_demand = 0
        total_cost = 0
        
        self.log(f"Результати аукціону (від найдешевшого):")
        for name, cost, capacity, agent in sorted_bids:
            if covered_demand < total_demand:
                self.log(f"ПЕРЕМОЖЕЦЬ: {name} (Ціна: {cost}$) - вмикаємо.", 'result_msg')
                agent.coordinate("ON", capacity)
                covered_demand += capacity
                total_cost += capacity * cost
            else:
                self.log(f"ВІДХИЛЕНО (дорого): {name} (Ціна: {cost}$) - вимкнено.", 'agent_msg')
                agent.coordinate("OFF")
                
        self.log(f"--- Результат Аукціону ---", 'header_msg')
        self.log(f"Покрито: {covered_demand:.0f} / {total_demand:.0f} МВт.", 'result_msg')
        self.log(f"Загальна вартість (погодинна): ${total_cost:,.0f}", 'result_msg')
        return total_cost

def create_tab(tab_control):
    """
    Створює вміст для четвертої вкладки (Функції Агентів)
    і прикріплює його до 'tab_control'.
    """
    
    tab4 = ttk.Frame(tab_control, padding=(10, 10))
    tab_control.add(tab4, text='Завдання 4: Функції Агентів')

    main_frame = ttk.Frame(tab4)
    main_frame.pack(fill="both", expand=True)

    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side="left", fill="y", padx=10, expand=False) 
    
    right_frame = ttk.LabelFrame(main_frame, text="Симуляція: Архітектура Smart Grid", padding=10)
    right_frame.pack(side="right", fill="both", expand=True)

    theory_frame = ttk.LabelFrame(left_frame, text="Аналіз функцій агентів (Теоретична відповідь)", padding=10)
    theory_frame.pack(fill="x", pady=(0, 10))

    table_data = [
        ['Спеціалізація', 'Агент виконує вузьке, чітко визначене завдання.', 'Агент-Генератор (тільки генерує), Агент-Споживач (тільки споживає).'],
        ['Координація', 'Централізоване управління для досягнення мети (через "Оператора").', "Оператор системи віддає прямий наказ ('Coordinate') генераторам для балансування мережі."],
        ['Кооперація', 'Агенти добровільно працюють разом над спільною метою (часто децентралізовано).', 'Агенти-Споживачі "кооперують", добровільно знижуючи попит при високій ціні.'],
        ['Колективне рішення', 'Агенти спільно приймають рішення (напр., голосування, аукціон).', 'Аукціон "на добу наперед", де генератори пропонують ціни, і система *колективно* обирає найдешевший мікс.']
    ]
    
    definitions_container = ttk.Frame(theory_frame)
    definitions_container.pack(fill="x", expand=True)
    
    WRAP_WIDTH = 580 

    for item in table_data:
        func_name, description, example = item
        
        func_frame = ttk.LabelFrame(definitions_container, text=func_name, padding=7)
        func_frame.pack(fill="x", pady=(0, 5))
        
        desc_label = ttk.Label(func_frame, text=f"Опис: {description}", wraplength=WRAP_WIDTH, justify="left")
        desc_label.pack(anchor="w")
        
        ex_label = ttk.Label(func_frame, text=f"Приклад: {example}", wraplength=WRAP_WIDTH, justify="left", font=("Helvetica", 9, "italic"))
        ex_label.pack(anchor="w", pady=(2,0))

    agents_frame = ttk.LabelFrame(left_frame, text="Наявні Агенти в мережі [СПЕЦІАЛІЗАЦІЯ]", padding=10)
    agents_frame.pack(fill="both", expand=True, pady=(10,0)) 

    agents_cols = ('name', 'type', 'capacity', 'cost')
    tree_agents = ttk.Treeview(agents_frame, columns=agents_cols, show='headings', height=8)
    
    tree_agents.column('name', width=150, anchor='w'); tree_agents.heading('name', text='Назва Агента')
    tree_agents.column('type', width=100, anchor='w'); tree_agents.heading('type', text='Тип')
    tree_agents.column('capacity', width=150, anchor='e'); tree_agents.heading('capacity', text='Потужність/Попит (МВт)')
    tree_agents.column('cost', width=180, anchor='e'); tree_agents.heading('cost', text='Вартість/Чутливість')
    
    tree_agents.pack(fill="both", expand=True) 
    
    control_frame = ttk.Frame(right_frame)
    control_frame.pack(fill="x", pady=5)
    
    ttk.Label(control_frame, text="Поточний Попит (МВт):").pack(side="left", padx=(0, 5))
    demand_var = tk.IntVar(value=1800)
    ttk.Entry(control_frame, textvariable=demand_var, width=10).pack(side="left", padx=5)

    log_frame = ttk.Frame(right_frame) 
    log_frame.pack(fill="both", expand=True, pady=10)
    
    log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical")
    log_widget = tk.Text(log_frame, wrap=tk.WORD, width=60, 
                         font=("Courier", 10), 
                         state='disabled', 
                         yscrollcommand=log_scrollbar.set)
    log_scrollbar.config(command=log_widget.yview)
    
    log_scrollbar.pack(side="right", fill="y")
    log_widget.pack(side="left", fill="both", expand=True)

    log_widget.tag_config('operator_msg', foreground='blue', font=("Courier", 10, "bold"))
    log_widget.tag_config('agent_msg', foreground='#333333') 
    log_widget.tag_config('header_msg', foreground='#006400', font=("Courier", 10, "bold underline")) 
    log_widget.tag_config('result_msg', foreground='#b30059', font=("Courier", 10, "bold")) 
    log_widget.tag_config('error_msg', foreground='red', font=("Courier", 10, "bold"))

    generators = [
        GeneratorAgent(name="АЕС-1 (База)", capacity_mw=1000, cost_per_mw=15),
        GeneratorAgent(name="ТЕС-1 (Вугілля)", capacity_mw=300, cost_per_mw=50, min_output=0.4),
        GeneratorAgent(name="ТЕС-2 (Газ)", capacity_mw=250, cost_per_mw=75, min_output=0.2),
        GeneratorAgent(name="ГЕС-1 (Пік)", capacity_mw=150, cost_per_mw=30),
        GeneratorAgent(name="СЕС (День)", capacity_mw=200, cost_per_mw=20)
    ]
    consumers = [
        ConsumerAgent(name="Місто (Побутовий)", base_demand_mw=1200, price_sensitivity=0.1),
        ConsumerAgent(name="Завод (Промисловий)", base_demand_mw=800, price_sensitivity=0.4)
    ]
    
    operator = SystemOperatorAgent(name="ОПЕРАТОР СИСТЕМИ", generators=generators, consumers=consumers)
    
    operator.log_widget = log_widget
    for agent in generators + consumers:
        agent.log_widget = log_widget

    for g in generators:
        tree_agents.insert('', 'end', values=(g.name, 'Генератор', g.capacity, f"{g.cost}$/МВт"))
    for c in consumers:
        tree_agents.insert('', 'end', values=(c.name, 'Споживач', c.base_demand, f"{c.price_sensitivity*100:.0f}% чутл."))
        
    def simulation_wrapper(func_name):
        log_widget.config(state='normal')
        log_widget.delete('1.0', tk.END) 
        
        try:
            demand = demand_var.get()
        except tk.TclError:
            demand = 1800
            demand_var.set(1800)
            operator.log("Невірний формат попиту! Встановлено 1800.", 'error_msg')
            
        if func_name == "coordinate":
            operator.run_coordination(demand)
        elif func_name == "auction":
            operator.run_collective_decision(demand)
        
        log_widget.config(state='disabled')

    button_frame = ttk.Frame(control_frame)
    button_frame.pack(side="right", fill="x", expand=True)

    btn1 = ttk.Button(button_frame, text="1. КООРДИНАЦІЯ (Наказ)", 
                      command=lambda: simulation_wrapper("coordinate"))
    btn1.pack(fill="x", expand=True, padx=5, pady=2)
    
    btn2 = ttk.Button(button_frame, text="2. КОЛЕКТИВНЕ РІШЕННЯ (Аукціон)", 
                      command=lambda: simulation_wrapper("auction"))
    btn2.pack(fill="x", expand=True, padx=5, pady=2)
    
    operator.log("Симуляція готова. Оберіть режим управління.", 'header_msg')
