import tkinter as tk
from tkinter import ttk, scrolledtext
import random
import time
import threading
import collections # Для зберігання історії даних

# Для графіків
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# ##################################################################
# ---             КЛАСИ АГЕНТІВ (З ПРЯМИМ ЗВ'ЯЗКОМ)               ---
# ##################################################################

class BaseAgent:
    """Базовий клас для всіх агентів з можливістю зв'язку."""
    def __init__(self, name):
        self.name = name
        self.log_widget = None
        self.peers = []
        self.current_value = 0 # Поточна генерація або 0 для датчика/оператора

    def add_peer(self, agent):
        if agent != self and agent not in self.peers:
            self.peers.append(agent)
            if self not in agent.peers: # Уникаємо дублювання
                agent.peers.append(self)

    def send_message(self, recipient_name, message_type, data=None):
        recipient = None
        for peer in self.peers:
            if peer.name == recipient_name: recipient = peer; break
        if recipient:
            if self.log_widget and self.log_widget.winfo_exists():
                # Зменшуємо затримку для швидкої реакції
                self.log_widget.after(10, lambda r=recipient, s=self.name, mt=message_type, d=data: r.receive_message(s, mt, d))
                self.log(f"Надсилаю '{message_type}' до {recipient_name}")
        else: self.log(f"ПОМИЛКА: Не знайдено '{recipient_name}'.", 'error_msg')

    def broadcast_message(self, message_type, data=None):
        self.log(f"Надсилаю '{message_type}' всім (Broadcast)")
        for peer in self.peers:
             if self.log_widget and self.log_widget.winfo_exists():
                 # Зменшуємо затримку
                 self.log_widget.after(10, lambda p=peer, s=self.name, mt=message_type, d=data: p.receive_message(s, mt, d))

    def receive_message(self, sender_name, message_type, data):
        self.log(f"Отримав '{message_type}' від {sender_name}")

    def log(self, message, tag='agent_msg'):
        if self.log_widget and self.log_widget.winfo_exists():
            try:
                self.log_widget.after(0, lambda m=message, t=tag: self._update_log(m, t))
            except tk.TclError: pass

    def _update_log(self, message, tag):
         if self.log_widget and self.log_widget.winfo_exists():
            self.log_widget.config(state='normal')
            self.log_widget.insert(tk.END, f"[{self.name}]: {message}\n", tag)
            self.log_widget.see(tk.END)
            self.log_widget.config(state='disabled')

    def reset(self):
        self.current_value = 0

# --- АГЕНТ: Датчик Захисту ---
class ProtectionSensorAgent(BaseAgent):
    """Агент, що симулює датчик релейного захисту."""
    def __init__(self, name):
        super().__init__(name)

    def simulate_fault(self):
        """Симулює виявлення короткого замикання."""
        self.log("!!! ВИЯВЛЕНО КОРОТКЕ ЗАМИКАННЯ НА ЛІНІЇ !!!", 'error_msg')
        # --- ПРЯМИЙ ЗВ'ЯЗОК ---
        # Негайно надсилаємо сигнал на відключення генераторам
        self.broadcast_message("URGENT_TRIP", {'reason': 'Short Circuit'})

    def reset(self):
        self.log("Стан датчика скинуто.")


class HydroPlantAgentDC(BaseAgent):
    """Агент ГЕС (дуже швидка реакція на TRIP)."""
    def __init__(self, name, capacity):
        super().__init__(name)
        self.capacity = capacity
        self.current_value = capacity # Починаємо працюючими

    def trip(self, reason=""):
        """Миттєво знижує/відключає генерацію."""
        if self.current_value > 0:
            reduction = self.current_value # Відключаємо повністю
            self.current_value = 0
            self.log(f"РЕАКЦІЯ (НАДШВИДКО)! Отримав TRIP ({reason}). Відключаю {reduction:.0f} МВт!", 'warn_msg')
            self.send_message("Оператор Системи", "ACTION_TAKEN", {'action': 'Tripped', 'value': reduction})
        else:
            self.log("Вже був вимкнений.")

    def receive_message(self, sender_name, message_type, data):
        super().receive_message(sender_name, message_type, data)
        if message_type == "URGENT_TRIP":
            reason = data.get('reason', 'Unknown') if data else 'Unknown'
            # ГЕС реагує дуже швидко (майже миттєво)
            if self.log_widget and self.log_widget.winfo_exists():
                self.log_widget.after(20, lambda r=reason: self.trip(r)) # Дуже мала затримка

    def reset(self):
        self.current_value = self.capacity # Повертаємо до робочого стану
        self.log(f"Стан скинуто (генерація {self.current_value} МВт).")


class GasTurbineAgentDC(BaseAgent):
    """Агент Газової Турбіни (також реагує на TRIP)."""
    def __init__(self, name, capacity, ramp_down_time_sec=2):
        super().__init__(name)
        self.capacity = capacity
        self.ramp_down_time = ramp_down_time_sec # Час на зниження
        self.current_value = capacity # Починаємо працюючими
        self.target_output = capacity
        self._timer = None

    def _ramp_down(self):
        """Внутрішня функція, що імітує зниження."""
        reduction = self.current_value - self.target_output
        if reduction > 0:
            self.current_value = self.target_output
            self.log(f"РЕАКЦІЯ (ПОВІЛЬНО)! Завершив зниження на {reduction:.0f} МВт. Поточна: {self.current_value:.0f} МВт.", 'ok_msg')
            self.send_message("Оператор Системи", "ACTION_TAKEN", {'action': 'Ramped Down (Gas)', 'value': reduction})
        self._timer = None

    def start_ramp_down(self, reason=""):
        """Запускає процес зниження потужності (з затримкою)."""
        if self._timer is not None:
            self.log("Процес зміни потужності вже триває.")
            return

        if self.current_value > 0:
            self.target_output = 0 # Знижуємо до нуля
            reduction_amount = self.current_value - self.target_output
            self.log(f"Отримав TRIP ({reason}). Починаю зниження на {reduction_amount:.0f} МВт ({self.ramp_down_time} сек)...")
            self._timer = threading.Timer(self.ramp_down_time, self._ramp_down)
            self._timer.daemon = True
            self._timer.start()
        else:
             self.log("Вже вимкнений.")

    def receive_message(self, sender_name, message_type, data):
        super().receive_message(sender_name, message_type, data)
        if message_type == "URGENT_TRIP":
            reason = data.get('reason', 'Unknown') if data else 'Unknown'
            # ТЕЦ реагує повільніше
            self.start_ramp_down(reason=reason)

    def reset(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self.current_value = self.capacity # Повертаємо до робочого стану
        self.target_output = self.capacity
        self.log(f"Стан скинуто (генерація {self.current_value} МВт).")


class SystemOperatorAgentDC(BaseAgent):
    """Агент Оператора (Моніторингу)."""
    def __init__(self, name, all_agents, graph_update_callback):
        super().__init__(name)
        self.all_agents = all_agents
        # log_widget встановлюється пізніше через set_log_widget
        self.graph_update_callback = graph_update_callback
        self.data_history = collections.defaultdict(lambda: collections.deque(maxlen=60))
        self.is_collecting = False
        self._collection_timer_id = None

    def set_log_widget(self, log_widget):
        self.log_widget = log_widget
        for agent in self.all_agents:
            agent.log_widget = self.log_widget

    def log(self, message, tag='operator_msg'):
        super().log(message, tag)

    def receive_message(self, sender_name, message_type, data):
        # Оператор отримує повідомлення пізніше
        delay = 200 # мс затримки
        if self.log_widget and self.log_widget.winfo_exists():
            self.log_widget.after(delay, lambda s=sender_name, mt=message_type, d=data: self._process_delayed_message(s, mt, d))

    def _process_delayed_message(self, sender_name, message_type, data):
        """Обробка повідомлення з імітацією затримки."""
        super().receive_message(sender_name, message_type, data) # Логуємо отримання
        if message_type == "ACTION_TAKEN":
            self.log(f"Агент {sender_name} відзвітував: {data.get('action', '')} ({data.get('value', '')})")
        elif message_type == "URGENT_TRIP":
             self.log(f"Отримав сигнал TRIP від {sender_name} (причина: {data.get('reason', 'N/A')}). Генератори вже мали відреагувати!", 'warn_msg')

    def start_data_collection(self):
        for key in self.data_history: self.data_history[key].clear()
        self.is_collecting = True
        self._collect_data_loop()

    def stop_data_collection(self):
        self.is_collecting = False
        if self._collection_timer_id:
            if self.log_widget and self.log_widget.winfo_exists():
                try: self.log_widget.after_cancel(self._collection_timer_id)
                except tk.TclError: pass
            self._collection_timer_id = None
        self.log("Збір даних для графіку припинено.", 'operator_msg')

    def _collect_data_loop(self):
        if not self.is_collecting: return
        if not (self.log_widget and self.log_widget.winfo_exists()):
             self.is_collecting = False; return

        current_time_step = len(self.data_history['time']) # Крок часу

        self.data_history['time'].append(current_time_step)
        # Змінюємо індекси відповідно до нового списку агентів
        # all_agents_list = [sensor, hydro_plant, gas_turbine]
        self.data_history['hydro'].append(self.all_agents[1].current_value) # ГЕС
        self.data_history['gas'].append(self.all_agents[2].current_value)   # ТЕЦ

        if self.graph_update_callback:
            self.graph_update_callback(self.data_history)

        # Оновлюємо кожні 200 мс
        self._collection_timer_id = self.log_widget.after(200, self._collect_data_loop)

    def reset(self):
        self.log("Стан оператора не змінюється.", 'operator_msg')


# ##################################################################
# ---               ГОЛОВНА ФУНКЦІЯ ДЛЯ СТВОРЕННЯ ВКЛАДКИ         ---
# ##################################################################

def create_tab(tab_control):
    """Створює вміст для сьомої вкладки."""

    tab7 = ttk.Frame(tab_control, padding=(10, 10))
    tab_control.add(tab7, text='Завдання 7: Прямий Зв\'язок')

    # --- 1. Створення фреймів ---
    main_frame = ttk.Frame(tab7)
    main_frame.pack(fill="both", expand=True)

    left_column_frame = ttk.Frame(main_frame, width=450)
    left_column_frame.pack(side="left", fill="y", padx=(0, 10), expand=False)
    left_column_frame.pack_propagate(False)

    right_column_frame = ttk.Frame(main_frame)
    right_column_frame.pack(side="right", fill="both", expand=True)

    # --- 2. Ліва колонка (Теорія) ---
    #
    # ---- ВЕЛИКИЙ БЛОК ОНОВЛЕННЯ: Прибрано `**` з тексту ----
    #
    theory_frame = ttk.LabelFrame(left_column_frame, text="Аналіз Методу з Прямим Зв'язком", padding=10)
    theory_frame.pack(fill="x")

    def create_theory_section(parent, title, content):
        frame = ttk.LabelFrame(parent, text=title, padding=7)
        frame.pack(fill="x", pady=5)
        msg = tk.Message(frame, text=content, width=430, justify="left", font=("Helvetica", 10))
        msg.pack(fill='x', expand=True)

    features_content = (
        "- Децентралізація: Агенти можуть обмінюватися інформацією та координувати дії без центрального контролера (або з мінімальним його втручанням).\n"
        "- Асинхронність: Агенти діють і спілкуються незалежно, у власному темпі.\n"
        "- Локальні знання: Кожен агент має інформацію лише про своє оточення та сусідів.\n"
        "- Виникнення поведінки: Глобальна поведінка системи (напр., стабілізація) виникає з локальних взаємодій."
    )
    create_theory_section(theory_frame, "Особливості", features_content)

    pros_content = (
        "- Швидкість реакції: Агенти можуть миттєво реагувати на локальні зміни (як ВЕС на вітер), не чекаючи команди з центру.\n"
        "- Масштабованість: Легше додавати нових агентів, не перевантажуючи центральний контролер.\n"
        "- Надійність (частково): Вихід з ладу одного агента не обов'язково зупиняє всю систему (якщо є дублювання).\n"
        "- Гнучкість: Система може адаптуватися до змінних умов."
    )
    create_theory_section(theory_frame, "Переваги", pros_content)

    cons_content = (
        "- Складність координації: Досягнення глобальної мети (напр., мінімальна вартість) може бути складним без центрального планування.\n"
        "- Ризик конфліктів: Агенти з різними цілями можуть приймати суперечливі рішення.\n"
        "- Неоптимальність: Локально оптимальні рішення не завжди є глобально оптимальними.\n"
        "- Надмірність комунікацій: Велика кількість агентів може генерувати забагато повідомлень.\n"
        "- Безпека: Більше точок для можливих атак або помилок."
    )
    create_theory_section(theory_frame, "Недоліки", cons_content)
    #
    # ---- КІНЕЦЬ БЛОКУ ОНОВЛЕННЯ ----
    #

    # --- 3. Лог та кнопки управління ---
    control_log_frame = ttk.LabelFrame(left_column_frame, text="Лог Симуляції та Управління", padding=10)
    control_log_frame.pack(fill="both", expand=True, pady=10)

    btn_frame = ttk.Frame(control_log_frame)
    btn_frame.pack(fill='x', pady=(0, 10))

    run_button_fault = ttk.Button(btn_frame, text="Симулювати Аварійний Сигнал (TRIP)", state='disabled')
    run_button_fault.pack(fill='x', expand=True)

    log_widget = scrolledtext.ScrolledText(control_log_frame, wrap=tk.WORD, width=60, height=15,
                                            font=("Courier", 9), state='disabled')
    log_widget.pack(fill="both", expand=True)
    log_widget.tag_config('operator_msg', foreground='blue', font=("Courier", 9, "bold"))
    log_widget.tag_config('agent_msg', foreground='#333333')
    log_widget.tag_config('warn_msg', foreground='orange', font=("Courier", 9, "bold"))
    log_widget.tag_config('ok_msg', foreground='green', font=("Courier", 9, "bold"))
    log_widget.tag_config('error_msg', foreground='red', font=("Courier", 9, "bold"))


    # --- 4. Графік (у правій колонці) ---
    graph_frame = ttk.LabelFrame(right_column_frame, text="Динаміка Генерації під час Аварії (МВт)", padding=10)
    graph_frame.pack(fill="both", expand=True)

    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    line_hydro, = ax.plot([], [], 'b-', label='Генерація ГЕС', linewidth=2)
    line_gas, = ax.plot([], [], 'o-', label='Генерація ТЕЦ', markersize=4)

    ax.legend(loc='upper right')
    ax.set_title("Реакція Генераторів на Сигнал TRIP")
    ax.set_xlabel("Час симуляції (0.2 с крок)")
    ax.set_ylabel("Генерація (МВт)")
    ax.grid(True)
    INITIAL_Y_MAX = 120
    ax.set_ylim(-10, INITIAL_Y_MAX)


    # --- 5. Створення агентів ---

    simulation_running = False

    # --- Функція оновлення графіку ---
    def update_graph(data_history):
        if not (log_widget and log_widget.winfo_exists()): return

        times = np.array(list(data_history['time']))
        hydro_values = np.array(list(data_history['hydro']))
        gas_values = np.array(list(data_history['gas']))

        if times.size == 0: return

        line_hydro.set_data(times, hydro_values)
        line_gas.set_data(times, gas_values)

        ax.set_xlim(-0.5, times[-1] + 0.5)
        min_y = -10 # Фіксований мінімум
        # Максимум беремо з початкової потужності + невеликий запас
        max_y = max(INITIAL_Y_MAX, np.max(hydro_values) if hydro_values.size > 0 else 0, np.max(gas_values) if gas_values.size > 0 else 0) + 10
        ax.set_ylim(min_y, max_y)

        try: canvas.draw_idle()
        except tk.TclError: pass

    # --- Функція запуску симуляції ---
    def simulate_event(event_type):
        nonlocal simulation_running
        if simulation_running:
            operator.log("Симуляція вже триває...", 'error_msg')
            return
        simulation_running = True
        run_button_fault.config(state='disabled')

        log_widget.config(state='normal')
        log_widget.delete('1.0', tk.END)

        operator.log("--- СКИДАННЯ СТАНУ АГЕНТІВ ---", 'ok_msg')
        sensor.reset()
        hydro_plant.reset()
        gas_turbine.reset()
        operator.reset()
        operator.log("--- СТАН СКИНУТО ---", 'ok_msg')

        operator.stop_data_collection()
        operator.start_data_collection()

        finish_delay_ms = 1000

        if event_type == 'fault':
            operator.log("--- ПОЧАТОК: Аварійний Сигнал TRIP ---", 'error_msg')
            if log_widget.winfo_exists():
                 log_widget.after(500, lambda: sensor.simulate_fault())
            finish_delay_ms = int((gas_turbine.ramp_down_time + 1) * 1000)

        def finish_simulation():
             nonlocal simulation_running
             if log_widget and log_widget.winfo_exists():
                operator.log("--- КІНЕЦЬ СИМУЛЯЦІЇ ---", 'ok_msg')
                operator.stop_data_collection()
                run_button_fault.config(state='normal')
             simulation_running = False

        if log_widget and log_widget.winfo_exists():
            log_widget.after(finish_delay_ms, finish_simulation)
        else:
             simulation_running = False
             operator.stop_data_collection()

        log_widget.config(state='disabled')

    # Створюємо агентів ПІСЛЯ визначення функцій
    sensor = ProtectionSensorAgent("Датчик ЛЕП-110")
    hydro_plant = HydroPlantAgentDC("ГЕС-Швидка", capacity=100)
    gas_turbine = GasTurbineAgentDC("ТЕЦ-Маневр (Газ)", capacity=80, ramp_down_time_sec=2)
    operator = SystemOperatorAgentDC("Оператор Системи", [sensor, hydro_plant, gas_turbine], update_graph)

    # Налаштовуємо зв'язки
    sensor.add_peer(hydro_plant); sensor.add_peer(gas_turbine); sensor.add_peer(operator)
    hydro_plant.add_peer(operator); gas_turbine.add_peer(operator)
    # ГЕС і ТЕЦ можуть не знати одна одну в цьому сценарії

    # Передаємо log_widget Оператору та іншим агентам
    operator.set_log_widget(log_widget)
    sensor.log_widget = log_widget; hydro_plant.log_widget = log_widget;
    gas_turbine.log_widget = log_widget;

    # --- 6. Кнопки запуску (Прив'язка команд після визначення функції) ---
    run_button_fault.config(command=lambda: simulate_event('fault'), state='normal')

    # Початкове повідомлення та запуск збору даних
    operator.log("Система готова. Натисніть кнопку.", 'ok_msg')
    operator.start_data_collection() # Починаємо збір даних одразу