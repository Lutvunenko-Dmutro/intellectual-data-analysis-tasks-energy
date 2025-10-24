import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Для 3D графіку
import threading # Для малювання поверхні в окремому потоці

# --- ВИКОРИСТОВУЄМО ТІЛЬКИ SCIKIT-FUZZY ---
FUZZY_AVAILABLE = False # Починаємо з False
FUZZY_IMPORT_ERROR = ""
ctrl = None # Зберігатимемо імпортований control
fuzz = None # Зберігатимемо імпортований fuzz

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    FUZZY_AVAILABLE = True
    print("INFO: Бібліотека 'scikit-fuzzy' успішно імпортована.")
except ImportError as e:
    FUZZY_AVAILABLE = False
    FUZZY_IMPORT_ERROR = f"ПОМИЛКА ІМПОРТУ: Не вдалося завантажити 'scikit-fuzzy'.\nДеталі: {e}\nПереконайтесь, що її встановлено ('pip install scikit-fuzzy')."
    print(FUZZY_IMPORT_ERROR)
except Exception as e:
    FUZZY_AVAILABLE = False
    FUZZY_IMPORT_ERROR = f"ПОМИЛКА ІМПОРТУ: Неочікувана помилка при імпорті 'scikit-fuzzy'.\nДеталі: {e}"
    print(FUZZY_IMPORT_ERROR)

print(f"INFO: Статус доступності scikit-fuzzy: {FUZZY_AVAILABLE}")
# --- КІНЕЦЬ БЛОКУ ІМПОРТУ ---


# ##################################################################
# ---         ЛОГІКА НЕЧІТКОЇ СИСТЕМИ (ЗАМІСТЬ ANFIS)            ---
# ##################################################################

# Глобальні змінні для нечіткої системи та симуляції
fuzzy_ctrl_system = None
load_simulation = None

def build_fuzzy_system():
    """Створює систему нечіткого висновку типу Сугено."""
    if not FUZZY_AVAILABLE or ctrl is None or fuzz is None:
        raise ImportError("Модулі scikit-fuzzy недоступні для створення системи.")

    global fuzzy_ctrl_system

    hour = ctrl.Antecedent(np.arange(0, 24, 1), 'Година')
    temp = ctrl.Antecedent(np.arange(-10, 35, 1), 'Температура')
    load = ctrl.Consequent(np.arange(500, 2501, 10), 'Навантаження', defuzzify_method='centroid')

    hour.automf(names=['ранок', 'день', 'вечір'])
    temp.automf(names=['холодно', 'нормально', 'спекотно'])

    load['низьке'] = fuzz.trimf(load.universe, [500, 800, 1100])
    load['середнє'] = fuzz.trimf(load.universe, [1000, 1500, 2000])
    load['високе'] = fuzz.trimf(load.universe, [1800, 2150, 2500])

    rule1 = ctrl.Rule(hour['ранок'] & temp['холодно'], load['середнє'])
    rule2 = ctrl.Rule(hour['ранок'] & temp['нормально'], load['середнє'])
    rule3 = ctrl.Rule(hour['ранок'] & temp['спекотно'], load['високе'])
    rule4 = ctrl.Rule(hour['день'] & temp['холодно'], load['середнє'])
    rule5 = ctrl.Rule(hour['день'] & temp['нормально'], load['низьке'])
    rule6 = ctrl.Rule(hour['день'] & temp['спекотно'], load['високе'])
    rule7 = ctrl.Rule(hour['вечір'] & temp['холодно'], load['високе'])
    rule8 = ctrl.Rule(hour['вечір'] & temp['нормально'], load['високе'])
    rule9 = ctrl.Rule(hour['вечір'] & temp['спекотно'], load['високе'])

    load_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    fuzzy_ctrl_system = load_ctrl
    return load_ctrl

# ##################################################################
# ---               ГОЛОВНА ФУНКЦІЯ ДЛЯ СТВОРЕННЯ ВКЛАДКИ         ---
# ##################################################################

def create_tab(tab_control):
    """
    Створює вміст для восьмої вкладки (Нейро-нечіткі мережі)
    """
    global load_simulation

    tab8 = ttk.Frame(tab_control, padding=(10, 10))
    tab_control.add(tab8, text='Завдання 8: Нечітка Система')

    # --- 1. Створення фреймів ---
    main_frame = ttk.Frame(tab8)
    main_frame.pack(fill="both", expand=True)

    LEFT_WIDTH = 450
    left_frame = ttk.Frame(main_frame, width=LEFT_WIDTH)
    left_frame.pack(side="left", fill="y", padx=10, expand=False)
    left_frame.pack_propagate(False)
    left_frame.grid_rowconfigure(0, weight=1)
    left_frame.grid_columnconfigure(0, weight=1)

    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side="right", fill="both", expand=True)

    # --- 2. Ліва колонка (Теорія) ---
    #
    # ---- ВЕЛИКИЙ БЛОК ОНОВЛЕННЯ: Прибрано `**` з тексту ----
    #
    theory_frame = ttk.LabelFrame(left_frame, text="Архітектури та Навчання Нейро-Нечітких Мереж", padding=10)
    theory_frame.grid(row=0, column=0, sticky='nsew')
    theory_frame.grid_columnconfigure(0, weight=1)

    WRAP_WIDTH = LEFT_WIDTH - 40

    def create_theory_grid_section(parent, row_index, title, content):
        frame = ttk.LabelFrame(parent, text=title, padding=7)
        frame.grid(row=row_index, column=0, sticky='new', pady=5)
        frame.grid_columnconfigure(0, weight=1)

        lbl = ttk.Label(frame, text=content, wraplength=WRAP_WIDTH, justify="left", font=("Helvetica", 10))
        lbl.grid(row=0, column=0, sticky='ew')

    # --- Блок 1: Загальна ідея ---
    intro_content = (
        "Нейро-нечіткі системи поєднують переваги:\n"
        "- Нечіткої логіки: Можливість використовувати лінгвістичні правила ('ЯКЩО температура ВИСОКА...') та працювати з невизначеністю.\n"
        "- Нейронних мереж: Здатність до навчання на даних та адаптації параметрів."
    )
    create_theory_grid_section(theory_frame, 0, "Загальна ідея", intro_content)

    # --- Блок 2: Мамдані ---
    mamdani_content = (
        "Апроксиматор Мамдані: Класичний тип нечіткої системи.\n\n"
        "- Правила: 'ЯКЩО вхід1 є A1 І вхід2 є B1 ТОДІ вихід є C1'.\n"
        "- Висновок: Нечітка множина (напр., 'навантаження ВИСОКЕ').\n"
        "- Дефазифікація: Потрібен крок для перетворення нечіткого висновку на чітке число (напр., метод центроїду).\n"
        "- Навчання: Нейронна мережа може налаштовувати параметри функцій належності (A1, B1, C1)."
    )
    create_theory_grid_section(theory_frame, 1, "Апроксиматор Мамдані", mamdani_content)

    # --- Блок 3: TSK ---
    tsk_content = (
        "Мережа Такагі-Сугено-Канга (TSK): Популярний гібридний тип.\n\n"
        "- Правила: 'ЯКЩО вхід1 є A1 І вхід2 є B1 ТОДІ вихід = f(вхід1, вхід2)'.\n"
        "- Висновок (f): Зазвичай лінійна функція від входів (напр., `y = p*вхід1 + q*вхід2 + r`).\n"
        "- Дефазифікація: Проста - середньозважене значення висновків правил.\n"
        "- Навчання: НМ навчає параметри ФН (A1, B1) та коефіцієнти лінійних функцій (p, q, r)."
    )
    create_theory_grid_section(theory_frame, 2, "Мережа Такагі-Сугено (TSK)", tsk_content)

    # --- Блок 4: ANFIS ---
    anfis_content = (
        "ANFIS (Adaptive Neuro-Fuzzy Inference System): Найвідоміша реалізація TSK.\n\n"
        "- Архітектура: 5-шарова НМ, де кожен шар виконує крок нечіткого висновку (Фазифікація, Сила правил, Нормалізація, Висновок правила, Загальний вихід).\n"
        "- Навчання: Гібридне: градієнтний спуск для ФН + МНК для коефіцієнтів висновків.\n\n"
        "Примітка: Через проблеми з бібліотекою 'anfis', ця вкладка демонструє базову Нечітку Систему (FIS) за допомогою 'scikit-fuzzy'. FIS є основою для ANFIS."
    )
    create_theory_grid_section(theory_frame, 3, "ANFIS (Теорія)", anfis_content)
    #
    # ---- КІНЕЦЬ БЛОКУ ОНОВЛЕННЯ ----
    #


    # --- 3. Права колонка (Симуляція FIS) ---
    sim_frame = ttk.LabelFrame(right_frame, text="Практика: Прогнозування навантаження (Нечітка Система)", padding=10)
    sim_frame.pack(fill="both", expand=True)

    # 3.1 Керування симуляцією
    control_frame = ttk.Frame(sim_frame)
    control_frame.pack(fill="x", pady=(0, 10))

    ttk.Label(control_frame, text="Година:").grid(row=0, column=0, padx=5, sticky='w')
    hour_var = tk.DoubleVar(value=18.0)
    hour_scale = ttk.Scale(control_frame, from_=0, to=23, orient='horizontal', variable=hour_var, length=200, command=lambda v: update_simulation())
    hour_scale.grid(row=0, column=1, padx=5, sticky='ew')
    hour_label = ttk.Label(control_frame, text=f"{hour_var.get():.1f}")
    hour_label.grid(row=0, column=2, padx=5)

    ttk.Label(control_frame, text="Темп-ра (°C):").grid(row=1, column=0, padx=5, sticky='w')
    temp_var = tk.DoubleVar(value=5.0)
    temp_scale = ttk.Scale(control_frame, from_=-10, to=35, orient='horizontal', variable=temp_var, length=200, command=lambda v: update_simulation())
    temp_scale.grid(row=1, column=1, padx=5, sticky='ew')
    temp_label = ttk.Label(control_frame, text=f"{temp_var.get():.1f}")
    temp_label.grid(row=1, column=2, padx=5)

    control_frame.grid_columnconfigure(1, weight=1)

    # 3.2 Графік (3D Поверхня)
    fig_frame = ttk.Frame(sim_frame)
    fig_frame.pack(fill="both", expand=True, pady=10)

    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    canvas = FigureCanvasTkAgg(fig, master=fig_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True)

    result_label = ttk.Label(sim_frame, text="Прогноз навантаження: N/A", justify="center", font=("Helvetica", 12, "bold"))
    result_label.pack(fill="x")

    # --- 4. Логіка FIS ---
    def plot_surface():
        """Малює 3D поверхню відгуку системи."""
        if not FUZZY_AVAILABLE or fuzzy_ctrl_system is None: return
        print("Малюю поверхню відгуку FIS...")
        # (Код малювання поверхні без змін)
        x_hour = np.arange(0, 24, 1); y_temp = np.arange(-10, 35, 1)
        xx, yy = np.meshgrid(x_hour, y_temp); zz = np.zeros_like(xx)
        temp_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl_system)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                try:
                    temp_sim.input['Година'] = xx[i, j]
                    temp_sim.input['Температура'] = yy[i, j]
                    temp_sim.compute()
                    zz[i, j] = temp_sim.output['Навантаження']
                except ValueError: zz[i, j] = np.nan
        ax.clear()
        surf = ax.plot_surface(xx, yy, zz, cmap='viridis', edgecolor='none')
        ax.set_xlabel('Година доби'); ax.set_ylabel('Температура (°C)'); ax.set_zlabel('Прогноз (МВт)')
        ax.set_title('Поверхня Відгуку Нечіткої Системи')
        try: canvas.draw()
        except tk.TclError: print("Помилка малювання canvas (можливо, вікно закрито)")
        print("Малювання поверхні завершено.")


    def update_simulation(event=None):
        """Оновлює прогноз при зміні повзунків."""
        if not FUZZY_AVAILABLE or load_simulation is None: return
        current_hour = hour_var.get(); current_temp = temp_var.get()
        hour_label.config(text=f"{current_hour:.1f}"); temp_label.config(text=f"{current_temp:.1f}")
        try:
            load_simulation.input['Година'] = current_hour
            load_simulation.input['Температура'] = current_temp
            load_simulation.compute()
            predicted_load = load_simulation.output['Навантаження']
            result_label.config(text=f"Прогноз навантаження: {predicted_load:.0f} МВт")
        except ValueError: result_label.config(text="Прогноз: Вхід поза діапазоном")
        except Exception as e: result_label.config(text=f"Помилка: {e}")


    # --- 5. Ініціалізація FIS ---
    if FUZZY_AVAILABLE:
        try:
            print("Створення нечіткої системи...")
            control_system = build_fuzzy_system()
            load_simulation = ctrl.ControlSystemSimulation(control_system)
            print("Нечітка система створена.")
            # Запускаємо малювання поверхні в окремому потоці
            plot_thread = threading.Thread(target=plot_surface, daemon=True)
            plot_thread.start()
            update_simulation()
        except Exception as e:
            INIT_ERROR = f"ПОМИЛКА при створенні FIS: {e}"
            print(INIT_ERROR)
            result_label.config(text=INIT_ERROR, foreground="red", wraplength=500)
            hour_scale.config(state='disabled'); temp_scale.config(state='disabled')
    else:
        result_label.config(text=FUZZY_IMPORT_ERROR, foreground="red", wraplength=500)
        hour_scale.config(state='disabled'); temp_scale.config(state='disabled')