import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import curve_fit # <--- Ядро методу LM

# ##################################################################
# ---                ЛОГІКА ПРОГНОЗУВАННЯ (LM)                   ---
# ##################################################################

def create_load_data():
    """Створює симульований добовий графік навантаження (24 точки)."""
    x_data = np.arange(0, 24, 1.0)
    y_ideal = 1500 - 500 * np.cos((x_data - 2) * np.pi / 12) + 300 * np.sin(x_data * np.pi / 12)
    y_noise = np.random.normal(0, 50, x_data.shape)
    y_data = y_ideal + y_noise
    return x_data, y_data

def fitting_function(x, a, b, c, d, offset):
    """Наша нелінійна модель для прогнозування."""
    return a * np.sin(np.pi/12 * x + b) + c * np.sin(np.pi/6 * x + d) + offset

# ##################################################################
# ---               ГОЛОВНА ФУНКЦІЯ ДЛЯ СТВОРЕННЯ ВКЛАДКИ         ---
# ##################################################################

def create_tab(tab_control):
    """
    Створює вміст для шостої вкладки (Метод Левенберга-Марквардта)
    """

    tab6 = ttk.Frame(tab_control, padding=(10, 10))
    tab_control.add(tab6, text='Завдання 6: Прогнозування (LM)')

    # --- 1. Створення фреймів ---
    main_frame = ttk.Frame(tab6)
    main_frame.pack(fill="both", expand=True)

    # --- ВИПРАВЛЕНО: Використовуємо grid для лівої колонки ---
    # Задаємо фіксовану ширину для лівої колонки
    LEFT_WIDTH = 450
    left_frame = ttk.Frame(main_frame, width=LEFT_WIDTH)
    left_frame.pack(side="left", fill="y", padx=10, expand=False)
    left_frame.pack_propagate(False) # Забороняємо стискатись
    # Налаштовуємо grid всередині left_frame
    left_frame.grid_rowconfigure(0, weight=0) # Теорія
    left_frame.grid_rowconfigure(1, weight=1) # Порожній простір (якщо потрібен)
    left_frame.grid_rowconfigure(2, weight=0) # Кнопка знизу
    left_frame.grid_columnconfigure(0, weight=1) # Колонка займає всю ширину

    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side="right", fill="both", expand=True)

    # --- 2. Ліва колонка (Теорія) ---
    #
    # ---- ВЕЛИКИЙ БЛОК ОНОВЛЕННЯ: ttk.Label + grid() ----
    #
    theory_frame = ttk.LabelFrame(left_frame, text="Метод Левенберга-Марквардта (Теорія)", padding=10)
    theory_frame.grid(row=0, column=0, sticky='new') # Розміщуємо рамку теорії
    theory_frame.grid_columnconfigure(0, weight=1) # Дозволяємо вмісту розтягуватись

    # --- Блок 1: Призначення методу ---
    purpose_frame = ttk.LabelFrame(theory_frame, text="Призначення методу", padding=7)
    purpose_frame.grid(row=0, column=0, sticky='new', pady=5)
    purpose_frame.grid_columnconfigure(0, weight=1) # Колонка всередині purpose_frame

    purpose_content = (
        "Метод Левенберга-Марквардта (LM) — це ітераційний алгоритм, який використовується для розв'язання **нелінійних задач найменших квадратів**.\n\n"
        "**ПРОСТОЮ МОВОЮ:** Коли у вас є 'шумні' дані (як добовий графік навантаження) і складна математична модель (наша `fitting_function`), метод LM допомагає знайти **найкращі параметри** цієї моделі, щоб вона максимально точно 'лягла' на дані.\n\n"
        "Він є **гібридом** двох інших методів:"
        "\n  - **Методу Гаусса-Ньютона:** Швидкий, але може 'розбігтися', якщо початкове наближення погане."
        "\n  - **Методу градієнтного спуску:** Надійний (завжди рухається до мінімуму), але повільний біля оптимуму."
        "\nLM 'перемикається' між ними: далеко від оптимуму діє як градієнтний спуск, а близько — як Гаусс-Ньютон."
    )
    # Використовуємо ttk.Label з wraplength, розміщений через grid
    lbl_purpose = ttk.Label(purpose_frame, text=purpose_content, wraplength=LEFT_WIDTH-40, justify="left", font=("Helvetica", 10))
    lbl_purpose.grid(row=0, column=0, sticky='ew')
    #
    # ---- КІНЕЦЬ БЛОКУ ОНОВЛЕННЯ ----
    #

    # --- Кнопка запуску апроксимації ---
    run_button = ttk.Button(left_frame, text="Виконати апроксимацію (Знайти модель)", command=lambda: run_fitting())
    run_button.grid(row=2, column=0, sticky='sew', pady=20) # sticky='sew' = South-East-West


    # --- 3. Права колонка (Графік та Таблиця) ---

    # 3.1 Графік (зверху)
    plot_frame = ttk.LabelFrame(right_frame, text="Практика: Апроксимація Добового Графіка", padding=10)
    plot_frame.pack(side="top", fill="both", expand=True)

    fig = Figure(figsize=(6, 5), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    result_label = ttk.Label(plot_frame, text="Знайдені параметри моделі: N/A",
                             font=("Helvetica", 10), wraplength=500, justify="left", padding=(5, 5))
    result_label.pack(side=tk.BOTTOM, fill="x")

    # 3.2 Порівняльна таблиця (знизу)
    compare_frame = ttk.LabelFrame(right_frame, text="Порівняння з іншими градієнтними методами", padding=10)
    compare_frame.pack(side="bottom", fill="x", expand=False, pady=(10, 0))

    table_headers = ['Метод', 'Швидкість', 'Надійність', 'Вимоги до початк. набл.', 'Тип задач']
    table_data = [
        ['Градієнтний спуск', 'Повільна біля оптимуму', 'Висока (завжди збігається)', 'Низькі', 'Загальна оптимізація'],
        ['Гаусса-Ньютона', 'Дуже швидка біля оптимуму', 'Низька (може розбігтися)', 'Високі', 'Нелін. найм. квадрати'],
        ['Левенберга-Марквардта (LM)', 'Швидка (як ГН біля оптимуму)', 'Висока (як ГС далеко)', 'Середні', 'Нелін. найм. квадрати']
    ]

    cols = ('col1', 'col2', 'col3', 'col4', 'col5')
    tree_compare = ttk.Treeview(compare_frame, columns=cols, show='headings', height=4)

    tree_compare.column('col1', width=180, anchor='w'); tree_compare.heading('col1', text=table_headers[0])
    tree_compare.column('col2', width=180, anchor='w'); tree_compare.heading('col2', text=table_headers[1])
    tree_compare.column('col3', width=180, anchor='w'); tree_compare.heading('col3', text=table_headers[2])
    tree_compare.column('col4', width=200, anchor='w'); tree_compare.heading('col4', text=table_headers[3])
    tree_compare.column('col5', width=180, anchor='w'); tree_compare.heading('col5', text=table_headers[4])

    tree_compare.tag_configure('oddrow', background='#f0f0f0')
    tree_compare.tag_configure('evenrow', background='white')

    for i, row in enumerate(table_data):
        tag = 'evenrow' if i % 2 == 0 else 'oddrow'
        tree_compare.insert('', 'end', values=row, tags=(tag,))

    tree_compare.pack(fill="x", expand=True)


    # --- 4. Функція запуску апроксимації ---
    # Тепер функція run_fitting визначена ПІСЛЯ створення всіх віджетів
    def run_fitting():
        # (Код функції run_fitting залишається без змін)
        print("Запуск апроксимації (Вкладка 6)...")
        x_data, y_data = create_load_data()
        try:
            initial_guess = [500, -1, 300, 0, 1500]
            params, covariance = curve_fit(fitting_function, x_data, y_data, p0=initial_guess)
            y_fit = fitting_function(x_data, *params)
            ax.clear()
            ax.scatter(x_data, y_data, label='Реальні дані (з шумом)', color='blue', s=50)
            ax.plot(x_data, y_fit, label='Апроксимація (модель LM)', color='red', linewidth=2)
            ax.set_title("Апроксимація добового графіка навантаження")
            ax.set_xlabel("Година доби")
            ax.set_ylabel("Навантаження, МВт")
            ax.legend()
            ax.grid(True)
            canvas.draw()
            param_names = ['a', 'b', 'c', 'd', 'offset']
            param_str = ", ".join([f"{name}={val:.2f}" for name, val in zip(param_names, params)])
            result_label.config(text=f"Знайдені параметри: {param_str}")
            print("Апроксимація завершена.")
        except RuntimeError:
            result_label.config(text="Помилка: Не вдалося знайти параметри.")
            print("Помилка апроксимації.")
        except Exception as e:
             result_label.config(text=f"Виникла помилка: {e}")
             print(f"Виникла помилка: {e}")

    # Переприв'язуємо команду кнопці
    run_button.config(command=run_fitting)

    # Запускаємо 1 раз при старті
    run_fitting()