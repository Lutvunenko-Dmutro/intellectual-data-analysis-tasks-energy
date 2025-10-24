
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import sys
import os

try:
    from tab1_feature_selection import create_energy_dataset as create_energy_dataset_orig
    def create_energy_dataset(n_samples=1000):
        df = create_energy_dataset_orig(n_samples)
        feature_name_mapping = {
            'time_of_day': 'Година доби', 'temperature': 'Температура', 'day_of_week': 'День тижня',
            'industrial_activity': 'Пром. активність', 'solar_generation': 'Генерація СЕС',
            'noise_1_market_price': 'Шум_1_Ціна', 'noise_2_wind_speed': 'Шум_2_Вітер',
            'noise_3_network_freq': 'Шум_3_Частота', 'noise_4_transformer_id': 'Шум_4_ID',
            'noise_5_random_noise': 'Шум_5', 'system_load': 'Завантаження'
        }
        df_renamed = df.rename(columns=feature_name_mapping)
        return df_renamed
    print("INFO: Використовується функція create_energy_dataset з tab1_feature_selection.")

except (ImportError, ModuleNotFoundError):
    print("Попередження: Не вдалося імпортувати create_energy_dataset. Використовується локальна копія.")
    def create_energy_dataset(n_samples=1000):
        data = {}; data['time_of_day'] = np.random.randint(0, 24, n_samples); data['temperature'] = np.random.normal(15, 10, n_samples); data['day_of_week'] = np.random.randint(0, 7, n_samples); data['industrial_activity'] = np.random.uniform(0.5, 1.0, n_samples); data['solar_generation'] = np.clip(np.sin(data['time_of_day'] / 24 * np.pi) * np.random.uniform(50, 100, n_samples), 0, None); data['noise_1_market_price'] = np.random.uniform(1, 5, n_samples); data['noise_2_wind_speed'] = np.random.uniform(0, 15, n_samples); data['noise_3_network_freq'] = np.random.normal(50.0, 0.05, n_samples); data['noise_4_transformer_id'] = np.random.randint(1, 50, n_samples); data['noise_5_random_noise'] = np.random.randn(n_samples); df = pd.DataFrame(data); load = 1000; load += -500 * np.cos(df['time_of_day'] / 24 * 2 * np.pi) + 200 * np.sin(df['time_of_day'] / 24 * 4 * np.pi); load += 5 * (df['temperature'] - 18)**2; load -= df['day_of_week'].isin([5, 6]) * 300; load += df['industrial_activity'] * 250; load -= df['solar_generation']; load += np.random.normal(0, 20, n_samples); df['system_load'] = load;
        feature_name_mapping = {'time_of_day': 'Година доби', 'temperature': 'Температура', 'day_of_week': 'День тижня', 'industrial_activity': 'Пром. активність', 'solar_generation': 'Генерація СЕС', 'noise_1_market_price': 'Шум_1_Ціна', 'noise_2_wind_speed': 'Шум_2_Вітер', 'noise_3_network_freq': 'Шум_3_Частота', 'noise_4_transformer_id': 'Шум_4_ID', 'noise_5_random_noise': 'Шум_5', 'system_load': 'Завантаження'}; df.rename(columns=feature_name_mapping, inplace=True); return df


def create_tab(tab_control):
    """Створює вміст для вкладки Кореляційного аналізу."""

    tab10 = ttk.Frame(tab_control, padding=(10, 10))
    tab_control.add(tab10, text='Завд. 2 (10): Кореляція')

    main_frame = ttk.Frame(tab10)
    main_frame.pack(fill="both", expand=True)

    LEFT_WIDTH = 400
    left_frame = ttk.Frame(main_frame, width=LEFT_WIDTH)
    left_frame.pack(side="left", fill="y", padx=(0, 10), expand=False)
    left_frame.pack_propagate(False)
    left_frame.grid_rowconfigure(0, weight=0) 
    left_frame.grid_rowconfigure(1, weight=0) 
    left_frame.grid_rowconfigure(2, weight=1) 
    left_frame.grid_columnconfigure(0, weight=1)

    right_frame = ttk.LabelFrame(main_frame, text="Візуалізація: Діаграма розсіювання", padding=10)
    right_frame.pack(side="right", fill="both", expand=True)

    theory_frame = ttk.LabelFrame(left_frame, text="Теорія: Коефіцієнт Кореляції Пірсона", padding=10)
    theory_frame.grid(row=0, column=0, sticky='new')
    theory_frame.grid_columnconfigure(0, weight=1)

    MESSAGE_WIDTH = LEFT_WIDTH - 40

    def create_theory_section(parent, title, content):
        frame = ttk.LabelFrame(parent, text=title, padding=7)
        frame.pack(fill="x", pady=5) 
        msg = tk.Message(frame, text=content, width=MESSAGE_WIDTH, justify="left", font=("Helvetica", 10))
        msg.pack(fill='x', expand=True)

    theory_content_1 = (
        "Коефіцієнт кореляції Пірсона (r) - це міра того, наскільки добре точки даних лягають на **пряму лінію**.\n\n"
        "Він показує силу і напрямок **ЛІНІЙНОГО** зв'язку між двома змінними.\n\n"
        "Значення:\n"
        "• +1: Ідеальний позитивний лінійний зв'язок (точки на висхідній прямій).\n"
        "• -1: Ідеальний негативний лінійний зв'язок (точки на низхідній прямій).\n"
        "•  0: Немає лінійного зв'язку (точки розкидані хаотично АБО зв'язок є, але нелінійний, напр. U-подібний)."
    )
    create_theory_section(theory_frame, "Визначення та Значення", theory_content_1.replace("**", ""))

    theory_content_2 = (
        "Сила зв'язку (за модулем |r|):\n"
        "•  0.0 - 0.3: Дуже слабкий / Відсутній\n"
        "•  0.3 - 0.5: Слабкий\n"
        "•  0.5 - 0.7: Середній\n"
        "•  0.7 - 0.9: Сильний\n"
        "•  0.9 - 1.0: Дуже сильний"
    )
    create_theory_section(theory_frame, "Сила зв'язку", theory_content_2)

    theory_content_3 = (
        "❗ **ВАЖЛИВО:** Кореляція НЕ дорівнює причинності! Сильний зв'язок між двома змінними не означає, що одна викликає іншу. Може існувати третій фактор, що впливає на обидві."
    )
    create_theory_section(theory_frame, "Кореляція vs Причинність", theory_content_3.replace("**", "").replace("❗ ", ""))

    result_frame = ttk.LabelFrame(left_frame, text="Результат Обчислення", padding=10)
    result_frame.grid(row=1, column=0, sticky='new', pady=10)
    result_frame.grid_columnconfigure(0, weight=1)

    result_label = ttk.Label(result_frame, text="r = N/A", font=("Helvetica", 14, "bold"), anchor="center")
    result_label.grid(row=0, column=0, sticky='ew')
    result_explanation = ttk.Label(result_frame, text="↑ Цей коефіцієнт вимірює лише ЛІНІЙНУ складову зв'язку.", font=("Helvetica", 8, "italic"), anchor="center")
    result_explanation.grid(row=1, column=0, sticky='ew', pady=(5,0))

    analysis_frame = ttk.LabelFrame(left_frame, text="Аналіз та Висновок", padding=10)
    analysis_frame.grid(row=2, column=0, sticky='nsew', pady=(0, 10))
    analysis_frame.grid_columnconfigure(0, weight=1)
    analysis_frame.grid_rowconfigure(0, weight=1)

    analysis_text = scrolledtext.ScrolledText(analysis_frame, wrap=tk.WORD, height=10,
                                              font=("Helvetica", 10), state='disabled')
    analysis_text.pack(fill="both", expand=True)

    fig = Figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True)

    def calculate_and_plot_correlation():
        print("Обчислення кореляції (Вкладка 10)...")
        try:
            df = create_energy_dataset(n_samples=500)
            var1_name = 'Температура'
            var2_name = 'Завантаження'

            if var1_name not in df.columns or var2_name not in df.columns:
                 print(f"Помилка: колонки не знайдено. Наявні: {list(df.columns)}")
                 raise ValueError(f"Потрібні колонки '{var1_name}'/'{var2_name}' відсутні.")

            var1_data = df[var1_name]
            var2_data = df[var2_name]

            correlation_matrix = np.corrcoef(var1_data.astype(float), var2_data.astype(float))
            if np.isnan(correlation_matrix).any() or correlation_matrix.shape != (2, 2):
                raise ValueError("Не вдалося обчислити кореляційну матрицю.")
            correlation_coefficient = correlation_matrix[0, 1]

            result_label.config(text=f"r = {correlation_coefficient:.3f}")

            r = correlation_coefficient
            abs_r = abs(r)
            sign = "позитивний" if r > 0 else ("негативний" if r < 0 else "нульовий")
            strength = ""
            if abs_r < 0.3: strength = "дуже слабкий"
            elif 0.3 <= abs_r < 0.5: strength = "слабкий"
            elif 0.5 <= abs_r < 0.7: strength = "середній"
            elif 0.7 <= abs_r < 0.9: strength = "сильний"
            else: strength = "дуже сильний"

            analysis_conclusion = (
                f"Обчислений коефіцієнт Пірсона між '{var1_name}' та '{var2_name}' становить r = {r:.3f}.\n\n"
                f"Це вказує на {strength} {sign} ЛІНІЙНИЙ зв'язок між змінними.\n\n"
                f"Інтерпретація в контексті енергосистеми:\n"
            )
            if r < -0.3:
                 analysis_conclusion += (f"Спостерігається {strength} негативна лінійна тенденція: при підвищенні температури, завантаження лінійно дещо знижується (можливо, через зменшення опалення).\n\n")
            elif r > 0.3:
                 analysis_conclusion += (f"Спостерігається {strength} позитивна лінійна тенденція: при підвищенні температури, завантаження лінійно дещо зростає (можливо, через кондиціонування).\n\n")
            else:
                 analysis_conclusion += (f"Сила лінійного зв'язку класифікується як '{strength}'. Це НЕ означає відсутність впливу температури! ")

            analysis_conclusion += (
                f"Залежність завантаження від температури часто є НЕЛІНІЙНОЮ (U-подібною: високе споживання і в мороз, і в спеку). "
                f"Коефіцієнт Пірсона 'r' погано вловлює такі складні залежності. Діаграма розсіювання (графік праворуч) може дати краще уявлення про реальну форму зв'язку.\n\n"
                f"Висновок: Хоча лінійний зв'язок є {strength}, для точного моделювання слід використовувати нелінійні моделі (див. Завдання 3 та 4)."
            )

            analysis_text.config(state='normal')
            analysis_text.delete('1.0', tk.END)
            analysis_text.insert('1.0', analysis_conclusion)
            analysis_text.config(state='disabled')

            ax.clear()
            ax.scatter(var1_data, var2_data, alpha=0.5)
            ax.set_title(f"Діаграма розсіювання: {var1_name} vs {var2_name}\n(r = {r:.3f})")
            ax.set_xlabel(var1_name)
            ax.set_ylabel(var2_name)
            ax.grid(True)
            canvas.draw()

            print("Обчислення та візуалізація кореляції завершено.")

        except Exception as e:
            error_msg = f"Помилка: {e}"
            result_label.config(text="Помилка")
            analysis_text.config(state='normal')
            analysis_text.delete('1.0', tk.END)
            analysis_text.insert('1.0', error_msg)
            analysis_text.config(state='disabled')
            ax.clear(); ax.text(0.5, 0.5, error_msg, ha='center', va='center', color='red'); canvas.draw()
            print(error_msg)
            import traceback
            traceback.print_exc()

    update_button = ttk.Button(left_frame, text="Перерахувати (нові дані)", command=calculate_and_plot_correlation)
    update_button.grid(row=3, column=0, sticky='sew', pady=10)

    calculate_and_plot_correlation()