
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext 
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
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

def nonlinear_parabolic_model(temp, a, b, c):
    """Проста параболічна модель залежності навантаження від температури."""
    return a * (temp - b)**2 + c

def create_tab(tab_control):
    """Створює вміст для вкладки Нелінійної Моделі."""

    tab11 = ttk.Frame(tab_control, padding=(10, 10))
    tab_control.add(tab11, text='Завд. 3 (11): Нелінійна Модель')

    main_frame = ttk.Frame(tab11)
    main_frame.pack(fill="both", expand=True)

    LEFT_WIDTH = 400
    left_frame = ttk.Frame(main_frame, width=LEFT_WIDTH)
    left_frame.pack(side="left", fill="y", padx=(0, 10), expand=False)
    left_frame.pack_propagate(False)
    left_frame.grid_rowconfigure(0, weight=0) 
    left_frame.grid_rowconfigure(1, weight=1) 
    left_frame.grid_columnconfigure(0, weight=1)

    right_frame = ttk.LabelFrame(main_frame, text="Візуалізація: Нелінійна Залежність Навантаження від Температури", padding=10)
    right_frame.pack(side="right", fill="both", expand=True)

    theory_frame = ttk.LabelFrame(left_frame, text="Теорія: Нелінійні Моделі та Методи Оцінки", padding=10)
    theory_frame.grid(row=0, column=0, sticky='new')
    theory_frame.grid_columnconfigure(0, weight=1)

    MESSAGE_WIDTH = LEFT_WIDTH - 40

    def create_theory_section(parent, title, content):
        frame = ttk.LabelFrame(parent, text=title, padding=7)
        frame.pack(fill="x", pady=5)
        msg = tk.Message(frame, text=content, width=MESSAGE_WIDTH, justify="left", font=("Helvetica", 10))
        msg.pack(fill='x', expand=True)

    theory_content_1 = (
        "Часто зв'язок між змінними (напр., Температура -> Навантаження) не є прямою лінією. Для таких випадків використовують нелінійні регресійні моделі.\n\n"
        "Замість Y = aX + b, модель може мати вигляд Y = a*X^2 + b*X + c (поліноміальна), Y = a*exp(b*X) (експоненційна), Y = a*(X-b)^2 + c (параболічна) тощо."
    )
    create_theory_section(theory_frame, "Нелінійні моделі", theory_content_1)

    theory_content_2 = (
        "Параметри (a, b, c...) нелінійних моделей зазвичай знаходять ітераційними методами оптимізації, які мінімізують суму квадратів помилок:\n"
        "- Метод Градієнтного Спуску\n"
        "- Метод Гаусса-Ньютона\n"
        "- Метод Левенберга-Марквардта (LM) - гібридний, часто використовується (див. Завд. 6)."
    )
    create_theory_section(theory_frame, "Методи оцінювання параметрів", theory_content_2)

    theory_content_3 = (
        "Якість нелінійної моделі оцінюють за:\n"
        "- R² (Коефіцієнт детермінації): Частка поясненої варіації.\n"
        "- RMSE (Середньоквадратична помилка): Середня помилка прогнозу.\n"
        "- Візуальний аналіз: Наскільки добре крива моделі проходить через точки даних на діаграмі розсіювання."
    )
    create_theory_section(theory_frame, "Оцінка якості моделі", theory_content_3)

    analysis_frame = ttk.LabelFrame(left_frame, text="Аналіз та Результати", padding=10)
    analysis_frame.grid(row=1, column=0, sticky='nsew', pady=(10, 0))
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

    def fit_nonlinear_model_and_plot():
        print("Підгонка нелінійної моделі (Вкладка 11)...")
        try:
            df = create_energy_dataset(n_samples=500)
            x_var_name = 'Температура'
            y_var_name = 'Завантаження'

            if x_var_name not in df.columns or y_var_name not in df.columns:
                 raise ValueError(f"Колонки '{x_var_name}'/'{y_var_name}' відсутні.")

            x_data = df[x_var_name].values 
            y_data = df[y_var_name].values

            initial_guess = [5, 18, 1000]
            params, covariance = curve_fit(nonlinear_parabolic_model, x_data, y_data, p0=initial_guess)
            a_fit, b_fit, c_fit = params

            y_pred = nonlinear_parabolic_model(x_data, a_fit, b_fit, c_fit)
            r2 = r2_score(y_data, y_pred)
            rmse = np.sqrt(mean_squared_error(y_data, y_pred))

            analysis_conclusion = (
                f"Було виконано підгонку нелінійної параболічної моделі:\n"
                f"Навантаження ≈ a * (Температура - b)² + c\n\n"
                f"Знайдені параметри (метод Левенберга-Марквардта):\n"
                f"  a (крутизна) = {a_fit:.2f}\n"
                f"  b (опт. темп.) = {b_fit:.2f} °C\n"
                f"  c (базове навант.) = {c_fit:.0f} МВт\n\n"
                f"Оцінка якості моделі:\n"
                f"- R² = {r2:.3f} (модель пояснює {r2*100:.1f}% варіації навантаження)\n"
                f"- RMSE = {rmse:.1f} МВт (середня помилка прогнозу)\n\n"
                f"Висновок: Значення R² {'> 0.5' if r2 > 0.5 else '< 0.5'} вказує на {'задовільну' if r2 > 0.5 else 'посередню'} здатність простої параболічної моделі описати залежність навантаження від температури, ігноруючи інші фактори. Графік показує, що модель вловлює U-подібну тенденцію."
            )

            analysis_text.config(state='normal')
            analysis_text.delete('1.0', tk.END)
            analysis_text.insert('1.0', analysis_conclusion)
            analysis_text.config(state='disabled')

            ax.clear()

            sort_idx = np.argsort(x_data)
            x_sorted = x_data[sort_idx]
            y_sorted = y_data[sort_idx]
            y_pred_sorted = y_pred[sort_idx]

            ax.scatter(x_sorted, y_sorted, alpha=0.5, label='Реальні дані (з шумом)')
            ax.plot(x_sorted, y_pred_sorted, color='red', linewidth=2, label=f'Нелінійна модель (R²={r2:.3f})')
            ax.set_title(f"Нелінійна залежність: {y_var_name} від {x_var_name}")
            ax.set_xlabel(x_var_name + " (°C)")
            ax.set_ylabel(y_var_name + " (МВт)")
            ax.legend()
            ax.grid(True)
            canvas.draw()

            print("Підгонка нелінійної моделі завершена.")

        except Exception as e:
            error_msg = f"Помилка: {e}"
            analysis_text.config(state='normal')
            analysis_text.delete('1.0', tk.END)
            analysis_text.insert('1.0', error_msg)
            analysis_text.config(state='disabled')
            ax.clear(); ax.text(0.5, 0.5, error_msg, ha='center', va='center', color='red'); canvas.draw()
            print(error_msg)
            import traceback
            traceback.print_exc()

    update_button = ttk.Button(left_frame, text="Перерахувати (нові дані)", command=fit_nonlinear_model_and_plot)
    update_button.grid(row=3, column=0, sticky='sew', pady=10) 

    fit_nonlinear_model_and_plot()