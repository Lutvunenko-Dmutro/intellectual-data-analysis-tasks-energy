import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler 
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
    """Створює вміст для вкладки Регресійного аналізу."""

    tab12 = ttk.Frame(tab_control, padding=(10, 10))
    tab_control.add(tab12, text='Завд. 4 (12): Регресія')

    main_frame = ttk.Frame(tab12)
    main_frame.pack(fill="both", expand=True)

    LEFT_WIDTH = 450
    left_frame = ttk.Frame(main_frame, width=LEFT_WIDTH)
    left_frame.pack(side="left", fill="y", padx=(0, 10), expand=False)
    left_frame.pack_propagate(False)
    left_frame.grid_rowconfigure(0, weight=0)
    left_frame.grid_rowconfigure(1, weight=1)
    left_frame.grid_columnconfigure(0, weight=1)

    right_frame = ttk.LabelFrame(main_frame, text="Візуалізація: Реальні vs Прогнозовані Значення", padding=10)
    right_frame.pack(side="right", fill="both", expand=True)

    theory_frame = ttk.LabelFrame(left_frame, text="Теорія: Лінійна Регресія", padding=10)
    theory_frame.grid(row=0, column=0, sticky='new')
    theory_frame.grid_columnconfigure(0, weight=1)

    MESSAGE_WIDTH = LEFT_WIDTH - 40

    def create_theory_section(parent, title, content):
        frame = ttk.LabelFrame(parent, text=title, padding=7)
        frame.pack(fill="x", pady=5)
        msg = tk.Message(frame, text=content, width=MESSAGE_WIDTH, justify="left", font=("Helvetica", 10))
        msg.pack(fill='x', expand=True)

    theory_content_1 = (
        "Регресійний аналіз моделює зв'язок між залежною змінною (Y, 'Завантаження') та однією чи кількома незалежними (X, 'Година', 'Температура'...). Лінійна регресія припускає лінійний зв'язок:\n\n"
        "Y = β₀ + β₁X₁ + β₂X₂ + βn*Xn + ε\n\n" 
        "- β₀: Вільний член (intercept).\n"
        "- β₁, βn: Коефіцієнти регресії, показують вплив кожної X на Y." 
    )
    create_theory_section(theory_frame, "Рівняння Регресії", theory_content_1)

    theory_content_2 = (
        "Якість моделі оцінюється за:\n"
        "- R² (Коефіцієнт детермінації): Частка поясненої варіації (0-1).\n"
        "- Adjusted R²: R², скоригований на кількість X.\n"
        "- RMSE: Середня помилка прогнозу (в одиницях Y).\n"
        "- P-values коефіцієнтів: Статистична значущість впливу кожної X (p < 0.05 => значущий)."
    )
    create_theory_section(theory_frame, "Оцінка Якості Моделі", theory_content_2)

    analysis_frame = ttk.LabelFrame(left_frame, text="Аналіз Моделі та Висновок", padding=10)
    analysis_frame.grid(row=1, column=0, sticky='nsew', pady=(10, 0))
    analysis_frame.grid_columnconfigure(0, weight=1)
    analysis_frame.grid_rowconfigure(0, weight=1)

    analysis_text = scrolledtext.ScrolledText(analysis_frame, wrap=tk.WORD, height=15,
                                              font=("Helvetica", 10), state='disabled')
    analysis_text.pack(fill="both", expand=True)

    fig = Figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill="both", expand=True)

    def build_evaluate_regression():
        print("Побудова регресії (Вкладка 12)...")
        try:
            df = create_energy_dataset(n_samples=500)
            feature_cols = ['Година доби', 'Температура', 'Пром. активність', 'Генерація СЕС']
            df['Тип_Дня'] = df['День тижня'].apply(lambda x: 0 if x < 5 else 1)
            feature_cols.append('Тип_Дня')
            target_col = 'Завантаження'

            if not all(col in df.columns for col in feature_cols + [target_col]):
                 raise ValueError(f"Не всі колонки є: {feature_cols + [target_col]}")

            X = df[feature_cols]
            y = df[target_col]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = LinearRegression()
            model.fit(X_train_scaled, y_train)

            intercept = model.intercept_
            coeffs = model.coef_

            equation = f"Завантаження_станд ≈ {intercept:.2f}"
            coefficients_info = []
            for name, coef in zip(feature_cols, coeffs):
                equation += f" {'+' if coef >= 0 else '-'} {abs(coef):.2f}*{name}_станд"
                coefficients_info.append(f"- {name}: {coef:.2f}")

            y_pred = model.predict(X_test_scaled)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            analysis_conclusion = (
                f"Побудовано лінійну регресійну модель для прогнозування Завантаження.\n\n"
                f"Рівняння регресії (для стандартизованих даних):\n{equation}\n\n"
                f"Аналіз впливу змінних (коефіцієнти для стандартизованих даних):\n"
                f"Коефіцієнти показують зміну Завантаження (в од. станд. відхилення) при зміні ознаки на 1 її станд. відхилення:\n"
                + "\n".join(coefficients_info) + "\n\n"
                f"Оцінка якості моделі:\n"
                f"- R² = {r2:.3f} (модель пояснює ~{r2*100:.1f}% варіації).\n"
                f"- RMSE = {rmse:.1f} МВт (середня помилка).\n\n"
                f"Висновок: Якість лінійної моделі (R²) є {('дуже низькою', 'низькою', 'помірною', 'високою', 'дуже високою')[int(r2*5)] if 0<=r2<=1 else 'некоректною'}. "
                f"Найбільший вплив (за модулем коеф.) має '{feature_cols[np.argmax(np.abs(coeffs))]}'. "
                f"Лінійна модель може бути недостатньо точною через нелінійні залежності Завантаження."
            )

            analysis_text.config(state='normal')
            analysis_text.delete('1.0', tk.END)
            analysis_text.insert('1.0', analysis_conclusion)
            analysis_text.config(state='disabled')

            ax.clear()
            ax.scatter(y_test, y_pred, alpha=0.5)
            min_val = min(y_test.min(), y_pred.min()); max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ідеальний прогноз')
            ax.set_title(f"Реальні vs Прогнозовані Значення\n(R²={r2:.3f}, RMSE={rmse:.1f})")
            ax.set_xlabel("Реальне Завантаження (МВт)")
            ax.set_ylabel("Прогнозоване Завантаження (МВт)")
            ax.legend()
            ax.grid(True)
            canvas.draw()

            print("Побудова та оцінка регресії завершено.")

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

    update_button = ttk.Button(left_frame, text="Перерахувати (нові дані)", command=build_evaluate_regression)
    update_button.grid(row=2, column=0, sticky='sew', pady=10)

    build_evaluate_regression()