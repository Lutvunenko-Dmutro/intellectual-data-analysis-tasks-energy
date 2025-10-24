import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
import re
import collections
import random
import datetime # Потрібно для генерації логів
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Використовуємо scikit-learn для TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# ##################################################################
# ---                ЛОГІКА АНАЛІЗУ ТЕКСТУ                       ---
# ##################################################################

def generate_log_file():
    """Створює симульований лог-файл."""
    # (Код генерації логів залишається без змін)
    logs = []
    start_time = datetime.datetime(2025, 10, 24, 14, 30, 0)
    common_events = [
        "INFO: Connection stable substation_3", "INFO: Load normal substation_1",
        "WARN: Voltage fluctuation line_12", "INFO: Connection stable substation_2",
        "INFO: Load normal substation_4", "WARN: Voltage fluctuation line_18",
    ]
    cascade_event = [
        (1, "WARN: Voltage fluctuation line_18"), (2, "WARN: Voltage fluctuation line_18"),
        (3, "ERROR: Breaker trip line_18"), (4, "CRITICAL: Overload transformer_A substation_7"),
        (5, "ALERT: High temperature transformer_A substation_7"), (6, "CRITICAL: Overload transformer_A substation_7"),
        (7, "CRITICAL: FIRE detected substation_7 physical_damage"), (8, "ERROR: Communication loss substation_7"),
        (9, "ERROR: Breaker trip line_45"), (10, "CRITICAL: Cascade failure imminent network_sector_B")
    ]
    current_time = start_time
    cascade_index = 0
    for i in range(1000):
        current_time += datetime.timedelta(seconds=random.randint(0, 1))
        if cascade_index < len(cascade_event) and i == 500 + cascade_event[cascade_index][0]:
            event = cascade_event[cascade_index][1]
            cascade_index += 1
        else: event = random.choice(common_events)
        logs.append(f"[{current_time}] {event}")
    return logs

def tokenize(text):
    """Оновлений токенізатор."""
    # (Код токенізації залишається без змін)
    text = re.sub(r'\[.*?\]', '', text)
    text = text.lower()
    text = re.sub(r'\b(info|warn|alert|critical|error)\b', '', text)
    tokens = re.findall(r'\b[a-z_0-9]+\b', text)
    tokens = [token for token in tokens if not token.isdigit()]
    return tokens

# ##################################################################
# ---               ГОЛОВНА ФУНКЦІЯ ДЛЯ СТВОРЕННЯ ВКЛАДКИ         ---
# ##################################################################

def create_tab(tab_control):
    """Створює вміст для п'ятої вкладки."""

    tab5 = ttk.Frame(tab_control, padding=(10, 10))
    tab_control.add(tab5, text='Завдання 5: Аналіз Логів (NLP)')

    # --- 1. Створення фреймів ---
    main_frame = ttk.Frame(tab5)
    main_frame.pack(fill="both", expand=True)

    # Задаємо фіксовану ширину і забороняємо стискання
    LEFT_WIDTH = 500
    left_frame = ttk.Frame(main_frame, width=LEFT_WIDTH)
    left_frame.pack(side="left", fill="y", padx=10, expand=False)
    left_frame.pack_propagate(False)
    # Налаштовуємо grid всередині left_frame
    left_frame.grid_rowconfigure(0, weight=0) # Теорія
    left_frame.grid_rowconfigure(1, weight=1) # Логи розтягуються
    left_frame.grid_columnconfigure(0, weight=1) # Колонка займає всю ширину


    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side="right", fill="both", expand=True)

    # --- 2. Ліва колонка (Теорія та Лог) ---
    #
    # ---- ВЕЛИКИЙ БЛОК ОНОВЛЕННЯ: ttk.Label + grid() для теорії ----
    #
    theory_frame = ttk.LabelFrame(left_frame, text="Закон Ципфа та TF-IDF (Теорія)", padding=10)
    theory_frame.grid(row=0, column=0, sticky='new') # Розміщуємо рамку теорії
    theory_frame.grid_columnconfigure(0, weight=1) # Дозволяємо вмісту розтягуватись

    # Задаємо ширину для переносу тексту (трохи менше ширини колонки)
    WRAP_WIDTH = LEFT_WIDTH - 40

    # --- Блок 1: Закон Ципфа ---
    zipf_theory_frame = ttk.LabelFrame(theory_frame, text="1. У чому полягає закон Ципфа?", padding=7)
    zipf_theory_frame.grid(row=0, column=0, sticky='new', pady=5)
    zipf_theory_frame.grid_columnconfigure(0, weight=1) # Колонка всередині

    zipf_content = (
        "Закон Ципфа — це емпіричний закон, який стверджує, що частота слова обернено пропорційна його рангу (його порядковому номеру у списку за спаданням частоти).\n\n"
        "ПРОСТОЮ МОВОЮ: У будь-якому тексті (включно з логами) буде декілька ДУЖЕ ЧАСТИХ слів ('INFO', 'WARN') і величезна кількість рідкісних слів ('FIRE'). Графік (Ранг-Частота) у лог-координатах буде прямою лінією."
    )
    # Використовуємо ttk.Label з wraplength, розміщений через grid
    lbl_zipf = ttk.Label(zipf_theory_frame, text=zipf_content, wraplength=WRAP_WIDTH, justify="left", font=("Helvetica", 10))
    lbl_zipf.grid(row=0, column=0, sticky='ew')


    # --- Блок 2: Ваги слів ---
    tfidf_theory_frame = ttk.LabelFrame(theory_frame, text="2. Яким чином обчислити ваги слів?", padding=7)
    tfidf_theory_frame.grid(row=1, column=0, sticky='new', pady=5)
    tfidf_theory_frame.grid_columnconfigure(0, weight=1) # Колонка всередині

    tfidf_content = (
        "A) Проста частота (Frequency): Просто порахувати, скільки разів зустрічається слово. Мінус: 'INFO' матиме найвищу вагу, хоча воно неважливе.\n\n"
        "Б) TF-IDF (Term Frequency-Inverse Document Frequency): Це найкращий метод. Вага слова = (Як часто слово в цьому повідомленні?) × (Наскільки рідкісне це слово у всіх інших повідомленнях?).\n\n"
        "РЕЗУЛЬТАТ: TF-IDF дає найвищу вагу словам, які часто зустрічаються в *одному конкретному* повідомленні про аварію, але *рідко* зустрічаються в усьому іншому \"шумному\" лог-файлі (напр., 'FIRE', 'substation_7')."
    )
    # Використовуємо ttk.Label з wraplength, розміщений через grid
    lbl_tfidf = ttk.Label(tfidf_theory_frame, text=tfidf_content, wraplength=WRAP_WIDTH, justify="left", font=("Helvetica", 10))
    lbl_tfidf.grid(row=0, column=0, sticky='ew')
    #
    # ---- КІНЕЦЬ БЛОКУ ОНОВЛЕННЯ ----
    #

    # --- Кнопка запуску ---
    run_button = ttk.Button(theory_frame, text="Згенерувати та Проаналізувати Лог-файл", command=lambda: run_analysis(log_widget, ax_zipf, canvas_zipf, tree_tfidf))
    run_button.grid(row=2, column=0, sticky='ew', pady=10)

    # --- Лог-файл ---
    log_frame = ttk.LabelFrame(left_frame, text="Симульований Журнал Тривог (N=1000)", padding=10)
    log_frame.grid(row=1, column=0, sticky='nsew', pady=10)

    log_widget = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=20,
                                            font=("Courier", 9), state='disabled')
    log_widget.pack(fill="both", expand=True)


    # --- 3. Права колонка (Результати) ---
    # (Код для правої колонки залишається без змін)
    zipf_frame = ttk.LabelFrame(right_frame, text="Доведення Закону Ципфа (Графік Ранг-Частота)", padding=10)
    zipf_frame.pack(fill="both", expand=True)
    fig_zipf = Figure(figsize=(6, 4), dpi=100)
    ax_zipf = fig_zipf.add_subplot(111)
    canvas_zipf = FigureCanvasTkAgg(fig_zipf, master=zipf_frame)
    canvas_zipf.get_tk_widget().pack(fill="both", expand=True)

    tfidf_frame = ttk.LabelFrame(right_frame, text="Аналіз Ваги Слів (Frequency vs. TF-IDF)", padding=10)
    tfidf_frame.pack(fill="both", expand=True, pady=10)
    tree_cols = ('rank', 'freq_word', 'freq_val', 'tfidf_word', 'tfidf_val')
    tree_tfidf = ttk.Treeview(tfidf_frame, columns=tree_cols, show='headings', height=10)
    tree_tfidf.column('rank', width=50, anchor='center'); tree_tfidf.heading('rank', text='Ранг')
    tree_tfidf.column('freq_word', width=120, anchor='w'); tree_tfidf.heading('freq_word', text='Топ за Частотою (Шум)')
    tree_tfidf.column('freq_val', width=80, anchor='e'); tree_tfidf.heading('freq_val', text='Частота')
    tree_tfidf.column('tfidf_word', width=150, anchor='w'); tree_tfidf.heading('tfidf_word', text='Топ за TF-IDF (Сигнал)')
    tree_tfidf.column('tfidf_val', width=80, anchor='e'); tree_tfidf.heading('tfidf_val', text='Вага TF-IDF')
    tree_tfidf.pack(fill="both", expand=True)

    # --- 4. Функція Аналізу ---
    def run_analysis(log_widget, ax_zipf, canvas_zipf, tree_tfidf):
        # (Код аналізу залишається без змін)
        print("Запуск аналізу логів (Вкладки 5)...")
        log_data = generate_log_file()
        log_widget.config(state='normal')
        log_widget.delete('1.0', tk.END)
        log_widget.insert('1.0', "\n".join(log_data))
        log_widget.config(state='disabled')
        full_log_text = " ".join(log_data)
        zipf_tokens = re.findall(r'\b[a-z_0-9]+\b', full_log_text.lower())
        word_counts = collections.Counter(zipf_tokens)
        sorted_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        ranks = np.arange(1, len(sorted_counts) + 1)
        frequencies = [count for word, count in sorted_counts]
        ax_zipf.clear()
        ax_zipf.plot(np.log(ranks), np.log(frequencies), 'r-')
        ax_zipf.set_title("Закон Ципфа для логів енергосистеми")
        ax_zipf.set_xlabel("Log(Ранг слова)")
        ax_zipf.set_ylabel("Log(Частота слова)")
        ax_zipf.grid(True)
        canvas_zipf.draw()
        vectorizer = TfidfVectorizer(tokenizer=tokenize)
        tfidf_matrix = vectorizer.fit_transform(log_data)
        feature_names = np.array(vectorizer.get_feature_names_out())
        total_tfidf_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
        tfidf_indices = total_tfidf_scores.argsort()[::-1]
        sorted_tfidf_words = [(feature_names[i], total_tfidf_scores[i]) for i in tfidf_indices]
        tree_tfidf.delete(*tree_tfidf.get_children())
        for i in range(10):
            rank = i + 1
            freq_word, freq_val = sorted_counts[i]
            if i < len(sorted_tfidf_words):
                 tfidf_word, tfidf_val = sorted_tfidf_words[i]
            else:
                 tfidf_word, tfidf_val = "-", 0.0
            tree_tfidf.insert('', 'end', values=(rank, freq_word, freq_val, tfidf_word, f"{tfidf_val:.4f}"))
        print("Аналіз логів завершено.")

    # --- 5. Кнопка запуску ---
    # Вже створена вище

    # Запускаємо 1 раз при старті
    run_analysis(log_widget, ax_zipf, canvas_zipf, tree_tfidf)