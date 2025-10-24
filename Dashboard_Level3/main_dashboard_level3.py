
import tkinter as tk
from tkinter import ttk
import sys
import os

sys.path.append(os.path.dirname(__file__))

import tab9_decision_tree
import tab10_correlation
import tab11_nonlinear_model 
import tab12_regression_analysis 

def main():
    """
    Головна функція для створення та запуску ДРУГОГО дашборду (Рівень 3).
    """

    root = tk.Tk()
    root.title("Дашборд (Рівень 3): Моніторинг Енергосистеми")
    root.geometry("1100x850") 

    style = ttk.Style()
    style.theme_use('clam')

    tab_control = ttk.Notebook(root)
    tab_control.pack(expand=1, fill="both", padx=10, pady=10)

    try:

        # Завдання 1 (Вкладка 9)
        tab9_decision_tree.create_tab(tab_control)

        # Завдання 2 (Вкладка 10)
        tab10_correlation.create_tab(tab_control)

        # Завдання 3 (Вкладка 11)
        tab11_nonlinear_model.create_tab(tab_control) 

        # Завдання 4 (Вкладка 12)
        tab12_regression_analysis.create_tab(tab_control) 


    except Exception as e:
        error_frame = ttk.Frame(tab_control, padding=(10, 10))
        tab_control.add(error_frame, text='Помилка Завантаження')
        ttk.Label(error_frame, text=f"Не вдалося завантажити вкладку:\n{e}",
                  font=("Helvetica", 12), foreground="red").pack(padx=20, pady=20)
        print(f"Помилка при створенні вкладки: {e}")
        import traceback
        traceback.print_exc()

    print("Дашборд Рівня 3 запущено.")
    root.mainloop()
    print("Програму закрито.")

if __name__ == "__main__":
    main()