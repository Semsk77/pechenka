import pyautogui
import numpy as np
from ultralytics import YOLO
import time
import keyboard
import threading

# Загрузка модели
model = YOLO('runs/detect/cookie_detector-5/weights/best.pt')

# --- НАСТРОЙКИ ---
CLICKS_PER_LOOP = 50          # Сколько раз кликнуть по печеньке за один цикл
COOKIE_CLICK_DELAY = 0     # Задержка между кликами по печеньке
CONFIDENCE_THRESHOLD = 0.4    # Порог уверенности модели

COOKIE_CLASS_NAME = 'game'    
UPGRADE_CLASS_NAME = 'upgrades'

# Координаты для прокрутки магазина 
SHOP_X = 1753                 # Координата X (по горизонтали) внутри окна магазина, чтобы туда навести мышку для прокрутки
SHOP_Y = 543                  
SCROLL_AMOUNT = -500          # Прокрутка вниз

# Переменные состояния
last_upgrade_time = 0
UPGRADE_CHECK_INTERVAL = 15.0  # Проверяем магазин раз в 3 секунды
is_paused = False             # Флаг паузы

def click(x, y):
    pyautogui.click(int(x), int(y))

def toggle_pause():
    """Функция, которая переключает паузу при нажатии пробела"""
    global is_paused
    is_paused = not is_paused
    if is_paused:
        print("\n⏸️ ПРОГРАММА НА ПАУЗЕ. Нажмите пробел, чтобы продолжить...")
    else:
        print("\n▶️ ПРОГРАММА ПРОДОЛЖАЕТ РАБОТУ.")

# Запускаем слушатель клавиш в отдельном потоке (чтобы он работал параллельно с основным циклом)
keyboard.on_press_key("space", lambda _: toggle_pause())

def click_cookie_and_find_upgrades():
    global last_upgrade_time, is_paused
    
    # Если стоит пауза - просто ждем и не делаем ничего
    while is_paused:
        time.sleep(0.1)
        
    # 1. Делаем скриншот
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    
    # 2. Отправляем в нейросеть
    results = model(screenshot_np, verbose=False)

    cookie_center = None
    upgrade_centers = []

    # 3. Разбираем результаты
    for result in results:
        for box in result.boxes:
            confidence = box.conf[0]
            
            if confidence > CONFIDENCE_THRESHOLD:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                x1, y1, x2, y2 = box.xyxy[0]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                if class_name == COOKIE_CLASS_NAME:
                    cookie_center = (cx, cy)
                elif class_name == UPGRADE_CLASS_NAME:
                    upgrade_centers.append((cx, cy))

    current_time = time.time()

    # --- ЛОГИКА ---
    
    # ШАГ 1: Кликаем печеньку
    if cookie_center:
        for _ in range(CLICKS_PER_LOOP):
            # Проверка на экстренный выход
            if keyboard.is_pressed('q'):
                print("Остановлено пользователем (нажата Q).")
                exit()
                
            # Проверка на паузу внутри цикла кликов
            while is_paused:
                time.sleep(0.1)
                
            click(cookie_center[0], cookie_center[1])
            time.sleep(COOKIE_CLICK_DELAY)
    else:
        print("⚠️ Печенька не найдена на экране!")

    # ШАГ 2: Проверяем магазин
    if current_time - last_upgrade_time > UPGRADE_CHECK_INTERVAL:
        if upgrade_centers:
            print(f"🛒 Найдено кнопок: {len(upgrade_centers)}. Покупаем...")
            upgrade_centers.sort(key=lambda p: p[1])
            for ux, uy in upgrade_centers:
                while is_paused: # Проверка на паузу перед каждым кликом
                    time.sleep(0.1)
                click(ux, uy)
                time.sleep(0.05)
        else:
            print("🔍 Кнопки не найдены. Прокручиваем магазин...")
            pyautogui.moveTo(SHOP_X, SHOP_Y)
            pyautogui.scroll(SCROLL_AMOUNT)
            time.sleep(0.3) 
            
        last_upgrade_time = current_time

# Главный цикл
print("Бот запущен!")
print("Нажмите ПРОБЕЛ, чтобы поставить на паузу / снять с паузы.")
print("Нажмите Q, чтобы полностью остановить программу.")

while True:
    try:
        click_cookie_and_find_upgrades()
        time.sleep(0.05)
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Ошибка: {e}")
        time.sleep(1)