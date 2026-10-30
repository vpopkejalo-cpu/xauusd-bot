#!/bin/bash
# ===================================================================
# Автоматический скрипт установки и запуска торгового помощника XAU/USD
# Разработано для Google Cloud e2-micro (Ubuntu / Debian Linux)
# ===================================================================

set -e

echo ""
echo "==================================================================="
echo "   АВТОМАТИЧЕСКАЯ УСТАНОВКА ТОРГОВОГО ПОМОЩНИКА XAU/USD (ЗОЛОТО)   "
echo "==================================================================="
echo ""

# Определение рабочей директории и пользователя
INSTALL_DIR=$(pwd)
CURRENT_USER=$(whoami)

echo ">>> [1/6] Проверка и настройка файла подкачки (SWAP 2GB)..."
# На серверах e2-micro всего 1 ГБ оперативной памяти. SWAP предотвращает сбои из-за нехватки памяти.
if [ ! -f /swapfile ]; then
    echo "Создание файла подкачки (SWAP) для стабильной работы 1GB RAM..."
    sudo fallocate -l 2G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=2048
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "SWAP 2GB успешно подключен!"
else
    echo "SWAP уже настроен."
fi

echo ""
echo ">>> [2/6] Обновление пакетов и установка системных утилит..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv git curl

echo ""
echo ">>> [3/6] Создание виртуального окружения Python..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Виртуальное окружение venv создано."
fi

source venv/bin/activate
pip install --upgrade pip

echo ""
echo ">>> [4/6] Установка необходимых библиотек..."
pip install -r requirements.txt

echo ""
echo ">>> [5/6] Настройка конфигурации (.env)..."
if [ ! -f ".env" ]; then
    echo "-------------------------------------------------------------------"
    echo "Файл конфигурации не найден. Давайте настроим его прямо сейчас!"
    echo "-------------------------------------------------------------------"
    
    echo -n "Введите ваш API ключ Twelve Data (например, a1b2c3d4...): "
    read -r TD_KEY
    
    echo -n "Введите ваш Telegram Bot Token (например, 1234567890:ABCdef...): "
    read -r TG_TOKEN
    
    echo -n "Введите ваш Telegram Chat ID (например, 987654321): "
    read -r TG_CHAT
    
    cat <<EOF > .env
TWELVE_DATA_API_KEY=${TD_KEY}
TELEGRAM_BOT_TOKEN=${TG_TOKEN}
TELEGRAM_CHAT_ID=${TG_CHAT}
SYMBOL=XAU/USD
TIMEFRAME=15min
POLL_INTERVAL_SECONDS=300
MIN_PROBABILITY=0.60
ATR_PERIOD=14
SL_ATR_MULTIPLIER=1.5
TP_ATR_MULTIPLIER=2.5
FILTER_ASIAN_SESSION=true
ASIAN_SESSION_START_HOUR=23
ASIAN_SESSION_END_HOUR=8
EOF
    echo "Файл .env успешно создан!"
else
    echo "Файл .env уже существует. Используем текущую конфигурацию."
fi

echo ""
echo ">>> [6/6] Создание службы автозапуска systemd (24/7)..."
SERVICE_FILE="/etc/systemd/system/xauusd_bot.service"

sudo bash -c "cat <<EOF > ${SERVICE_FILE}
[Unit]
Description=Autonomous XAU/USD Telegram Trading Assistant
After=network.target

[Service]
Type=simple
User=${CURRENT_USER}
WorkingDirectory=${INSTALL_DIR}
ExecStart=${INSTALL_DIR}/venv/bin/python3 ${INSTALL_DIR}/main.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF"

echo "Перезагрузка демона systemd и запуск службы..."
sudo systemctl daemon-reload
sudo systemctl enable xauusd_bot.service
sudo systemctl restart xauusd_bot.service

echo ""
echo "==================================================================="
echo "🎉 ПОЗДРАВЛЯЕМ! ТОРГОВЫЙ БОТ УСПЕШНО РАЗВЕРНУТ И ЗАПУЩЕН 24/7!"
echo "==================================================================="
echo ""
echo "В ваш Telegram должно было прийти приветственное сообщение от бота."
echo ""
echo "📌 ПОЛЕЗНЫЕ КОМАНДЫ ДЛЯ УПРАВЛЕНИЯ:"
echo "• Посмотреть статус бота:     sudo systemctl status xauusd_bot"
echo "• Посмотреть логи в реале:    sudo journalctl -u xauusd_bot -f"
echo "• Перезапустить бота:         sudo systemctl restart xauusd_bot"
echo "• Остановить бота:            sudo systemctl stop xauusd_bot"
echo ""
