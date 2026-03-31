#!/bin/bash
set -e

APP_DIR="/home/maxzhuyt/police-web-app"
VENV_DIR="$APP_DIR/venv"
USER="maxzhuyt"

echo "=== Creating systemd service ==="
sudo tee /etc/systemd/system/police-web-app.service > /dev/null <<EOF
[Unit]
Description=Police Report Generator Web App
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
ExecStart=$VENV_DIR/bin/gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 300 app:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "=== Enabling and starting service ==="
sudo systemctl daemon-reload
sudo systemctl enable police-web-app.service
sudo systemctl start police-web-app.service

echo ""
echo "=== Done! ==="
sudo systemctl status police-web-app.service --no-pager
