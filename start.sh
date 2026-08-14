#!/bin/bash
cd "$(dirname "$0")"
while true; do
  python3 app.py >> /tmp/cardcrm.log 2>&1
  echo "Server crashed, restarting in 2 seconds..."
  sleep 2
done
