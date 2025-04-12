#!/bin/bash
while true; do
  # 任务1：每周日到周四晚8:30执行
  if { [[ $(date +%u) -eq 7 ]] || [[ $(date +%u) -le 4 ]]; } && [[ $(date +%H:%M) == "20:30" ]]; then
    echo "===== 任务1开始 $(date) =====" >> /root/stock/cron.log
    /root/stock/venv/bin/python3 /root/stock/stock-select-vps.py --market-cap --debug >> /root/stock/cron.log 2>&1
    rm -f /root/stock/process_status.json /root/stock/amount_result.txt
    /root/stock/venv/bin/python3 /root/stock/stock-select-vps.py --amount --debug >> /root/stock/cron.log 2>&1
    echo "===== 任务1完成 $(date) =====" >> /root/stock/cron.log
  fi

  # 任务2：每周一到周五早7:30执行
  if [[ $(date +%u) -le 5 ]] && [[ $(date +%H:%M) == "07:30" ]]; then
    echo "===== 任务2开始 $(date) =====" >> /root/stock/cron.log
    /root/stock/venv/bin/python3 /root/stock/stock-select-vps.py --qiangrizhangfu --debug >> /root/stock/cron.log 2>&1
    /root/stock/venv/bin/python3 /root/stock/stock-select-vps.py --technical --debug >> /root/stock/cron.log 2>&1
    /root/stock/venv/bin/python3 /root/stock/stock-select-vps.py --send-report --debug >> /root/stock/cron.log 2>&1
    echo "===== 任务2完成 $(date) =====" >> /root/stock/cron.log
  fi

  sleep 60
done