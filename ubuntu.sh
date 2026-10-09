#!/usr/bin/env bash

# 添加官方驱动源（如果还没加）
# sudo add-apt-repository ppa:graphics-drivers/ppa
# sudo apt update
# sudo apt install nvidia-driver-580-open


cd /home/my/openpilot
source /home/my/openpilot/.venv/bin/activate

# --- rear camera restart loop (every 20min1s = 1201s) ---
restart_rear() {
  return
  PROC_NAME=manager
  ProcNumber=`ps -ef |grep -w $PROC_NAME|grep -v grep|wc -l`
  if [ $ProcNumber -le 0 ];then
    echo "op is not running.."
    return
  else
    pkill -f "build/left" 2>/dev/null
    pkill -f "build/right" 2>/dev/null
    sleep 1
    nohup /home/my/rear/build/left 'usb-0000:04:00.3-2.1' > left.txt 2>&1 &
    nohup /home/my/rear/build/right 'usb-0000:04:00.3-2.2' > right.txt 2>&1 &
    echo "$(date): rear processes restarted"
  fi

}

# restart_rear
while true; do
  sleep 1202
  # restart_rear
done &
REAR_WATCHER_PID=$!

ProcNumber=`ps -ef |grep -w "left"|grep -v grep|wc -l`
if [ $ProcNumber -le 0 ];then
   # initial start handled by restart_rear above
   :
fi

nohup /home/my/rear/delete.sh > del.txt 2>&1 &

export FINGERPRINT="TOYOTA_HIGHLANDER_TSS2"
export SKIP_FW_QUERY="1"

PROC_NAME=manager1
ProcNumber=`ps -ef |grep -w $PROC_NAME|grep -v grep|wc -l`
if [ $ProcNumber -le 0 ];then
   export ATHENA_HOST='ws://385770bs19.zicp.vip:6899'
   export API_HOST='http://385770bs19.zicp.vip:6898'
   export MAPBOX_TOKEN='pk.eyJ1Ijoiam5ld2IiLCJhIjoiY2xxNW8zZXprMGw1ZzJwbzZneHd2NHljbSJ9.gV7VPRfbXFetD-1OVF0XZg'
   cd /home/my/openpilot

   cd /home/my/openpilot/system/manager
    DP_YOLO_EVAL_ENABLED=1 DP_YOLO_USE_ONLY_LEAD=0 USE_WEBCAM=1 ./manager.py
   #./launch_openpilot.sh
   cd /home/my/openpilot
else
   echo "op is running.."
fi

