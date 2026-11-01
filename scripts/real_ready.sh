#!/bin/bash
# Before a real attempt (rehearse.py --preflight-cmd): stop the driver if the arm is at home, take a fresh CAN
# preflight with it stopped, start the driver again. Refuses to stop the driver with the arm away from home:
# that would drop it.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CTL="${REAL_CTL:-$HERE/../outputs/summon/ctl}"
NS=rebotarm_arm_hold_trial
HOME_TOLERANCE=0.25    # rad on every arm joint
mkdir -p "$CTL"
cd "$HERE"
unset CONDA_PREFIX
source ./ros2_env.sh || exit 1
export PATH=/usr/bin:$PATH

if [ -f "$CTL/driver.pid" ] && kill -0 "$(cat "$CTL/driver.pid")" 2>/dev/null; then
    timeout 8 ros2 topic echo --once --field position "/$NS/joint_states" > "$CTL/q.txt" 2>/dev/null
    python3 - "$CTL/q.txt" "$HOME_TOLERANCE" <<'PY' || { echo "アームがホームにいないのでドライバーを止めない" >&2; exit 1; }
import re, sys
q = [float(v) for v in re.findall(r"-?\d+\.\d+(?:e-?\d+)?", open(sys.argv[1]).read())]
sys.exit(0 if len(q) >= 6 and max(abs(v) for v in q[:6]) < float(sys.argv[2]) else 1)
PY
    kill -INT -"$(cat "$CTL/driver.pid")" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$(cat "$CTL/driver.pid")" 2>/dev/null || break; sleep 0.5; done
    rm -f "$CTL/driver.pid"
    sleep 2
fi

if timeout 3 ros2 topic echo --once --field position "/$NS/joint_states" > /dev/null 2>&1; then
    echo "別のドライバーが動いている。止めてからやり直す" >&2
    exit 1
fi

n=$(date +%s)
/usr/bin/python3 check_can_stability_readonly.py --output-dir "$CTL/pf_$n" --seconds 30 || exit 1
cp "$CTL/pf_$n/summary.json" "$CTL/P.json"

setsid ros2 launch rebotarm_bringup driver.launch.py hardware_config:="$HERE/config/arm_pick_wide_trial.yaml" \
    arm_namespace:=$NS joint_state_rate:=30.0 > "$CTL/driver_$n.log" 2>&1 &
echo $! > "$CTL/driver.pid"
for _ in $(seq 1 40); do
    if timeout 2 ros2 topic echo --once --field position "/$NS/joint_states" > "$CTL/q.txt" 2>/dev/null; then
        echo "ドライバー起動: $(tr -d '\n' < "$CTL/q.txt" | cut -c1-90)"
        exit 0
    fi
    kill -0 "$(cat "$CTL/driver.pid")" 2>/dev/null || break
    sleep 1
done
echo "ドライバーが起動しない: $CTL/driver_$n.log" >&2
exit 1
