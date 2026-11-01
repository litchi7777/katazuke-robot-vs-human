# 実機での動かし方

必ず人がそばにいて、いつでも止められる状態で動かしてください。

1. **通信の確認**（ドライバーを止めた状態で。読み取りだけで、アームは動きません）

   ```bash
   source scripts/ros2_env.sh
   python3 scripts/check_can_stability_readonly.py --output-dir outputs/preflight
   ```

2. **ドライバーの起動**（起動するとモーターに力が入ります）

   ```bash
   ros2 launch rebotarm_bringup driver.launch.py \
     hardware_config:=$PWD/scripts/config/arm_pick_wide_trial.yaml \
     arm_namespace:=rebotarm_arm_hold_trial joint_state_rate:=30.0
   ```

3. **片付け**（1回ずつ挑み、失敗は後回しにして、最後にやり直す）

   ```bash
   python3 scripts/tidy_desk.py --preflight outputs/preflight/summary.json \
     --boxes scripts/config/desk_boxes.json --driver-config scripts/config/arm_pick_wide_trial.yaml \
     --max-speed 0.35 --approve web --host 0.0.0.0 --port 8765 \
     --rounds 4 --attempts 1 --out outputs/run_1
   ```

   迷う物の確認は、表示された Web ページで答えます。

カメラの校正値（`scripts/calib/`）と箱の位置（`scripts/config/desk_boxes.json`）は、この机のものです。別の机では測り直してください。
