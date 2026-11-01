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

3. **片付け**（`scripts/` で実行。通信チェックとドライバーの起動は `real_ready.sh` が回のたびに行う）

   ```bash
   cd scripts
   source /opt/ros/humble/setup.bash && source ../rebotarm_ros2/install/setup.bash
   export ROS_DOMAIN_ID=42 FASTDDS_BUILTIN_TRANSPORTS=UDPv4 MUJOCO_GL=egl MJ_FINGER_KP=1200
   mkdir -p ../outputs/ctl
   REAL_CTL=$PWD/../outputs/ctl python3 -u rehearse.py --real --out ../outputs/run_1 --execute --real-first \
     --preflight ../outputs/ctl/P.json --preflight-cmd "REAL_CTL=$PWD/../outputs/ctl bash real_ready.sh" \
     --tidy-args "--host 0.0.0.0 --port 8765 --max-speed 0.35"
   ```

   迷う物の確認は `http://localhost:8765/` で答えます。アームがホームにいない時は、ドライバーを止めません。

電源を入れ直すと、グリッパーのゼロ点が1回転分ずれることがあります。動かす前のチェックで止まるので、手で閉じてから、モーター7の `set_zero_position` を実行してください。

カメラの校正値（`scripts/calib/`）と箱の位置（`scripts/config/desk_boxes.json`）は、この机のものです。別の机では測り直してください。
