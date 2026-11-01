# 実機での動かし方

必ず人がそばにいて、いつでも止められる状態で動かしてください。

## 片付ける

`rehearse.py` が、通信の確認とドライバーの起動（`real_ready.sh`）を回のたびに行います。ドライバーを手で起動しないでください。別のドライバーが動いていると `real_ready.sh` は止まります。

```bash
cd scripts
source /opt/ros/humble/setup.bash && source ../rebotarm_ros2/install/setup.bash
export ROS_DOMAIN_ID=42 FASTDDS_BUILTIN_TRANSPORTS=UDPv4 MUJOCO_GL=egl MJ_FINGER_KP=1200
mkdir -p ../outputs/ctl
REAL_CTL=$PWD/../outputs/ctl python3 -u rehearse.py --real --out ../outputs/run_1 --execute --real-first \
  --preflight ../outputs/ctl/P.json --preflight-cmd "REAL_CTL=$PWD/../outputs/ctl bash real_ready.sh" \
  --tidy-args "--port 8765 --max-speed 0.35"
```

- 迷う物の確認は `http://localhost:8765/` で答えます。別の端末から答える時は `--tidy-args` に `--host 0.0.0.0` を足せますが、ページに認証はありません。同じネットワークの人が誰でも「捨てる」を押せます。
- `--out` は毎回新しいフォルダにします。

## 止める

電源を切るとアームは落ちます。緊急時以外は次の手順で止めてください。

1. `rehearse.py` を Ctrl-C で止めます。アームはその場で保持します。
2. 残った `arm_goto.py` を、プロセス番号を調べて `kill -9` します。
3. 指に物があれば、手で受けられるようにします。
4. 次を実行し、`q=` の行が出るのを待ってから、`arm`、`home`、`finish`、`quit` を1つずつ入力します。

```bash
cd scripts
python3 -u arm_goto.py --execute --resume --gripper --grip-open-max 4.5 --max-speed 0.2 \
  --driver-config config/arm_pick_wide_trial.yaml --preflight ../outputs/ctl/P.json \
  --min-tcp-z 0.0 --output-dir ../outputs/recover_$(date +%s)
```

`P.json` が1時間より古いと拒否されるので、その時は `touch` します。アームがホームにいない時、`real_ready.sh` はドライバーを止めません。

## グリッパーのゼロ点

電源を入れ直すと、グリッパーのゼロ点が1回転分ずれることがあります。動かす前のチェックで止まるので、次の手順で直します。

1. ドライバーを止めて、指を手で閉じます。
2. 次を実行します。モーターは動きません。

```bash
source scripts/ros2_env.sh
python3 -c "
import time
from motorbridge import Controller
c = Controller(channel='can0'); m = c.add_robstride_motor(7, 0xFD, 'rs-00')
m.set_zero_position(); time.sleep(0.5)
print(m.robstride_get_param_f32(0x7019, 200)); c.close()"
```

表示が0付近になれば成功です。

## 別の机で使う時

カメラの校正値（`scripts/calib/`）と箱の位置（`scripts/config/desk_boxes.json`）は、この机のものです。測り直してください。
