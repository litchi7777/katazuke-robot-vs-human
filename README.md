# 絶対に片付けるロボット vs 絶対に散らかす男

6軸アーム（Seeed reBot Arm B601-RS）と Claude で、机の上の物を片付けるロボットです。
カメラで机を見て、物ごとに「ゴミか残すか」「どうつかむか」を判断し、ゴミ箱と保管箱へ運びます。

もう一つのテーマは「**Claude でできる片付けの限界はどこか**」です。「散らかす男」は人間です。人間が実際の机に物を置き、ロボットがそれを片付けます。シミュレーションでは、人間の代わりの練習相手として Claude が弱点を突く机を作り、失敗の原因を直しては、また挑みます。

## 仕組み

1. **見る**：上のカメラの写真を Claude が読み、物の一覧を作る。SAM で物の輪郭を切り出す。
2. **判断する**：物ごとに、ゴミ／残す、握る力、つかみ方を決める（`claude -p` を並列で呼ぶ）。迷う物は Web ページで人に確認する。先に片付けられる物から進める。
3. **つかむ**：形に合わせて作戦を変える。細い物は机の2 mm上まで下ろす。幅の広い物は取っ手や縁をつまむ。柔らかい物は短い辺を挟む。
4. **確かめる**：持ち上げたあと、元の場所から物が消えたかをカメラで見る。消えていなければ運ばない。
5. **運ぶ**：途中の物に当たらない経路を選び、箱の上で放す。
6. **やり直す**：失敗した物は後回しにして、全部運んだあと、写真を撮り直してもう一度挑む。

### 実機とシミュレーションの往復（`scripts/rehearse.py`）

1. まず実機で試す。動画は1回目から番号順に残す。
2. 失敗したら、写真から物を3Dモデルにしてシミュレーションに召喚する（Hunyuan3D、`scripts/summon.py`）。
3. Claude が作戦（握る力、開き幅、向き、縁をつまむ、など）を出してシミュレーションで試し、失敗を振り返って次の作戦を出す。シミュレーションは何回でも回せる。
4. 実機で再挑戦する。実機の挑戦は5回まで。

### 安全の決まり

ソフトの層を分けて、次の決まりを固定しています。

- **残す物は捨てない。精密機器はつかまない**（触れている物は指を離してつかむ）。
- グリッパーの開きは 4.5 rad まで。ドライバーの設定と `arm_goto.py` の両方で止める。
- 動かす前に、グリッパーの読みが正常な範囲かを確かめる。外れていたら動かない。
- 実機の前に、CAN 通信の異常がないことを読み取りだけで確かめる。

## 実験

| スクリプト | 内容 |
|---|---|
| `scripts/tidy_desk.py` | 片付けの本体。実機とシミュレーションの両方で動く |
| `scripts/rehearse.py`、`summon.py` | 写真から3Dで召喚し、シミュレーションと実機を往復する |
| `scripts/battle.py` | シミュレーションでの練習試合。人間の代わりに Claude が机を作り、ロボットが片付ける |
| `scripts/object_study.py` | 物を1個ずつ机に置いて、物そのものの難しさを測る |
| `scripts/mj_campaign.py` | 決まった机のセットで成績を測る（改善で悪化していないかの確認） |
| `scripts/desk_memory.py` | 経験ノート。失敗から教訓を貯めて次に使う |
| `scripts/mj_scene.py`、`mj_arm.py` | MuJoCo の机とアーム。カメラは実機の校正値を使う |
| `scripts/make_final_video.py`、`make_final_edit.py`、`make_flow_steps.py` | 提出動画の編集に使ったもの。実験の録画（`outputs/`、リポジトリには含まない）が必要なので、そのままでは動かない |

## 動かし方

Ubuntu 22.04、ROS 2 Humble で確認しています。シミュレーションだけなら実機は要りません。

### 準備

アームのモデルとドライバーは再配布せず、元のリポジトリから固定コミットで取り寄せて、差分を当てます。実機では、差分に入っているグリッパーの開き幅の上限が必須です。

```bash
vcs import < dependencies.repos    # rebotarm_ros2、reBotArm_control_py、motorbridge を取得
git -C rebotarm_ros2 apply ../patches/rebotarm_ros2-local.patch
git -C rebotarm_ros2/third_party/reBotArm_control_py -c core.whitespace=cr-at-eol apply ../../../patches/rebotarm-sdk-safety.patch
bash scripts/build_motorbridge_local.sh    # 実機のみ。Rust が要る
(cd rebotarm_ros2 && colcon build)         # 実機のみ
```

Python の依存は `mujoco`、`pinocchio`、`opencv-python`、`numpy`、`scipy`、`Pillow`、`PyYAML`、`pytest` です。ほかに次の物を使います。

- 判定：[Claude Code](https://claude.com/claude-code) の `claude` コマンド
- 輪郭：conda 環境 `tidyvision` に `ultralytics` を入れ、SAM の重み `sam2.1_b.pt` を `scripts/` に置く
- 写真からの3D：`~/Hunyuan3D-2.1` の [Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)（なければ輪郭の押し出しで代用）
- 録画と動画：`ffmpeg`、Noto Sans CJK フォント

### シミュレーション

```bash
cd scripts
source /opt/ros/humble/setup.bash
export MUJOCO_GL=egl MJ_FINGER_KP=1200
python3 -m pytest -q tests                                       # テスト
python3 object_study.py --out ../outputs/study --kind gso_sponge --pose 0   # 物を1個だけ
python3 battle.py --out ../outputs/battle                        # シミュレーションの練習試合
```

### 実機

実機では、必ず人がそばにいる状態で動かしてください。止め方を含めた手順は [`docs/real_arm.md`](docs/real_arm.md) にあります。

環境変数で、人から渡す情報を指定できます。

| 変数 | 内容 |
|---|---|
| `TIDY_PERSON_NOTES` | 人が判定に伝える事実のファイル（例：「このコップは樹脂」）。あると記憶より優先する |
| `TIDY_LEARNED` | 物の名前の一部と作戦の対応表（JSON）。計画がない物にだけ効く |
| `TIDY_RIM_INSET` | 縁をつまむ位置の、外周からの距離（m、既定 0.008） |
| `TIDY_REAL_HEIGHTS` | 実機で測った高さ（`x,y,h;...`）。召喚に使う |

## ライセンス

このリポジトリのコードは MIT ライセンスです（[`LICENSE`](LICENSE)）。素材は次の通りです。

- `scripts/mj_assets/objects/gso_*` は [Google Scanned Objects](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/)（CC-BY 4.0）を変換したものです。詳細は [`GSO_LICENSE.txt`](scripts/mj_assets/objects/GSO_LICENSE.txt) を見てください。
- `scripts/mj_assets/objects/` の `paper`、`paper_b`、`purple_marker` は、机の写真から Hunyuan3D-2.1 で作ったものです。Hunyuan3D のライセンスに従います。
- アームのモデルとドライバーは [Seeed-Projects/reBotArmController_ROS2](https://github.com/Seeed-Projects/reBotArmController_ROS2) のものです。このリポジトリには含みません。
