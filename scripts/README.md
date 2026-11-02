# scripts の案内

ファイルが多く見えますが、入口は次の5本です。残りはそこから読み込まれる部品です。
部品どうしが同じフォルダを前提に読み込み合っているので、1つのフォルダにまとめています。

## 入口

| ファイル | 内容 |
|---|---|
| `tidy_desk.py` | 片付けの本体。実機とシミュレーションの両方で動く |
| `rehearse.py` | 写真から再現して練習し、実機と往復する |
| `object_study.py` | 物を1個ずつ置いて難しさを測る |
| `battle.py` | Claude が作る机でのシミュレーション |
| `mj_campaign.py`、`rehearse_eval.py` | 成績の測定と、練習の有無の比較 |

## 部品

| 役割 | ファイル |
|---|---|
| 判断 | `desk_judge.py`（Claude による判定）、`desk_memory.py`（経験ノート）、`vision_seg.py`（SAM）、`confirm_server.py`（人に聞く Web ページ） |
| 写真からの再現 | `summon.py`、`summon_worker.py` |
| シミュレーション | `mj_arm.py`、`mj_scene.py`、`mj_objects.py`、`sim_arm.py`、`sim_world.py`、`sim_scenarios.py`、`sim_campaign.py` |
| 掴む計画 | `plan_pen_pick.py`、`pick_pen_auto.py`、`topdown_ik.py`、`reach_map.py`、`box_fetch.py`、`handeye_solve.py` |
| 実機（ROS 2） | `arm_goto.py`、`joint1_range_trial.py`、`joint1_moveit.py` |
| 実機の確認 | `check_can_stability_readonly.py`、`probe_can_readonly.py`、`real_ready.sh` |
| 環境 | `ros2_env.sh`、`build_motorbridge_local.sh` |

`joint1_*` や `pick_pen_auto` は最初の実験の名前が残ったもので、今も実機の動作で使っています。

## フォルダ

| フォルダ | 内容 |
|---|---|
| `config/` | ドライバーの設定と箱の位置 |
| `calib/` | カメラの校正値 |
| `launch/` | MoveIt の起動 |
| `mj_assets/` | シミュレーションの物と机 |
| `tests/` | テスト |
