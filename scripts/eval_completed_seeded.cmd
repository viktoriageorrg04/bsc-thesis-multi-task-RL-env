@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM cross-eval completed seeded single-task checkpoints
REM usage:
REM   scripts\eval_completed_seeded.cmd [num_envs] [num_episodes] [output_dir]
REM
REM recommended credible run:
REM   scripts\eval_completed_seeded.cmd 256 1024 results_seeded_1024

set "NUM_ENVS=%~1"
if "%NUM_ENVS%"=="" set "NUM_ENVS=256"

set "NUM_EPISODES=%~2"
if "%NUM_EPISODES%"=="" set "NUM_EPISODES=1024"

set "OUTPUT_DIR=%~3"
if "%OUTPUT_DIR%"=="" set "OUTPUT_DIR=results_seeded_1024"

set "ISAACLAB_BAT=%ISAACLAB_BAT%"
if "%ISAACLAB_BAT%"=="" set "ISAACLAB_BAT=C:\Users\pavel\OneDrive\Desktop\IsaacLab\isaaclab.bat"

set "ROOT=%CD%"

echo [INFO] NUM_ENVS=%NUM_ENVS%
echo [INFO] NUM_EPISODES=%NUM_EPISODES%
echo [INFO] OUTPUT_DIR=%OUTPUT_DIR%
echo [INFO] ISAACLAB_BAT=%ISAACLAB_BAT%

call :eval_policy A1_forward_s0 "logs\rsl_rl\unitree_go2_a1_legacy_1024_seeds\2026-05-20_20-33-57_A1_forward_legacy_1024_s0\model_1499.pt"
call :eval_policy A1_forward_s1 "logs\rsl_rl\unitree_go2_a1_legacy_1024_seeds\2026-05-20_20-35-33_A1_forward_legacy_1024_s1\model_1499.pt"
call :eval_policy A1_forward_s2 "logs\rsl_rl\unitree_go2_a1_legacy_1024_seeds\2026-05-20_20-35-33_A1_forward_legacy_1024_s2\model_1499.pt"

call :eval_policy A2_omni_s0 "logs\rsl_rl\unitree_go2_a2_baseline_1024_seeds\2026-05-20_23-25-26_A2_omni_1024_s0\model_1499.pt"
call :eval_policy A2_omni_s1 "logs\rsl_rl\unitree_go2_a2_baseline_1024_seeds\2026-05-21_00-42-03_A2_omni_1024_s1\model_1499.pt"
call :eval_policy A2_omni_s2 "logs\rsl_rl\unitree_go2_a2_baseline_1024_seeds\2026-05-21_00-47-45_A2_omni_1024_s2\model_1499.pt"

call :eval_policy B1_rough_s0 "logs\rsl_rl\unitree_go2_b1_baseline_1024_seeds\2026-05-21_00-47-45_B1_rough_1024_s0\model_1499.pt"
call :eval_policy B1_rough_s1 "logs\rsl_rl\unitree_go2_b1_baseline_1024_seeds\2026-05-21_00-48-44_B1_rough_1024_s1\model_1499.pt"
call :eval_policy B1_rough_s2 "logs\rsl_rl\unitree_go2_b1_baseline_1024_seeds\2026-05-21_00-56-43_B1_rough_1024_s2\model_1499.pt"

call :eval_policy B2_stairs_s0 "logs\rsl_rl\unitree_go2_b2_local_apr15_recheck_1024_seeds\2026-05-22_23-51-11_B2_stairs_local_apr15_recheck_1024_s0\model_1440.pt"
call :eval_policy B2_stairs_s1 "logs\rsl_rl\unitree_go2_b2_local_apr15_recheck_1024_seeds\2026-05-23_09-29-34_B2_stairs_local_apr15_recheck_1024_s1\model_1440.pt"
call :eval_policy B2_stairs_s2 "logs\rsl_rl\unitree_go2_b2_local_apr15_recheck_1024_seeds\2026-05-23_18-57-36_B2_stairs_local_apr15_recheck_1024_s2\model_1440.pt"

call :eval_policy C2_gap_s0 "logs\rsl_rl\unitree_go2_c2_from_a2_refined_1024_seeds\2026-05-21_23-19-13_C2_gap_from_a2_refined_1024_s0\model_2998.pt"
call :eval_policy C2_gap_s1 "logs\rsl_rl\unitree_go2_c2_from_a2_refined_1024_seeds\2026-05-22_01-37-17_C2_gap_from_a2_refined_1024_s1\model_2998.pt"
call :eval_policy C2_gap_s2 "logs\rsl_rl\unitree_go2_c2_from_a2_refined_1024_seeds\2026-05-22_03-20-49_C2_gap_from_a2_refined_1024_s2\model_2998.pt"

echo [DONE] Seeded eval finished.
echo [NEXT] Aggregate:
echo   python scripts\aggregate_seeded_eval.py --results_root %OUTPUT_DIR% --metric success_rate
exit /b 0

:eval_policy
set "TRAIN_LABEL=%~1"
set "CHECKPOINT=%~2"
if not exist "%CHECKPOINT%" (
  echo [ERROR] Missing checkpoint for %TRAIN_LABEL%: %CHECKPOINT%
  exit /b 2
)

call :eval_task "%TRAIN_LABEL%" "%CHECKPOINT%" A1_forward MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0
if errorlevel 1 exit /b %errorlevel%
call :eval_task "%TRAIN_LABEL%" "%CHECKPOINT%" A2_omni MTL-Velocity-Flat-Unitree-Go2-A2-Omni-v0
if errorlevel 1 exit /b %errorlevel%
call :eval_task "%TRAIN_LABEL%" "%CHECKPOINT%" B1_rough MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-v0
if errorlevel 1 exit /b %errorlevel%
call :eval_task "%TRAIN_LABEL%" "%CHECKPOINT%" B2_stairs MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-v0
if errorlevel 1 exit /b %errorlevel%
call :eval_task "%TRAIN_LABEL%" "%CHECKPOINT%" C2_gap MTL-Custom-Gap-Unitree-Go2-C2-v0
if errorlevel 1 exit /b %errorlevel%
exit /b 0

:eval_task
set "TRAIN_LABEL=%~1"
set "CHECKPOINT=%~2"
set "EVAL_SHORT=%~3"
set "EVAL_TASK=%~4"

echo.
echo [EVAL] %TRAIN_LABEL% on %EVAL_SHORT%
call "%ISAACLAB_BAT%" -p scripts\evaluate.py ^
  --task "%EVAL_TASK%" ^
  --checkpoint "%ROOT%\%CHECKPOINT%" ^
  --report_train_task "%TRAIN_LABEL%" ^
  --headless ^
  --num_envs "%NUM_ENVS%" ^
  --num_episodes "%NUM_EPISODES%" ^
  --output_dir "%OUTPUT_DIR%" ^
  --eval_task "%EVAL_SHORT%"
exit /b %errorlevel%
