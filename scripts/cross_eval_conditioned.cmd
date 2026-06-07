@echo off
setlocal

REM cross-eval one conditioned MTL checkpoint on all benchmark tasks
REM usage:
REM   scripts\cross_eval_conditioned.cmd C:\path\to\model.pt [num_episodes] [num_envs] [output_dir]

if "%~1"=="" (
  echo Usage: %~nx0 CHECKPOINT [NUM_EPISODES] [NUM_ENVS]
  exit /b 2
)

set "CHECKPOINT=%~1"
set "NUM_EPISODES=%~2"
set "NUM_ENVS=%~3"
set "OUTPUT_DIR=%~4"
if "%NUM_EPISODES%"=="" set "NUM_EPISODES=128"
if "%NUM_ENVS%"=="" set "NUM_ENVS=32"
if "%OUTPUT_DIR%"=="" set "OUTPUT_DIR=results"

set "ISAACLAB_BAT=C:\Users\pavel\OneDrive\Desktop\IsaacLab\isaaclab.bat"
set "TRAIN_TASK=MTL-Conditioned-Unitree-Go2-AllTerrains-v0"

echo [INFO] Checkpoint: %CHECKPOINT%
echo [INFO] Episodes: %NUM_EPISODES%
echo [INFO] Num envs: %NUM_ENVS%
echo [INFO] Output dir: %OUTPUT_DIR%

call "%ISAACLAB_BAT%" -p scripts\evaluate.py --task MTL-Velocity-Flat-Unitree-Go2-A1-Forward-v0 --checkpoint "%CHECKPOINT%" --report_train_task %TRAIN_TASK% --headless --num_envs %NUM_ENVS% --num_episodes %NUM_EPISODES% --output_dir "%OUTPUT_DIR%" --eval_task A1_forward
if errorlevel 1 exit /b %errorlevel%

call "%ISAACLAB_BAT%" -p scripts\evaluate.py --task MTL-Velocity-Flat-Unitree-Go2-A2-Omni-v0 --checkpoint "%CHECKPOINT%" --report_train_task %TRAIN_TASK% --headless --num_envs %NUM_ENVS% --num_episodes %NUM_EPISODES% --output_dir "%OUTPUT_DIR%" --eval_task A2_omni
if errorlevel 1 exit /b %errorlevel%

call "%ISAACLAB_BAT%" -p scripts\evaluate.py --task MTL-Velocity-Rough-Unitree-Go2-B1-RoughWalk-v0 --checkpoint "%CHECKPOINT%" --report_train_task %TRAIN_TASK% --headless --num_envs %NUM_ENVS% --num_episodes %NUM_EPISODES% --output_dir "%OUTPUT_DIR%" --eval_task B1_rough
if errorlevel 1 exit /b %errorlevel%

call "%ISAACLAB_BAT%" -p scripts\evaluate.py --task MTL-Velocity-Rough-Unitree-Go2-B2-StairClimb-v0 --checkpoint "%CHECKPOINT%" --report_train_task %TRAIN_TASK% --headless --num_envs %NUM_ENVS% --num_episodes %NUM_EPISODES% --output_dir "%OUTPUT_DIR%" --eval_task B2_stairs
if errorlevel 1 exit /b %errorlevel%

call "%ISAACLAB_BAT%" -p scripts\evaluate.py --task MTL-Custom-Gap-Unitree-Go2-C2-v0 --checkpoint "%CHECKPOINT%" --report_train_task %TRAIN_TASK% --headless --num_envs %NUM_ENVS% --num_episodes %NUM_EPISODES% --output_dir "%OUTPUT_DIR%" --eval_task C2_gap
if errorlevel 1 exit /b %errorlevel%

echo [DONE] Cross-eval complete.
