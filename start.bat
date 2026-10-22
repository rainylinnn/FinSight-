@echo off
rem ======================================================================
rem  FinSight 酒研助手 · Windows 一键启动
rem  用法：双击本文件，或在命令行执行  start.bat
rem ======================================================================
setlocal

cd /d "%~dp0"

rem ---- 1. 依赖检查 -----------------------------------------------------
where uv >nul 2>nul
if errorlevel 1 (
    echo [错误] 没有找到 uv，请先安装： https://docs.astral.sh/uv/
    echo         或用 pip install uv
    pause
    exit /b 1
)

rem ---- 2. 环境变量检查 -------------------------------------------------
if "%DEEPSEEK_API_KEY%"=="" (
    if exist ".env" (
        echo [提示] 未检测到系统环境变量 DEEPSEEK_API_KEY，将尝试从 .env 读取
    ) else (
        echo [错误] 缺少 DEEPSEEK_API_KEY
        echo        请设置系统环境变量，或复制 .env.example 为 .env 并填写
        pause
        exit /b 1
    )
)

rem ---- 3. 同步依赖 -----------------------------------------------------
echo [1/2] 同步依赖（uv sync）
uv sync
if errorlevel 1 (
    echo [错误] 依赖同步失败
    pause
    exit /b 1
)

rem ---- 4. 启动服务 -----------------------------------------------------
echo [2/2] 启动服务：http://127.0.0.1:8003
echo        首次启动要加载 Milvus 集合和 BGE-M3 模型，约 30-60 秒属正常
echo.
uv run python app.py

pause
