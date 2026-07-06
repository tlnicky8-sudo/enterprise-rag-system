@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 企业知识库智能问答系统

echo.
echo ============================================
echo   企业知识库智能问答系统 ^(Web 版^)
echo ============================================
echo.

REM --- 第一步：数据预处理 ---
echo [1/2] 检查数据并预处理...
.venv\Scripts\python setup_data.py
if %errorlevel% neq 0 (
    echo.
    echo 数据预处理失败，请检查 Milvus 是否已启动
    pause
    exit /b 1
)

REM --- 第二步：启动 Web 服务 ---
echo.
echo [2/2] 启动 Web 服务...
echo.
echo   浏览器访问: http://127.0.0.1:5000
echo   按 Ctrl+C 停止服务
echo.

start "" http://127.0.0.1:5000

.venv\Scripts\python web_app.py

pause
