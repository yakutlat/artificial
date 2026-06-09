@echo off
chcp 65001 > nul
echo ================================================
echo    金融投研 RAG 智能问答系统 - 启动脚本
echo ================================================
echo.

set PYTHON=C:\Users\86136\.workbuddy\binaries\python\envs\default\Scripts\python.exe
set STREAMLIT=C:\Users\86136\.workbuddy\binaries\python\envs\default\Scripts\streamlit.exe
set APP_DIR=C:\Users\86136\WorkBuddy\2026-06-08-21-05-28\finance_rag

echo [1/2] 检查依赖...
%PYTHON% -c "import streamlit, sentence_transformers, faiss, openai; print('所有依赖正常')" 2>nul
if errorlevel 1 (
    echo 正在安装依赖（约1-2分钟）...
    %PYTHON% -m pip install streamlit sentence-transformers faiss-cpu openai -q
)

echo [2/2] 启动 Streamlit 应用...
echo.
echo 应用将在浏览器中自动打开：http://localhost:8501
echo 按 Ctrl+C 停止服务
echo.

cd /d %APP_DIR%
%STREAMLIT% run app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false --server.headless true

pause
