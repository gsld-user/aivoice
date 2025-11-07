@echo off
REM 設定 Python 3.10 環境
SET PATH=%PATH%;D:\home\Python310\Scripts;D:\home\Python310

REM 使用 Azure 指定的 PORT 啟動 Flask
IF "%PORT%"=="" SET PORT=8080

REM 啟動 Flask App
python apikey\app.py
