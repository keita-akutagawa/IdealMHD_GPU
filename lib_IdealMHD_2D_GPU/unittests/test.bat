@echo off
setlocal enabledelayedexpansion

REM オプションの設定
set options=-rdc=true
set programfile=..\*.cu
set constfile=test_const.cu

REM テストファイルを取得
for /f "delims=" %%i in ('dir /b /s test_*.cu') do (
    set "testfile=%%i"

    REM ベース名の取得
    for %%j in ("%%~nxi") do (
        set "basename=%%~nxj"
    )

    REM constfile と一致するか確認
    if /i not "!basename!" == "%constfile%" (
        nvcc %options% %constfile% %programfile% !testfile!
        if errorlevel 1 (
            echo nvcc failed for !testfile!
            exit /b 1
        )
        a.exe
        if errorlevel 1 (
            echo a.exe failed for !testfile!
            exit /b 1
        )
    )
)

endlocal
