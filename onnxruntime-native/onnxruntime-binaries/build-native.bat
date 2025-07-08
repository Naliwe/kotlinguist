@echo off
setlocal
set BUILD_DIR=%~dp0build

if not exist onnxruntime (
    git clone --recursive https://github.com/microsoft/onnxruntime
)

cd onnxruntime

call build.bat ^
  --config Release ^
  --build_shared_lib ^
  --use_cuda ^
  --skip_tests

copy build\Windows\Release\onnxruntime.dll %BUILD_DIR%\libonnxruntime-windows-x86_64.dll
endlocal
