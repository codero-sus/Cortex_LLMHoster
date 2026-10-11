@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Cortex LLMHoster source updater. Updates only the current project branch.
set "UPDATE_BRANCH=arena/b3072048-cortex-llmhoster"
set "PROJECT_ROOT="

for /f "delims=" %%R in ('git -C "%~dp0." rev-parse --show-toplevel 2^>nul') do set "PROJECT_ROOT=%%R"
if not defined PROJECT_ROOT (
    echo Error: this updater must run inside a Git checkout.
    exit /b 1
)
pushd "%PROJECT_ROOT%" || exit /b 1

set "CURRENT_BRANCH="
for /f "delims=" %%B in ('git branch --show-current 2^>nul') do set "CURRENT_BRANCH=%%B"
if not "%CURRENT_BRANCH%"=="%UPDATE_BRANCH%" (
    echo Error: expected branch "%UPDATE_BRANCH%"; currently on "%CURRENT_BRANCH%".
    echo Cortex LLMHoster updater will not switch to or update main.
    popd
    exit /b 1
)
git remote get-url origin >nul 2>nul
if errorlevel 1 (
    echo Error: this checkout has no Git remote named origin.
    popd
    exit /b 1
)

set "STAMP="
for /f "delims=" %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMddTHHmmssZ" 2^>nul') do set "STAMP=%%T"
if not defined STAMP set "STAMP=%RANDOM%-%RANDOM%"
set "BACKUP_ROOT=%PROJECT_ROOT%\backups\json-%STAMP%"
if exist "%BACKUP_ROOT%" set "BACKUP_ROOT=%BACKUP_ROOT%-%RANDOM%"
mkdir "%BACKUP_ROOT%" 2>nul
if errorlevel 1 (
    echo Error: could not create JSON backup directory "%BACKUP_ROOT%".
    popd
    exit /b 1
)

rem Copy every project JSON file while excluding Git, environments, caches, and prior backups.
robocopy "%PROJECT_ROOT%" "%BACKUP_ROOT%" *.json /S /COPY:DAT /R:1 /W:1 /XD "%PROJECT_ROOT%\.git" "%PROJECT_ROOT%\.venv" "%PROJECT_ROOT%\venv" "%PROJECT_ROOT%\backups" "%PROJECT_ROOT%\node_modules" "%PROJECT_ROOT%\build" "%PROJECT_ROOT%\dist" >nul
set "ROBOCOPY_CODE=%ERRORLEVEL%"
if %ROBOCOPY_CODE% GEQ 8 (
    echo Error: JSON backup failed with robocopy exit code %ROBOCOPY_CODE%.
    popd
    exit /b 1
)
echo JSON backup created at: %BACKUP_ROOT%

set "REMOTE_REF=refs/remotes/origin/%UPDATE_BRANCH%"
echo Fetching all tracked project-file updates from origin/%UPDATE_BRANCH% ...
git fetch origin "%UPDATE_BRANCH%:%REMOTE_REF%"
if errorlevel 1 (
    echo Error: could not fetch branch "%UPDATE_BRANCH%". No branch switch was attempted.
    popd
    exit /b 1
)
git merge --ff-only "%REMOTE_REF%"
if errorlevel 1 (
    echo Error: fast-forward update failed. Local changes are preserved; resolve them and rerun.
    popd
    exit /b 1
)
echo Cortex LLMHoster repository files are updated from %UPDATE_BRANCH%.

set "BOOTSTRAP_PYTHON="
for /f "tokens=1,* delims==" %%A in ('set 2PY2 2^>nul') do if /i "%%A"=="2PY2" set "BOOTSTRAP_PYTHON=%%B"
if not defined BOOTSTRAP_PYTHON if exist "%PROJECT_ROOT%\.env" for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%PROJECT_ROOT%\.env") do if /i "%%A"=="2PY2" set "BOOTSTRAP_PYTHON=%%B"
if not defined BOOTSTRAP_PYTHON if exist "%PROJECT_ROOT%\python.env" for /f "usebackq eol=# delims=" %%L in ("%PROJECT_ROOT%\python.env") do if not defined BOOTSTRAP_PYTHON set "BOOTSTRAP_PYTHON=%%L"
for /f "tokens=1,* delims==" %%A in ("%BOOTSTRAP_PYTHON%") do if /i "%%A"=="2PY2" set "BOOTSTRAP_PYTHON=%%B"
set "BOOTSTRAP_PYTHON=%BOOTSTRAP_PYTHON:"=%"
set "BOOTSTRAP_PYTHON=%BOOTSTRAP_PYTHON:'=%"
if not defined BOOTSTRAP_PYTHON if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" set "BOOTSTRAP_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"
if not defined BOOTSTRAP_PYTHON for /f "delims=" %%P in ('where python.exe 2^>nul') do if not defined BOOTSTRAP_PYTHON set "BOOTSTRAP_PYTHON=%%P"
if not defined BOOTSTRAP_PYTHON (
    echo Error: source files are updated, but Python was not found to refresh the install.
    popd
    exit /b 1
)
set "PYTHONPATH=%PROJECT_ROOT%\src;%PYTHONPATH%"
set "SELECTED_PYTHON="
for /f "delims=" %%P in ('"%BOOTSTRAP_PYTHON%" -c "from pathlib import Path; from cortex_llmhoster.python_runtime import resolve_python_executable; print(resolve_python_executable(project_root=Path.cwd()))"') do set "SELECTED_PYTHON=%%P"
if not defined SELECTED_PYTHON (
    echo Error: could not resolve the configured Python interpreter.
    popd
    exit /b 1
)
"%SELECTED_PYTHON%" -m pip install --upgrade --no-input --editable "%PROJECT_ROOT%"
if errorlevel 1 (
    echo Error: source files updated, but package installation failed.
    popd
    exit /b 1
)

echo Cortex LLMHoster update complete. Restart the running server to load the new files.
echo JSON backup retained at: %BACKUP_ROOT%
popd
exit /b 0
