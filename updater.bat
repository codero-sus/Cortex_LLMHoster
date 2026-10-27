@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem Clone and fast-forward only the fixed Cortex LLMHoster update branch.
set "UPDATE_BRANCH=arena/b3072048-cortex-llmhoster"
set "REPOSITORY_URL=https://github.com/codero-sus/Cortex_LLMHoster.git"
set "PROJECT_ROOT="

for /f "delims=" %%R in ('git -C "%~dp0." rev-parse --show-toplevel 2^>nul') do set "PROJECT_ROOT=%%R"
if not defined PROJECT_ROOT (
    echo Error: run this updater from a Git checkout of Cortex LLMHoster.
    exit /b 1
)
pushd "%PROJECT_ROOT%" || exit /b 1

set "CURRENT_BRANCH="
for /f "delims=" %%B in ('git branch --show-current 2^>nul') do set "CURRENT_BRANCH=%%B"
if not "%CURRENT_BRANCH%"=="%UPDATE_BRANCH%" (
    echo Error: expected branch "%UPDATE_BRANCH%"; currently on "%CURRENT_BRANCH%".
    echo The updater will not switch to or update main.
    popd
    exit /b 1
)

rem Keep every repository JSON file before source files are fast-forwarded.
set "BACKUP_ROOT=%PROJECT_ROOT%\backups\json-%RANDOM%-%RANDOM%-%RANDOM%"
if exist "%BACKUP_ROOT%" set "BACKUP_ROOT=%BACKUP_ROOT%-%RANDOM%"
mkdir "%BACKUP_ROOT%" 2>nul
if errorlevel 1 (
    echo Error: could not create JSON backup directory "%BACKUP_ROOT%".
    popd
    exit /b 1
)
robocopy "%PROJECT_ROOT%" "%BACKUP_ROOT%" *.json /S /COPY:DAT /R:1 /W:1 /XD "%PROJECT_ROOT%\.git" "%PROJECT_ROOT%\.venv" "%PROJECT_ROOT%\venv" "%PROJECT_ROOT%\backups" "%PROJECT_ROOT%\node_modules" "%PROJECT_ROOT%\build" "%PROJECT_ROOT%\dist" "%PROJECT_ROOT%\.cache" "%PROJECT_ROOT%\.pytest_cache" "%PROJECT_ROOT%\.mypy_cache" "%PROJECT_ROOT%\.ruff_cache" "%PROJECT_ROOT%\.tox" "%PROJECT_ROOT%\.nox" "%PROJECT_ROOT%\__pycache__" "%PROJECT_ROOT%\coverage" "%PROJECT_ROOT%\out" "%PROJECT_ROOT%\target" "%PROJECT_ROOT%\.next" "%PROJECT_ROOT%\.nuxt" "%PROJECT_ROOT%\.svelte-kit" "%PROJECT_ROOT%\.turbo" "%PROJECT_ROOT%\.vite" "%PROJECT_ROOT%\.parcel-cache" >nul
set "ROBOCOPY_CODE=%ERRORLEVEL%"
if %ROBOCOPY_CODE% GEQ 8 (
    echo Error: JSON backup failed with robocopy exit code %ROBOCOPY_CODE%.
    popd
    exit /b 1
)
echo JSON backup created at: %BACKUP_ROOT%

set "CLONE_ROOT=%TEMP%\Cortex_LLMHoster-update-%RANDOM%-%RANDOM%-%RANDOM%"
if exist "%CLONE_ROOT%" set "CLONE_ROOT=%CLONE_ROOT%-%RANDOM%"
echo Cloning branch %UPDATE_BRANCH% from %REPOSITORY_URL% ...
git clone --branch "%UPDATE_BRANCH%" "%REPOSITORY_URL%" "%CLONE_ROOT%"
if errorlevel 1 goto :clone_failed

set "CLONED_BRANCH="
for /f "delims=" %%B in ('git -C "%CLONE_ROOT%" branch --show-current 2^>nul') do set "CLONED_BRANCH=%%B"
if not "%CLONED_BRANCH%"=="%UPDATE_BRANCH%" (
    echo Error: Git cloned unexpected branch "%CLONED_BRANCH%". No checkout files were changed.
    goto :update_failed
)

echo Fast-forwarding the existing checkout from the cloned branch...
git fetch --no-tags "%CLONE_ROOT%" "refs/heads/%UPDATE_BRANCH%:refs/remotes/origin/%UPDATE_BRANCH%"
if errorlevel 1 goto :update_failed
git merge --ff-only "refs/remotes/origin/%UPDATE_BRANCH%"
if errorlevel 1 goto :update_failed

rmdir /s /q "%CLONE_ROOT%" 2>nul
echo Cortex LLMHoster source updated from %UPDATE_BRANCH%.
echo This updater uses Git only; it does not require Python, pip, or a virtual environment.
echo Restart Cortex LLMHoster to load the updated source.
echo JSON backup retained at: %BACKUP_ROOT%
popd
exit /b 0

:clone_failed
echo Error: could not clone branch "%UPDATE_BRANCH%". No checkout files were changed.
echo JSON backup retained at: %BACKUP_ROOT%
if exist "%CLONE_ROOT%" rmdir /s /q "%CLONE_ROOT%" 2>nul
popd
exit /b 1

:update_failed
echo Error: fast-forward update failed. Local changes are preserved; resolve them and rerun.
echo JSON backup retained at: %BACKUP_ROOT%
if exist "%CLONE_ROOT%" rmdir /s /q "%CLONE_ROOT%" 2>nul
popd
exit /b 1
