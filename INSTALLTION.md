# Cortex LLMHoster installation guide

This guide covers source/ZIP installation on Linux, macOS, Windows Command
Prompt, and Windows PowerShell, plus an optional `pipx` install. Cortex runs
inference locally: it does **not** require Ollama, a cloud inference account, or
a cloud API key. Read the [license](LICENSE) first: the project permits
unmodified personal, non-commercial use only.

## What you need

- Python 3.11 or newer.
- A local runtime and its model weights. The built-in adapter uses a local
  `llama-server` executable and GGUF weights. Other runtimes can be launched
  through the local `command` adapter.
- Disk space for Python dependencies, runtime files, model weights, and temporary
  media uploads. Model weights and runtimes are not included or auto-downloaded
  by Cortex; check their individual license terms.

The initial Python package installation needs access to PyPI unless you have
already cached the dependencies. **Inference itself stays on your computer**
and does not call a Cortex cloud service. An optional `CORTEX_API_KEY` is a
secret you create for protecting your local server/dashboard; it is not a cloud
provider credential and is not forwarded to the model process.

## 1. Get the Cortex files

You can clone the repository:

```text
git clone https://github.com/codero-sus/Cortex_LLMHoster.git
cd Cortex_LLMHoster
```

Or download the repository ZIP from GitHub, extract it, and open a terminal in
the extracted `Cortex_LLMHoster` folder. Do not edit or redistribute the
software; see [`LICENSE`](LICENSE).

## 2. Install Python dependencies

Use one of the platform-specific commands below from the project folder. The
virtual environment keeps Cortex's Python packages separate from system Python.

### Linux / macOS (Terminal)

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
cp cortex.example.toml cortex.toml
```

If `python3.11` is not installed, install Python 3.11 or newer from your OS
package manager or the official Python distribution, then rerun these commands
with that interpreter.

### Windows Command Prompt (`cmd.exe`)

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install .
copy cortex.example.toml cortex.toml
```

Keep the same Command Prompt open after activation so the `.venv` remains active.
If `py -3.11` is not found, install Python 3.11 or newer and enable the Python
Launcher during setup.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
Copy-Item cortex.example.toml cortex.toml
```

If PowerShell blocks activation scripts, either use Command Prompt or allow the
activation script for the current PowerShell process only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Optional: install the CLI with `pipx`

If you already use `pipx`, you can install Cortex in its own managed
virtual environment from the project folder:

```bash
pipx install .
```

Run the CLI from a folder containing `cortex.toml`, or pass its absolute path
with `--config`. If `pipx` is not on your `PATH`, run `pipx ensurepath` and open
a new terminal. You still need to install the local model runtime and weights
separately.

### Optional: select a portable or embedded Python

This setting is cross-platform. Create a `python.env` file in the project root
with one interpreter path (an absolute path, or one relative to the project
root). For example, on Linux/macOS:

```bash
printf '%s\n' '/opt/portable-python/bin/python3' > python.env
/opt/portable-python/bin/python3 -m pip install .
python -m cortex_llmhoster
```

On Windows, the file can contain a path such as
`C:\portable-python\python.exe`; install Cortex into that interpreter before
starting it through the project CLI:

```bat
> python.env echo C:\portable-python\python.exe
C:\portable-python\python.exe -m pip install .
python -m cortex_llmhoster
```

Alternatively, set the environment variable `2PY2` to the interpreter path. It
is an alternate-Python setting, **not Python 2**. On Linux/macOS, pass it for a
single command like this (shell variable names starting with a digit cannot be
exported normally):

```bash
env '2PY2=/opt/portable-python/bin/python3' python -m cortex_llmhoster
```

On Windows Command Prompt use `set "2PY2=C:\portable-python\python.exe"`;
in PowerShell use `Set-Item -Path Env:2PY2 -Value 'C:\portable-python\python.exe'`.
`2PY2` overrides `python.env`; without either setting, Cortex uses the Python
that launched it. Cortex re-executes its CLI under the selected interpreter and
uses it for updater pip installs and the `{python}` local-runtime command
placeholder. Install Cortex and its dependencies into the selected interpreter
first. Paths with spaces may be quoted in `python.env`.

## 3. Install a local runtime and model

### Built-in llama.cpp runtime

Install a CPU-only or hardware-accelerated llama.cpp build that includes
`llama-server`. Choose a build matching your OS and hardware, and verify it:

```text
llama-server --help
```

Keep `llama-server` on `PATH`, or set `llama_server_path` under `[server]` in
`cortex.toml`. You can also set `CORTEX_LLAMA_SERVER` to its full executable
path. On Windows, use a path such as
`C:/tools/llama.cpp/llama-server.exe` in TOML.

Obtain compatible GGUF model weights from their publisher and save them on the
same machine. Cortex does not ship or download weights. Confirm that the
model's own license permits your intended non-commercial use.

### Another local backend

Install the local engine and weights according to that engine's instructions.
Configure one model with `runtime = "command"` and an argv array that launches
its local OpenAI-compatible server. Cortex supplies `{host}` and `{port}` and
requires the command to bind to `127.0.0.1`; no remote inference URL is accepted.
The health endpoint configured by `health_path` must return HTTP 2xx.

## 4. Configure Cortex

Open `cortex.toml` in a text editor. Replace the example model path with the
absolute path to a real local GGUF file. The default listener is
`127.0.0.1:8624`, which is accessible from the same computer only.

Example settings:

```toml
[server]
host = "127.0.0.1"
port = 8624
workers = 1
default_model = "qwen-local"

[[models]]
id = "qwen-local"
runtime = "llama.cpp"
model_path = "/absolute/path/to/your-model.gguf" # replace this
capabilities = ["text_generation"]
default = true
gpu_layers = 0       # 0 = CPU; -1 = maximum GPU offload if supported
threads = 0          # 0 = automatic
context_size = 2048
batch_size = 256
ubatch_size = 64
parallel = 1
optimization_level = -1 # -1 Efficient, 0 Balanced, 1 ULTRA
```

On Windows, forward slashes are accepted for paths, for example:
`model_path = "C:/models/your-model.gguf"`.

**Efficiency starting points:**

- CPU-only: `gpu_layers = 0`, `parallel = 1`, and a moderate context size.
- GPU: try `gpu_layers = -1` only if the selected build supports your GPU and it
  has enough VRAM; lower the layer count if the model does not start.
- Low memory: choose optimization level `-1` (Efficient), reduce
  `context_size`, `batch_size`, and `ubatch_size`, and stop other models.
- ULTRA (`optimization_level = 1`) can use more scratch memory; it is not a
  universal speed switch. See the [tuning guide](README.md#cpu-and-gpu-tuning).

## 5. Start Cortex

For dashboard model management, set a local bearer key. This key is generated by
you and protects this Cortex instance only; **it is not a cloud API key**.

### Linux / macOS

```bash
export CORTEX_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m cortex_llmhoster
```

### Windows PowerShell

```powershell
$env:CORTEX_API_KEY = python -c "import secrets; print(secrets.token_urlsafe(32))"
python -m cortex_llmhoster
```

### Windows Command Prompt

In an interactive `cmd.exe` window, generate a key and save it in that window's
environment:

```bat
for /f %i in ('py -c "import secrets; print(secrets.token_urlsafe(32))"') do set CORTEX_API_KEY=%i
python -m cortex_llmhoster
```

If you put the command in a `.bat` file, write `%%i` instead of `%i` in the
`for` loop. To keep the key across restarts, set it in your user environment or
generate a new one when you launch Cortex.

Open <http://127.0.0.1:8624/> and enter the **same key value** in the dashboard
key prompt (do not type `Bearer ` in the prompt). The key is stored in that
browser's local storage. The configured default model starts automatically; use
**Models** to add, check, start, stop, or edit other local models.

Without `CORTEX_API_KEY`, inference API authentication is disabled, but the
admin/dashboard management APIs stay locked. This is a local access-control
choice; no cloud credentials are involved.

## 6. Verify installation

Check that Cortex is responding:

```bash
curl http://127.0.0.1:8624/health
```

Expected result:

```json
{"status":"ok"}
```

With a model configured and the local bearer key enabled, send a chat request:

```bash
curl http://127.0.0.1:8624/v1/chat/completions \
  -H "Authorization: Bearer $CORTEX_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-local","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

For Windows Command Prompt, use `curl.exe` (not the PowerShell `curl` alias).
Writing the JSON to a small file avoids command-line quoting issues:

```bat
echo {"model":"qwen-local","messages":[{"role":"user","content":"Hello"}],"stream":false}>request.json
curl.exe http://127.0.0.1:8624/v1/chat/completions ^
  -H "Authorization: Bearer %CORTEX_API_KEY%" ^
  -H "Content-Type: application/json" ^
  -d @request.json
```

Replace `qwen-local` with the public model ID in `cortex.toml`. To use an
OpenAI-compatible client library, set its `base_url` to
`http://127.0.0.1:8624/v1`; its `api_key` parameter is either the local
`CORTEX_API_KEY` or any placeholder when server authentication is disabled.

## Software updater

After launching Cortex, open **Settings → Software updates** and choose **Check
for updates**. The check is manual, sends only version/commit metadata to GitHub,
and does not run during startup. The update channel currently follows
`arena/b3072048-cortex-llmhoster`; the updater installs only after you click
**Install update** and confirm. Installation is supported only with one Cortex
worker to prevent simultaneous pip operations. The project version must be
incremented for a new branch update to be reported.

The updater installs the selected immutable commit using the same Python
interpreter that runs Cortex. It may need access to GitHub and PyPI and write
access to the active Python environment. After success, stop and restart Cortex
to load the updated code. Your model weights and TOML/UI configuration are not
replaced. Do not close the process while pip is installing.

## Network access and safety

Cortex binds to loopback by default and the managed model subprocesses bind to
loopback. Do not change the listen host to `0.0.0.0` unless you intentionally
need LAN/container access. If you expose the listener, configure the local
bearer key and firewall rules first. Cortex itself does not send prompts, media,
or the bearer key to a cloud inference service. Update checks are a separate,
user-triggered request to GitHub for project version metadata.

## More help

- [Troubleshooting common installation and runtime issues](TROUBLESHOOTING.md)
- [Runtime, API, capability, and optimization reference](README.md)
- [License and permitted use](LICENSE)
