# Cortex LLMHoster troubleshooting

Cortex manages local inference runtimes. It does not require Ollama or a cloud
provider API key. A configured `CORTEX_API_KEY` is only a bearer secret for your
local Cortex server. For platform-specific setup, start with the
[installation guide](INSTALLTION.md).

## Find the useful error first

1. Open **Models** in the dashboard and inspect the affected model's status and
   recent runtime output. Use **Check setup** to validate its configuration.
2. Check the terminal where `python -m cortex_llmhoster` is running.
3. Check the configured model ID, executable path, local model file, and runtime
   health path. Only the configured default model starts automatically.

Runtime output is captured by Cortex and can include filesystem paths or
backend diagnostics; review it before sharing logs publicly.

## Installation and Python errors

### `python` or `cortex-llmhoster` is not found

- Confirm Python 3.11 or newer is installed (`python --version`, `py -3.11
  --version` on Windows).
- Activate the project's `.venv` in the same terminal where you installed the
  package.
- From the project folder, run `python -m pip install .` again.
- As an alternative to the console command, run `python -m cortex_llmhoster`.

### `externally-managed-environment` during `pip install`

This is a system-Python protection (PEP 668). Do not force installation into
system Python. Create and activate a virtual environment, then install inside
it:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

On Windows Command Prompt, activate with `.venv\Scripts\activate.bat`; in
PowerShell use `.\.venv\Scripts\Activate.ps1`.

### Dependency download or install fails

The first installation needs PyPI access for Cortex's Python dependencies. It
does not download a model or contact a cloud inference API. Check that your
network or package index permits PyPI access, and use a supported Python version.
The OS/model runtime and model weights are installed separately.

## Model does not start

### `llama-server was not found`

- Run `llama-server --help` (Windows: `llama-server.exe --help`) in the same
  environment that starts Cortex.
- Add the executable to `PATH`, or set `llama_server_path` in `[server]` in
  `cortex.toml`; `CORTEX_LLAMA_SERVER` can also provide the path.
- Confirm that the executable is for your OS and that the GPU build, if used,
  supports your installed driver/device. CPU-only llama.cpp builds are valid.

### `GGUF model file not found`

- Check that `model_path` points to an existing `.gguf` file, not a directory,
  download page, or placeholder from `cortex.example.toml`.
- Use an absolute path. On Windows, TOML paths such as
  `C:/models/model.gguf` avoid backslash escaping issues.
- Cortex does not download weights. Obtain the file from its publisher and check
  that model's license separately.

### A `command` runtime exits or never becomes ready

- `runtime_command` must be a JSON/TOML argv array, not one shell command
  string. It must include `{host}` and `{port}` so it binds to Cortex's local
  loopback port.
- Ensure the executable is installed and resolvable from `PATH` (or use its
  full path) and that placeholders are spelled correctly.
- The configured `health_path` must return HTTP 2xx once the runtime is ready.
  If its API is not under `/v1`, set `api_base_path` to the backend's actual
  local API prefix.
- The runtime must implement the route and payload format you are sending;
  Cortex forwards requests but does not translate every backend dialect.

## Memory, GPU, and performance problems

### The runtime reports an out-of-memory error or exits during startup

1. Stop other hosted models; only one default model starts automatically, but
   manually started models also use memory.
2. For llama.cpp, set `gpu_layers = 0` to use CPU, or lower a positive
   `gpu_layers` count if VRAM is the limit. `-1` requests maximum available
   offload; it does not guarantee that every layer fits.
3. Reduce `context_size` first, then reduce `batch_size` and `ubatch_size`.
4. Keep `parallel = 1` on low-memory hosts and choose optimization level `-1`
   (Efficient). ULTRA (`1`) increases some batch/cache settings and can require
   more RAM/VRAM.
5. Ensure the model quantization and the model's required memory fit your
   hardware. Cortex cannot reduce the memory footprint of incompatible weights.

For other local engines, use their documented CPU/GPU and memory parameters in
`runtime_command`; the llama.cpp tuning fields are not automatically applied to
them.

### GPU is not detected or offload fails

Cortex's hardware panel is best-effort. GPU support depends on the selected
runtime build, matching drivers, and OS. Confirm the backend's own GPU test or
startup output first. If it fails, use `gpu_layers = 0` for llama.cpp or the
backend's CPU mode. A GPU displayed by the OS does not guarantee the installed
runtime can use it.

### Output is slow

Performance depends on hardware, weights, quantization, prompt/context length,
and backend configuration. Try a smaller context, fewer active model processes,
`parallel = 1`, and a suitable quantized model. Compare configurations on your
own machine; Cortex does not promise a specific tokens-per-second rate.

## Dashboard, authentication, and connection errors

### Dashboard says admin connection is needed or returns 401

- `CORTEX_API_KEY` must be set in the environment of the running Cortex process
  before startup if you want dashboard/admin management.
- Enter the exact secret value into the dashboard key prompt; do not include the
  `Bearer ` prefix there. The browser stores it in local storage.
- If the key changed, restart Cortex and re-enter the new value in the browser.
- This is local access control, not a cloud credential. If the key is unset,
  inference routes are unauthenticated but admin/config/model-management routes
  stay locked.

### `Address already in use` or the browser cannot connect

- Confirm the configured port (default `8624`) is not occupied. Try a different
  port with `server.port`, `CORTEX_PORT`, or `--port`.
- With the default `127.0.0.1` listener, open Cortex from the same computer at
  `http://127.0.0.1:<port>/`.
- For a container or intentional LAN access, configure the appropriate listen
  host (often `0.0.0.0`) and port mapping/firewall. Set a local bearer key before
  exposing it; do not expose an unauthenticated instance to an untrusted network.
- Use `GET /health` to check Cortex liveness. `GET /ready` requires a configured
  and running local model.

### Dashboard settings do not save

Check the server response, `cortex.local.json` write permissions, and the
terminal logs. Environment variables override corresponding values in the UI.
The dashboard writes its overlay next to `cortex.toml` by default; set
`CORTEX_ADMIN_CONFIG` if that location is read-only or unsuitable.

## API and model errors

### `model_not_found`

Use the public model `id` from the **Models** page or `GET /v1/models`; do not
use a local filename or backend's private name unless that is the configured
public ID. If Cortex authentication is enabled, include the local bearer key.

### `unsupported_model_capability` (HTTP 422)

The selected model has not declared the task required by the endpoint. Add the
correct capability only if that model/backend really supports it, or select a
model configured for that task. Capability labels do not implement missing
model functionality.

### Backend returns 404 or an unsupported-route error

The request reached the local process, but that runtime may not implement the
path. Confirm `api_base_path`, endpoint spelling, and the backend's OpenAI-
compatible API. Some local servers implement only a subset of OpenAI endpoints.
Unknown backend-defined routes are forwarded, but Cortex does not invent them.

### `local_model_not_ready` (HTTP 503)

Start the model in the dashboard and wait for its health check to pass. Check
its runtime error/log output. If no default is selected, provide a `model` in
the request or configure a default model. Only the default is started at
application startup.

### `request_too_large` (HTTP 413)

The JSON or media request exceeded its configured bound. Increase
`max_json_body_bytes` or `max_media_body_bytes` only as needed. Multipart media
uploads use temporary disk storage; ensure the temp volume has sufficient free
space as well as raising the size limit.

## Still stuck?

Record the OS, Python version, Cortex version/commit, selected runtime, endpoint,
HTTP status, and relevant sanitized runtime output. Do not post API keys, model
weights, private prompts, or personal media. Check that the model and runtime
licenses permit your intended use before sharing or deploying them.
