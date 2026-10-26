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

### Cortex cannot use the selected Python interpreter

- `2PY2` takes precedence over the optional root-level `python.env` file. Clear
  the environment variable or correct the path if it points to an old interpreter.
- `python.env` should contain exactly one Python executable path; relative paths
  are resolved from the project root. Confirm that the file exists and that the
  interpreter is executable (Linux/macOS).
- Install Cortex and its dependencies into the selected interpreter before
  starting the CLI; Cortex re-executes under that interpreter when needed.
- On Linux/macOS, set the digit-leading variable for a single launch with
  `env '2PY2=/path/to/python3' python -m cortex_llmhoster` rather than `export`.

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

### Cortex native C++ host or source conversion is unavailable

- Build the host from the repository with CMake, Git, and a C++17 compiler:
  `cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON`
  followed by `cmake --build native/build --target cortex-gguf-host --parallel 2`.
  The default executable path is `native/build/bin/cortex-gguf-host`.
- If the executable is elsewhere, put it on `PATH` or set `CORTEX_CPP_HOST` to
  its full path. It must be executable by the Cortex service account.
- Direct GGUF needs no converter. For HF SafeTensors or `pytorch_model.bin`,
  point `model_path` at a supported model directory with `config.json`; build
  the native target so its pinned converter scripts are available and install
  the upstream converter's optional dependencies in the interpreter selected by
  Cortex. Override it with `CORTEX_CPP_CONVERTER_PYTHON`; set
  `CORTEX_LLAMA_CPP_SOURCE_DIR` if the source tree is elsewhere.
- Legacy `.pt`, `.pth`, `.bin`, or `.ckpt` pickle checkpoints may execute code
  during conversion. Only enable `allow_unsafe_legacy_checkpoint_conversion`
  for a trusted legacy LLaMA checkpoint. Arbitrary training `.ckpt` files are
  not universally supported; export them to a supported HF directory or GGUF.
- Converted files use `CORTEX_CPP_CACHE_DIR` or the user's cache directory by
  default. Ensure enough free disk space for the source and converted model.
- The adapter currently supports text generation and embeddings, not
  projector-based vision or arbitrary task endpoints. Use a separate build
  directory for each target GPU; CUDA, Metal, or Vulkan needs the matching SDK
  and llama.cpp options. See `native/README.md`; `gpu_layers = 0` selects CPU.

### Legacy `runtime = "cortex"` Python backend is unavailable

- Existing configs use `llama-cpp-python` inside a separate supervised process.
  Install `cortex-llmhoster[embedded]` in the selected interpreter only if you
  need to keep that legacy runtime. New configurations should use `cortex-cpp`.

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
4. Keep `parallel = 1` on low-memory hosts. In the Models editor, a higher
   mode such as High+ uses smaller batches and a quantized KV cache to aim for
   lower memory use; quantized caches can affect output quality. ULTRA is
   optional and may use more scratch RAM/VRAM.
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

- Authentication is controlled by `CORTEX_API_AUTH_ENABLED` or
  `[server].api_auth_enabled`. When unset, it defaults on if a non-empty
  `CORTEX_API_KEY` or `CORTEX_ADMIN_USERNAME` is configured and off otherwise.
  When enabled, a non-empty API key is required; admin account variables are
  optional and add username/password dashboard sign-in. `CORTEX_API_KEY` takes
  precedence when non-empty. To use another variable, set `server.api_key_env`
  or `CORTEX_API_KEY_ENV` to a valid environment name; it cannot reuse the
  admin credential or auth-toggle variables.
- For account sign-in, configure both `CORTEX_ADMIN_USERNAME` and
  `CORTEX_ADMIN_PASSWORD` alongside the API key in the environment or ignored
  project-root `.env`. If no account is configured, the login form defaults to
  key mode; enter the exact local key without quotes or the `Bearer ` prefix.
- To run without a key and without authentication, remove/unset
  `CORTEX_API_KEY`, then launch with `python -m cortex_llmhoster --no-auth`
  (`--no-api-auth` remains supported) or set `CORTEX_API_AUTH_ENABLED=false`.
  This opens dashboard and inference APIs, so only use it on a trusted local
  listener.
- Successful sign-in creates a signed 30-day HttpOnly cookie; passwords and API
  keys are not stored in browser storage or the cookie. API clients can continue
  sending `Authorization: Bearer <key>`. Sign-in is limited to 10 attempts per
  client per minute; honor the `Retry-After` header before trying again. Admin
  API responses are sent with `Cache-Control: no-store`.
- If a cookie-authenticated browser mutation returns `csrf_origin_rejected`,
  the request needs same-origin `Origin`, `Referer`, or Fetch Metadata. Configure
  a reverse proxy to preserve the external scheme/host with `X-Forwarded-Proto`
  and `X-Forwarded-Host`; non-browser API clients can use a bearer key instead.
- If credentials change, restart Cortex and sign in again. Changing the API key
  or either account credential invalidates old sessions; use **Log out** to remove
  the current browser session.
- If the page still rejects correct credentials, confirm it is the preview for
  the same Uvicorn port serving the backend. The dashboard and login route are
  same-origin; opening a different port may connect to another server instance.
- This is local access control, not a cloud credential. Disabling auth is unsafe
  on a network reachable by untrusted clients.

### `Address already in use` or the browser cannot connect

- Confirm the configured port (default `8624`) is not occupied. Try a different
  port with `server.port`, `CORTEX_PORT`, or `--port`.
- The listen host must be an IP address or DNS hostname such as `127.0.0.1`,
  `::1`, `localhost`, or `0.0.0.0`; do not include a URL scheme or port.
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

Every HTTP response includes an `X-Request-ID`. For local-runtime transport
errors, the server log includes the same ID; include it when correlating a client
failure with terminal output.

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

### `model_request_failed` (HTTP 500)

Cortex hit an unexpected error while preparing the local-runtime request. The
response intentionally omits internal details; correlate `X-Request-ID` with
the terminal traceback and avoid sharing logs that contain private paths or
secrets.

### `model_server_unavailable` (HTTP 502)

Cortex could not reach the local runtime. Check that the runtime is still
running, its configured health path responds, and the selected model's port has
not been taken by another process.

### `request_too_large` (HTTP 413)

The JSON or media request exceeded its configured bound. Increase
`max_json_body_bytes` or `max_media_body_bytes` only as needed. Multipart media
uploads use temporary disk storage; ensure the temp volume has sufficient free
space as well as raising the size limit.

## Updater problems

### No update is available

- Check that **Settings → Software updates** is checking the configured
  `arena/b3072048-cortex-llmhoster` branch.
- The updater compares numeric package versions. The project must increment
  `version` in `pyproject.toml` for a new update to appear; a new commit with the
  same version is not offered as an update.
- Update checks run only when you click the button. They need outbound access to
  the GitHub API; unauthenticated GitHub rate limits may require waiting before
  trying again.

### Update check or installation failed

- The built-in installer requires one active Cortex worker. If Cortex runs with
  multiple workers, stop it and relaunch with `--workers 1` before installing.
- Confirm this machine can reach GitHub. Installing an update also uses pip and
  may need PyPI access if dependencies changed.
- Ensure Cortex was installed in a writable virtual environment. A system
  installation may reject package replacement; reinstall Cortex into a venv
  rather than running pip with elevated privileges.
- The update installs into the Python environment used by the running Cortex
  process. After success, stop and restart Cortex. Do not close the process
  while pip is installing.
- The root-level `updater.sh` / `updater.bat` scripts require Git and network
  access to GitHub, but not Python, pip, or a virtual environment. The checkout
  must already be on `arena/b3072048-cortex-llmhoster`; they never switch to
  `main`. Check `git branch --show-current` before running.
- Each script backs up repository `*.json` files under `backups/` before cloning.
  The scripts use `git clone --branch arena/b3072048-cortex-llmhoster
  https://github.com/codero-sus/Cortex_LLMHoster.git` into a temporary folder,
  then fast-forward the existing checkout from that clone.
- A local edit or divergent commit prevents the fast-forward; the scripts do not
  force-reset. The JSON backup remains available. Resolve or stash local changes
  and rerun the updater.
- The root scripts synchronize source files only; they do not install Python
  package changes. If an update adds a dependency, install it separately into
  the environment Cortex uses, then restart Cortex LLMHoster.
- If the built-in installer cannot complete, stop Cortex LLMHoster, obtain the
  official source archive, activate the same environment, and run `python -m
  pip install --upgrade --force-reinstall .` from the extracted project folder.
  Local weights and configuration are separate files and are not replaced by
  this command.

## Still stuck?

Record the OS, Python version, Cortex version/commit, selected runtime, endpoint,
HTTP status, and relevant sanitized runtime output. Do not post API keys, model
weights, private prompts, or personal media. Check that the model and runtime
licenses permit your intended use before sharing or deploying them.
