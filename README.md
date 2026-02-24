# nanohub-submit

`nanohub-submit` is a standalone, modern Python package for building and
executing NanoHUB submit requests over a direct socket/TLS connection to the
submit server, without importing any legacy modules.

## Highlights

- Typed request model (`SubmitRequest`)
- Deterministic command argument builder
- Programmatic client API (`NanoHUBSubmitClient`) with direct server protocol
- CLI entrypoint: `nanohub-submit`
- Python compatibility: 3.7+

## Install (editable for development)

```bash
python3 -m pip install -e .
```

Notebook extras (includes `ipywidgets<8`):

```bash
python3 -m pip install -e ".[notebook]"
```

## Python-Only Tests

```bash
python3 -m pytest -q
```

Live integration tests (against a real submit server) are opt-in:

```bash
NANOHUBSUBMIT_LIVE=1 python3 -m pytest -q tests/test_live_client.py
```

Optional non-default config path:

```bash
NANOHUBSUBMIT_LIVE=1 NANOHUBSUBMIT_CONFIG_PATH=/etc/submit/submit-client.conf \
python3 -m pytest -q tests/test_live_client.py
```

Optional remote live sweep test (long-running, disabled by default):

```bash
NANOHUBSUBMIT_LIVE=1 NANOHUBSUBMIT_LIVE_REMOTE=1 \
python3 -m pytest -q tests/test_live_client.py -k remote_parameter_sweep
```

## CLI examples

```bash
nanohub-submit submit --verbose --venue workspace -- python3 run.py
nanohub-submit status --verbose 12345 12346
nanohub-submit kill --verbose 12345
nanohub-submit venue-status --verbose
nanohub-submit catalog --verbose --format json
nanohub-submit explore --verbose --format json
```

## Python usage

```python
from nanohubsubmit import NanoHUBSubmitClient, SubmitRequest

client = NanoHUBSubmitClient(
    verbose=True,
)
request = SubmitRequest(
    command="python3",
    command_arguments=["run.py", "--size", "100"],
    venues=["workspace"],
    n_cpus=4,
    wall_time="01:30:00",  # also accepts integer minutes
    wait=True,
)
result = client.submit(request)
print(result.returncode)
print(result.stdout)
```

Equivalent to:
`submit --local --runName=echotest --progress submit -s, -p @@name=hub1,hub2,hub3 echo @@name`

```python
from nanohubsubmit import NanoHUBSubmitClient, ProgressMode, SubmitRequest

client = NanoHUBSubmitClient(verbose=True)
result = client.submit(
    SubmitRequest(
        command="echo",
        command_arguments=["@@name"],
        local=True,
        run_name="echotest",
        separator=",",
        parameters=["@@name=hub1,hub2,hub3"],
        progress=ProgressMode.SUBMIT,
    ),
    operation_timeout=180.0,
)
print(result.returncode)
print(result.stdout)
```

With `run_name="echotest"` this creates `echotest/` with
`parameterCombinations.csv`, per-instance subdirectories (`01`, `02`, ...), and
`echotest_XX.stdout` files.

`run_name` must be unique in the current working directory. If the path already
exists, `client.submit(...)` raises `CommandExecutionError`.

Equivalent to:
`submit --runName=runtest --progress submit -p @@Vin=1,2,3,4,5 /apps/pegtut/current/examples/capacitor_voltage/sim1.py --Vin @@Vin`

```python
from nanohubsubmit import NanoHUBSubmitClient, ProgressMode, SubmitRequest

client = NanoHUBSubmitClient(verbose=True)
result = client.submit(
    SubmitRequest(
        command="/apps/pegtut/current/examples/capacitor_voltage/sim1.py",
        command_arguments=["--Vin", "@@Vin"],
        run_name="runtest",
        parameters=["@@Vin=1,2,3,4,5"],
        progress=ProgressMode.SUBMIT,
    ),
    operation_timeout=600.0,
)

print("returncode:", result.returncode)
timed_out = getattr(result, "timed_out", False)
process_ids = getattr(result, "process_ids", None)
if process_ids is None:
    process_ids = [result.job_id] if result.job_id is not None else []

print("timed_out:", timed_out)
print("job_id:", result.job_id)
print("process_ids:", process_ids)
print(result.stdout)

progress_lines = [
    line for line in result.stdout.splitlines()
    if line.startswith("=SUBMIT-PROGRESS=>")
]
print("progress lines:", len(progress_lines))

# If timeout happened after launch, keep tracking by known IDs.
if hasattr(client, "monitor_tracked_runs"):
    tracking = client.monitor_tracked_runs(
        root=".",
        include_live_status=True,
        operation_timeout=60.0,
    ).to_dict()
    for item in tracking["runs"]:
        run = item["run"]
        if run["run_name"] == "runtest":
            print("derived_state:", item["derived_state"])
            print("latest_progress:", item["latest_progress"])
            break
```

In Jupyter, this submit pattern (`run_name` + `progress=submit`) automatically
shows live `ipywidgets` progress bars while the call is running:
- overall submit progress from `=SUBMIT-PROGRESS=>` frames
- per-instance bars from `parameterCombinations.csv`

Disable this behavior with:
`NanoHUBSubmitClient(jupyter_auto_progress=False)`.

## Metadata utilities

```python
from nanohubsubmit import NanoHUBSubmitClient
from nanohubsubmit.utils import (
    load_available_catalog,
    explore_submit_server,
    find_catalog_entries,
)

client = NanoHUBSubmitClient(
    verbose=True,
)

catalog = load_available_catalog(client, operation_timeout=60.0)
print(catalog.tools)
print(catalog.venues)
print(catalog.managers)

exploration = explore_submit_server(client, operation_timeout=60.0).to_dict()
print(exploration["doctor"])

# Filter catalog entries by name (server-backed):
print(find_catalog_entries(client, "espresso", details=["tools"], limit=10))
```

`operation_timeout` prevents metadata discovery calls from hanging forever if the
submit server does not emit an exit frame (default is 60 seconds).

`NanoHUBSubmitClient` uses `/etc/submit/submit-client.conf` by default. Pass
`config_path=...` only when you need a non-default config file.

`SubmitRequest.progress` supports: `auto`, `curses`, `submit`, `text`,
`pegasus`, and `silent`.

Local submissions (`SubmitRequest(..., local=True)`) use a direct fast path by
default for immediate execution. This path also supports parameter sweeps and
`progress=submit` output. The legacy wire-protocol local mode is not fully
implemented, so local requests always use the fast path.

## Preflight Validation And Run Tracking

```python
from nanohubsubmit import NanoHUBSubmitClient, SubmitRequest

client = NanoHUBSubmitClient()

request = SubmitRequest(
    command="abacus",
    venues=["workspace"],
    input_files=["input.dat"],
)

# Includes command shape, file existence, and catalog checks (tool/venue/manager).
validation = client.preflight_submit_request(
    request,
    extra_existing_paths=["settings.yaml"],
).to_dict()
print(validation["ok"])

result = client.submit(request, operation_timeout=120.0)
print(result.returncode, result.job_id, result.run_name)
print(result.process_ids, result.timed_out)

# Track runs during the client session:
for run in client.list_tracked_runs():
    print(run.to_dict())

# Merge tracked runs with discovered run directories and progress:
tracking = client.monitor_tracked_runs(root=".", include_live_status=True).to_dict()
for item in tracking["runs"]:
    print(item["run"]["run_name"], item["derived_state"], item["latest_progress"])
```

If a long-running remote `submit` reaches `operation_timeout` after launch, the
client now returns a partial `CommandResult` (`returncode=124`,
`timed_out=True`) with any known IDs in `process_ids` so you can continue
tracking with `status(...)` or `monitor_tracked_runs(...)`.

## Jupyter Tutorial

Open and run:

- `notebooks/nanohubsubmit_tutorial.ipynb`

The notebook includes:
- End-to-end API usage examples
- Local and remote submit patterns
- Catalog/explore/status/kill/session examples
- Interactive `ipywidgets` controls compatible with `ipywidgets<8`
