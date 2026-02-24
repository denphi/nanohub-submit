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
    wait=True,
)
result = client.submit(request)
print(result.returncode)
print(result.stdout)
```

## Metadata utilities

```python
from nanohubsubmit import NanoHUBSubmitClient
from nanohubsubmit.utils import load_available_catalog, explore_submit_server

client = NanoHUBSubmitClient(
    verbose=True,
)

catalog = load_available_catalog(client, operation_timeout=60.0)
print(catalog.tools)
print(catalog.venues)
print(catalog.managers)

exploration = explore_submit_server(client, operation_timeout=60.0).to_dict()
print(exploration["doctor"])
```

`operation_timeout` prevents metadata discovery calls from hanging forever if the
submit server does not emit an exit frame (default is 60 seconds).

`NanoHUBSubmitClient` uses `/etc/submit/submit-client.conf` by default. Pass
`config_path=...` only when you need a non-default config file.

Local submissions (`SubmitRequest(..., local=True)`) use a direct fast path by
default for immediate execution. Set `local_fast_path=False` when creating the
client to force local mode through the submit server protocol.

## Jupyter Tutorial

Open and run:

- `notebooks/nanohubsubmit_tutorial.ipynb`

The notebook includes:
- End-to-end API usage examples
- Local and remote submit patterns
- Catalog/explore/status/kill/session examples
- Interactive `ipywidgets` controls compatible with `ipywidgets<8`
