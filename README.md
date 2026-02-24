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

## CLI examples

```bash
nanohub-submit submit --config /etc/submit/submit-client.conf --venue workspace -- python3 run.py
nanohub-submit status 12345 12346
nanohub-submit kill 12345
nanohub-submit venue-status
nanohub-submit catalog --format json
nanohub-submit explore --format json
```

## Python usage

```python
from nanohubsubmit import NanoHUBSubmitClient, SubmitRequest

client = NanoHUBSubmitClient(config_path="/etc/submit/submit-client.conf")
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

client = NanoHUBSubmitClient(config_path="/etc/submit/submit-client.conf")

catalog = load_available_catalog(client)
print(catalog.tools)
print(catalog.venues)
print(catalog.managers)

exploration = explore_submit_server(client).to_dict()
print(exploration["doctor"])
```
