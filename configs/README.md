# Environment configurations

Configuration resolves in four layers, lowest precedence first:

1. built-in defaults (`config.py`) — the canonical parameterisation;
2. a TOML/JSON file from this directory;
3. `APGI_*` environment variables;
4. explicit CLI/API overrides.

```bash
# by file
APGI_CONFIG=configs/prod.toml python main.py

# by environment variable
APGI_SEED=42 APGI_STRICT=true python main.py
```

An `APGI_*` variable that does not name a known configuration key raises
rather than being ignored — silently dropping a misspelled override would give
you a run you did not configure.

| File | Purpose |
| --- | --- |
| `dev.toml` | Fast iteration. Small runs, verbose logging, unbounded history. |
| `staging.toml` | Full feature surface enabled, to exercise every code path before release. |
| `prod.toml` | Publication runs. Fixed seed, strict validation, canonical parameters only. |

## Publication runs

Use `prod.toml`. It pins a seed and sets `strict = true`, so any spec violation
aborts rather than producing a plausible-looking but non-conformant figure.
Write a run manifest alongside every output:

```python
from core.config_io import load_config
from pipeline import APGIPipeline

config = load_config("configs/prod.toml")
pipeline = APGIPipeline(config)
...
pipeline.write_manifest("outputs/figure_07.manifest.json", n_steps=10_000)
```

The manifest records the seed, the resolved config and its SHA-256, the git
commit and working-tree cleanliness, the interpreter, and every numerical
dependency version — everything needed to regenerate the figure exactly.
