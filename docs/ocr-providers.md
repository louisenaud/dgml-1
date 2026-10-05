# OCR providers

`--text-mode ocr` (and the OCR half of `--text-mode hybrid`) turns each rendered
page image into a stream of positioned words. Which engine does that is
**pluggable**: the `[ocr]` section of your config names a provider, and DGML
resolves it at runtime.

DGML bundles three providers, and you can point at your own class instead —
resolved from its dotted path exactly like the bundled ones, with no change to
DGML and no repo checkout.

## Configuring a provider

`ocr.provider` takes either a **short name** for a bundled provider or a dotted
`"module.path:ClassName"`. Everything else in the section is that provider's own
options.

```toml
[ocr]
provider = "azure"
endpoint = "https://<resource>.cognitiveservices.azure.com/"
api_key_env = "AZURE_DOCINTEL_KEY"
```

```toml
[ocr]
provider = "my_pkg.tesseract:TesseractProvider"
lang = "eng"
```

The short names are aliases, nothing more — `"aws"` and
`"dgml_core.ocr_aws:AwsProvider"` select the same class by the same mechanism.

On macOS an absent `[ocr]` section defaults to the on-device Apple Vision
provider (with a warning). On other platforms there is no built-in engine, so a
missing section is an error.

## Bundled providers

| Short name | Class | Install |
| --- | --- | --- |
| `macos` | `dgml_core.ocr_macos:MacosProvider` | `pip install dgml[macos]` |
| `azure` | `dgml_core.ocr_azure:AzureProvider` | `pip install dgml[azure]` |
| `aws` | `dgml_core.ocr_aws:AwsProvider` | `pip install dgml[aws]` |

### macOS — Apple Vision (`macos`)

On-device, no network, no API keys, no per-page cost. Takes no options. macOS
only; constructing it elsewhere raises an actionable error.

### Azure Document Intelligence (`azure`)

| Option | Required | Meaning |
| --- | --- | --- |
| `endpoint` | yes | Resource endpoint URL. |
| `api_key` | no | Literal key. Mutually exclusive with `api_key_env`. |
| `api_key_env` | no | **Name** of an env var holding the key. |

With neither key option set, auth falls through to `DefaultAzureCredential`
(env vars, managed identity, `az login`). Setting both yields
`OCR_CONFIG_INVALID`; a referenced-but-unset env var yields `AUTH_ERROR`.

### AWS Textract (`aws`)

| Option | Required | Meaning |
| --- | --- | --- |
| `region` | yes | AWS region. |
| `profile` | no | boto3 profile name; omitted means the default credential chain. |

Textract is invoked once per rendered page image (5 MB sync limit).

## Writing your own provider

You don't need the repo source — the `dgml` wheel resolves your class at runtime
from its dotted path.

1. Install the library that defines the base class: `pip install dgml-core` (or
   `pip install dgml`, which pulls it in).
2. Write a class subclassing `OcrProvider`:

   ```python
   from typing import Any, ClassVar

   from dgml_core import OcrConfig, OcrProvider
   from dgml_core.errors import MissingExtra, OcrConfigInvalid, OcrFailed

   class TesseractProvider(OcrProvider):
       name: ClassVar[str] = "tesseract"
       config_fields: ClassVar[frozenset[str]] = frozenset({"lang"})

       @classmethod
       def parse_config(cls, config: OcrConfig) -> OcrConfig:
           if not isinstance(config.options.get("lang"), str):
               raise OcrConfigInvalid("tesseract OCR requires 'ocr.lang'")
           return config          # must return it — see the contract below

       def __init__(self, config: OcrConfig) -> None:
           ...   # lazy-import your SDK; raise MissingExtra if it isn't installed

       def analyze_image(
           self,
           image_bytes: bytes,
           image_dims_px: tuple[int, int],
           page_num: int,
       ) -> list[dict[str, Any]]:
           ...   # return [{"t": text, "l": [left, top, right, bottom]}, …]
   ```

3. Make it importable by the same interpreter running `dgml`. Cleanest is your
   own `pip install`-ed package in that venv; the quick path is a loose `.py`
   file on `PYTHONPATH`, referenced by module name (the filename without `.py`).
4. Point `ocr.provider` at the dotted path, with your options alongside it.

### The contract your `parse_config` must honor

- **Return the config.** Validate, then `return config` (or a normalized copy).
  A `parse_config` that validates and falls off the end returns `None`, which is
  rejected with `OCR_CONFIG_INVALID` rather than handing your provider nothing.
- **Be pure and idempotent.** It runs twice — once when the config is loaded
  (the up-front gate `file add` relies on) and again when the provider is
  constructed. Don't open connections, read files, or mutate state in it; do
  that in `__init__`.
- **Raise `OcrConfigInvalid`** for a missing or malformed option of your own.
  Unknown keys are already handled for you.

### The contract your `analyze_image` must honor

- **Return shape** is a list of `{"t": <text>, "l": [left, top, right, bottom]}`,
  one per word. If your engine also reports page skew, return an
  `OcrPageResult(words=…, angle=…)` instead (degrees, clockwise-positive) and
  the loop will deskew the page image and rotate the boxes to match.
- **Coordinates are pixels** in the supplied `image_dims_px`, top-left origin.
  If your engine reports normalized (0–1) boxes, multiply through by those dims
  — that is what `image_dims_px` is for.
- **Order is reading order.** Grounding matches the DGML tree against this word
  stream, so a shuffled stream degrades `dg:origin` quality.
- **Thread safety.** The shared loop calls `analyze_image` from several threads
  against one provider instance — `ocr.max_concurrency` pages at a time
  (default 5).
- **Failures** raise `OcrFailed` (or `AuthError` from `__init__`). The loop does
  not retry; the first failure cancels pending pages.
- `config_fields` lists every option key you accept. DGML rejects anything else
  before calling you, with an "unknown fields" error naming the allowed keys —
  that is what catches typos and options left behind after switching provider,
  and you don't have to implement it. The universal keys — `provider` and
  `max_concurrency` — are DGML's own and never reach you, so leave them out of
  `config_fields`.

**Note on trust:** a dotted `provider` path runs arbitrary code from your
config, as you — the same trust model as `[conversion]` and `[storage]`. Keep
your `config.toml` under your own control.

## How OCR fits the workflow

- Page images are rendered at add time; OCR reads them from `page_images/` and
  writes one `page_text/page_N.json` per page — the same shape the digital
  extractor produces, so nothing downstream cares which mode produced it.
- OCR config is validated **up front**, before `file add` touches the
  filesystem: a bad `[ocr]` table fails with `OCR_CONFIG_MISSING` /
  `OCR_CONFIG_INVALID` and no file record is created. Resolving your provider
  class (and running its `parse_config`) happens there, so a broken custom
  provider fails the add rather than a corpus halfway through.
- Provider and auth failures during extraction are recorded on the file record
  as permanent; `dgml check --retry-errors` is the recovery path once the
  underlying problem is fixed.

## Troubleshooting

| Message | Cause |
| --- | --- |
| `'ocr.provider' must be one of […] or a dotted path` | A bare name that isn't a bundled short name. Use a full `module:Class` path for your own. |
| `could not import ocr module …` | The module isn't importable by the interpreter running `dgml` — check the venv and `PYTHONPATH`. |
| `module … has no attribute …` | The class name after `:` is wrong. |
| `… is not a OcrProvider subclass` | The path resolved to something else (a storage backend, a function). |
| `… resolved to abstract class …` | The path names the base class, or a subclass that doesn't implement every abstract method (the message lists which). |
| `….parse_config must return an OcrConfig` | Your `parse_config` validated but didn't `return config`. |
| `unknown fields in 'ocr' for provider …` | An option the provider doesn't declare in `config_fields` — a typo, or left over from a previous provider. |
