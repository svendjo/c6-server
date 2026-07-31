"""Per-environment storage for each prediction's artifacts (input.jpg and
prediction.json, or error.txt when the prediction failed).

Backend is chosen from config/<APP_ENV>.yaml (`results.backend`):
  local -> results/<id>/...         on local disk
  s3    -> s3://<bucket>/<id>/...   (boto3)

`/predict` writes its files into a working directory, then `commit()` finalizes it
(local: nothing to do; s3: upload the directory's files, then drop the local temp).
Saving is what makes a bad prediction reproducible afterwards -- without it the
input image is gone the moment the request ends.
"""
import shutil
import tempfile
from pathlib import Path

from config import ConfigError


class LocalResultStore:
    """results/<id>/... on local disk (the default / dev behavior)."""

    def __init__(self, dir="results"):
        self.root = Path(dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def new_working_dir(self, result_id):
        d = self.root / result_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def commit(self, result_id, working_dir):
        pass  # files are already under results/<id>/

    def describe(self, result_id):
        return str(self.root / result_id)


class S3ResultStore:
    """s3://<bucket>/<id>/... -- used in aws-prod (App Runner disk is ephemeral and
    not shared across instances, so results must live in shared object storage)."""

    def __init__(self, bucket, region=None):
        import boto3
        self.bucket = bucket
        self._s3 = boto3.client("s3", region_name=region)

    def new_working_dir(self, result_id):
        # Stage to a temp dir so the request writes plain files either way;
        # commit() uploads its contents.
        return Path(tempfile.mkdtemp(prefix=f"{result_id}-"))

    def commit(self, result_id, working_dir):
        wd = Path(working_dir)
        try:
            for p in sorted(wd.rglob("*")):
                if p.is_file():
                    key = f"{result_id}/{p.relative_to(wd).as_posix()}"
                    self._s3.upload_file(str(p), self.bucket, key)
        finally:
            shutil.rmtree(wd, ignore_errors=True)

    def describe(self, result_id):
        return f"s3://{self.bucket}/{result_id}/"


def make_store(results):
    """Build the result store from the `results:` block of the env config."""
    backend = (results or {}).get("backend", "local")
    if backend == "s3":
        return S3ResultStore(results["bucket"], region=results.get("region"))
    if backend == "local":
        return LocalResultStore(results.get("dir", "results"))
    # ConfigError, not SystemExit: the store is built lazily on the first request,
    # where SystemExit would bypass the endpoint's error handling (config.validate
    # catches this at startup anyway).
    raise ConfigError(f"Unknown results.backend {backend!r} (use 'local' or 's3').")
