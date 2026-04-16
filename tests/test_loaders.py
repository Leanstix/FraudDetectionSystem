from __future__ import annotations

from pathlib import Path

from src.data.loaders import DatasetLoader
from src.data.schemas import REQUIRED_FILE_NAMES, validate_json_list, validate_transactions_schema


def _dataset_zip() -> str:
    candidates = [
        Path("Brave New World - train.zip"),
        Path("hackTheCode/Brave New World - train.zip"),
        Path("hackTheCode/Brave+New+World+-+train.zip"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    found = list(Path(".").glob("**/Brave*train.zip"))
    if not found:
        raise FileNotFoundError("Brave New World train zip not found")
    return str(found[0])


def test_dataset_zip_loads_and_required_files_found():
    loader = DatasetLoader(_dataset_zip())
    paths = loader.load_all()
    root_files = {p.name for p in Path(paths.dataset_root).iterdir() if p.is_file()}
    assert REQUIRED_FILE_NAMES.issubset(root_files)


def test_schemas_validate():
    loader = DatasetLoader(_dataset_zip())
    tx = loader.load_transactions()
    users = loader.load_users()
    locations = loader.load_locations()
    sms = loader.load_sms()
    mails = loader.load_mails()

    validate_transactions_schema(tx)
    validate_json_list("users.json", users, {"first_name", "last_name", "iban", "residence"})
    validate_json_list("locations.json", locations, {"biotag", "timestamp", "lat", "lng", "city"})
    validate_json_list("sms.json", sms, {"sms"})
    validate_json_list("mails.json", mails, {"mail"})
