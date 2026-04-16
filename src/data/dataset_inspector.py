from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders import DatasetLoader


class DatasetInspector:
    def __init__(self, input_path: str):
        self.loader = DatasetLoader(input_path)

    def inspect(self) -> dict:
        paths = self.loader.load_all()
        tx = self.loader.load_transactions()
        users = self.loader.load_users()
        locations = self.loader.load_locations()
        sms = self.loader.load_sms()
        mails = self.loader.load_mails()
        return {
            "dataset_name": paths.dataset_name,
            "dataset_root": paths.dataset_root,
            "transactions": len(tx),
            "users": len(users),
            "locations": len(locations),
            "sms_threads": len(sms),
            "mail_threads": len(mails),
            "transaction_columns": list(tx.columns),
        }
