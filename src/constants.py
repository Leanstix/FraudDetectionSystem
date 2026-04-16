from __future__ import annotations

REQUIRED_FILES = {
    "transactions": "transactions.csv",
    "users": "users.json",
    "locations": "locations.json",
    "sms": "sms.json",
    "mails": "mails.json",
}

TRANSACTION_REQUIRED_COLUMNS = [
    "transaction_id",
    "sender_id",
    "recipient_id",
    "transaction_type",
    "amount",
    "location",
    "payment_method",
    "sender_iban",
    "recipient_iban",
    "balance_after",
    "description",
    "timestamp",
]

SUSPICIOUS_KEYWORDS = [
    "urgent",
    "verify",
    "suspended",
    "locked",
    "customs",
    "confirm payment",
    "click",
    "password",
    "account",
    "security",
    "action required",
    "benefit",
    "social security",
]

PHISHING_DOMAIN_HINTS = [
    "paypa1",
    "amaz0n",
    "netfl1x",
    "secure-pay",
    "verify",
    "billing",
]

DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_RANDOM_SEED = 42
