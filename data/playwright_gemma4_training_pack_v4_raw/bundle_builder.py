
import json, zipfile, re

def clean_text(s):
    if not s:
        return None
    return re.sub(r"\s+", " ", str(s)).strip() or None

def build_bundle_from_zip(zip_path):
    """
    Example skeleton:
    1) unzip
    2) iterate pages/*
    3) read static.json / trigger-candidates.json / trigger-results/*.json / final-report.json
    4) serialize evidence into the same schema as train_raw.jsonl
    """
    raise NotImplementedError
