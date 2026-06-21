"""Shared constants for cervical cytology classification."""

CERVICAL_LABELS = ("HSIL", "ASC-H", "LSIL", "ASC-US", "Normal")

NORMAL_LABEL = "Normal"

LABEL_ALIASES = {
    "hsil": "HSIL",
    "high-grade squamous intraepithelial lesion": "HSIL",
    "high grade squamous intraepithelial lesion": "HSIL",
    "asc-h": "ASC-H",
    "asch": "ASC-H",
    "atypical squamous cells cannot exclude hsil": "ASC-H",
    "lsil": "LSIL",
    "low-grade squamous intraepithelial lesion": "LSIL",
    "low grade squamous intraepithelial lesion": "LSIL",
    "asc-us": "ASC-US",
    "ascus": "ASC-US",
    "atypical squamous cells of undetermined significance": "ASC-US",
    "normal": "Normal",
    "negative": "Normal",
    "nilm": "Normal",
}

LABEL_PRIORS = {
    "HSIL": {
        "nucleus": "markedly enlarged hyperchromatic nucleus",
        "chromatin": "coarse chromatin and high nuclear-to-cytoplasmic ratio",
        "decision": "high-grade intraepithelial lesion",
    },
    "ASC-H": {
        "nucleus": "atypical nucleus suspicious for high-grade change",
        "chromatin": "irregular chromatin that is not definitive enough for HSIL",
        "decision": "atypical squamous cell, cannot exclude HSIL",
    },
    "LSIL": {
        "nucleus": "mildly enlarged nucleus with low-grade atypia",
        "chromatin": "chromatin change consistent with low-grade lesion",
        "decision": "low-grade intraepithelial lesion",
    },
    "ASC-US": {
        "nucleus": "borderline nuclear enlargement",
        "chromatin": "subtle atypia without definitive low-grade or high-grade pattern",
        "decision": "atypical squamous cell of undetermined significance",
    },
    "Normal": {
        "nucleus": "regular nucleus with preserved cytoplasm",
        "chromatin": "uniform chromatin and normal cell arrangement",
        "decision": "normal cervical squamous cell",
    },
}
