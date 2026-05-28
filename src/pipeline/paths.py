from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

LANDING = DATA / "landing"
LANDING_TX = LANDING / "Transactions"
LANDING_PRODUCTS = LANDING / "Products"

BRONZE = DATA / "bronze"
SILVER = DATA / "silver"
GOLD = DATA / "gold"

for p in [BRONZE, SILVER, GOLD]:
    p.mkdir(parents=True, exist_ok=True)
