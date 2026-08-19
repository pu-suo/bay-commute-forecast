import sys, logging, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
from forecast.features_stations import build_seasonal
s = build_seasonal("~/traffic-data/stations")
out = os.path.expanduser("~/traffic-data/stations/_seasonal.parquet")
s.to_parquet(out, index=False, compression="zstd")
print(f"wrote {len(s):,} cells -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")
