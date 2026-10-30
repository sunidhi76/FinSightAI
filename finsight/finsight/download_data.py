
"""Optional sample-report downloader. Public URLs can change, so failures are non-fatal."""
from __future__ import annotations
from pathlib import Path
import requests
from config import RAW_DIR
URLS={
    # These are intentionally examples; SEC/IR sites can change filenames.
    "README_sample_source.txt":"https://www.sec.gov/edgar.shtml"
}
def main():
    RAW_DIR.mkdir(parents=True,exist_ok=True)
    print("FinSight sample-data helper")
    for name,url in URLS.items():
        try:
            r=requests.get(url,timeout=15,headers={"User-Agent":"FinSight demo contact@example.com"})
            r.raise_for_status()
            if name.endswith(".pdf"):
                (RAW_DIR/name).write_bytes(r.content); print("Downloaded",name)
            else:
                print("Public source reachable:",url)
        except Exception as exc:
            print(f"Download failed for {name}: {exc.__class__.__name__}")
    print("If no PDFs were downloaded, manually place public annual-report PDFs in data/raw/.")
if __name__=="__main__": main()
