"""
Download JPMorgan Chase 10-K filings from SEC.GOV (last 5 years)
"""

import os
import time
import requests

CIK = "0000019617"  # JPMorgan Chase CIK on SEC.GOV
OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "RAG-Assignment research@example.com"
}

# Fetch the last N 10-K filing metadata from SEC.GOV
def get_10k_filings(cik, num_filings: int = 5):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()

    filings = data["filings"]["recent"]
    results = []
    for i, form in enumerate(filings["form"]):
        if form == "10-K":
            results.append({
                "accessionNumber": filings["accessionNumber"][i].replace("-", ""),
                "filingDate": filings["filingDate"][i],
                "primaryDocument": filings["primaryDocument"][i],
            })
        if len(results) >= num_filings:
            break
    return results

# Download the primary document for a given 10-K filing
def download_filing(cik, accession, primary_doc, filing_date):
    base_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession}/{primary_doc}"
    )
    resp = requests.get(base_url, headers=HEADERS)
    resp.raise_for_status()

    year = filing_date[:4]
    ext = primary_doc.split(".")[-1]
    out_path = os.path.join(OUTPUT_DIR, f"JPM_10K_{year}.{ext}")

    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"  Saved: {out_path}  ({len(resp.content) // 1024} KB)")
    return out_path


def main():
    print("Starting Download")
    filings = get_10k_filings(CIK, num_filings=5)

    for filing in filings:
        print(f"  Date: {filing['filingDate']}  Doc: {filing['primaryDocument']}")
        try:
            download_filing(
                CIK,
                filing["accessionNumber"],
                filing["primaryDocument"],
                filing["filingDate"],
            )
        except Exception as e:
            print(f"  ERROR: {e}")
        time.sleep(0.6)

    print("\nDownload complete")


if __name__ == "__main__":
    main()
