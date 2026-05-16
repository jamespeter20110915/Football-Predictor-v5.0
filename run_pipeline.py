"""One-click data collection pipeline for Football Predictor v5.0."""

from scrapers.football_data_scraper import download_all as download_fd
from scrapers.understat_scraper import download_all as download_us
from scrapers.merge import merge_and_save

if __name__ == "__main__":
    print("=" * 60)
    print("Step 1/3: Downloading Football-Data.co.uk CSVs ...")
    print("=" * 60)
    download_fd()

    print("\n" + "=" * 60)
    print("Step 2/3: Downloading Understat data ...")
    print("=" * 60)
    download_us()

    print("\n" + "=" * 60)
    print("Step 3/3: Merging into unified parquet ...")
    print("=" * 60)
    merge_and_save()

    print("\nDone! Output: data/processed/matches.parquet")
