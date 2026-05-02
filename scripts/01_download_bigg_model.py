import requests

from metabolic_gap_filling.config import BIGG_JSON_URL, RAW_DATA_DIR


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DATA_DIR / "iJO1366.json"

    response = requests.get(BIGG_JSON_URL, timeout=60)
    response.raise_for_status()
    output_path.write_bytes(response.content)

    print(f"Downloaded {BIGG_JSON_URL}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
