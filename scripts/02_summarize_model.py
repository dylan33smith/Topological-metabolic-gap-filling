from metabolic_gap_filling.config import RAW_DATA_DIR, RESULTS_DIR
from metabolic_gap_filling.data import load_cobra_model, reaction_table, summarize_model


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = load_cobra_model(RAW_DATA_DIR / "iJO1366.json")

    summarize_model(model).to_csv(RESULTS_DIR / "model_summary.csv", index=False)
    reaction_table(model).to_csv(RESULTS_DIR / "reaction_table.csv", index=False)

    print("Wrote model summary and reaction table.")


if __name__ == "__main__":
    main()
