import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    exclude = {"anon_id", "id", "name"}
    quasi_ids = []
    for c in anon_df.columns:
        if c in aux_df.columns and c not in exclude:
            quasi_ids.append(c)
    merged = anon_df.merge(aux_df, on=quasi_ids, how="inner")
    unique_matches = merged.groupby("anon_id").filter(lambda g: len(g) == 1)

    return unique_matches[["anon_id", "name"]].rename(columns={"name": "matched_name"})


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    return len(matches_df) / len(anon_df)
