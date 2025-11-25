import pandas as pd
import ast
import re

# def load_and_clean_games_csv(path: str) -> pd.DataFrame:
#     """
#     Load and clean the games CSV.
    
#     Fixes:
#     - Converts stringified lists (themes, keywords, involved_companies)
#       from: "[""Action"", ""Kids""]"
#       to:   ["Action", "Kids"]
#     - Ensures rating is float and year is int
#     - Strips whitespace
#     """

#     df = pd.read_csv(path)

#     list_columns = ["themes", "keywords", "involved_companies"]

#     def parse_list_field(x):
#         if pd.isna(x) or x == "":
#             return []
#         cleaned = x.replace('""', '"')
#         try:
#             return ast.literal_eval(cleaned)
#         except Exception:
#             return re.findall(r'"(.*?)"', cleaned)

#     # Parse all list-like columns

#     for col in list_columns:
#         df[col] = df[col].apply(parse_list_field)

#     # Fix simple numeric types
#     df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
#     df["first_release_year"] = pd.to_numeric(df["first_release_year"], errors="coerce").astype("Int64")

#     # Strip name column
#     df["name"] = df["name"].astype(str).str.strip()

#     return df
import pandas as pd
import ast
import re

# def clean_list_column(x):
#     """
#     Convert any of the following → Python list:
#     - "['Action', 'Kids']"
#     - '[''Action'', ''Kids'']'
#     - ['Action', 'Kids']
#     - "['Action']"
#     - ['Action']
#     """

#     if pd.isna(x):
#         return []

#     # Already parsed into a list by pandas?
#     if isinstance(x, list):
#         return x

#     # Normalize double-double quotes → single
#     s = str(x).strip().replace('""', '"')

#     # Ensure outer quotes are valid Python by enforcing quote style:
#     # Convert all double quotes to single quotes, except where apostrophes appear
#     # (This lets literal_eval work more reliably)
#     # Example: ["Action"] → ['Action']
#     if s.startswith("[") and s.endswith("]"):
#         # Replace double quotes only when they appear as list delimiters
#         s = re.sub(r'"([^"]*)"', r"'\1'", s)

#     # Final attempt: parse as Python literal
#     try:
#         parsed = ast.literal_eval(s)
#         if isinstance(parsed, list):
#             return parsed
#     except Exception:
#         pass

#     # Fallback: manual extraction of items inside quotes
#     items = re.findall(r"'([^']*)'", s)
#     return items


# def load_and_clean_games_csv(path: str) -> pd.DataFrame:
#     df = pd.read_csv(path)

#     list_columns = ["themes", "keywords", "involved_companies"]

#     for col in list_columns:
#         df[col] = df[col].apply(clean_list_column)

#     # Convert numeric fields
#     df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
#     df["first_release_year"] = pd.to_numeric(df["first_release_year"], errors="coerce").astype("Int64")

#     # Strip text fields
#     df["name"] = df["name"].astype(str).str.strip()

#     return df


import pandas as pd

def strip_outer_quotes(x):
    if isinstance(x, str) and len(x) >= 2:
        if x.startswith('"') and x.endswith('"'):
            return x[1:-1]
    return x

def clean_csv_strip_outer_quotes(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)

    # apply ONLY outer-quote removal to every cell
    return df.applymap(strip_outer_quotes)

import re
import ast
import pandas as pd


import pandas as pd
import json
import re

def fix_json_list(s: str):
    """Convert CSV-escaped list into real Python list."""
    if pd.isna(s) or s.strip() == "":
        return []
    
    # Example input:
    #   "[""Action"", ""Kids""]"
    #
    # Step 1: Replace double double-quotes → single double-quotes
    s = s.replace('""', '"')

    # Step 2: Ensure it's valid JSON
    # Some rows might already be valid JSON, so wrap in try/except
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Last resort fallbacks
        # Extract things in quotes: "text"
        items = re.findall(r'"(.*?)"', s)
        return items


def load_clean_games_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    list_cols = ["themes", "keywords", "involved_companies"]

    for col in list_cols:
        df[col] = df[col].apply(fix_json_list)

    # Convert numbers
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["first_release_year"] = pd.to_numeric(df["first_release_year"], errors="coerce")

    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on 'id' column, keeping the first occurrence."""
    return df.drop_duplicates(subset=["id"], keep="first")


if __name__ == "__main__":
    raw_df = pd.read_csv("dataset_construction/data/metadata.csv")
    print(f"Raw metadata rows: {len(raw_df)}")
    cleaned_df = remove_duplicates(raw_df)
    print(f"Cleaned metadata rows: {len(cleaned_df)}")
    cleaned_df.to_csv("dataset_construction/data/metadata_cleaned.csv", index=False)