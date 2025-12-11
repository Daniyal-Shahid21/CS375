#!/usr/bin/env bash

# Base file that defines the canonical set of Pokémon
BASE_FILE="pokemon_df.csv"

# All CSVs to synchronize to pokemon_df's first column
FILES=(
  "pokemon_df.csv"
  "pokemon_types_df.csv"
  "pokemon_abilities_df.csv"
  "pokemon_learnsets_df.csv"
  "pokemon_data.csv"
  "pokemon_data_final.csv"
  "pokemon_removed_df.csv"
)

# Temporary file for the whitelist of names
WHITELIST="_pokemon_whitelist.tmp"

# 1) Build whitelist from first column of pokemon_df.csv (excluding header)
if [[ ! -f "$BASE_FILE" ]]; then
  echo "Base file $BASE_FILE not found. Run this script in the directory with your CSVs."
  exit 1
fi

# Extract first column (name), skip header
tail -n +2 "$BASE_FILE" | cut -d',' -f1 > "$WHITELIST"

echo "Whitelist built from $BASE_FILE (first column)."

# 2) Filter each CSV in FILES by the whitelist
for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "Skipping $f (file not found)."
    continue
  fi

  echo "Filtering $f ..."

  tmp="${f}.tmp"

  # awk logic:
  #  - First pass (NR==FNR): read whitelist into array 'ok'
  #  - Second pass:
  #      * keep header (FNR==1)
  #      * keep a row only if its first column ($1) is in 'ok'
  awk -F',' '
    NR==FNR { ok[$1]=1; next }
    FNR==1 { print; next }
    ($1 in ok) { print }
  ' "$WHITELIST" "$f" > "$tmp" && mv "$tmp" "$f"
done

# 3) Clean up
rm -f "$WHITELIST"

echo "Done. All listed CSVs now only contain Pokémon present in the first column of pokemon_df.csv."
