import pandas as pd

# 1️⃣ Load the original CSV (raw file)
df = pd.read_csv("historical_positions.csv")

# 2️⃣ Clean & standardize columns

# Parse dates
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")

# Ensure Shares is numeric
df["Shares"] = pd.to_numeric(df["Shares"], errors="coerce")

# Strip spaces from Type (so "Sell " becomes "Sell")
df["Type"] = df["Type"].astype(str).str.strip()

# Clean entry: remove $, commas, spaces → then convert to numeric
df["entry"] = (
    df["entry"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)
df["entry"] = pd.to_numeric(df["entry"], errors="coerce")

# # (Optional) Clean Total Transaction Amount too, if you want it numeric
# if "Total Transaction Amount" in df.columns:
#     df["Total Transaction Amount"] = (
#         df["Total Transaction Amount"]
#         .astype(str)
#         .str.replace("$", "", regex=False)
#         .str.replace(",", "", regex=False)
#         .str.strip()
#     )
#     df["Total Transaction Amount"] = pd.to_numeric(
#         df["Total Transaction Amount"], errors="coerce"
#     )

# 3️⃣ Drop rows missing critical fields
df = df.dropna(subset=["Date", "Ticker", "Type", "Shares"])

# 4️⃣ SAVE the cleaned version
# Option A: save to a NEW file
df.to_csv("historical_positions_clean.csv", index=False)

# Option B (if you want the app to use the cleaned version directly):
# df.to_csv("historical_positions.csv", index=False)
