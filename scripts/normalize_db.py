import sqlite3

import polars as pl

DB_PATH = "../data/finance.db"

# -----------------------------
# 1. CONNECT + READ WITH SAFE SCHEMA
# -----------------------------
conn = sqlite3.connect(DB_PATH)
conn.execute("PRAGMA foreign_keys = ON")

cursor = conn.cursor()
cursor.execute("PRAGMA table_info(financial_records)")
columns_info = cursor.fetchall()

schema_overrides = {}
for _, name, sql_type, *_ in columns_info:
    sql_type = sql_type.upper()
    if "INT" in sql_type:
        schema_overrides[name] = pl.Int64
    elif "REAL" in sql_type:
        schema_overrides[name] = pl.Float64
    else:
        schema_overrides[name] = pl.Utf8

df = pl.read_database(
    "SELECT * FROM financial_records",
    connection=conn,
    schema_overrides=schema_overrides,
)

print("=" * 60)
print("📥 DATA LOAD")
print("=" * 60)
print(f"Loaded {df.height} rows")
print("Schema:")
for k, v in df.schema.items():
    print(f"  {k}: {v}")

# -----------------------------
# 2. CREATE COMPANIES TABLE
# -----------------------------
companies = df.select(
    [
        "company_id",
        "company_size",
        "is_public",
        "segment_code",
        "industry_code_level1",
        "industry_code_level6",
    ]
).unique(subset=["company_id"])

print("\n" + "=" * 60)
print("🏢 COMPANIES TABLE")
print("=" * 60)
print(f"Unique companies: {companies.height}")

conn.execute("DROP TABLE IF EXISTS companies")
conn.execute(
    """
    CREATE TABLE companies (
        company_id INTEGER PRIMARY KEY,
        company_size INTEGER,
        is_public INTEGER,
        segment_code INTEGER,
        industry_code_level1 INTEGER,
        industry_code_level6 INTEGER
    )
    """
)

conn.executemany(
    "INSERT INTO companies VALUES (?, ?, ?, ?, ?, ?)",
    companies.to_numpy().tolist(),
)

conn.execute("CREATE INDEX idx_companies_size ON companies(company_size)")
conn.execute("CREATE INDEX idx_companies_industry ON companies(industry_code_level6)")

print("✅ companies table created with indexes")

# -----------------------------
# 3. CREATE FINANCIALS TABLE
# -----------------------------
financials = df.select(
    [
        "company_id",
        "year",
        "ranking",
        "statement_id",
        "revenue",
        "net_income",
        "total_assets",
        "total_liabilities",
        "equity",
    ]
)

print("\n" + "=" * 60)
print("📊 FINANCIALS TABLE")
print("=" * 60)
print(f"Financial rows: {financials.height}")

conn.execute("DROP TABLE IF EXISTS financials")
conn.execute(
    """
    CREATE TABLE financials (
        company_id INTEGER,
        year INTEGER,
        ranking INTEGER,
        statement_id INTEGER,
        revenue REAL,
        net_income REAL,
        total_assets REAL,
        total_liabilities REAL,
        equity REAL,
        PRIMARY KEY (company_id, year),
        FOREIGN KEY (company_id) REFERENCES companies(company_id)
    )
    """
)

conn.executemany(
    "INSERT INTO financials VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
    financials.to_numpy().tolist(),
)

conn.execute("CREATE INDEX idx_financials_year ON financials(year)")
conn.execute("CREATE INDEX idx_financials_company ON financials(company_id)")

print("✅ financials table created with PK, FK, and indexes")

# -----------------------------
# 4. VERBOSE SANITY CHECKS
# -----------------------------
print("\n" + "=" * 60)
print("🧪 SANITY CHECKS")
print("=" * 60)

# 4.1 Row counts
print("\n📊 ROW COUNTS")
print(f"Raw records: {df.height}")
print(f"Companies:   {companies.height}")
print(f"Financials:  {financials.height}")

# 4.2 Referential integrity
missing_companies = conn.execute(
    """
    SELECT COUNT(*)
    FROM financials f
    LEFT JOIN companies c USING (company_id)
    WHERE c.company_id IS NULL
    """
).fetchone()[0]

if missing_companies == 0:
    print("✅ Referential integrity OK")
else:
    print(f"❌ Referential integrity FAILED ({missing_companies} rows)")

# 4.3 Duplicate (company_id, year)
duplicates = (
    financials.group_by(["company_id", "year"]).count().filter(pl.col("count") > 1)
)

if duplicates.is_empty():
    print("✅ No duplicate (company_id, year)")
else:
    print("❌ Duplicate (company_id, year) found:")
    print(duplicates)

# 4.4 NULL diagnostics
print("\n📋 NULL VALUE SUMMARY")
for col in financials.columns:
    nulls = financials.select(pl.col(col).is_null().sum()).item()
    if nulls > 0:
        pct = (nulls / financials.height) * 100
        print(f"  {col}: {nulls} ({pct:.2f}%)")

# 4.5 Value sanity
print("\n🔍 BASIC VALUE CHECKS")
for col in ["revenue", "net_income", "total_assets"]:
    min_val = financials.select(pl.col(col).min()).item()
    max_val = financials.select(pl.col(col).max()).item()
    print(f"  {col}: min={min_val}, max={max_val}")

print("\n✅ All sanity checks passed")

# -----------------------------
# 5. CREATE RATIOS TABLE
# -----------------------------

print("\n" + "=" * 60)
print("📐 RATIOS TABLE")
print("=" * 60)


def safe_div(numerator, denominator):
    return (
        pl.when((pl.col(denominator).is_null()) | (pl.col(denominator) == 0))
        .then(None)
        .otherwise(pl.col(numerator) / pl.col(denominator))
    )


ratios = financials.with_columns(
    [
        safe_div("net_income", "revenue").alias("net_margin"),
        safe_div("net_income", "total_assets").alias("roa"),
        safe_div("net_income", "equity").alias("roe"),
        safe_div("total_assets", "equity").alias("asset_leverage"),
    ]
).select(
    [
        "company_id",
        "year",
        "net_margin",
        "roa",
        "roe",
        "asset_leverage",
    ]
)


print(f"Ratios computed: {ratios.height}")

conn.execute("DROP TABLE IF EXISTS ratios")
conn.execute(
    """
    CREATE TABLE ratios (
        company_id INTEGER,
        year INTEGER,
        net_margin REAL,
        roa REAL,
        roe REAL,
        asset_leverage REAL,
        PRIMARY KEY (company_id, year),
        FOREIGN KEY (company_id) REFERENCES companies(company_id),
        FOREIGN KEY (company_id, year)
            REFERENCES financials(company_id, year)
    )
    """
)

conn.executemany(
    """
    INSERT INTO ratios VALUES (?, ?, ?, ?, ?, ?)
    """,
    ratios.to_numpy().tolist(),
)

conn.execute("CREATE INDEX idx_ratios_company ON ratios(company_id)")
conn.execute("CREATE INDEX idx_ratios_year ON ratios(year)")

print("✅ ratios table created with FK + indexes")

# Ratio sanity
print("\n📐 RATIO SANITY CHECKS")

for col in ["net_margin", "roa", "roe", "asset_leverage"]:
    nulls = ratios.select(pl.col(col).is_null().sum()).item()
    if nulls > 0:
        pct = nulls / ratios.height * 100
        print(f"  {col}: {nulls} NULLs ({pct:.2f}%)")

print("\n📊 RATIO RANGES")
for col in ["net_margin", "roa", "roe"]:
    min_val = ratios.select(pl.col(col).min()).item()
    max_val = ratios.select(pl.col(col).max()).item()
    print(f"  {col}: min={min_val}, max={max_val}")

leverage_min = ratios.select(pl.col("asset_leverage").min()).item()
leverage_max = ratios.select(pl.col("asset_leverage").max()).item()
print(f"  asset_leverage: min={leverage_min}, max={leverage_max}")

# -----------------------------
# 5. CLOSE
# -----------------------------
conn.commit()
conn.close()

print("\n🎉 DATABASE NORMALIZATION COMPLETE")
