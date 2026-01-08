import sqlite3

import polars as pl

DB_PATH = "../data/finance.db"
CSV_PATH = "../data/raw.csv"

# -----------------------------
# 0. READ CSV (RAW INGEST)
# -----------------------------

df = pl.read_csv(
    CSV_PATH,
    separator=";",
    null_values=["", " "],
    infer_schema_length=0,
)

print(f"CSV loaded with {len(df.columns)} columns")

# -----------------------------------------
# 0.1 DROP RAW FLAGS & LAGGED DUPLICATES
# -----------------------------------------

cols_to_drop = [c for c in df.columns if c.startswith("missing_") or c.endswith("_1")]

df = df.drop(cols_to_drop)
print(f"Dropped {len(cols_to_drop)} raw helper columns")

# -----------------------------
# 1. RENAME COLUMNS
# -----------------------------

COLUMN_RENAME_MAP = {
    "anio": "year",
    "expediente": "company_id",
    "posicion_general": "ranking",
    "cia_imvalores": "is_public",
    "id_estado_financiero": "statement_id",
    "tamaño_e": "company_size",
    "ingresos_ventas": "revenue",
    "ingresos_totales": "total_income",
    "utilidad_neta": "net_income",
    "utilidad_ejercicio": "operating_income",
    "utilidad_an_imp": "income_before_tax",
    "impuesto_renta": "income_tax",
    "activos": "total_assets",
    "pasivo": "total_liabilities",
    "patrimonio": "equity",
    "deuda_total": "total_debt",
    "deuda_total_c_plazo": "short_term_debt",
    "liquidez_corriente": "current_ratio",
    "prueba_acida": "quick_ratio",
    "end_activo": "asset_leverage",
    "end_patrimonial": "equity_leverage",
    "end_activo_fijo": "fixed_assets",
    "end_corto_plazo": "short_term_assets",
    "end_largo_plazo": "long_term_assets",
    "cobertura_interes": "interest_coverage",
    "apalancamiento": "leverage",
    "apalancamiento_financiero": "financial_leverage",
    "end_patrimonial_ct": "current_equity",
    "end_patrimonial_nct": "non_current_equity",
    "apalancamiento_c_l_plazo": "short_long_term_leverage",
    "rot_cartera": "receivables_turnover",
    "rot_activo_fijo": "fixed_assets_turnover",
    "rot_ventas": "asset_turnover",
    "per_med_cobranza": "avg_collection_period",
    "per_med_pago": "avg_payment_period",
    "impac_gasto_a_v": "sales_expense_impact",
    "impac_carga_finan": "financial_burden",
    "rent_neta_activo": "roe_assets",
    "rent_ope_patrimonio": "roe_equity",
    "rent_ope_activo": "roe_assets_calc",
    "roe": "roe",
    "roa": "roa",
    "margen_bruto": "gross_margin",
    "margen_operacional": "operating_margin",
    "rent_neta_ventas": "net_sales_margin",
    "fortaleza_patrimonial": "equity_strength",
    "gastos_financieros": "financial_expenses",
    "gastos_admin_ventas": "sgna_expenses",
    "depreciaciones": "depreciation",
    "amortizaciones": "amortization",
    "costos_ventas_prod": "cost_of_goods_sold",
    "total_gastos": "total_expenses",
    "n_empleados": "employee_count",
    "n": "n",
    "max": "max",
    "cod_segmento": "segment_code",
    "ciiu_n1": "industry_code_level1",
    "ciiu_n6": "industry_code_level6",
    "x": "x",
    "y": "y",
}

df = df.rename({k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns})
print("Columns renamed")

# -----------------------------
# 2. TYPE DEFINITIONS
# -----------------------------

INT_COLS = [
    "year",
    "company_id",
    "ranking",
    "is_public",
    "statement_id",
    "company_size",
    "employee_count",
    "segment_code",
    "industry_code_level1",
    "industry_code_level6",
    "n",
    "max",
]

FLOAT_COLS = [
    "revenue",
    "total_income",
    "net_income",
    "operating_income",
    "income_before_tax",
    "income_tax",
    "total_assets",
    "total_liabilities",
    "equity",
    "equity_strength",
    "total_debt",
    "short_term_debt",
    "fixed_assets",
    "short_term_assets",
    "long_term_assets",
    "current_equity",
    "non_current_equity",
    "current_ratio",
    "quick_ratio",
    "asset_leverage",
    "equity_leverage",
    "leverage",
    "financial_leverage",
    "short_long_term_leverage",
    "interest_coverage",
    "asset_turnover",
    "receivables_turnover",
    "fixed_assets_turnover",
    "avg_collection_period",
    "avg_payment_period",
    "sales_expense_impact",
    "financial_burden",
    "roe_assets",
    "roe_equity",
    "roe_assets_calc",
    "roe",
    "roa",
    "gross_margin",
    "operating_margin",
    "net_sales_margin",
    "financial_expenses",
    "sgna_expenses",
    "depreciation",
    "amortization",
    "cost_of_goods_sold",
    "total_expenses",
    "x",
    "y",
]

# -----------------------------
# 3. EUROPEAN DECIMAL HANDLING
# -----------------------------


def euro_float(col: str) -> pl.Expr:
    return (
        pl.when(pl.col(col).is_null())
        .then(None)
        .otherwise(
            pl.col(col)
            .str.replace(r"^,", "0,", literal=False)
            .str.replace(".", "", literal=True)
            .str.replace(",", ".", literal=True)
            .cast(pl.Float64, strict=True)
        )
    )


df = df.with_columns(
    [
        pl.col(c).cast(pl.Int64, strict=False).alias(c)
        for c in INT_COLS
        if c in df.columns
    ]
)

df = df.with_columns([euro_float(c).alias(c) for c in FLOAT_COLS if c in df.columns])

print("Type conversion completed")

# -----------------------------
# 4. SQLITE TABLE CREATION
# -----------------------------


def polars_to_sql(dtype):
    if dtype in (pl.Int64, pl.Int32):
        return "INTEGER"
    if dtype in (pl.Float64, pl.Float32):
        return "REAL"
    return "TEXT"


columns_sql = [f'"{name}" {polars_to_sql(dtype)}' for name, dtype in df.schema.items()]

create_sql = f"""
CREATE TABLE IF NOT EXISTS financial_records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    {", ".join(columns_sql)}
);
"""

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute(create_sql)
conn.commit()

print("SQLite table ready")

# -----------------------------
# 5. INSERT DATA
# -----------------------------

rows = df.to_dicts()
placeholders = ", ".join(["?"] * len(df.columns))
columns = ", ".join(f'"{c}"' for c in df.columns)

insert_sql = f"""
INSERT INTO financial_records ({columns})
VALUES ({placeholders})
"""

cursor.executemany(insert_sql, [tuple(r[c] for c in df.columns) for r in rows])
conn.commit()
conn.close()

print("All data inserted successfully")
