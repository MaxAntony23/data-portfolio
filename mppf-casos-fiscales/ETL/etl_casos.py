import pandas as pd
import numpy as np
from pathlib import Path
import sqlite3
from dateutil import parser

# -------------------- Paths --------------------
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed"
DB_DIR  = BASE_DIR / "db"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Columnas esperadas --------------------
COLUMN_MAP = {
    "periodo": "periodo",
    "anio": "anio",
    "fecha_descarga": "fecha_descarga",
    "distrito fiscal": "distrito_fiscal",
    "tipo fiscalia": "tipo_fiscalia",
    "materia": "materia",
    "especialidad": "especialidad",
    "tipo_caso": "tipo_caso",
    "especializada": "especializada",
    "ingresado": "ingresado",
    "atendido": "atendido",
    "ubigeo_pjfs": "ubigeo_pjfs",
    "dpto_pjfs": "dpto_pjfs",
    "prov_pjfs": "prov_pjfs",
    "dist_pjfs": "dist_pjfs",
    "fecha_corte": "fecha_corte",
}
COLUMNS_FINAL = list(COLUMN_MAP.values())

# -------------------- Meses en español -> número --------------------
MONTH_MAP = {
    "enero":1, "febrero":2, "marzo":3, "abril":4, "mayo":5, "junio":6,
    "julio":7, "agosto":8, "setiembre":9, "septiembre":9, "octubre":10,
    "noviembre":11, "diciembre":12
}

# -------------------- Helpers de fechas --------------------
def _parse_date_safe(x, dayfirst=True):
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        return x
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "nul", "null"}:
        return pd.NaT
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y"):
        try:
            return pd.to_datetime(s, format=fmt, errors="raise")
        except Exception:
            pass
    try:
        return pd.to_datetime(parser.parse(s, dayfirst=dayfirst))
    except Exception:
        return pd.NaT

def _month_from_periodo(periodo):
    if pd.isna(periodo):
        return np.nan
    s = str(periodo).strip().lower()
    if s.isdigit():
        m = int(s)
        return m if 1 <= m <= 12 else np.nan
    for name, num in MONTH_MAP.items():
        if name in s:
            return num
    tokens = [t for t in s.replace("_", "-").replace(" ", "-").split("-") if t]
    for t in tokens:
        if t.isdigit():
            val = int(t)
            if 1 <= val <= 12:
                return val
    return np.nan

# -------------------- Carga y limpieza base --------------------
def load_csv(file: Path) -> pd.DataFrame:
    df = pd.read_csv(file, encoding="utf-8", low_memory=False)
    rename = {c.strip().lower(): COLUMN_MAP[c.strip().lower()]
              for c in df.columns if c.strip().lower() in COLUMN_MAP}
    df = df.rename(columns=rename)
    for col in COLUMNS_FINAL:
        if col not in df.columns:
            df[col] = np.nan
    return df[COLUMNS_FINAL].copy()

def clean_types(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["ingresado", "atendido"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")
    df["fecha_corte"]    = df["fecha_corte"].apply(_parse_date_safe)
    df["fecha_descarga"] = df["fecha_descarga"].apply(_parse_date_safe)

    exclude = {"anio", "ingresado", "atendido", "fecha_corte", "fecha_descarga"}
    text_cols = [c for c in df.columns if c not in exclude]
    for c in text_cols:
        if not pd.api.types.is_datetime64_any_dtype(df[c]) and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = (df[c].astype("string")
                           .str.strip()
                           .replace({"nan": pd.NA, "None": pd.NA, "null": pd.NA}, regex=False))
            df[c] = df[c].str.title()

    for c in ["distrito_fiscal","tipo_fiscalia","materia","especialidad",
              "tipo_caso","especializada","ubigeo_pjfs","dpto_pjfs",
              "prov_pjfs","dist_pjfs","periodo"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    return df

# -------------------- Features y KPIs --------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["mes_from_periodo"] = df["periodo"].apply(_month_from_periodo)

    def _derive_fecha(row):
        if pd.notna(row["fecha_corte"]):
            return row["fecha_corte"]
        if pd.notna(row["fecha_descarga"]):
            return row["fecha_descarga"]
        anio = row["anio"]
        mes  = row["mes_from_periodo"]
        if pd.notna(anio) and not pd.isna(mes):
            try:
                return pd.Timestamp(year=int(anio), month=int(mes), day=1)
            except Exception:
                pass
        if pd.notna(anio):
            return pd.Timestamp(year=int(anio), month=12, day=31)
        return pd.NaT

    df["fecha_corte_final"] = df.apply(_derive_fecha, axis=1)

    df["tasa_atencion"] = np.where(df["ingresado"] > 0,
                                   df["atendido"] / df["ingresado"], np.nan)
    df["backlog"] = df["ingresado"] - df["atendido"]
    df["datos_invalidos"] = (df["atendido"] > df["ingresado"]).astype(int)

    esp = df["especializada"].astype("string").fillna("").str.strip().str.lower()
    NEGATIVOS = {"", "no", "0", "false", "f", "n", "sin", "ninguno",
                 "no especializada", "no-especializada"}
    df["es_especializado"] = np.where(~esp.isin(NEGATIVOS), 1, 0).astype(int)

    # Fechas para análisis temporal
    df["mes_final"]  = df["fecha_corte_final"].dt.month
    df["anio_final"] = df["fecha_corte_final"].dt.year
    df["anio_mes_date"] = pd.to_datetime(
        dict(year=df["anio_final"].fillna(df["anio"]).astype("Int64"),
             month=df["mes_final"].fillna(1).astype("Int64"),
             day=1),
        errors="coerce"
    )
    df["anio_mes_key"] = (df["anio_mes_date"].dt.year * 100 +
                          df["anio_mes_date"].dt.month).astype("Int64")

    # eliminar columnas duplicadas
    df.drop(columns=["fecha_corte", "anio_mes"], errors="ignore", inplace=True)
    df.drop(columns=["mes_from_periodo", "mes_final", "anio_final"], inplace=True)

    return df

# -------------------- Salida CSV + SQLite --------------------
def save_outputs(df: pd.DataFrame):
    csv_path = OUT_DIR / "casos_fiscales_2019_2023.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"✅ CSV limpio → {csv_path}")

    db_path = DB_DIR / "casos_fiscales.sqlite"
    with sqlite3.connect(db_path) as con:
        df.to_sql("casos_fiscales", con, if_exists="replace", index=False)
        con.execute("CREATE INDEX IF NOT EXISTS idx_anio ON casos_fiscales(anio);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_distrito ON casos_fiscales(distrito_fiscal);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_materia ON casos_fiscales(materia);")
        con.execute("CREATE INDEX IF NOT EXISTS idx_meskey ON casos_fiscales(anio_mes_key);")
    print(f"✅ SQLite → {db_path}")

    qc = {
        "registros": len(df),
        "nulls_fecha_corte_final": int(df["fecha_corte_final"].isna().sum()),
        "errores_tipo": int(df["datos_invalidos"].sum()),
        "rango_fechas": (str(df["fecha_corte_final"].min()), str(df["fecha_corte_final"].max()))
    }
    print("🔎 QC:", qc)

# -------------------- Main --------------------
def main():
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No hay CSV en {RAW_DIR}")
    frames = []
    for f in files:
        print(f"📥 Procesando {f.name}")
        df = load_csv(f)
        df = clean_types(df)
        df = add_features(df)
        df["fuente"] = f.name
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    save_outputs(full)

if __name__ == "__main__":
    main()