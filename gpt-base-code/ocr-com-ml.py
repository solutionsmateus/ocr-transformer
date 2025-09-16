
"""
• Busca recursiva por imagens no diretório fixo (suporte a acentos/OneDrive)
• Pré-processamento multi-variante + OCR multi-PSM (melhora recall)
• Detecção de preços:
    - Regex padrão (R$ 9,99 / 9,99 / 9.99)
    - Reconstrução quando inteiros e centavos vêm separados (ex.: "4" + "09")
• Associação descrição ↔ preço considerando linhas acima/abaixo
• Flag de preço App se “APP” estiver próximo ao preço (na linha ou vizinhança)
• Deduplicação por similaridade de nome + mesmo preço
• Exporta XLSX com colunas no padrão

Ajuste os blocos CONFIG e META conforme necessário.
"""

from __future__ import annotations
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
import pytesseract
from pytesseract import Output

try:
    from rapidfuzz import fuzz
except ImportError:
    fuzz = None


# ========== CONFIG ==========
TESSERACT_EXE = r"C:\Users\pedro.gomes\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
TESSDATA_DIR  = r"C:\Users\pedro.gomes\AppData\Local\Programs\Tesseract-OCR\tessdata"

ENCARTE_DIR   = Path(r"C:\Users\pedro.gomes\Desktop\Encartes\Encartes-M")
DESTINO_XLSX  = Path(r"C:\Users\pedro.gomes\Desktop\Encartes\Leitura_Encartes.xlsx")

META: Dict[str, str] = {
    "empresa": "Assaí Atacadista",
    "estado": "Pernambuco",
    "cidade": "Recife",
    "data_publicacao": "18/08/2025",
    "validade": "18/08/2025 – 22/08/2025",
    "campanha": "Especial do Comerciante",
    "lang": "por",
}

# ========== INIT ==========
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR
if not Path(TESSERACT_EXE).exists():
    sys.stderr.write(f"[ERRO] Tesseract não encontrado em {TESSERACT_EXE}\n")
    sys.exit(1)

# ========== REGEX ==========
PRICE_RE   = re.compile(r'(?:R\s*\$|\$)?\s*([0-9]{1,3}(?:[\.\s][0-9]{3})*(?:[,\.][0-9]{2})|[0-9]+[,\.][0-9]{2})', re.I)
CENT2_RE   = re.compile(r'^\d{2}$')    # 2 dígitos (centavos)
INT_RE     = re.compile(r'^\d{1,4}$')  # 1–4 dígitos (inteiro do preço)
APP_HINTRE = re.compile(r'\bapp\b', re.I)


# ========== IO / OCR HELPERS ==========
def ensure_dir(p: Path) -> None:
    if not p.exists():
        sys.stderr.write(f"[ERRO] Diretório não encontrado: {p}\n")
        sys.exit(2)

def imread_u(path: Path) -> Optional[np.ndarray]:
    """Leitura robusta (caminhos com acentos/OneDrive)."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except Exception:
        return None

def upscale(img: np.ndarray, scale: float = 1.6) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)

def preprocess_variants(img_bgr: np.ndarray) -> List[np.ndarray]:
    """Gera variantes para aumentar o recall do OCR."""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)

    bin1  = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 15)
    bin2  = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 35, 10)
    inv1  = cv2.bitwise_not(bin1)
    up1   = upscale(bin1, 1.6)
    up2   = upscale(inv1, 1.6)

    return [bin1, bin2, inv1, up1, up2]

def ocr_words_multi(img_binaries: List[np.ndarray], lang: str = "por") -> pd.DataFrame:
    """Roda OCR em múltiplas variantes/PSMs e une resultados (palavras com bbox)."""
    cfgs = ["--oem 3 --psm 6", "--oem 3 --psm 11"]  # psm 6: blocks, psm 11: sparse
    bags: List[pd.DataFrame] = []

    for im in img_binaries:
        for cfg in cfgs:
            df = pytesseract.image_to_data(im, lang=lang, config=cfg, output_type=Output.DATAFRAME)
            if df is None or df.empty:
                continue
            df = df.dropna(subset=["text"])
            df["text"] = df["text"].astype(str).str.strip()
            df = df[df["text"] != ""]
            bags.append(df[["text","left","top","width","height"]])

    if not bags:
        return pd.DataFrame(columns=["text","left","top","width","height"])

    allw = pd.concat(bags, ignore_index=True)
    # Dedup approximate: arredonda coords para reduzir overlap
    allw["k"] = (
        allw["text"].str.lower().str.replace(r"\s+", " ", regex=True) + "|" +
        (allw["left"]//2).astype(str) + "|" + (allw["top"]//2).astype(str)
    )
    allw = allw.drop_duplicates(subset=["k"]).drop(columns=["k"]).reset_index(drop=True)
    return allw

def to_lines(words: pd.DataFrame, y_tol: int = 8) -> pd.DataFrame:
    """Agrupa palavras em linhas (aproximação) pelo eixo Y."""
    if words.empty:
        return pd.DataFrame(columns=["line_id","text","x1","y1","x2","y2"])

    df = words.sort_values(["top","left"]).reset_index(drop=True)
    lines, buf, cy, lid = [], [], None, 0
    for r in df.itertuples(index=False):
        top = int(r.top); left = int(r.left); width = int(r.width); height = int(r.height)
        if cy is None or abs(top - cy) > y_tol:
            if buf:
                lines.append((lid, buf)); lid += 1
            buf = [(r.text, left, top, width, height)]
            cy  = top
        else:
            buf.append((r.text, left, top, width, height))
    if buf: lines.append((lid, buf))

    out = []
    for lid, ws in lines:
        xs = [w[1] for w in ws]; ys = [w[2] for w in ws]
        ws_ = [w[3] for w in ws]; hs = [w[4] for w in ws]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max([x + w for x, w in zip(xs, ws_)]), max([y + h for y, h in zip(ys, hs)])
        text = " ".join(str(w[0]) for w in ws)
        out.append((lid, text, x1, y1, x2, y2))
    return pd.DataFrame(out, columns=["line_id","text","x1","y1","x2","y2"])


# ========== PRICE DETECTION ==========
def parse_float_brl(s: str) -> Optional[float]:
    s = s.strip().replace(" ", "")
    if not s:
        return None
    # remove milhares e normaliza decimal para "."
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def prices_from_lines(lines: pd.DataFrame) -> List[Dict[str, Any]]:
    """Preços via regex nas linhas completas."""
    out: List[Dict[str, Any]] = []
    for r in lines.itertuples(index=False):
        txt = r.text.replace("O", "0")  # corrige O↔0
        for m in PRICE_RE.finditer(txt):
            val = parse_float_brl(m.group(1))
            out.append({
                "line_id": r.line_id, "text": r.text,
                "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2,
                "price_value": val, "app_flag": bool(APP_HINTRE.search(r.text))
            })
    return out

def prices_from_words(words: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Reconstrói preços quando inteiros e centavos aparecem separados (ex.: "4" "09").
    Procura par (int: 1–4 dígitos) seguido de (cent: 2 dígitos) na mesma linha (y próximo) e perto no x.
    """
    out: List[Dict[str, Any]] = []
    if words.empty: return out

    # Ordena por Y depois X
    w = words.sort_values(["top","left"]).reset_index(drop=True)
    wn = w.to_dict("records")

    for i, a in enumerate(wn):
        atxt = str(a["text"]).strip()
        if not INT_RE.match(atxt):  # inteiro candidato
            continue
        ay, ah = int(a["top"]), int(a["height"])
        ax1, ax2 = int(a["left"]), int(a["left"]) + int(a["width"])

        # Busca próximo token à direita com 2 dígitos (centavos)
        for j in range(i+1, min(i+8, len(wn))):  # janela pequena
            b = wn[j]
            btxt = str(b["text"]).strip()
            if not CENT2_RE.match(btxt):
                continue
            by, bh = int(b["top"]), int(b["height"])
            bx1, bx2 = int(b["left"]), int(b["left"]) + int(b["width"])

            # Alinhamento vertical aproximado (mesma "linha")
            same_line = abs((ay + ah//2) - (by + bh//2)) <= max(ah, bh, 20)
            # Proximidade horizontal razoável
            gap_x = bx1 - ax2
            close_x = (0 <= gap_x <= 120)

            if same_line and close_x:
                price = f"{atxt},{btxt}"
                val = parse_float_brl(price)
                if val is None:
                    continue
                x1, y1 = min(ax1, bx1), min(ay, by)
                x2, y2 = max(ax2, bx2), max(ay+ah, by+bh)
                # Busca "APP" por perto (janela ao redor do preço)
                app_flag = False
                win_x1, win_y1 = x1 - 80, y1 - 40
                win_x2, win_y2 = x2 + 80, y2 + 40
                for k in range(max(0, i-6), min(len(wn), j+6)):
                    t = str(wn[k]["text"])
                    if APP_HINTRE.search(t):
                        lx, ly = int(wn[k]["left"]), int(wn[k]["top"])
                        if (win_x1 <= lx <= win_x2) and (win_y1 <= ly <= win_y2):
                            app_flag = True
                            break

                out.append({
                    "line_id": -1,  # não vem de uma "linha" consolidada
                    "text": f"{atxt} {btxt}",
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "price_value": val, "app_flag": app_flag
                })
                break  # achou um par para este inteiro, segue

    return out

def merge_price_candidates(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> pd.DataFrame:
    """Une listas de preços removendo duplicatas aproximadas por centroide e valor."""
    allc = (a or []) + (b or [])
    if not allc:
        return pd.DataFrame(columns=["x1","y1","x2","y2","price_value","app_flag","text","line_id"])

    df = pd.DataFrame(allc)
    # Centro aproximado arredondado para dedup
    cx = ((df["x1"] + df["x2"]) // 2) // 10
    cy = ((df["y1"] + df["y2"]) // 2) // 10
    pv = df["price_value"].round(2)
    df["key"] = cx.astype(str) + "|" + cy.astype(str) + "|" + pv.astype(str)
    df = df.sort_values(["price_value"]).drop_duplicates(subset=["key"]).drop(columns=["key"]).reset_index(drop=True)
    return df


# ========== LINK DESC <-> PRICE ==========
def score_link(pr: Any, row: Any, max_hgap: int = 500) -> float:
    """Menor é melhor (distância vertical + penalidade horizontal)."""
    if pr.x1 > row.x2:
        dx = pr.x1 - row.x2
    elif row.x1 > pr.x2:
        dx = row.x1 - pr.x2
    else:
        dx = 0
    dy = abs((pr.y1 + pr.y2)//2 - (row.y1 + row.y2)//2)
    return dy + min(dx, max_hgap) * 0.35

def link_desc(lines: pd.DataFrame, prices: pd.DataFrame) -> List[Dict[str, str]]:
    """Associa cada preço à descrição mais plausível (acima/abaixo próximo)."""
    if prices.empty:
        return []
    items: List[Dict[str, str]] = []

    for pr in prices.itertuples(index=False):
        # Candidatos acima/abaixo dentro de uma janela vertical
        window = 190
        cands = lines[(lines.y1 >= pr.y1 - window) & (lines.y2 <= pr.y2 + window)]

        if not cands.empty:
            best = min(cands.itertuples(index=False, name="Row"), key=lambda r: score_link(pr, r))
            desc = best.text
        else:
            desc = pr.text  # fallback

        # Limpeza e formatação
        desc = PRICE_RE.sub("", desc).strip(" -–—|:;")
        preco = "" if pd.isna(pr.price_value) else f"{float(pr.price_value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        app   = f"R$ {preco}" if getattr(pr, "app_flag", False) and preco else ""

        items.append({
            "Produto (Descrição)": desc,
            "Preço (Do Encarte)": preco,
            "App": app
        })
    return items


# ========== PÓS ==========
def dedup_items(items: List[Dict[str, str]], thresh: int = 92) -> List[Dict[str, str]]:
    """Remove duplicados por similaridade de nome + mesmo preço."""
    if not items:
        return items
    out: List[Dict[str, str]] = []
    for it in items:
        name = it["Produto (Descrição)"].lower().strip()
        price = it["Preço (Do Encarte)"]
        dup = False
        for ex in out:
            if ex["Preço (Do Encarte)"] != price:
                continue
            if fuzz:
                if fuzz.token_set_ratio(ex["Produto (Descrição)"].lower(), name) >= thresh:
                    dup = True; break
            else:
                if ex["Produto (Descrição)"].lower() == name:
                    dup = True; break
        if not dup:
            out.append(it)
    return out

def to_dataframe(items: List[Dict[str, str]], meta: Dict[str, str]) -> pd.DataFrame:
    cols = ["Empresa","Estado","Cidade","Data da Publicação","Validade","Campanha",
            "Produto (Descrição)","Preço (Do Encarte)","App"]
    rows = [
        [meta.get("empresa",""), meta.get("estado",""), meta.get("cidade",""),
         meta.get("data_publicacao",""), meta.get("validade",""), meta.get("campanha",""),
         it.get("Produto (Descrição)",""), it.get("Preço (Do Encarte)",""), it.get("App","")]
        for it in items
    ]
    return pd.DataFrame(rows, columns=cols)

def export_xlsx(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as wr:
        df.to_excel(wr, index=False, sheet_name="ENCARTE")
        ws = wr.sheets["ENCARTE"]
        for i, col in enumerate(df.columns):
            width = max(12, min(60, int(df[col].astype(str).map(len).max()) + 2))
            ws.set_column(i, i, width)


# ========== MAIN ==========
def main() -> None:
    ensure_dir(ENCARTE_DIR)

    patterns = ("*.jpg","*.jpeg","*.png","*.webp","*.tif","*.tiff",
                "*.JPG","*.JPEG","*.PNG","*.WEBP","*.TIF","*.TIFF")
    files: List[Path] = []
    for pat in patterns:
        files.extend(ENCARTE_DIR.rglob(pat))
    files = sorted(set(files))

    if not files:
        sys.stderr.write(f"[ERRO] Nenhuma imagem encontrada em: {ENCARTE_DIR}\n")
        sys.exit(3)

    print(f"[INFO] {len(files)} arquivo(s) de imagem encontrado(s).")

    all_items: List[Dict[str, str]] = []
    skipped = 0

    for p in files:
        img = imread_u(p)
        if img is None:
            skipped += 1
            continue

        variants = preprocess_variants(img)
        words = ocr_words_multi(variants, META.get("lang","por"))
        if words.empty:
            continue

        lines  = to_lines(words)
        # Coleta preços por duas estratégias e une
        p1 = prices_from_lines(lines)
        p2 = prices_from_words(words)
        prices = merge_price_candidates(p1, p2)

        # Associa descrição↔preço
        items = link_desc(lines, prices)
        all_items.extend(items)

    if skipped:
        print(f"[INFO] Imagens não lidas: {skipped}")

    # Dedup + Planilha
    all_items = dedup_items(all_items)
    df = to_dataframe(all_items, META)
    export_xlsx(df, DESTINO_XLSX)
    print(f"[OK] Salvo: {DESTINO_XLSX} | Linhas: {len(df)}")


if __name__ == "__main__":
    main()
