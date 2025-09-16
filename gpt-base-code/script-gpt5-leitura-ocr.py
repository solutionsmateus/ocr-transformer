import os, re, json, glob
from pathlib import Path

import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import pandas as pd
from rapidfuzz import process

# --- ajuste para Windows ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\pedro.gomes\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# =========================
# Configurações
# =========================
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PDF_EXTS = {".pdf"}

PT_OCR = "--psm 6 -l por"
NUM_OCR = "--psm 6 -l por -c tessedit_char_whitelist=0123456789,.$R$ "

PRICE_RE = re.compile(r"R\$\s*\d{1,3}(?:\.\d{3})*,\d{2}")
STOPWORDS = ["validade", "a partir", "apartir", "cada", "oferta", "promoção", "congel"]

# Vocabulário base para correção fuzzy
VOCAB_BASE = [
    "Arroz", "Feijão", "Leite", "Café", "Óleo", "Açúcar", "Sabão", "Linguiça",
    "Presuntado", "Queijo", "Nescau", "Aurora", "Friboi", "Nestlé", "Italac",
    "Mussarela", "Macarrão", "Farinha", "Sadia", "Biscoito", "Amaciante", "Detergente", "Bovina"
]

# =========================
# Utils
# =========================
def load_pages(path: Path):
    ext = path.suffix.lower()
    if ext in IMG_EXTS:
        return [Image.open(path).convert("RGB")]
    elif ext in PDF_EXTS:
        images = []
        with fitz.open(str(path)) as doc:
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
        return images
    else:
        raise ValueError(f"Formato não suportado: {ext}")

def iter_inputs(in_path: Path):
    if in_path.is_file():
        yield in_path
    else:
        for ext in list(IMG_EXTS) + list(PDF_EXTS):
            yield from (Path(p) for p in glob.glob(str(in_path / f"**/*{ext}"), recursive=True))

def ocr_text(pil_img, config=PT_OCR):
    return pytesseract.image_to_string(pil_img, config=config)

def ocr_digits(pil_img):
    return pytesseract.image_to_string(pil_img, config=NUM_OCR)

# =========================
# Limpeza e Correção
# =========================
def clean_price(text):
    txt = text.replace("O", "0").replace("o", "0")
    txt = txt.replace("S", "5").replace("$", "R$")
    m = PRICE_RE.findall(txt.replace("\n", " "))
    if m:
        return m[0]
    num = re.findall(r"\d+[,.]\d{2}", txt)
    if num:
        return "R$ " + num[0].replace(".", ",")
    return ""

def clean_product(text):
    t = re.sub(r"\s+", " ", text).strip()
    t = PRICE_RE.sub("", t)
    t = re.sub(r"(a partir.*?un\.?|cada\s*\d+)", "", t, flags=re.I)
    t = re.sub(r"[^A-Za-zÀ-ú0-9\s,\.]", " ", t)
    t = " ".join([p for p in t.split() if len(p) > 2])
    return t.title()[:140]

def is_lixo(texto):
    t = texto.lower()
    return any(sw in t for sw in STOPWORDS)

def corrigir_produto(produto):
    palavras = produto.split()
    corrigidas = []
    for p in palavras:
        match = process.extractOne(p, VOCAB_BASE)
        if match and match[1] > 80:
            corrigidas.append(match[0])
        else:
            corrigidas.append(p)
    return " ".join(corrigidas)

# =========================
# OCR de página
# =========================
def detect_price_boxes(cv_img):
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 80, 80]); upper1 = np.array([12, 255, 255])
    lower2 = np.array([170, 80, 80]); upper2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if w*h < 3000: continue
        ar = w/float(h)
        if 1.1 < ar < 5.0 and h > 40:
            boxes.append((x,y,w,h))
    boxes.sort(key=lambda b: (b[1]//100, b[0]))
    return boxes

def crop_pil(cv_img, box, pad=0):
    x,y,w,h = box
    roi = cv_img[max(y-pad,0):y+h+pad, max(x-pad,0):x+w+pad]
    return Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

def product_region(cv_img, box, up_pixels=120, margin=10):
    x,y,w,h = box
    y0 = max(y-up_pixels,0)
    y1 = max(y-margin,0)
    roi = cv_img[y0:y1, x:x+w]
    if roi.size != 0:
        roi = cv2.resize(roi, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

def parse_page(pil_img, empresa, cidade, campanha="Ofertas", data=""):
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    items = []
    for box in detect_price_boxes(cv_img):
        preco = clean_price(ocr_digits(crop_pil(cv_img, box, pad=6)))
        produto = clean_product(ocr_text(product_region(cv_img, box)))
        produto = corrigir_produto(produto)
        if not produto or is_lixo(produto):
            continue
        items.append({
            "Empresa": empresa,
            "Data": data,
            "Campanha": campanha,
            "Produto": produto,
            "Preço (Do Encarte)": preco,
            "App": "",
            "Cidade": cidade
        })
    return items

# =========================
# Loop principal
# =========================
def process_folder(input_dir: str, output_dir: str, empresa: str, cidade: str):
    in_dir, out_dir = Path(input_dir), Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arquivos = list(iter_inputs(in_dir))
    print(f"{len(arquivos)} arquivos encontrados.")

    all_rows = []
    for idx, fpath in enumerate(arquivos, start=1):
        print(f"[{idx}/{len(arquivos)}] {fpath}")
        for i, pg in enumerate(load_pages(fpath), start=1):
            rows = parse_page(pg, empresa, cidade)
            all_rows.extend(rows)
            with open(out_dir / f"{Path(fpath).stem}_p{i}.json", "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)

    if all_rows:
        df = pd.DataFrame(all_rows, columns=["Empresa","Data","Campanha","Produto","Preço (Do Encarte)","App","Cidade"])
        xlsx_out = out_dir / "encartes_consolidado.xlsx"
        df.to_excel(xlsx_out, index=False)
        print(f"Extração concluída: {len(df)} itens")
        print(f"Excel salvo em: {xlsx_out}")
    else:
        print("Nenhum item encontrado.")

# =========================
# Exemplo de uso
# =========================
if __name__ == "__main__":
    process_folder(
        input_dir=r"C:\Users\pedro.gomes\Desktop\Encartes\Encartes-A",   # pasta com encartes
        output_dir=r"C:\Users\pedro.gomes\Desktop\Encartes",    # saída
        empresa="Nome do Supermercado",
        cidade="Nome da cidade"
    )
