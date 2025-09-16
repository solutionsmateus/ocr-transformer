import cv2, numpy as np, pytesseract, re, pandas as pd
from pathlib import Path

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\pedro.gomes\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

RX_PRECO = re.compile(r'(?:R?\$?\s*)?(\d{1,3}(?:[.,]\d{2}))')  # 12,90 | R$ 12,90 | 129,90
def norm_preco(txt):
    m = RX_PRECO.search(txt.replace('O','0').replace('S','$'))
    if not m: return None
    valor = m.group(1).replace('.',',')  # força vírgula
    return f'R$ {valor}' if 'R' not in txt and '$' not in txt else f'R$ {valor}'

def ocr(img, conf="--oem 3 --psm 6 -l por"):
    return pytesseract.image_to_string(img, config=conf)

def ocr_num(img):
    cfg = "--oem 3 --psm 7 -l por -c tessedit_char_whitelist=0123456789R$,."
    return pytesseract.image_to_string(img, config=cfg)


def detectar_precos_bboxes(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    low1, high1 = (0,120,70), (10,255,255)
    low2, high2 = (170,120,70), (180,255,255)
    mask = cv2.inRange(hsv, np.array(low1), np.array(high1)) | cv2.inRange(hsv, np.array(low2), np.array(high2))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H,W = bgr.shape[:2]
    boxes=[]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = w*h
        ar = w/float(h)
        if area<1500 or ar<1.2 or h<20: 
            continue
        if y>H*0.92 or y< H*0.05: 
            continue
        boxes.append((x,y,w,h))

    boxes = sorted(boxes, key=lambda b:(b[1]//40, b[0]))
    return boxes


def crop_descricao(bgr, price_box, pad_y=10, altura_rel=1.6):
    x,y,w,h = price_box
    H,W = bgr.shape[:2]
    top = max(0, int(y - altura_rel*h) - pad_y)
    bot = max(0, y - 2) 
    left = max(0, x - 10)
    right = min(W, x + w + 10)
    return bgr[top:bot, left:right]

def preprocess(bgr):
    bgr = cv2.resize(bgr, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)
    return bgr, eq

def extrair_itens_encarte(
        caminho_img,
        empresa, data_inicio, data_fim, campanha, cidade, estado,
        categoria_default="Perecíveis"):
    
    bgr = cv2.imread(str(caminho_img))
    assert bgr is not None, f"Não abri a imagem: {caminho_img}"
    bgr, eq = preprocess(bgr)

    price_boxes = detectar_precos_bboxes(bgr)

    itens = []
    for bx in price_boxes:
        x,y,w,h = bx
        price_roi = eq[y:y+h, x:x+w]
        preco_raw = ocr_num(price_roi)
        preco = norm_preco(preco_raw)
        if not preco:
            ker = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            price_roi2 = cv2.morphologyEx(price_roi, cv2.MORPH_DILATE, ker, iterations=1)
            preco = norm_preco(ocr_num(price_roi2))
        if not preco:
            continue 
        desc_roi = crop_descricao(bgr, bx)
        if desc_roi.size == 0:
            continue
      
        desc_txt = ocr(desc_roi, conf="--oem 3 --psm 6 -l por")
        linhas = [l.strip(" .,:;|") for l in desc_txt.splitlines() if l.strip()]
        boas = []
        for l in linhas:
            if len(l) < 3: 
                continue
            if RX_PRECO.search(l):  
                continue
            if l.lower() in {"quilo","cada","kg","pc","pct","pc/kg","pc/pc"}:
                continue
            boas.append(l)
        if not boas:
            continue
        descricao = " ".join(boas[-2:]) if len(boas)>=2 else boas[-1]
        descricao = re.sub(r'\s{2,}',' ',descricao)
        descricao = descricao.replace("RESFRVA","RESERVA").replace("RESFRIADO","RESFRIADO")

        itens.append({
            "Empresa (Nome da Empresa)": empresa,
            "Data": f"{data_inicio} - {data_fim}",
            "Data Inicio": data_inicio,
            "Data Fim": data_fim,
            "Campanha": campanha,  # ex.: "4ª e 5ª FeirassAí - QUI/QUI - PI"
            "Categoria do Produto": categoria_default,
            "Produto (Descrição)": descricao,
            "Preço (Do Encarte)": preco,
            "App (Preço para Usuários do App)": "",
            "Cidade": cidade,
            "Estado": estado
        })

    df = pd.DataFrame(itens).drop_duplicates(subset=["Produto (Descrição)","Preço (Do Encarte)"])
    df = df[df["Produto (Descrição)"].str.split().str.len().fillna(0) >= 2].reset_index(drop=True)
    return df

def salvar_xlsx(df, saida_path):
    saida_path = Path(saida_path)
    saida_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(saida_path, index=False)
    return saida_path


if __name__ == "__main__":
    img = r"C:\Users\pedro.gomes\Desktop\Encartes\Encartes-M\campanha-37534-cluster-693-pagina-2.jpeg"  # caminho da sua imagem
    df = extrair_itens_encarte(
        caminho_img=img,
        empresa="Assaí Atacadista",
        data_inicio="03/09/2025",
        data_fim="04/09/2025",
        campanha="Perecíveis - QUI/SEX - PI",
        cidade="Teresina",
        estado="PI",
        categoria_default="Perecíveis"
    )
    print(df.head(10))
    salvar_xlsx(df, "saida/ASSAI_Teresina_Pereciveis_2025-09-03_04.xlsx")
