from __future__ import annotations
import pandas
import tesseract
import pytesseract
import pathlib
import sys
import numpy as np
import cv2
import re
import rapidfuzz
import os

from typing import List, Dict, Tuple, Optional, Any
from tesseract import Output
from rapidfuzz import fuzz
from pathlib import Path

#Execution of Tesseract
TESSERACT_EXE = r"C:\Users\pedro.gomes\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
TESSDATA_DIR  = r"C:\Users\pedro.gomes\AppData\Local\Programs\Tesseract-OCR\tessdata"

#Destination Dict of Code
ENCARTE_DIR   = Path(r"C:\Users\pedro.gomes\Desktop\Encartes\Encartes-M")
DESTINO_XLSX  = Path(r"C:\Users\pedro.gomes\Desktop\Encartes\Leitura_Encartes.xlsx")

#Regex Configuration of Prices
PRICE_RE   = re.compile(r'(?:R\s*\$|\$)?\s*([0-9]{1,3}(?:[\.\s][0-9]{3})*(?:[,\.][0-9]{2})|[0-9]+[,\.][0-9]{2})', re.I)
CENT2_RE   = re.compile(r'^\d{2}$')    # 2 dígitos (centavos)
INT_RE     = re.compile(r'^\d{1,4}$')  # 1–4 dígitos (inteiro do preço)
APP_HINTRE = re.compile(r'\bapp\b', re.I)

#Execution of tesseract (cmd)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
os.environ("TESSERACT_PREFIX") = TESSDATA_DIR

if not Path(TESSERACT_EXE).exists():
    sys.stderr.write(f"[ERRO] Tesseract não encontrado em {TESSERACT_EXE}\n")
    sys.exit(1)

#Meta of the image (data of image)
META = List(dict) = {
    "Empresa": {empresa},
    "Data Validade": {data_validade}, 
    "Data Inicio": {data_inicio},
    "Data Fim": {data_fim},
    "Campanha": {campanha},
    "Nome da Campanha": {nome_campanha},
    "Dia (Nome da Campanha, Estado e Cidade)": {dia_campanha},
    "App (Preço App)": {app},
    "Cidade": {cidade}
}