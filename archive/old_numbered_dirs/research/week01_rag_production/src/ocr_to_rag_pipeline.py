from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Protocol

import cv2
import numpy as np
from PIL import Image
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from sqlalchemy import create_engine, text
from tqdm import tqdm

from src.chunking.tokenizer import build_counter


@dataclass
class Settings:
    input_dir: str = "data/pages_images"
    book_id: str = "bidaya_nihaya"

    ocr_engine: str = "surya"
    ocr_langs: List[str] = field(default_factory=lambda: ["ar"])
    ocr_pass2_conf_threshold: float = 0.90
    max_pages: Optional[int] = None
    bbox_normalized: bool = False

    dpi: int = 400
    do_deskew: bool = True
    do_binarize: bool = True
    do_denoise: bool = True
    crop_margins: bool = True

    pages_per_parent: int = 8
    child_max_tokens: int = 550
    child_overlap_tokens: int = 100
    child_min_tokens: int = 120

    pg_dsn: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/books"

    es_url: str = "http://localhost:9200"
    es_index_pages_raw: str = "pages_raw"
    es_index_children_norm: str = "children_norm"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_children: str = "children_vectors"
    qdrant_collection_summaries: str = "summaries_vectors"

    embedding_dim: int = 1024
@dataclass
class OcrLine:
    text: str
    bbox: List[float]
    conf: float
    line_no: int


@dataclass
class OcrPageResult:
    lines: List[OcrLine]
    page_conf_mean: float
    raw_engine_output: dict | None = None


class OcrEngine(Protocol):
    def ocr_image(self, image: Image.Image, langs: List[str]) -> OcrPageResult: ...


class SuryaEngine:
    def __init__(self):
        from surya.ocr import run_ocr  # type: ignore

        self._run_ocr = run_ocr

    def ocr_image(self, image: Image.Image, langs: List[str]) -> OcrPageResult:
        res = self._run_ocr([image], lang=langs)
        page = res[0]
        blocks = getattr(page, "blocks", None) or getattr(page, "lines", None) or []
        lines: List[OcrLine] = []
        confs = []
        for i, b in enumerate(blocks):
            text = getattr(b, "text", "") or ""
            bbox = list(getattr(b, "bbox", [0, 0, 0, 0]))
            conf = float(getattr(b, "confidence", getattr(b, "conf", 0.0)) or 0.0)
            lines.append(OcrLine(text=text, bbox=bbox, conf=conf, line_no=i))
            confs.append(conf)
        page_conf = sum(confs) / len(confs) if confs else 0.0
        return OcrPageResult(lines=lines, page_conf_mean=page_conf, raw_engine_output=None)


def build_ocr_engine(name: str) -> OcrEngine:
    if name == "surya":
        return SuryaEngine()
    raise ValueError(f"Unknown OCR engine: {name}")
# Preprocess helpers

def pil_to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_pil(arr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def deskew(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray < 200))
    if coords.size == 0:
        return image_bgr
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image_bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def denoise(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image_bgr, None, 7, 7, 7, 21)


def binarize(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15)
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)


def crop_margins(image_bgr: np.ndarray, pad: int = 10) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(bw)
    if coords is None:
        return image_bgr
    x, y, w, h = cv2.boundingRect(coords)
    x = max(x - pad, 0)
    y = max(y - pad, 0)
    return image_bgr[y : y + h + 2 * pad, x : x + w + 2 * pad]


def preprocess_image(img: Image.Image, do_deskew: bool, do_denoise: bool, do_binarize: bool, do_crop: bool) -> Image.Image:
    bgr = pil_to_cv(img)
    if do_crop:
        bgr = crop_margins(bgr)
    if do_deskew:
        bgr = deskew(bgr)
    if do_denoise:
        bgr = denoise(bgr)
    if do_binarize:
        bgr = binarize(bgr)
    return cv_to_pil(bgr)
# Quality
import re

ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")
BAD_CHARS = re.compile(r"[^\u0600-\u06FF0-9\s\.\,\;\:\!\؟\-\(\)\[\]«»ـ]+")


def garbage_ratio(text: str) -> float:
    if not text:
        return 1.0
    bad = len(BAD_CHARS.findall(text))
    return bad / max(len(text), 1)


def page_quality_score(page_conf_mean: float, text: str) -> float:
    g = garbage_ratio(text)
    return (0.7 * page_conf_mean) + (0.3 * max(0.0, 1.0 - min(g * 10, 1.0)))
# Postprocess
import regex as re

ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\uFEFF]")
DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")
TATWEEL = re.compile(r"ـ+")
MULTI_SPACE = re.compile(r"[ \t]{2,}")
SPACE_BEFORE_PUNCT = re.compile(r"\s+([،؛:!؟\.\)\]\}»])")
SPACE_AFTER_OPEN = re.compile(r"([\(\[\{«])\s+")
LATIN_PUNCT = [
    (re.compile(r"\?"), "؟"),
    (re.compile(r";"), "؛"),
    (re.compile(r","), "،"),
]
FOOTNOTE_REF = re.compile(r"[\(（]\s*(\d{1,3})\s*[\)）]")
FOOTNOTE_REF2 = re.compile(r"\[\s*(\d{1,3})\s*\]")
STRONG_END = re.compile(r"[\.؟!؛:]$|[»\)\]]$")
HEADING_START = re.compile(r"^(باب|فصل|ذكر|وفي سنة|ثم دخلت سنة|قال|وروى|وفي الصحيح)\b")
ENUM_START = re.compile(r"^\(?\d+\)?\s*[-–—]?\s*")


@dataclass
class ProcessedPage:
    raw_fixed: str
    norm_search: str
    paragraphs: List[str]
    line_map: List[dict]


def clean_line_text(s: str) -> str:
    s = ZERO_WIDTH.sub("", s)
    for rgx, repl in LATIN_PUNCT:
        s = rgx.sub(repl, s)
    s = MULTI_SPACE.sub(" ", s).strip()
    s = SPACE_BEFORE_PUNCT.sub(r"\1", s)
    s = SPACE_AFTER_OPEN.sub(r"\1", s)
    return s


def vertical_gap(b1: List[float], b2: List[float]) -> float:
    return max(0.0, float(b2[1]) - float(b1[3]))


def merge_lines_to_paragraphs(lines: List[OcrLine], bbox_normalized: bool = False) -> tuple[List[str], List[List[int]]]:
    if not lines:
        return [], []
    cleaned = [(i, clean_line_text(l.text), l.bbox) for i, l in enumerate(lines) if l.text and l.text.strip()]
    if not cleaned:
        return [], []
    heights = [max(1.0, float(b[3]) - float(b[1])) for _, _, b in cleaned]
    heights_sorted = sorted(heights)
    med_h = heights_sorted[len(heights_sorted) // 2]
    paras: List[str] = []
    maps: List[List[int]] = []
    cur_text, cur_map = cleaned[0][1], [cleaned[0][0]]
    prev_bbox, prev_text = cleaned[0][2], cleaned[0][1]
    for idx, text, bbox in cleaned[1:]:
        gap = vertical_gap(prev_bbox, bbox)
        starts_heading = bool(HEADING_START.match(text)) and len(text) < 140
        starts_enum = bool(ENUM_START.match(text))
        can_merge = (not STRONG_END.search(prev_text)) and (not starts_heading) and (not starts_enum)
        if not bbox_normalized:
            can_merge = can_merge and (gap < 0.35 * med_h)
        if can_merge:
            cur_text = cur_text + " " + text
            cur_map.append(idx)
        else:
            paras.append(cur_text.strip())
            maps.append(cur_map)
            cur_text, cur_map = text, [idx]
        prev_bbox, prev_text = bbox, text
    paras.append(cur_text.strip())
    maps.append(cur_map)
    return paras, maps


def normalize_for_search(text: str) -> str:
    t = clean_line_text(text)
    t = FOOTNOTE_REF.sub(r"〔\1〕", t)
    t = FOOTNOTE_REF2.sub(r"〔\1〕", t)
    t = DIACRITICS.sub("", t)
    t = re.sub(r"[إأآٱ]", "ا", t)
    t = t.replace("ى", "ي")
    t = TATWEEL.sub("", t)
    t = MULTI_SPACE.sub(" ", t).strip()
    return t


def postprocess_page(lines: List[OcrLine], bbox_normalized: bool) -> ProcessedPage:
    paras, maps = merge_lines_to_paragraphs(lines, bbox_normalized=bbox_normalized)
    raw_fixed = "\n\n".join(paras)
    norm_paras = [normalize_for_search(p) for p in paras]
    norm_search = "\n\n".join(norm_paras)
    line_map = [{"para_no": pi, "line_indices": li} for pi, li in enumerate(maps)]
    return ProcessedPage(raw_fixed=raw_fixed, norm_search=norm_search, paragraphs=norm_paras, line_map=line_map)
# Chunking
import regex as re

SENT_SPLIT = re.compile(r"(?<=[؟!\.;؛])\s+")


def split_sentences_ar(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    return [p.strip() for p in SENT_SPLIT.split(text) if p.strip()]


def build_parents(pages: List[tuple[int, int, str, str]], pages_per_parent: int, book_id: str) -> list[dict]:
    parents = []
    for i in range(0, len(pages), pages_per_parent):
        chunk = pages[i : i + pages_per_parent]
        v0, p0, _, _ = chunk[0]
        v1, p1, _, _ = chunk[-1]
        raw = "\n\n".join([x[2] for x in chunk])
        norm = "\n\n".join([x[3] for x in chunk])
        pid = f"{book_id}:v{v0}:p{p0}-{p1}"
        parents.append(
            {
                "id": pid,
                "book_id": book_id,
                "parent_type": "page_window",
                "title": None,
                "section_path": None,
                "volume_start": v0,
                "page_start": p0,
                "volume_end": v1,
                "page_end": p1,
                "text_raw": raw,
                "text_norm": norm,
            }
        )
    return parents


def build_children(parent: dict, max_tokens: int, overlap_tokens: int, min_tokens: int) -> list[dict]:
    counter = build_counter(prefer_tiktoken=True)
    sents = split_sentences_ar(parent["text_norm"])
    chunks = []
    cur: List[str] = []
    cur_tokens = 0
    chunk_no = 0

    def flush():
        nonlocal chunk_no, cur, cur_tokens
        if not cur:
            return
        text_norm = " ".join(cur).strip()
        tok = counter.count(text_norm)
        if tok < min_tokens:
            return
        text_raw = text_norm
        cid = f"{parent['id']}:c{chunk_no}"
        chunks.append(
            {
                "id": cid,
                "book_id": parent["book_id"],
                "parent_id": parent["id"],
                "chunk_no": chunk_no,
                "text_raw": text_raw,
                "text_norm": text_norm,
                "token_count": tok,
                "overlap_tokens": overlap_tokens,
                "page_start": parent["page_start"],
                "page_end": parent["page_end"],
                "offsets": {},
            }
        )
        chunk_no += 1

    for s in sents:
        t = counter.count(s)
        if cur_tokens + t <= max_tokens:
            cur.append(s)
            cur_tokens += t
        else:
            flush()
            overlap: List[str] = []
            overlap_tok = 0
            for prev in reversed(cur):
                overlap.insert(0, prev)
                overlap_tok += counter.count(prev)
                if overlap_tok >= overlap_tokens:
                    break
            cur = overlap + [s]
            cur_tokens = overlap_tok + t

    flush()
    return chunks
# Storage (Postgres)
SCHEMA_SQL = r'''
CREATE TABLE IF NOT EXISTS ocr_pages (
  id SERIAL PRIMARY KEY,
  book_id TEXT NOT NULL,
  volume INT NOT NULL,
  page_no INT NOT NULL,
  image_uri TEXT,
  image_hash TEXT,
  ocr_engine TEXT,
  ocr_version TEXT,
  text_raw TEXT,
  text_norm TEXT,
  layout_json JSONB,
  quality JSONB,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  UNIQUE(book_id, volume, page_no)
);

CREATE TABLE IF NOT EXISTS parents (
  id TEXT PRIMARY KEY,
  book_id TEXT NOT NULL,
  parent_type TEXT NOT NULL,
  title TEXT,
  section_path TEXT,
  volume_start INT NOT NULL,
  page_start INT NOT NULL,
  volume_end INT NOT NULL,
  page_end INT NOT NULL,
  text_raw TEXT,
  text_norm TEXT,
  token_count_est INT,
  hash_raw TEXT,
  hash_norm TEXT,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS children (
  id TEXT PRIMARY KEY,
  book_id TEXT NOT NULL,
  parent_id TEXT NOT NULL REFERENCES parents(id),
  chunk_no INT NOT NULL,
  text_raw TEXT,
  text_norm TEXT,
  token_count INT,
  overlap_tokens INT,
  page_start INT,
  page_end INT,
  offsets JSONB,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
''' 


def init_db(engine):
    with engine.begin() as con:
        con.execute(text(SCHEMA_SQL))


def upsert_ocr_page(engine, row: dict):
    sql = text(
        """
    INSERT INTO ocr_pages (book_id, volume, page_no, image_uri, image_hash, ocr_engine, ocr_version,
                          text_raw, text_norm, layout_json, quality)
    VALUES (:book_id, :volume, :page_no, :image_uri, :image_hash, :ocr_engine, :ocr_version,
            :text_raw, :text_norm, CAST(:layout_json AS JSONB), CAST(:quality AS JSONB))
    ON CONFLICT (book_id, volume, page_no) DO UPDATE SET
      image_uri=EXCLUDED.image_uri,
      image_hash=EXCLUDED.image_hash,
      ocr_engine=EXCLUDED.ocr_engine,
      ocr_version=EXCLUDED.ocr_version,
      text_raw=EXCLUDED.text_raw,
      text_norm=EXCLUDED.text_norm,
      layout_json=EXCLUDED.layout_json,
      quality=EXCLUDED.quality;
    """
    )
    with engine.begin() as con:
        con.execute(sql, row)


def upsert_parent(engine, row: dict):
    sql = text(
        """
    INSERT INTO parents (id, book_id, parent_type, title, section_path, volume_start, page_start, volume_end, page_end,
                         text_raw, text_norm, token_count_est, hash_raw, hash_norm)
    VALUES (:id, :book_id, :parent_type, :title, :section_path, :volume_start, :page_start, :volume_end, :page_end,
            :text_raw, :text_norm, :token_count_est, :hash_raw, :hash_norm)
    ON CONFLICT (id) DO UPDATE SET
      title=EXCLUDED.title,
      section_path=EXCLUDED.section_path,
      text_raw=EXCLUDED.text_raw,
      text_norm=EXCLUDED.text_norm,
      token_count_est=EXCLUDED.token_count_est,
      hash_raw=EXCLUDED.hash_raw,
      hash_norm=EXCLUDED.hash_norm;
    """
    )
    with engine.begin() as con:
        con.execute(sql, row)


def upsert_child(engine, row: dict):
    sql = text(
        """
    INSERT INTO children (id, book_id, parent_id, chunk_no, text_raw, text_norm, token_count, overlap_tokens,
                          page_start, page_end, offsets)
    VALUES (:id, :book_id, :parent_id, :chunk_no, :text_raw, :text_norm, :token_count, :overlap_tokens,
            :page_start, :page_end, CAST(:offsets AS JSONB))
    ON CONFLICT (id) DO UPDATE SET
      text_raw=EXCLUDED.text_raw,
      text_norm=EXCLUDED.text_norm,
      token_count=EXCLUDED.token_count;
    """
    )
    with engine.begin() as con:
        con.execute(sql, row)
# Indexing

def build_es(es_url: str) -> Elasticsearch:
    return Elasticsearch(es_url)


def ensure_indices(es: Elasticsearch, idx_pages_raw: str, idx_children_norm: str) -> None:
    if not es.indices.exists(index=idx_pages_raw):
        es.indices.create(
            index=idx_pages_raw,
            mappings={
                "properties": {
                    "book_id": {"type": "keyword"},
                    "volume": {"type": "integer"},
                    "page_no": {"type": "integer"},
                    "text_raw": {"type": "text"},
                    "quality_score": {"type": "float"},
                }
            },
        )
    if not es.indices.exists(index=idx_children_norm):
        es.indices.create(
            index=idx_children_norm,
            mappings={
                "properties": {
                    "book_id": {"type": "keyword"},
                    "child_id": {"type": "keyword"},
                    "parent_id": {"type": "keyword"},
                    "page_start": {"type": "integer"},
                    "page_end": {"type": "integer"},
                    "text_norm": {"type": "text"},
                }
            },
        )


def index_page_raw(es: Elasticsearch, index: str, doc_id: str, body: dict) -> None:
    es.index(index=index, id=doc_id, document=body)


def index_child_norm(es: Elasticsearch, index: str, doc_id: str, body: dict) -> None:
    es.index(index=index, id=doc_id, document=body)


def build_qdrant(url: str) -> QdrantClient:
    return QdrantClient(url=url)


def ensure_collections(q: QdrantClient, children_col: str, summaries_col: str, dim: int) -> None:
    existing = {c.name for c in q.get_collections().collections}
    if children_col not in existing:
        q.create_collection(
            collection_name=children_col,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
    if summaries_col not in existing:
        q.create_collection(
            collection_name=summaries_col,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )


def upsert_vectors(q: QdrantClient, collection: str, points: List[qm.PointStruct]) -> None:
    q.upsert(collection_name=collection, points=points)
# Utils

def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_volume_page(filename: str) -> tuple[int, int]:
    stem = Path(filename).stem.lower()
    if "v" in stem and "p" in stem:
        v_part = stem.split("v", 1)[1]
        if "_" in v_part:
            v_str, rest = v_part.split("_", 1)
        else:
            v_str, rest = v_part, ""
        p_str = rest.split("p", 1)[1] if "p" in rest else "0"
        return int(v_str.lstrip("0") or 0), int(p_str.lstrip("0") or 0)
    parts = stem.split("_")
    return int(parts[0]), int(parts[1])


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")
# Embedding stub

def embed_texts(texts: List[str], dim: int) -> List[List[float]]:
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        vec = [(h[i % len(h)] / 255.0) for i in range(dim)]
        out.append(vec)
    return out
# Main pipeline

def run_pipeline(cfg: Settings) -> None:
    img_paths = sorted(Path(cfg.input_dir).glob("*.png")) + sorted(Path(cfg.input_dir).glob("*.jpg"))
    if cfg.max_pages:
        img_paths = img_paths[: cfg.max_pages]
    print(f"Found {len(img_paths)} pages")

    engine = create_engine(cfg.pg_dsn)
    init_db(engine)

    es = build_es(cfg.es_url)
    ensure_indices(es, cfg.es_index_pages_raw, cfg.es_index_children_norm)

    q = build_qdrant(cfg.qdrant_url)
    ensure_collections(q, cfg.qdrant_collection_children, cfg.qdrant_collection_summaries, cfg.embedding_dim)

    ocr = build_ocr_engine(cfg.ocr_engine)

    pages_for_parents: List[tuple[int, int, str, str]] = []

    for path in tqdm(img_paths, desc="OCR pages"):
        v, p = parse_volume_page(path.name)
        img_hash = sha1_file(str(path))
        img = load_image(str(path))

        img1 = preprocess_image(img, cfg.do_deskew, cfg.do_denoise, cfg.do_binarize, cfg.crop_margins)
        r1 = ocr.ocr_image(img1, cfg.ocr_langs)
        raw_join = "\n".join([ln.text for ln in r1.lines])
        score1 = page_quality_score(r1.page_conf_mean, raw_join)

        if r1.page_conf_mean < cfg.ocr_pass2_conf_threshold:
            img2 = preprocess_image(img, True, True, True, True)
            r2 = ocr.ocr_image(img2, cfg.ocr_langs)
            raw_join2 = "\n".join([ln.text for ln in r2.lines])
            score2 = page_quality_score(r2.page_conf_mean, raw_join2)
            if score2 > score1:
                r1, raw_join, score1 = r2, raw_join2, score2

        processed = postprocess_page(r1.lines, bbox_normalized=cfg.bbox_normalized)

        upsert_ocr_page(
            engine,
            {
                "book_id": cfg.book_id,
                "volume": v,
                "page_no": p,
                "image_uri": str(path),
                "image_hash": img_hash,
                "ocr_engine": cfg.ocr_engine,
                "ocr_version": "v1",
                "text_raw": processed.raw_fixed,
                "text_norm": processed.norm_search,
                "layout_json": "{}",
                "quality": json.dumps({"page_conf_mean": r1.page_conf_mean, "quality_score": score1}),
            },
        )

        es_doc_id = f"{cfg.book_id}:v{v}:p{p}"
        index_page_raw(
            es,
            cfg.es_index_pages_raw,
            es_doc_id,
            {
                "book_id": cfg.book_id,
                "volume": v,
                "page_no": p,
                "text_raw": processed.raw_fixed,
                "quality_score": score1,
            },
        )

        pages_for_parents.append((v, p, processed.raw_fixed, processed.norm_search))

    parents = build_parents(pages_for_parents, cfg.pages_per_parent, cfg.book_id)
    for parent in tqdm(parents, desc="Store parents"):
        upsert_parent(
            engine,
            {
                "id": parent["id"],
                "book_id": parent["book_id"],
                "parent_type": parent["parent_type"],
                "title": parent["title"],
                "section_path": parent["section_path"],
                "volume_start": parent["volume_start"],
                "page_start": parent["page_start"],
                "volume_end": parent["volume_end"],
                "page_end": parent["page_end"],
                "text_raw": parent["text_raw"],
                "text_norm": parent["text_norm"],
                "token_count_est": len(parent["text_norm"].split()),
                "hash_raw": "",
                "hash_norm": "",
            },
        )

        children = build_children(parent, cfg.child_max_tokens, cfg.child_overlap_tokens, cfg.child_min_tokens)
        child_texts = [c["text_norm"] for c in children]
        vectors = embed_texts(child_texts, cfg.embedding_dim)
        points = []
        for c, vec in zip(children, vectors):
            upsert_child(
                engine,
                {
                    "id": c["id"],
                    "book_id": c["book_id"],
                    "parent_id": c["parent_id"],
                    "chunk_no": c["chunk_no"],
                    "text_raw": c["text_raw"],
                    "text_norm": c["text_norm"],
                    "token_count": c["token_count"],
                    "overlap_tokens": c["overlap_tokens"],
                    "page_start": c["page_start"],
                    "page_end": c["page_end"],
                    "offsets": json.dumps(c["offsets"]),
                },
            )

            index_child_norm(
                es,
                cfg.es_index_children_norm,
                c["id"],
                {
                    "book_id": c["book_id"],
                    "child_id": c["id"],
                    "parent_id": c["parent_id"],
                    "page_start": c["page_start"],
                    "page_end": c["page_end"],
                    "text_norm": c["text_norm"],
                },
            )

            points.append(
                qm.PointStruct(
                    id=c["id"],
                    vector=vec,
                    payload={
                        "book_id": c["book_id"],
                        "parent_id": c["parent_id"],
                        "page_start": c["page_start"],
                        "page_end": c["page_end"],
                        "token_count": c["token_count"],
                    },
                )
            )

        if points:
            upsert_vectors(q, cfg.qdrant_collection_children, points)

    print("Done: OCR -> Postprocess -> Parents/Children -> ES/Qdrant/PG")
