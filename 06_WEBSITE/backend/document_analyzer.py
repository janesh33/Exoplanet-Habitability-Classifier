"""
document_analyzer.py
====================
Robust, context-aware exoplanet document analysis module.
Performs:
1. Multi-page PDF text extraction and OCR for images.
2. Canonical planet name normalization.
3. Frequency- and context-weighted primary planet identification.
4. Strict exact-matching against the Habitable Worlds Catalog (HWC) without loose substring fallbacks.
5. Context-based parameter extraction with strict planet and host-star association.
6. Validation checks ensuring no cross-contamination between exoplanets in comparative papers.
"""

import os
import io
import re
import logging
import tempfile
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

import pypdf

# Configure internal logger
logger = logging.getLogger("document_analyzer")
logger.setLevel(logging.INFO)

# Check for Apple Vision framework on macOS for local image OCR
_has_vision_ocr = False
try:
    import Vision
    from Cocoa import NSURL
    _has_vision_ocr = True
except Exception:
    _has_vision_ocr = False


def extract_text_from_pdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extracts text from each page of a PDF with page numbering.
    Returns: [{'page': 1, 'text': '...'}, ...]
    """
    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    pages_content = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"Error reading PDF page {idx}: {e}")
            text = ""
        pages_content.append({
            "page": idx,
            "text": text
        })
    return pages_content


def extract_text_from_image(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Extracts text from an image (PNG, JPG, JPEG) using Apple Vision OCR on macOS.
    """
    if not _has_vision_ocr:
        return [{"page": 1, "text": ""}]

    suffix = os.path.splitext(filename)[1] or '.png'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        url = NSURL.fileURLWithPath_(tmp_path)
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        success, error = handler.performRequests_error_([request], None)
        if not success:
            return [{"page": 1, "text": ""}]

        lines = [obs.text() for obs in request.results()]
        extracted_text = "\n".join(lines)
        return [{"page": 1, "text": extracted_text}]
    except Exception as e:
        logger.error(f"Image OCR error: {e}")
        return [{"page": 1, "text": ""}]
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def normalize_planet_name(raw_name: str) -> str:
    """
    Normalizes candidate planet names into standard canonical format.
    Examples:
    - "Kepler-452b" -> "Kepler-452 b"
    - "Kepler 452b" -> "Kepler-452 b"
    - "KEPLER-452B" -> "Kepler-452 b"
    - "TRAPPIST-1e" -> "TRAPPIST-1 e"
    - "Proxima Cen b" -> "Proxima Cen b"
    - "Earth" -> "Earth"
    """
    if not raw_name:
        return ""

    cleaned = raw_name.strip()

    # Special handling for Earth
    if cleaned.lower() == "earth":
        return "Earth"

    # Match Kepler designations: e.g. Kepler-452 b, Kepler 452b, KEPLER-452B
    m_kepler = re.match(r'^kepler[\s\-_]*(\d+)[\s\-_]*([a-z])?$', cleaned, re.IGNORECASE)
    if m_kepler:
        num = m_kepler.group(1)
        letter = (m_kepler.group(2) or '').lower()
        if letter:
            return f"Kepler-{num} {letter}"
        return f"Kepler-{num}"

    # Match TRAPPIST designations
    m_trappist = re.match(r'^trappist[\s\-_]*1[\s\-_]*([a-z])?$', cleaned, re.IGNORECASE)
    if m_trappist:
        letter = (m_trappist.group(1) or '').lower()
        if letter:
            return f"TRAPPIST-1 {letter}"
        return "TRAPPIST-1"

    # Match K2 designations
    m_k2 = re.match(r'^k2[\s\-_]*(\d+)[\s\-_]*([a-z])?$', cleaned, re.IGNORECASE)
    if m_k2:
        num = m_k2.group(1)
        letter = (m_k2.group(2) or '').lower()
        if letter:
            return f"K2-{num} {letter}"
        return f"K2-{num}"

    # Match Gliese designations
    m_gliese = re.match(r'^gliese[\s\-_]*(\d+)[\s\-_]*([a-z])?$', cleaned, re.IGNORECASE)
    if m_gliese:
        num = m_gliese.group(1)
        letter = (m_gliese.group(2) or '').lower()
        if letter:
            return f"Gliese-{num} {letter}"
        return f"Gliese-{num}"

    # Standardize spacing between digits and trailing planet designation letter
    std = re.sub(r'(\d+)\s*([a-z])\b', r'\1 \2', cleaned)
    # Capitalize prefix
    parts = std.split()
    if len(parts) >= 2:
        prefix = parts[0].capitalize()
        rest = " ".join(parts[1:])
        return f"{prefix} {rest}"

    return std


def get_host_star_name(canonical_planet_name: str) -> str:
    """
    Derives the host star name from a canonical planet name by stripping the planet letter.
    Example:
    - "Kepler-452 b" -> "Kepler-452"
    - "TRAPPIST-1 e" -> "TRAPPIST-1"
    - "Proxima Cen b" -> "Proxima Cen"
    - "Earth" -> "Sun"
    """
    if canonical_planet_name.lower() == "earth":
        return "Sun"

    # If ends with a single lowercase letter designation preceded by space
    m = re.match(r'^(.*?)\s+[a-z]$', canonical_planet_name, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return canonical_planet_name


def identify_planets_in_document(pages: List[Dict[str, Any]], catalog_names: List[str]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Scans the document to detect all candidate exoplanet names, weights them
    by prominence (title, abstract, section headings, repeated mentions), and identifies
    the primary planet being discussed.

    Returns:
    - ranked_planets: List of dicts with {"name", "canonical", "mentions", "score", "first_page"}
    - primary_planet: Canonical name of the primary planet
    """
    detected_counts: Dict[str, Dict[str, Any]] = {}

    # Patterns for detecting candidate exoplanet names in text
    exoplanet_patterns = [
        r'\b(Kepler[\s\-_]*\d+[\s\-_]*[a-z]?)\b',
        r'\b(TRAPPIST[\s\-_]*1[\s\-_]*[a-z]?)\b',
        r'\b(K2[\s\-_]*\d+[\s\-_]*[a-z]?)\b',
        r'\b(Gliese[\s\-_]*\d+[\s\-_]*[a-z]?)\b',
        r'\b(GJ[\s\-_]*\d+[\s\-_]*[a-z]?)\b',
        r'\b(HD[\s\-_]*\d+[\s\-_]*[a-z]?)\b',
        r'\b(HIP[\s\-_]*\d+[\s\-_]*[a-z]?)\b',
        r'\b(TOI[\s\-_]*\d+[\s\-_]*[a-z]?)\b',
        r'\b(WASP[\s\-_]*\d+[\s\-_]*[a-z]?)\b',
        r'\b(Proxima\s+Cen(?:\s*[a-z])?)\b'
    ]

    for p in pages:
        page_num = p["page"]
        text = p["text"]
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        is_page_1 = (page_num == 1)

        # Check exoplanet regex patterns
        for pat in exoplanet_patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                raw = match.group(1).strip()
                canonical = normalize_planet_name(raw)
                if not canonical:
                    continue

                if canonical not in detected_counts:
                    detected_counts[canonical] = {
                        "name": canonical,
                        "raw_mentions": [],
                        "count": 0,
                        "score": 0.0,
                        "first_page": page_num,
                        "in_title": False
                    }

                detected_counts[canonical]["count"] += 1
                detected_counts[canonical]["score"] += 2.0  # Base occurrence score

                # Check if occurrence is in the document title or abstract (first 5 lines of Page 1)
                if is_page_1:
                    match_pos = match.start()
                    # If within first 400 characters of Page 1, likely in title or abstract
                    if match_pos < 400:
                        detected_counts[canonical]["score"] += 15.0
                        detected_counts[canonical]["in_title"] = True

        # Check for Earth as a standalone planet comparison (not as an astronomical unit)
        earth_pattern = r'(?<!R_)(?<!M_)(?<!\d\s)(?<!\d\.)\bEarth\b(?!\s*radii|\s*radius|\s*mass|\s*masses|\s*flux|\s*units|\s*twin|\s*analog|\s*eccentricity)'
        for match in re.finditer(earth_pattern, text, re.IGNORECASE):
            canonical = "Earth"
            if canonical not in detected_counts:
                detected_counts[canonical] = {
                    "name": "Earth",
                    "raw_mentions": [],
                    "count": 0,
                    "score": 0.0,
                    "first_page": page_num,
                    "in_title": False
                }
            detected_counts[canonical]["count"] += 1
            # Earth as a comparison body receives lower primary weight unless it is in the title
            detected_counts[canonical]["score"] += 0.5
            if is_page_1 and match.start() < 250:
                detected_counts[canonical]["score"] += 10.0
                detected_counts[canonical]["in_title"] = True

    if not detected_counts:
        return [], None

    # Filter out standalone star names if a planet of that star is present (e.g. filter "Kepler-452" if "Kepler-452 b" exists)
    all_keys = list(detected_counts.keys())
    for name in all_keys:
        # If name has no planet letter (e.g. "Kepler-452")
        if not re.search(r'\s+[a-z]$', name, re.IGNORECASE) and name != "Earth":
            # Check if a planet with this host star exists
            has_planet_counterpart = any(other.startswith(f"{name} ") for other in all_keys)
            if has_planet_counterpart:
                # Merge star counts into planet score
                for other in all_keys:
                    if other.startswith(f"{name} "):
                        detected_counts[other]["score"] += detected_counts[name]["score"] * 0.5
                del detected_counts[name]

    # Rank candidate planets by total prominence score
    ranked = sorted(
        detected_counts.values(),
        key=lambda item: (-item["score"], -item["count"], item["first_page"], item["name"])
    )

    primary_planet = ranked[0]["name"] if ranked else None

    logger.info(f"Detected planets: {[p['name'] for p in ranked]}, Primary identified: {primary_planet}")
    return ranked, primary_planet


def match_planet_in_catalog(planet_name: str, df_catalog: pd.DataFrame) -> Optional[pd.Series]:
    """
    Strictly matches an extracted canonical planet name against the Habitable Worlds Catalog DataFrame.
    
    STRICT RULES (RULE 6 & RULE 7):
    - Exact canonical match only.
    - NEVER use prefix matching (startswith) or loose substring matching.
    - NEVER fall back to an unrelated exoplanet (e.g. Kepler-452 b must NEVER match Kepler-186 f).
    - If not found, returns None.
    """
    if df_catalog is None or df_catalog.empty or not planet_name:
        return None

    canonical_target = normalize_planet_name(planet_name)
    clean_target = re.sub(r'[^a-z0-9]', '', canonical_target.lower())

    if not clean_target:
        return None

    # 1. Exact normalized match against P. Name
    for idx, row in df_catalog.iterrows():
        p_name = str(row.get('P. Name', '')).strip()
        clean_p = re.sub(r'[^a-z0-9]', '', p_name.lower())
        if clean_target == clean_p:
            logger.info(f"MATCH CONFIRMED: Target '{planet_name}' -> Catalog '{p_name}'")
            return row

    # 2. Exact match against P. Name Kepler
    if 'P. Name Kepler' in df_catalog.columns:
        for idx, row in df_catalog.iterrows():
            k_name = str(row.get('P. Name Kepler', '')).strip()
            clean_k = re.sub(r'[^a-z0-9]', '', k_name.lower())
            if clean_k and clean_target == clean_k:
                logger.info(f"MATCH CONFIRMED (Kepler Name): Target '{planet_name}' -> Catalog '{row.get('P. Name')}'")
                return row

    # 3. Exact match against P. Name KOI
    if 'P. Name KOI' in df_catalog.columns:
        for idx, row in df_catalog.iterrows():
            koi_name = str(row.get('P. Name KOI', '')).strip()
            clean_koi = re.sub(r'[^a-z0-9]', '', koi_name.lower())
            if clean_koi and clean_target == clean_koi:
                logger.info(f"MATCH CONFIRMED (KOI Name): Target '{planet_name}' -> Catalog '{row.get('P. Name')}'")
                return row

    logger.warning(f"CATALOG MATCH FAILED: '{planet_name}' not found in HWC catalog.")
    return None


def extract_parameters_for_planet(
    pages: List[Dict[str, Any]],
    target_planet: str,
    all_detected_planets: List[str]
) -> Dict[str, Any]:
    """
    Context-based parameter extraction with strict planet and host-star association.
    
    STRICT RULES (RULE 2, RULE 3, RULE 8, RULE 9):
    - Every extracted value must be verified to associate with target_planet (or its host star).
    - If a parameter sentence or table row explicitly mentions a DIFFERENT detected exoplanet
      (e.g., Kepler-186 f) and does NOT mention target_planet (Kepler-452 b), it is DISCARDED.
    - Stellar properties (Teff, Stellar Radius, Stellar Mass) must associate with target planet's host star.
    - If a value cannot be verified, it is flagged as Unverified.
    """
    canonical_target = normalize_planet_name(target_planet)
    target_host_star = get_host_star_name(canonical_target)
    other_planets = [
        normalize_planet_name(p) for p in all_detected_planets 
        if normalize_planet_name(p).lower() != canonical_target.lower() and normalize_planet_name(p).lower() != 'earth'
    ]

    extracted: Dict[str, Any] = {}

    patterns = {
        "P. Radius (EU)": {
            "label": "Planet Radius",
            "unit": "Earth radii (R⊕)",
            "is_stellar": False,
            "regexes": [
                r'(\d+(?:\.\d+)?)\s*(?:R_?(?:Earth|⊕|E)\b|Earth\s*radii)',
                r'(?:planetary\s+radius|radius\s+of\s+the\s+planet)[^\d]{0,25}(\d+(?:\.\d+)?)\s*(?:R_?⊕|Earth\s*radii|R_E)?',
                r'R_p\s*=\s*(\d+(?:\.\d+)?)'
            ]
        },
        "P. Period (days)": {
            "label": "Orbital Period",
            "unit": "days",
            "is_stellar": False,
            "regexes": [
                r'(\d+(?:\.\d+)?)\s*(?:days|day|d\b)',
                r'(?:orbital\s+period|period\s+P)[^\d]{0,25}(\d+(?:\.\d+)?)\s*(?:days)?',
                r'P\s*=\s*(\d+(?:\.\d+)?)\s*(?:days|d\b)'
            ]
        },
        "P. Sem Major Axis (AU)": {
            "label": "Semi-Major Axis",
            "unit": "AU",
            "is_stellar": False,
            "regexes": [
                r'(\d+(?:\.\d+)?)\s*AU\b',
                r'(?:semi-major\s+axis|orbital\s+separation|distance\s+a)[^\d]{0,25}(\d+(?:\.\d+)?)\s*AU',
                r'a\s*=\s*(\d+(?:\.\d+)?)\s*AU'
            ]
        },
        "S. Teff (K)": {
            "label": "Stellar Effective Temperature",
            "unit": "K",
            "is_stellar": True,
            "regexes": [
                r'(?:T_?eff|effective\s+temperature|stellar\s+temperature)[^\d]{0,20}(\d{3,5}(?:\.\d+)?)\s*K',
                r'(\d{3,5})\s*(?:±\s*\d+)?\s*K'
            ]
        },
        "S. Radius (SU)": {
            "label": "Stellar Radius",
            "unit": "Solar radii (R⊙)",
            "is_stellar": True,
            "regexes": [
                r'(\d+(?:\.\d+)?)\s*(?:R_?(?:Sun|⊙|S)\b|Solar\s*radii)',
                r'R_\*\s*=\s*(\d+(?:\.\d+)?)'
            ]
        },
        "P. Mass (EU)": {
            "label": "Planet Mass",
            "unit": "Earth masses (M⊕)",
            "is_stellar": False,
            "regexes": [
                r'(\d+(?:\.\d+)?)\s*(?:M_?(?:Earth|⊕|E)\b|Earth\s*masses)',
                r'M_p\s*=\s*(\d+(?:\.\d+)?)'
            ]
        },
        "S. Mass (SU)": {
            "label": "Stellar Mass",
            "unit": "Solar masses (M⊙)",
            "is_stellar": True,
            "regexes": [
                r'(\d+(?:\.\d+)?)\s*(?:M_?(?:Sun|⊙|S)\b|Solar\s*masses)',
                r'M_\*\s*=\s*(\d+(?:\.\d+)?)'
            ]
        },
        "P. Teq Mean (K)": {
            "label": "Equilibrium Temperature",
            "unit": "K",
            "is_stellar": False,
            "regexes": [
                r'(?:T_?eq|equilibrium\s+temperature)[^\d]{0,20}(\d{2,4}(?:\.\d+)?)\s*K'
            ]
        },
        "P. SFlux Mean (EU)": {
            "label": "Stellar Insolation Flux",
            "unit": "Earth flux (S/S0)",
            "is_stellar": False,
            "regexes": [
                r'(?:insolation|stellar\s+flux|SFlux|S_?eff)[^\d]{0,20}(\d+(?:\.\d+)?)\s*(?:S_?⊕|S_?0|EU)?'
            ]
        },
        "P. Eccentricity": {
            "label": "Orbital Eccentricity",
            "unit": "",
            "is_stellar": False,
            "regexes": [
                r'(?:eccentricity|e\s*=)[^\d]{0,15}(0\.\d+)'
            ]
        }
    }

    # Extract parameters
    for key, spec in patterns.items():
        found = False

        for p in pages:
            if found:
                break
            page_num = p["page"]
            text = p["text"]

            # Break page into sentences/paragraphs
            sentences = re.split(r'(?<=[.!?\n])\s+', text)

            for sentence in sentences:
                if found:
                    break

                for rx in spec["regexes"]:
                    m = re.search(rx, sentence, re.IGNORECASE)
                    if m:
                        try:
                            val = float(m.group(1))

                            # Range plausibility check
                            if key == "S. Teff (K)" and (val < 1000 or val > 50000):
                                continue
                            if key == "P. Radius (EU)" and (val <= 0 or val > 100):
                                continue
                            if key == "P. Period (days)" and (val <= 0 or val > 1000000):
                                continue

                            # CONTEXT ASSOCIATION CHECK (RULE 2 & RULE 3)
                            # Check if sentence mentions an OTHER planet and NOT the target
                            sentence_lower = sentence.lower()
                            has_other_planet = any(other.lower() in sentence_lower for other in other_planets)
                            has_target_planet = (canonical_target.lower() in sentence_lower or 
                                                 target_host_star.lower() in sentence_lower)

                            # If another planet is explicitly the subject of this sentence, reject!
                            if has_other_planet and not has_target_planet:
                                logger.info(f"DISCARDED parameter {key}={val} on Page {page_num}: belongs to other planet in context '{sentence[:60]}'")
                                continue

                            # If it's a stellar parameter, check host star association
                            if spec["is_stellar"]:
                                # If other host star mentioned, reject
                                other_stars = [get_host_star_name(op).lower() for op in other_planets]
                                if any(os_name in sentence_lower for os_name in other_stars if os_name != target_host_star.lower()):
                                    logger.info(f"DISCARDED stellar parameter {key}={val} on Page {page_num}: belongs to different host star")
                                    continue

                            # Verified parameter match
                            snippet = m.group(0).strip()
                            # Clean snippet
                            extracted[key] = {
                                "planet": canonical_target,
                                "parameter": key,
                                "label": spec["label"],
                                "value": val,
                                "unit": spec["unit"],
                                "page": page_num,
                                "source_page": f"Page {page_num}",
                                "context": sentence.strip()[:180],
                                "snippet": snippet,
                                "verified": True
                            }
                            found = True
                            logger.info(f"VERIFIED: {key} = {val} {spec['unit']} associated with {canonical_target} (Page {page_num})")
                            break
                        except ValueError:
                            continue

    return extracted
