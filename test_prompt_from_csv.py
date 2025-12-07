#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

def load_glossary_csv(path: Path) -> Dict[str, Any]:
    df = pd.read_csv(path, dtype=str).fillna("")
    out: Dict[str, Any] = {}
    for _, row in df.iterrows():
        cat = row.get("category", "").strip() or "Uncategorized"
        lang = (row.get("language", "") or "en").strip()
        term = row.get("term", "").strip()
        if not term:
            continue
        aliases = [a.strip() for a in row.get("aliases", "").split(",") if a.strip()]
        notes = row.get("notes", "").strip()
        out.setdefault(cat, {"language": lang, "terms": []})
        out[cat]["language"] = lang
        out[cat]["terms"].append({"term": term, "aliases": aliases, "notes": notes})
    return out

def terms_by_lang(glossary: Dict[str, Any], lang: str = "en") -> List[str]:
    pool = []
    for _, block in glossary.items():
        if block.get("language", "en") == lang:
            for t in block["terms"]:
                pool.append(t["term"])
                pool.extend(t.get("aliases", []))
    seen, dedup = set(), []
    for w in pool:
        k = w.lower()
        if k not in seen:
            dedup.append(w)
            seen.add(k)
    return dedup

def pick_scene_pack(glossary: Dict[str, Any], incident_type: str = "general",
                    language: str = "en", extras: Optional[List[str]] = None,
                    max_terms: int = 120) -> List[str]:
    seeds = {
        "cardiac": ["chest pain","12-lead ECG","STEMI","ASA","nitroglycerin","ROSC",
                    "ventricular fibrillation","ventricular tachycardia","defibrillation","cardioversion","EKG","NTG"],
        "trauma": ["trauma alert","tourniquet","tension pneumothorax","needle decompression",
                   "occlusive dressing","C-spine","traction splint","hemorrhage"],
        "respiratory": ["respiratory distress","asthma","COPD","albuterol","ipratropium",
                        "CPAP","PEEP","ETCO2","SpO2","wheezing"],
        "neuro": ["stroke","TIA","Glasgow Coma Scale","GCS","last known well","Cincinnati stroke scale","BE FAST","PERRL"],
        "overdose": ["overdose","opioid","naloxone","Narcan","respiratory depression","pinpoint pupils","bag valve mask","BVM"],
        "general": ["vital signs stable","blood pressure","heart rate","oxygen saturation","pulse ox","primary survey","SAMPLE","OPQRST"]
    }
    base = seeds.get(incident_type, seeds["general"])
    pool = [w for w in terms_by_lang(glossary, language) if w not in base]
    chosen = list(base) + pool[:max(0, max_terms - len(base))]
    if extras:
        chosen.extend([e for e in extras if e])
        chosen = chosen[:max_terms]
    return chosen

def build_initial_prompt(scene_terms: List[str], max_words: int = 200) -> str:
    return " ".join(scene_terms[:max_words])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV glossary")
    ap.add_argument("--incident", default="general",
                    choices=["general","cardiac","trauma","respiratory","neuro","overdose"])
    ap.add_argument("--lang", default="en")
    ap.add_argument("--extras", nargs="*", default=None,
                    help="Extra proper nouns (hospitals, streets, agency names)")
    ap.add_argument("--max_terms", type=int, default=120)
    ap.add_argument("--out", type=Path, default=None,
                    help="Write the initial_prompt text here (UTF-8). If omitted, prints to stdout.")
    args = ap.parse_args()

    g = load_glossary_csv(Path(args.csv))
    scene_terms = pick_scene_pack(g, incident_type=args.incident, language=args.lang,
                                  extras=args.extras, max_terms=args.max_terms)
    prompt = build_initial_prompt(scene_terms)

    if args.out:
        args.out.write_text(prompt, encoding="utf-8")
        print(f"Wrote prompt to: {args.out}")
    else:
        print("\n--- initial_prompt (copy-paste into Whisper) ---\n")
        print(prompt)
        print("\n--- stats ---")
    print(f"incident={args.incident} lang={args.lang} terms={len(scene_terms)}")

if __name__ == "__main__":
    main()
