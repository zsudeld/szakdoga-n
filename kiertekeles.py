"""
kiertekeles.py
==============
Kiértékelő modul: a három sentiment-konfiguráció (szótári, HuBERT, hibrid)
összehasonlítása kézzel címkézett referenciaadaton.

A bemenet egy CSV fájl, amelynek van egy szöveges oszlopa és egy kézi címkét
tartalmazó oszlopa (pozitív / semleges / negatív). A modul

  1. a szövegeket egyszer lefuttatja az elemző motoron,
  2. a címkézett mintát fejlesztői és teszt részre osztja (rögzített seed),
  3. mindhárom konfigurációra kiszámítja a pontosságot, az osztályonkénti
     precision / recall / F1 értékeket és a makro-F1-et,
  4. elkészíti a hibrid módszer konfúziós mátrixát és a hibás besorolások listáját,
  5. opcionálisan kiszámítja két címkéző közötti Cohen-féle kappát,
  6. az eredményt Excel fájlba menti.

A küszöbértékek hangolását a fejlesztői részen kell végezni; a végső
számokat a teszt részről kell jelenteni.

Használat:
    python kiertekeles.py -f cimkezett.csv -c szoveg -g gold
    python kiertekeles.py -f cimkezett.csv -c szoveg -g gold --gold2 gold_masodik -o ertekeles.xlsx
"""

import argparse
import time
from datetime import datetime

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

CIMKEK = ["pozitív", "semleges", "negatív"]

# Az elemzőmotor eredményszótárának kulcsai a három konfigurációhoz.
KONFIGURACIOK = {
    "A) Szótári módszer": "kategoria",
    "B) NYTK HuBERT": "hunbert_kategoria",
    "C) Hibrid módszer": "hibrid_kategoria",
}

_CIMKE_ALIAS = {
    "pozitív": "pozitív", "pozitiv": "pozitív", "poz": "pozitív", "positive": "pozitív",
    "pos": "pozitív", "+": "pozitív",
    "semleges": "semleges", "sem": "semleges", "neutral": "semleges", "neu": "semleges",
    "negatív": "negatív", "negativ": "negatív", "neg": "negatív", "negative": "negatív",
    "-": "negatív",
}  # numerikus címkék (0/1/2) szándékosan nincsenek: a kódolásuk nem egységes


def cimke_normalizal(ertek):
    """
    A kézi címkét egységes alakra hozza (pozitív / semleges / negatív).
    Hiányzó értéknél None-t ad; ismeretlen címkénél ValueError-t dob.
    """
    if ertek is None or (isinstance(ertek, float) and pd.isna(ertek)):
        return None
    szoveg = str(ertek).strip().lower()
    if szoveg in ("", "nan", "none"):
        return None
    if szoveg not in _CIMKE_ALIAS:
        raise ValueError(f"Ismeretlen címke: {ertek!r} (várt: pozitív / semleges / negatív)")
    return _CIMKE_ALIAS[szoveg]


def felosztas(gold: list, teszt_arany: float = 0.7, seed: int = 42):
    """
    A mintát fejlesztői és teszt részre osztja (indexlistákat ad vissza).
    Ha minden osztályból legalább két elem van, rétegzett (stratified) a felosztás.
    """
    if not 0.0 < teszt_arany < 1.0:
        raise ValueError("A teszt arány 0 és 1 közé kell, hogy essen.")
    idx = list(range(len(gold)))
    if len(idx) < 2:
        raise ValueError("A felosztáshoz legalább két címkézett szöveg kell.")
    from collections import Counter
    darabok = Counter(gold)
    retegzett = gold if min(darabok.values()) >= 2 and len(darabok) > 1 else None
    try:
        dev, teszt = train_test_split(
            idx, test_size=teszt_arany, random_state=seed, stratify=retegzett
        )
    except ValueError:
        dev, teszt = train_test_split(idx, test_size=teszt_arany, random_state=seed)
    return sorted(dev), sorted(teszt)


def metrikak(gold: list, becsles: list) -> dict:
    """
    Osztályozási metrikák egy konfigurációra.

    A makro-F1 csak azokra az osztályokra átlagol, amelyeknek van valós előfordulása
    a mintában (különben a hiányzó osztály 0-s F1-je torzítaná az átlagot).
    A "hiba" vagy más, nem érvényes becslés tévesnek számít.
    """
    if len(gold) != len(becsles) or not gold:
        raise ValueError("A valós és a becsült címkék listájának azonos, nem nulla hosszúnak kell lennie.")
    p, r, f, tamogatas = precision_recall_fscore_support(
        gold, becsles, labels=CIMKEK, zero_division=0
    )
    osztaly = {
        c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
            "tamogatas": int(tamogatas[i])}
        for i, c in enumerate(CIMKEK)
    }
    jelenlevo = [o["f1"] for o in osztaly.values() if o["tamogatas"] > 0]
    return {
        "n": len(gold),
        "pontossag": float(accuracy_score(gold, becsles)),
        "makro_f1": float(sum(jelenlevo) / len(jelenlevo)) if jelenlevo else 0.0,
        "osztaly": osztaly,
        "ervenytelen_becslesek": sum(1 for b in becsles if b not in CIMKEK),
    }


def _f1_vagy_none(m: dict, cimke: str):
    o = m["osztaly"][cimke]
    return round(o["f1"], 4) if o["tamogatas"] > 0 else None


def konfuzios_matrix(gold: list, becsles: list) -> pd.DataFrame:
    """Konfúziós mátrix (sorok: valós, oszlopok: becsült); az érvénytelen becslés külön oszlop."""
    becsles_norm = [b if b in CIMKEK else "hiba" for b in becsles]
    oszlopok = CIMKEK + (["hiba"] if "hiba" in becsles_norm else [])
    m = confusion_matrix(gold, becsles_norm, labels=CIMKEK + ["hiba"])
    df = pd.DataFrame(m[: len(CIMKEK)], index=CIMKEK, columns=CIMKEK + ["hiba"])
    return df[oszlopok]


def kappa(gold1: list, gold2: list) -> float:
    """Cohen-féle kappa két címkéző között."""
    if len(gold1) != len(gold2) or not gold1:
        raise ValueError("A két címkelista hossza egyezzen meg, és ne legyen üres.")
    return float(cohen_kappa_score(gold1, gold2, labels=CIMKEK))


def eredmenyek_elemzese(motor, szovegek: list[str]) -> tuple[list[dict], float]:
    """Lefuttatja az elemzőmotort a szövegeken; visszaadja az eredményeket és a futási időt (mp)."""
    t0 = time.perf_counter()
    eredmenyek = list(motor.elemzes_batch(szovegek))
    return eredmenyek, time.perf_counter() - t0


def ertekel(
    df: pd.DataFrame,
    szoveg_oszlop: str,
    gold_oszlop: str,
    motor,
    gold2_oszlop: str | None = None,
    teszt_arany: float = 0.7,
    seed: int = 42,
) -> dict:
    """
    A teljes kiértékelés. A `motor` bármilyen objektum lehet, amelynek van
    `elemzes_batch(list[str])` generátor-metódusa (a FejlettSentimentElemzo vagy
    egy tesztben használt helyettesítő).
    """
    for oszlop in [szoveg_oszlop, gold_oszlop] + ([gold2_oszlop] if gold2_oszlop else []):
        if oszlop not in df.columns:
            raise KeyError(f"Nincs '{oszlop}' oszlop a fájlban. Oszlopok: {list(df.columns)}")

    munka = df[[szoveg_oszlop, gold_oszlop] + ([gold2_oszlop] if gold2_oszlop else [])].copy()
    munka["_gold"] = [cimke_normalizal(v) for v in munka[gold_oszlop]]
    ures_szoveg = munka[szoveg_oszlop].isna() | (munka[szoveg_oszlop].astype(str).str.strip() == "")
    kihagyott = int((munka["_gold"].isna() | ures_szoveg).sum())
    munka = munka[~(munka["_gold"].isna() | ures_szoveg)]
    if len(munka) < 2:
        raise ValueError("Legalább két címkézett, nem üres szöveg kell a kiértékeléshez.")

    szovegek = [str(s) for s in munka[szoveg_oszlop]]
    gold = list(munka["_gold"])
    eredmenyek, ido = eredmenyek_elemzese(motor, szovegek)
    if len(eredmenyek) != len(szovegek):
        raise RuntimeError("Az elemző nem minden szövegre adott eredményt.")

    # Az NYTK modell hiánya ne nézzen ki érvényes (semleges) HuBERT-becslésnek:
    # ilyenkor a 'B) NYTK HuBERT' konfiguráció sorai 'hiba'-ként jelennek meg.
    if hasattr(motor, "sentiment"):
        nytk_erheto = motor.sentiment is not None
        if not nytk_erheto:
            print(
                "FIGYELEM: Az NYTK (HuBERT) modell nem érhető el – degradált, "
                "szótár-alapú módban fut az elemzés. A 'B) NYTK HuBERT' "
                "konfiguráció eredményei 'hiba'-ként jelennek meg, nem semlegesként."
            )
    else:
        nytk_erheto = None

    dev_idx, teszt_idx = felosztas(gold, teszt_arany, seed)
    reszminta = {"fejlesztői": dev_idx, "teszt": teszt_idx, "összes": list(range(len(gold)))}

    sorok = []
    for resz, idx in reszminta.items():
        g = [gold[i] for i in idx]
        for nev, kulcs in KONFIGURACIOK.items():
            m = metrikak(g, [eredmenyek[i].get(kulcs, "hiba") for i in idx])
            sorok.append({
                "Részminta": resz, "Konfiguráció": nev, "N": m["n"],
                "Pontosság": round(m["pontossag"], 4), "Makro-F1": round(m["makro_f1"], 4),
                # ha az osztály nem fordul elő a részmintában, az F1 nem értelmezhető (üres cella)
                "F1 (pozitív)": _f1_vagy_none(m, "pozitív"),
                "F1 (semleges)": _f1_vagy_none(m, "semleges"),
                "F1 (negatív)": _f1_vagy_none(m, "negatív"),
                "Érvénytelen becslés": m["ervenytelen_becslesek"],
            })

    teszt_gold = [gold[i] for i in teszt_idx]
    konf = konfuzios_matrix(teszt_gold, [eredmenyek[i].get("hibrid_kategoria", "hiba") for i in teszt_idx])

    hibak = []
    for i in teszt_idx:
        e = eredmenyek[i]
        if e.get("hibrid_kategoria") != gold[i]:
            hibak.append({
                "Szöveg": szovegek[i][:300], "Valós": gold[i],
                "Szótári": e.get("kategoria"), "HuBERT": e.get("hunbert_kategoria"),
                "HuBERT conf.": e.get("hunbert_confidence"), "Szótári pont": e.get("pontszam"),
                "Hibrid": e.get("hibrid_kategoria"), "Megalapozás": e.get("hibrid_megalapozas"),
            })

    k = None
    if gold2_oszlop:
        g2 = [cimke_normalizal(v) for v in munka[gold2_oszlop]]
        pár = [(a, b) for a, b in zip(gold, g2) if b is not None]
        if pár:
            k = kappa([a for a, _ in pár], [b for _, b in pár])

    meta = {
        "Kiértékelés ideje": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Címkézett szövegek száma": len(gold),
        "Kihagyott sorok (hiányzó címke vagy szöveg)": kihagyott,
        "NYTK modell elérhető": (
            "n/a (nem FejlettSentimentElemzo motor)" if nytk_erheto is None
            else ("igen" if nytk_erheto else "NEM – a B) konfiguráció 'hiba'-ként jelentkezik")
        ),
        "Fejlesztői / teszt méret": f"{len(dev_idx)} / {len(teszt_idx)}",
        "Teszt arány": teszt_arany, "Seed": seed,
        "Elemzés futási ideje (mp)": round(ido, 2),
        "Szövegek / mp": round(len(szovegek) / ido, 2) if ido > 0 else None,
        "Cohen-kappa (két címkéző)": round(k, 4) if k is not None else "nincs megadva",
    }
    return {
        "osszehasonlitas": pd.DataFrame(sorok), "konfuzio": konf,
        "hibak": pd.DataFrame(hibak), "meta": meta, "kappa": k,
    }


def mentes_excel(eredmeny: dict, utvonal: str) -> None:
    """Az eredmény mentése többlapos Excel fájlba."""
    with pd.ExcelWriter(utvonal, engine="openpyxl") as iro:
        eredmeny["osszehasonlitas"].to_excel(iro, sheet_name="Összehasonlítás", index=False)
        eredmeny["konfuzio"].to_excel(iro, sheet_name="Konfúzió (hibrid, teszt)")
        eredmeny["hibak"].to_excel(iro, sheet_name="Hibás besorolások", index=False)
        pd.DataFrame(list(eredmeny["meta"].items()), columns=["Beállítás", "Érték"]).to_excel(
            iro, sheet_name="Beállítások", index=False
        )
        for lap in iro.book.worksheets:
            for oszlop in lap.columns:
                # Excel-képlet-injekció elleni védelem: a bemeneti szövegből
                # származó, '=', '+', '-', '@' jellel kezdődő cellák legyenek
                # szövegként tárolva, ne képletként.
                for c in oszlop:
                    if isinstance(c.value, str) and c.value[:1] in ('=', '+', '-', '@'):
                        c.data_type = 's'
                szeles = max((len(str(c.value)) for c in oszlop if c.value is not None), default=8)
                lap.column_dimensions[oszlop[0].column_letter].width = min(max(10, szeles + 2), 60)


def main() -> None:
    from huspacy_elemzo import FejlettSentimentElemzo
    from utils import intelligens_csv_beolvasas

    ap = argparse.ArgumentParser(description="Sentiment-kiértékelés kézi címkék ellen")
    ap.add_argument("-f", "--fajl", required=True, help="címkézett CSV fájl")
    ap.add_argument("-c", "--szoveg", required=True, help="a szöveges oszlop neve")
    ap.add_argument("-g", "--gold", required=True, help="a kézi címkék oszlopa")
    ap.add_argument("--gold2", help="második címkéző oszlopa (Cohen-kappához)")
    ap.add_argument("--teszt-arany", type=float, default=0.7, help="a teszt rész aránya (alapért.: 0,7)")
    ap.add_argument("--seed", type=int, default=42, help="a felosztás véletlenmagja")
    ap.add_argument("-o", "--kimenet", default=None, help="kimeneti xlsx")
    arg = ap.parse_args()

    df = intelligens_csv_beolvasas(arg.fajl)
    if df is None or df.empty:
        raise SystemExit("A CSV üres vagy nem olvasható.")

    motor = FejlettSentimentElemzo()
    er = ertekel(df, arg.szoveg, arg.gold, motor, arg.gold2, arg.teszt_arany, arg.seed)

    pd.set_option("display.width", 200)
    print("\n=== Összehasonlítás ===")
    print(er["osszehasonlitas"].to_string(index=False))
    print("\n=== Hibrid konfúziós mátrix (teszt rész) ===")
    print(er["konfuzio"])
    print("\n=== Beállítások ===")
    for k, v in er["meta"].items():
        print(f"{k}: {v}")

    ut = arg.kimenet or f"kiertekeles_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    mentes_excel(er, ut)
    print(f"\nMentve: {ut}")


if __name__ == "__main__":
    main()
