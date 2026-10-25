from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button, TextBox

try:
    import wfdb
except ImportError:
    wfdb = None


# Složka s převedenými TXT (prohledává se i v podsložkách).
SLOZKA_TXT = Path(r"C:\Zk_Furst_2\Vystup")

# Původní složka s .hea / .dat / .atr / .qrs (odtud se berou anotace).
SLOZKA_ZDROJ = Path(r"C:\Zk_Furst_2\files")

# Záznam, který se otevře jako první (název bez .txt).
POCATECNI_ZAZNAM = "00"

# Soubor s anotacemi rytmu (.atr = ručně ověřené anotace kardiologa).
PRIPONA_ANOTACI = "atr"

# Rytmy, které NEJSOU problém. Vše ostatní se zvýrazní červeně.
NORMALNI_RYTMY = {"N"}

# Uložit přehled problémových úseků každého záznamu do CSV.
ULOZIT_PREHLED = True

BARVA_PROBLEMU = "red"

POPISY_RYTMU = {
    "AFIB": "fibrilace síní",
    "AFL": "flutter síní",
    "AB": "síňová bigeminie",
    "B": "komorová bigeminie",
    "BII": "AV blok II. stupně",
    "IVR": "idioventrikulární rytmus",
    "J": "junkční rytmus",
    "NOD": "nodální rytmus",
    "P": "stimulovaný rytmus",
    "PREX": "preexcitace (WPW)",
    "SBR": "sinusová bradykardie",
    "SVTA": "supraventrikulární tachyarytmie",
    "T": "komorová trigeminie",
    "VFL": "flutter komor",
    "VT": "komorová tachykardie",
}


# ---------------------------------------------------------------------------
# Hledání a načítání souborů
# ---------------------------------------------------------------------------

def prirozeny_klic(cesta: Path):
    """Přirozené řazení: 2 < 10 < 100 (ne textové 10 < 100 < 2)."""
    text = str(cesta.relative_to(SLOZKA_TXT).with_suffix(""))
    return [int(cast) if cast.isdigit() else cast.lower()
            for cast in re.split(r"(\d+)", text)]


def najdi_zaznamy(slozka: Path) -> list[Path]:
    if not slozka.is_dir():
        raise FileNotFoundError(f"Složka neexistuje: {slozka}")
    return sorted(slozka.rglob("*.txt"), key=prirozeny_klic)


def nacti_zaznam(cesta: Path):
    print(f"\nNačítám {cesta.name}, chvíli to může trvat...")
    tabulka = pd.read_csv(cesta, sep="\t")

    if "time_s" not in tabulka.columns:
        raise ValueError("V souboru chybí sloupec time_s.")

    cas = tabulka["time_s"].to_numpy(dtype=float)
    kanaly = [sloupec for sloupec in tabulka.columns if sloupec != "time_s"]

    if len(cas) < 2 or not kanaly:
        raise ValueError("TXT musí obsahovat čas a alespoň jeden EKG kanál.")
    if not np.all(np.isfinite(cas)) or not np.all(np.diff(cas) > 0):
        raise ValueError("Čas musí obsahovat platné, rostoucí hodnoty.")

    signaly = tabulka[kanaly].to_numpy(dtype=float)
    print(f"Hotovo: {len(cas)} vzorků, {len(kanaly)} kanálů.")
    return cas, kanaly, signaly


def nacti_epizody(cesta_txt: Path, cas: np.ndarray):
    """
    Z .atr přečte změny rytmu a vrátí problémové úseky
    jako seznam (začátek_s, konec_s, zkratka_rytmu).

    Změna rytmu je v anotaci zapsaná v aux_note, např. "(AFIB" nebo "(N".
    Rytmus platí od své anotace až do další změny rytmu.
    """
    if wfdb is None:
        print("Knihovna wfdb není nainstalovaná, anotace se nenačtou.")
        return []

    relativni = cesta_txt.relative_to(SLOZKA_TXT).with_suffix("")
    zaklad = SLOZKA_ZDROJ / relativni
    soubor_anotaci = Path(f"{zaklad}.{PRIPONA_ANOTACI}")

    if not soubor_anotaci.exists():
        print("Anotace nenalezeny:", soubor_anotaci)
        return []

    anotace = wfdb.rdann(str(zaklad), PRIPONA_ANOTACI)

    zmeny = []
    for vzorek, poznamka in zip(anotace.sample, anotace.aux_note):
        poznamka = (poznamka or "").strip("\x00 \t")
        if poznamka.startswith("("):
            zmeny.append((int(vzorek), poznamka[1:]))

    posledni_index = len(cas) - 1
    epizody = []

    for i, (vzorek, rytmus) in enumerate(zmeny):
        if rytmus in NORMALNI_RYTMY:
            continue

        konec_vzorek = zmeny[i + 1][0] if i + 1 < len(zmeny) else posledni_index
        zacatek = float(cas[min(max(vzorek, 0), posledni_index)])
        konec = float(cas[min(max(konec_vzorek, 0), posledni_index)])
        if konec <= zacatek:
            continue

        # Dvě navazující anotace stejného rytmu spojíme do jedné epizody.
        if epizody and epizody[-1][2] == rytmus and epizody[-1][1] >= zacatek:
            epizody[-1][1] = konec
        else:
            epizody.append([zacatek, konec, rytmus])

    return [tuple(epizoda) for epizoda in epizody]


def popis_rytmu(zkratka: str) -> str:
    popis = POPISY_RYTMU.get(zkratka)
    return f"{zkratka} ({popis})" if popis else zkratka


def hh_mm_ss(sekundy: float) -> str:
    h, zbytek = divmod(sekundy, 3600)
    m, s = divmod(zbytek, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:05.2f}"


def vypis_a_uloz_epizody(cesta_txt: Path, epizody):
    nazev = cesta_txt.stem
    if not epizody:
        print(f"Záznam {nazev}: žádné problémové úseky v anotacích.")
        return

    celkem = sum(konec - zacatek for zacatek, konec, _ in epizody)
    print(f"Záznam {nazev}: {len(epizody)} problémových úseků, "
          f"celkem {hh_mm_ss(celkem)}")
    print(f"{'č.':>4}  {'od':>11}  {'do':>11}  {'trvání':>11}  rytmus")
    for i, (zacatek, konec, rytmus) in enumerate(epizody, start=1):
        print(f"{i:>4}  {hh_mm_ss(zacatek)}  {hh_mm_ss(konec)}  "
              f"{hh_mm_ss(konec - zacatek)}  {popis_rytmu(rytmus)}")

    if ULOZIT_PREHLED:
        prehled = pd.DataFrame(
            [
                {
                    "cislo": i,
                    "rytmus": rytmus,
                    "popis": POPISY_RYTMU.get(rytmus, ""),
                    "od_s": round(zacatek, 3),
                    "do_s": round(konec, 3),
                    "trvani_s": round(konec - zacatek, 3),
                    "od": hh_mm_ss(zacatek),
                    "do": hh_mm_ss(konec),
                    "trvani": hh_mm_ss(konec - zacatek),
                }
                for i, (zacatek, konec, rytmus) in enumerate(epizody, start=1)
            ]
        )
        slozka = SLOZKA_TXT / "_problemove_useky"
        slozka.mkdir(parents=True, exist_ok=True)
        cesta_csv = slozka / f"{nazev}_problemy.csv"
        prehled.to_csv(cesta_csv, sep=";", index=False, encoding="utf-8-sig")
        print("Přehled uložen do:", cesta_csv)


# ---------------------------------------------------------------------------
# Stav aplikace
# ---------------------------------------------------------------------------

zaznamy = najdi_zaznamy(SLOZKA_TXT)
if not zaznamy:
    raise FileNotFoundError(f"Ve složce nejsou žádné TXT soubory: {SLOZKA_TXT}")

nazvy = [cesta.stem for cesta in zaznamy]
print("Nalezené záznamy:", ", ".join(nazvy))

index_zaznamu = nazvy.index(POCATECNI_ZAZNAM) if POCATECNI_ZAZNAM in nazvy else 0

cas = np.array([0.0, 1.0])
signaly = np.zeros((2, 1))
kanaly = []
epizody = []
maska_problemu = np.zeros(2, dtype=bool)
zacatek_zaznamu = 0.0
konec_zaznamu = 1.0
delka_okna = 10.0

index_problemu = -1
posledni_skok = None

osy = []
krivky = []
krivky_problemu = []
stitky = []
znacky_posuvniku = []


# ---------------------------------------------------------------------------
# Okno a ovládací prvky
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(13, 8.5))
titulek = fig.suptitle("", y=0.985)
text_problemu = fig.text(0.5, 0.935, "", ha="center", fontsize=10)
OBLAST_GRAFU = dict(left=0.10, right=0.97, top=0.91, bottom=0.36)

# Posuvník určuje začátek zobrazeného úseku.
osa_posuvniku = fig.add_axes([0.18, 0.27, 0.70, 0.03])
posuvnik = Slider(osa_posuvniku, "Začátek [s]", 0.0, 1.0,
                  valinit=0.0, valfmt="%.2f")

# Volba délky časového okna.
osa_voleb = fig.add_axes([0.06, 0.03, 0.12, 0.17])
volby = RadioButtons(osa_voleb, ("5 s", "10 s", "30 s", "60 s"), active=1)
osa_voleb.set_title("Délka okna", fontsize=9)

# Přepínání záznamů.
tlacitko_zpet = Button(fig.add_axes([0.24, 0.14, 0.14, 0.055]), "<< Předchozí záznam")
tlacitko_dal = Button(fig.add_axes([0.39, 0.14, 0.14, 0.055]), "Další záznam >>")
pole_zaznamu = TextBox(fig.add_axes([0.63, 0.14, 0.07, 0.055]), "Záznam: ",
                       initial=nazvy[index_zaznamu])

# Skoky mezi problémovými úseky.
tlacitko_problem_zpet = Button(fig.add_axes([0.24, 0.07, 0.14, 0.055]),
                               "<< Předchozí problém", color="#f6d5d5",
                               hovercolor="#eeb0b0")
tlacitko_problem_dal = Button(fig.add_axes([0.39, 0.07, 0.14, 0.055]),
                              "Další problém >>", color="#f6d5d5",
                              hovercolor="#eeb0b0")

info_epizody = fig.text(0.56, 0.09, "", fontsize=9, color="dimgray")
info_text = fig.text(0.24, 0.025, "", fontsize=9)


def postav_osy(nove_kanaly):
    """Vytvoří jeden graf pro každý kanál (počet se může mezi záznamy lišit)."""
    for osa in osy:
        osa.remove()
    osy.clear()
    krivky.clear()
    krivky_problemu.clear()
    stitky.clear()

    mrizka = fig.add_gridspec(len(nove_kanaly), 1, hspace=0.35, **OBLAST_GRAFU)
    prvni_osa = None

    for i, nazev in enumerate(nove_kanaly):
        osa = fig.add_subplot(mrizka[i, 0], sharex=prvni_osa)
        if prvni_osa is None:
            prvni_osa = osa

        # Problémové úseky jako světle červené pozadí.
        for zacatek, konec, _ in epizody:
            osa.axvspan(zacatek, konec, color=BARVA_PROBLEMU, alpha=0.12,
                        linewidth=0, zorder=0)

        krivka, = osa.plot([], [], linewidth=0.8, color="tab:blue")
        krivka_problemu, = osa.plot([], [], linewidth=0.9, color=BARVA_PROBLEMU)

        osa.set_ylabel(nazev, fontsize=9)
        osa.grid(True, alpha=0.3)
        if i < len(nove_kanaly) - 1:
            osa.tick_params(labelbottom=False)

        osy.append(osa)
        krivky.append(krivka)
        krivky_problemu.append(krivka_problemu)

    osy[-1].set_xlabel("Čas od začátku záznamu [s]")


def aktualizuj(_=None):
    if not osy:
        return

    zacatek = float(posuvnik.val)
    konec = min(zacatek + delka_okna, konec_zaznamu)

    # U konce záznamu posuneme okno zpět, aby nezůstalo prázdné.
    zacatek = max(zacatek_zaznamu, konec - delka_okna)

    # Najdeme pouze vzorky v aktuálním časovém okně.
    prvni = np.searchsorted(cas, zacatek, side="left")
    posledni = np.searchsorted(cas, konec, side="right")
    cas_okna = cas[prvni:posledni]
    maska_okna = maska_problemu[prvni:posledni]

    for i, (osa, krivka, krivka_problemu) in enumerate(
            zip(osy, krivky, krivky_problemu)):
        hodnoty = signaly[prvni:posledni, i]
        krivka.set_data(cas_okna, hodnoty)
        # Červená křivka jen v problémových úsecích, jinde NaN (nekreslí se).
        krivka_problemu.set_data(cas_okna, np.where(maska_okna, hodnoty, np.nan))
        osa.set_xlim(zacatek, konec)
        osa.relim()
        osa.autoscale_view(scalex=False, scaley=True)

    # Popisky problémů v horním grafu.
    for stitek in stitky:
        stitek.remove()
    stitky.clear()

    viditelne = [(z, k, r) for z, k, r in epizody if z < konec and k > zacatek]
    for z, _, rytmus in viditelne:
        stitky.append(osy[0].text(
            max(z, zacatek), 0.97, f" {rytmus}",
            transform=osy[0].get_xaxis_transform(),
            color=BARVA_PROBLEMU, fontsize=9, fontweight="bold",
            va="top", ha="left", clip_on=True,
        ))

    titulek.set_text(
        f"EKG {nazvy[index_zaznamu]} — úsek {zacatek:.2f} až {konec:.2f} s "
        f"({hh_mm_ss(zacatek)} – {hh_mm_ss(konec)})"
    )

    if viditelne:
        casti = [
            f"{popis_rytmu(r)}: od {hh_mm_ss(z)} do {hh_mm_ss(k)} "
            f"(trvá {hh_mm_ss(k - z)})"
            for z, k, r in viditelne[:2]
        ]
        if len(viditelne) > 2:
            casti.append(f"+ další {len(viditelne) - 2}")
        text_problemu.set_text("Problém v okně: " + "  |  ".join(casti))
        text_problemu.set_color(BARVA_PROBLEMU)
    else:
        text_problemu.set_text("V tomto okně není žádný problémový úsek")
        text_problemu.set_color("seagreen")

    if epizody:
        celkem = sum(k - z for z, k, _ in epizody)
        poradi = ""
        if posledni_skok is not None and abs(posuvnik.val - posledni_skok) < 1e-9:
            poradi = f"Úsek {index_problemu + 1}/{len(epizody)}\n"
        info_epizody.set_text(
            f"{poradi}Problémových úseků: {len(epizody)}\n"
            f"Celkem: {hh_mm_ss(celkem)} "
            f"({100 * celkem / max(konec_zaznamu, 1e-9):.1f} % záznamu)"
        )
    else:
        info_epizody.set_text("Žádné problémové úseky\n(nebo chybí soubor .atr)")

    fig.canvas.draw_idle()


def otevri_zaznam(novy_index: int):
    global index_zaznamu, cas, kanaly, signaly, epizody, maska_problemu
    global zacatek_zaznamu, konec_zaznamu, index_problemu, posledni_skok

    novy_index %= len(zaznamy)
    cesta = zaznamy[novy_index]

    titulek.set_text(f"Načítám {cesta.name} ...")
    fig.canvas.draw()
    fig.canvas.flush_events()

    try:
        novy_cas, nove_kanaly, nove_signaly = nacti_zaznam(cesta)
    except Exception as chyba:
        print("CHYBA u souboru:", cesta)
        print(chyba)
        titulek.set_text(f"Chyba u {cesta.name}: {chyba}")
        pole_zaznamu.set_val(nazvy[index_zaznamu])
        fig.canvas.draw_idle()
        return

    try:
        nove_epizody = nacti_epizody(cesta, novy_cas)
    except Exception as chyba:
        print("CHYBA při čtení anotací:", chyba)
        nove_epizody = []

    index_zaznamu = novy_index
    cas, kanaly, signaly = novy_cas, nove_kanaly, nove_signaly
    epizody = nove_epizody
    zacatek_zaznamu = float(cas[0])
    konec_zaznamu = float(cas[-1])
    index_problemu = -1
    posledni_skok = None

    # Maska vzorků, které leží v problémovém úseku.
    maska_problemu = np.zeros(len(cas), dtype=bool)
    for z, k, _ in epizody:
        maska_problemu[np.searchsorted(cas, z, "left"):
                       np.searchsorted(cas, k, "right")] = True

    vypis_a_uloz_epizody(cesta, epizody)
    postav_osy(kanaly)

    # Nový rozsah posuvníku podle délky záznamu.
    posuvnik.valmin = zacatek_zaznamu
    posuvnik.valmax = konec_zaznamu
    posuvnik.ax.set_xlim(zacatek_zaznamu, konec_zaznamu)

    # Červené značky na posuvníku = mapa problémů v celém záznamu.
    for znacka in znacky_posuvniku:
        znacka.remove()
    znacky_posuvniku.clear()
    for z, k, _ in epizody:
        znacky_posuvniku.append(posuvnik.ax.axvspan(
            z, k, ymin=0.0, ymax=0.25, color=BARVA_PROBLEMU, linewidth=0))

    info_text.set_text(
        f"Záznam {index_zaznamu + 1}/{len(zaznamy)}  |  délka {hh_mm_ss(konec_zaznamu)}"
        f"  |  ←/→ posun   PageUp/PageDown záznam   B/N předchozí/další problém"
    )
    pole_zaznamu.set_val(nazvy[index_zaznamu])
    posuvnik.set_val(zacatek_zaznamu)
    aktualizuj()


def skoc_na_problem(smer: int):
    """Posune okno na začátek předchozího (-1) nebo dalšího (+1) problému."""
    global index_problemu, posledni_skok

    pocet = len(epizody)
    if pocet == 0:
        text_problemu.set_text("Tento záznam nemá žádné problémové úseky")
        text_problemu.set_color("seagreen")
        fig.canvas.draw_idle()
        return

    if posledni_skok is not None and abs(posuvnik.val - posledni_skok) < 1e-9:
        # Jsme na úseku z minulého skoku -> jen o jeden dál/zpět.
        novy = (index_problemu + smer) % pocet
    else:
        # Uživatel se mezitím posunul -> hledáme podle aktuální pozice.
        pozice = float(posuvnik.val) + 0.1 * delka_okna
        if smer > 0:
            novy = next((i for i, (z, _, _) in enumerate(epizody) if z > pozice), 0)
        else:
            novy = next((i for i in range(pocet - 1, -1, -1)
                         if epizody[i][0] < pozice), pocet - 1)

    index_problemu = novy
    zacatek_problemu = epizody[novy][0]

    # Začátek problému dáme kousek od levého okraje, ať je vidět i přechod.
    nejpozdeji = max(zacatek_zaznamu, konec_zaznamu - delka_okna)
    cil = float(np.clip(zacatek_problemu - 0.1 * delka_okna,
                        zacatek_zaznamu, nejpozdeji))
    posledni_skok = cil
    posuvnik.set_val(cil)


def vyber_podle_nazvu(text: str):
    text = text.strip().removesuffix(".txt")
    if not text or text == nazvy[index_zaznamu]:
        return

    if text in nazvy:
        otevri_zaznam(nazvy.index(text))
        return

    # Povolí i "5" místo "05".
    if text.isdigit():
        for i, nazev in enumerate(nazvy):
            if nazev.isdigit() and int(nazev) == int(text):
                otevri_zaznam(i)
                return

    print("Záznam nenalezen:", text)
    titulek.set_text(f"Záznam '{text}' nebyl nalezen")
    pole_zaznamu.set_val(nazvy[index_zaznamu])
    fig.canvas.draw_idle()


def zmen_delku(hodnota):
    global delka_okna
    delka_okna = float(hodnota.split()[0])
    aktualizuj()


def klavesa(udalost):
    # Šipky: posun v čase.
    if udalost.key in ("left", "right"):
        smer = 1 if udalost.key == "right" else -1
        nejpozdeji = max(zacatek_zaznamu, konec_zaznamu - delka_okna)
        aktualni = min(float(posuvnik.val), nejpozdeji)
        novy = np.clip(aktualni + smer * delka_okna / 2,
                       zacatek_zaznamu, nejpozdeji)
        posuvnik.set_val(float(novy))

    # PageUp / PageDown: předchozí / další záznam.
    elif udalost.key == "pageup":
        otevri_zaznam(index_zaznamu - 1)
    elif udalost.key == "pagedown":
        otevri_zaznam(index_zaznamu + 1)

    # B / N: předchozí / další problémový úsek.
    elif udalost.key == "b":
        skoc_na_problem(-1)
    elif udalost.key == "n":
        skoc_na_problem(+1)


posuvnik.on_changed(aktualizuj)
volby.on_clicked(zmen_delku)
tlacitko_zpet.on_clicked(lambda _: otevri_zaznam(index_zaznamu - 1))
tlacitko_dal.on_clicked(lambda _: otevri_zaznam(index_zaznamu + 1))
tlacitko_problem_zpet.on_clicked(lambda _: skoc_na_problem(-1))
tlacitko_problem_dal.on_clicked(lambda _: skoc_na_problem(+1))
pole_zaznamu.on_submit(vyber_podle_nazvu)
fig.canvas.mpl_connect("key_press_event", klavesa)

otevri_zaznam(index_zaznamu)
print("\nGraf je připraven.")
plt.show()