from pathlib import Path
import pandas as pd
import wfdb
import numpy as np
ZDROJOVA_CESTA = Path(r"C:\Zk_Furst_2\files")
VYSTUPNI_CESTA = Path(r"C:\Zk_Furst_2\Vystup")
LIMIT_ZAZNAMU = None

def preved_jeden_zaznam(cesta_k_hea: Path) -> None:
    zaklad_zaznamu = cesta_k_hea.with_suffix("") # # Odstranění přípony: například 00.hea -> 00.
    relativni_slozka = cesta_k_hea.parent.relative_to(ZDROJOVA_CESTA)
    cilova_slozka = VYSTUPNI_CESTA / relativni_slozka
    cilova_slozka.mkdir(parents=True, exist_ok=True)
    print("Načítám záznam:", zaklad_zaznamu.name)

    
    signaly, informace = wfdb.rdsamp(str(zaklad_zaznamu)) # Načtení signálů z .dat podle údajů v .hea.

    frekvence = float(informace["fs"])
    nazvy_kanalu = informace["sig_name"]
    jednotky = informace["units"]
    if signaly is None:
        raise ValueError(f"Záznam neobsahuje žádná data signálu: {zaklad_zaznamu}")

    pocet_vzorku = len(signaly)

    
    cas = np.arange(pocet_vzorku) / frekvence

    nazvy_sloupcu = [f"EKG_{i + 1}_{nazev}_{jednotka}"for i, (nazev, jednotka) in enumerate(zip(nazvy_kanalu, jednotky))]

    tabulka_signalu = pd.DataFrame(signaly, columns=nazvy_sloupcu)
    tabulka_signalu.insert(0, "time_s", cas)

    cesta_k_signalu_txt = cilova_slozka / f"{zaklad_zaznamu.name}.txt" ##################TADY SI DAL CILOVOU SLOŽKU MÍSTO VÝSTUPNI

    print("Ukládám TXT, chvíli to může trvat...")

    tabulka_signalu.to_csv(
        cesta_k_signalu_txt,
        sep="\t",
        index=False,
        encoding="utf-8",
        float_format="%.8f",
        chunksize=100_000,)

    print(
        "Hotovo:", zaklad_zaznamu.name,
        "| Počet vzorků:", pocet_vzorku,
        "| Počet kanálů:", len(nazvy_kanalu),
        "| Frekvence:", frekvence, "Hz",)
    print("Uloženo do:", cesta_k_signalu_txt)


def main() -> None:
    if not ZDROJOVA_CESTA.is_dir():
        print("CHYBA: Zdrojová složka neexistuje:")
        print(ZDROJOVA_CESTA)
        return

    VYSTUPNI_CESTA.mkdir(parents=True, exist_ok=True)

    # Vyhledání hlaviček i v podsložkách a seřazení podle cesty.
    seznam_hea = sorted(ZDROJOVA_CESTA.rglob("*.hea"))

    if LIMIT_ZAZNAMU is not None:
        seznam_hea = seznam_hea[:LIMIT_ZAZNAMU]

    if not seznam_hea:
        print("CHYBA: Nebyly nalezeny žádné soubory .hea.")
        return

    print("Záznamů k převodu:", len(seznam_hea))
    print("Výstupní složka:", VYSTUPNI_CESTA)

    pocet_uspesnych = 0
    pocet_chyb = 0

    for cesta_k_hea in seznam_hea:
        try:
            preved_jeden_zaznam(cesta_k_hea)
            pocet_uspesnych += 1

        except Exception as chyba:
            print("CHYBA u souboru:", cesta_k_hea)
            print(chyba)
            pocet_chyb += 1

    print("\nPřevod dokončen.")
    print("Úspěšně převedeno:", pocet_uspesnych)
    print("Počet chyb:", pocet_chyb)


if __name__ == "__main__":
    main()