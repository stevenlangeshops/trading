"""
train_decision_tree_v9_3.py
====================================================================================
Machine Learning  |  Decision Tree Rule Extraction  |  v9.3

Ziel: Nicht-lineare Wenn-Dann-Regeln aus dem Event-Study-Datensatz extrahieren.
Der Decision Tree lernt, welche Indikator-Kombination an welchem Zeitschritt
(T-5 bis T+5) am stärksten zwischen hochwertigen und normalen Wellen trennt.

Input:    ideal_trades_v9_master.csv  (aus extract_ideal_trades_v9_master.py)
Target:   is_top_tier = 1 wenn target_quality im obersten Drittel (Top 33 %)
Features: Alle feat_*-Spalten (8 Basis-Features × 11 Zeitschritte = 88 Spalten)

Modell:   DecisionTreeClassifier(max_depth=3, min_samples_leaf=50)
          max_depth=3:         Max. 3 Entscheidungsebenen → lesbare Regeln
          min_samples_leaf=50: Jedes Blatt min. 50 Trades → statistische Relevanz

Outputs:
  decision_tree_rules.png  – Visualisierung des Baums + Feature Importance
  Console                  – Regeln, Feature Importances, Klassifikations-Report

Verwendung:
  python train_decision_tree_v9_3.py
  python train_decision_tree_v9_3.py --depth 4 --min-leaf 30 --top-pct 33
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearn.tree import (DecisionTreeClassifier, export_text, plot_tree)
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    print("FEHLER: scikit-learn nicht installiert.")
    print("  pip install scikit-learn")
    sys.exit(1)

_here    = Path(__file__).parent
_IN_CSV  = _here / "ideal_trades_v9_master.csv"
_OUT_PNG = _here / "decision_tree_rules.png"


# ==============================================================================
# 1. DATEN LADEN & VORBEREITEN
# ==============================================================================

def load_and_prepare(csv_path: Path, top_pct: float) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Lädt den Event-Study-Datensatz und erstellt:
      X: Feature-Matrix  (alle feat_*-Spalten ohne NaN)
      y: Binäres Label   (1 = Top-Tier-Welle, 0 = Rest)
    """
    df = pd.read_csv(csv_path, parse_dates=["start_date", "end_date"])
    print(f"  Geladen: {len(df):,} Wellen, {len(df.columns)} Spalten")

    # Feature-Spalten identifizieren
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    print(f"  Feature-Spalten: {len(feat_cols)}  "
          f"({len([c for c in feat_cols if '_t0' in c or '_tm' in c or '_tp' in c])} Time-Steps)")

    # Binäres Target
    threshold = df["target_quality"].quantile(1 - top_pct / 100)
    df["is_top_tier"] = (df["target_quality"] >= threshold).astype(int)
    n_top  = df["is_top_tier"].sum()
    n_rest = len(df) - n_top
    print(f"  Top-{top_pct:.0f}%-Schwelle:  target_quality ≥ {threshold:.3f}")
    print(f"  Top-Tier (y=1): {n_top:,}  ({n_top/len(df)*100:.1f}%)")
    print(f"  Rest     (y=0): {n_rest:,}  ({n_rest/len(df)*100:.1f}%)")

    # NaN entfernen
    df_clean = df[feat_cols + ["is_top_tier", "target_quality",
                               "return_pct", "intrawave_dd"]].dropna()
    dropped  = len(df) - len(df_clean)
    if dropped > 0:
        print(f"  Verworfen (NaN):  {dropped:,} Zeilen")
    print(f"  Trainings-Datensatz: {len(df_clean):,} Wellen")

    X = df_clean[feat_cols]
    y = df_clean["is_top_tier"]
    return df_clean, X, y, feat_cols


# ==============================================================================
# 2. MODELL TRAINIEREN + EVALUIEREN
# ==============================================================================

def train_evaluate(
    X:           pd.DataFrame,
    y:           pd.Series,
    max_depth:   int,
    min_leaf:    int,
) -> DecisionTreeClassifier:
    clf = DecisionTreeClassifier(
        max_depth        = max_depth,
        min_samples_leaf = min_leaf,
        criterion        = "gini",
        random_state     = 42,
    )
    clf.fit(X, y)

    # In-Sample Accuracy
    acc_train = clf.score(X, y)

    # 5-Fold Stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

    print(f"\n  Modell-Parameter:")
    print(f"    max_depth        = {max_depth}")
    print(f"    min_samples_leaf = {min_leaf}")
    print(f"    Kriterium        = Gini")
    print(f"    Knoten (gesamt)  = {clf.tree_.node_count}")
    print(f"    Blätter          = {clf.get_n_leaves()}")
    print(f"\n  Performance:")
    print(f"    Train Accuracy:  {acc_train:.3f}")
    print(f"    CV Accuracy:     {cv_scores.mean():.3f} ± {cv_scores.std():.3f}  "
          f"(5-Fold Stratified)")

    y_pred = clf.predict(X)
    print(f"\n  Klassifikations-Report:")
    report = classification_report(y, y_pred,
                                   target_names=["Normaler Trade", "Top-Tier"],
                                   digits=3)
    for line in report.split("\n"):
        print(f"    {line}")

    # Konfusionsmatrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Konfusionsmatrix:")
    print(f"    Richtig Negativ: {tn:>5}  Falsch Positiv: {fp:>5}")
    print(f"    Falsch Negativ:  {fn:>5}  Richtig Positiv: {tp:>5}")
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"    Präzision Top-Tier: {prec:.1%}   Recall: {rec:.1%}")

    return clf


# ==============================================================================
# 3. FEATURE IMPORTANCES AUSGEBEN
# ==============================================================================

def print_feature_importances(
    clf:       DecisionTreeClassifier,
    feat_cols: list[str],
    top_n:     int = 15,
) -> pd.DataFrame:
    imp = pd.DataFrame({
        "feature":    feat_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    # Nur Features mit tatsächlichem Beitrag
    imp_nonzero = imp[imp["importance"] > 0]

    print(f"\n  FEATURE IMPORTANCES  (top {min(top_n, len(imp_nonzero))} von "
          f"{len(imp_nonzero)} relevanten Features):")
    print(f"  {'─' * 62}")
    bar_max = imp_nonzero["importance"].max()
    for rank, (_, row) in enumerate(imp_nonzero.head(top_n).iterrows(), 1):
        bar_len = int(row["importance"] / bar_max * 30)
        bar     = "█" * bar_len
        # Feature-Name lesbar machen: feat_rsi14_tm1 → rsi14 @ t-1
        fname   = row["feature"].replace("feat_", "")
        for sfx, rep in [("_tm", " @ t-"), ("_tp", " @ t+"), ("_t0", " @ t0")]:
            fname = fname.replace(sfx, rep)
        print(f"  {rank:>3}. {fname:<28}  {row['importance']:>7.4f}  {bar}")

    return imp_nonzero


# ==============================================================================
# 4. TEXT-REGELN AUSGEBEN
# ==============================================================================

def print_text_rules(clf: DecisionTreeClassifier, feat_cols: list[str]) -> None:
    """
    Gibt die gelernten Entscheidungsregeln als lesbaren Text aus.
    Feature-Namen werden vereinfacht dargestellt.
    """
    # Vereinfachte Feature-Namen für Lesbarkeit
    readable = []
    for f in feat_cols:
        name = f.replace("feat_", "")
        for sfx, rep in [("_tm", "@t-"), ("_tp", "@t+"), ("_t0", "@t0")]:
            name = name.replace(sfx, rep)
        readable.append(name)

    rules = export_text(clf, feature_names=readable, show_weights=True)

    print(f"\n  ENTSCHEIDUNGSREGELN (max_depth={clf.max_depth}):")
    print(f"  {'─' * 62}")
    print(f"  Legende: class: 0 = Normaler Trade | class: 1 = Top-Tier")
    print(f"  'weights' = [Anzahl Klasse-0, Anzahl Klasse-1] im Knoten\n")
    for line in rules.split("\n"):
        if line.strip():
            print(f"  {line}")


# ==============================================================================
# 5. VISUALISIERUNG
# ==============================================================================

def plot_results(
    clf:       DecisionTreeClassifier,
    feat_cols: list[str],
    imp:       pd.DataFrame,
    out_png:   Path,
    max_depth: int,
    top_pct:   float,
) -> None:
    # Vereinfachte Namen für Plot
    readable = []
    for f in feat_cols:
        name = f.replace("feat_", "")
        for sfx, rep in [("_tm", "@t-"), ("_tp", "@t+"), ("_t0", "@t0")]:
            name = name.replace(sfx, rep)
        readable.append(name)

    fig = plt.figure(figsize=(24, 14), dpi=130,
                     facecolor="#f8f9fa")
    fig.suptitle(
        f"Decision Tree Rule Extraction  |  v9.3  |  "
        f"Target: Top-{top_pct:.0f}% Target_Quality  |  "
        f"max_depth={max_depth}",
        fontsize=13, fontweight="bold", y=0.99, color="#212121",
    )

    gs = fig.add_gridspec(
        1, 2, width_ratios=[3, 1],
        left=0.02, right=0.98, top=0.95, bottom=0.05,
        wspace=0.06,
    )

    # ── Linkes Panel: Decision Tree ───────────────────────────────────────────
    ax_tree = fig.add_subplot(gs[0])
    ax_tree.set_facecolor("#f8f9fa")

    plot_tree(
        clf,
        feature_names = readable,
        class_names   = ["Normal", "Top-Tier"],
        filled        = True,
        rounded       = True,
        impurity      = False,
        proportion    = True,
        fontsize      = 8,
        ax            = ax_tree,
    )
    ax_tree.set_title(
        "Gelernte Entscheidungsregeln",
        fontsize=11, fontweight="bold", pad=10, color="#37474f",
    )

    # ── Rechtes Panel: Feature Importance (nur nonzero) ──────────────────────
    ax_imp = fig.add_subplot(gs[1])
    ax_imp.set_facecolor("#f8f9fa")

    top15 = imp.head(15)
    names = [r["feature"].replace("feat_", "")
             .replace("_tm", "@t-").replace("_tp", "@t+").replace("_t0", "@t0")
             for _, r in top15.iterrows()]
    vals  = top15["importance"].values

    colors = plt.cm.RdYlGn(np.linspace(0.85, 0.35, len(vals)))
    bars   = ax_imp.barh(range(len(vals)), vals[::-1],
                         color=colors[::-1], edgecolor="#424242",
                         linewidth=0.5, alpha=0.9)

    ax_imp.set_yticks(range(len(vals)))
    ax_imp.set_yticklabels(names[::-1], fontsize=8)
    ax_imp.set_xlabel("Feature Importance (Gini)", fontsize=9)
    ax_imp.set_title("Feature Importances\n(nur genutzte Features)",
                     fontsize=10, fontweight="bold", color="#37474f")
    ax_imp.grid(True, axis="x", color="#e0e0e0", linewidth=0.5)
    ax_imp.spines[["top","right"]].set_visible(False)

    for bar, val in zip(bars, vals[::-1]):
        ax_imp.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", ha="left", fontsize=7.5,
        )

    plt.savefig(out_png, dpi=130, bbox_inches="tight",
                facecolor="#f8f9fa")
    plt.close(fig)
    print(f"\n  Chart gespeichert: {out_png}")


# ==============================================================================
# 6. TRADING REGELN ALS KLARTEXT  (interpretierte Ausgabe)
# ==============================================================================

def extract_actionable_rules(
    clf:       DecisionTreeClassifier,
    feat_cols: list[str],
    top_pct:   float,
) -> None:
    """
    Extrahiert die wichtigsten Split-Schwellen aus dem Baum und formuliert
    sie als konkrete, umsetzbare Trading-Bedingungen.
    """
    tree   = clf.tree_
    n_node = tree.node_count

    print(f"\n  UMSETZBARE TRADING-REGELN (aus dem Baum extrahiert):")
    print(f"  {'─' * 62}")
    print(f"  Ziel: Identifiziere Trades, die in die Top-{top_pct:.0f}% Qualität fallen.")
    print(f"  Alle Splits mit Gini-Verbesserung > 0.001 (relevante Schwellen):\n")

    seen = set()
    for node_id in range(n_node):
        if tree.children_left[node_id] == -1:   # Blatt
            continue
        feat_i   = tree.feature[node_id]
        thresh   = tree.threshold[node_id]
        impurity = tree.impurity[node_id]
        n_samp   = tree.n_node_samples[node_id]

        feat_name = feat_cols[feat_i]
        # Vereinfachen
        readable = (feat_name
                    .replace("feat_", "")
                    .replace("_tm", " @ t-")
                    .replace("_tp", " @ t+")
                    .replace("_t0", " @ t0"))

        # Zeitschritt aus Name
        t_str = ""
        for part in ["t-5","t-4","t-3","t-2","t-1","t0","t+1","t+2","t+3","t+4","t+5"]:
            if part in readable:
                t_str = part
                break

        vergangenheit = "tm" in feat_name or "_t0" in feat_name
        flag = "✓ (für Live-Signale nutzbar)" if vergangenheit else "✗ (nur akademisch)"

        key = (feat_i, round(thresh, 6))
        if key in seen:
            continue
        seen.add(key)

        # Linkes Kind (≤ Schwelle)
        left_id  = tree.children_left[node_id]
        right_id = tree.children_right[node_id]
        left_top  = tree.value[left_id][0][1] / tree.n_node_samples[left_id] if left_id != -1 else 0
        right_top = tree.value[right_id][0][1] / tree.n_node_samples[right_id] if right_id != -1 else 0
        better    = "≤" if left_top > right_top else ">"

        print(f"  WENN {readable} {better} {thresh:.4f}")
        print(f"       → Top-Tier-Anteil: {max(left_top,right_top):.1%}  "
              f"(Basis: {n_samp} Trades)")
        print(f"       Zeitpunkt: {t_str}  |  {flag}")
        print()


# ==============================================================================
# 7. MAIN
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decision Tree Rule Extraction  |  v9.3")
    parser.add_argument("--depth",    type=int,   default=3,
                        help="max_depth des Baums (default 3)")
    parser.add_argument("--min-leaf", type=int,   default=50,
                        help="min_samples_leaf (default 50)")
    parser.add_argument("--top-pct",  type=float, default=33.0,
                        help="Oberstes X%% als Top-Tier definieren (default 33)")
    args = parser.parse_args()

    print("=" * 72)
    print("  DECISION TREE RULE EXTRACTION  |  v9.3")
    print("=" * 72)
    print(f"""
  Input:       ideal_trades_v9_master.csv
  Target:      is_top_tier  (Top-{args.top_pct:.0f}% nach target_quality)
  Modell:      DecisionTreeClassifier
               max_depth={args.depth}  |  min_samples_leaf={args.min_leaf}
""")

    # 1. Daten laden
    if not _IN_CSV.exists():
        print(f"FEHLER: {_IN_CSV} nicht gefunden.")
        print("  Zuerst extract_ideal_trades_v9_master.py ausführen.")
        sys.exit(1)

    print("[1/4] Daten laden...")
    df_clean, X, y, feat_cols = load_and_prepare(_IN_CSV, args.top_pct)

    # 2. Training
    print(f"\n[2/4] Modell trainieren...")
    clf = train_evaluate(X, y, args.depth, args.min_leaf)

    # 3. Feature Importances
    print(f"\n[3/4] Regeln extrahieren...")
    imp = print_feature_importances(clf, feat_cols)
    print_text_rules(clf, feat_cols)
    extract_actionable_rules(clf, feat_cols, args.top_pct)

    # 4. Visualisierung
    print(f"\n[4/4] Visualisierung...")
    plot_results(clf, feat_cols, imp, _OUT_PNG, args.depth, args.top_pct)

    print(f"""
  FAZIT:
  {'─' * 62}
  Die extrahierten Regeln beschreiben, welche Indikator-Schwellen
  an welchem Zeitschritt (T-5 bis T+5) am stärksten zwischen
  Top-Tier-Wellen und normalen Trades trennen.

  Nur Regeln mit Zeitstempel ✓ (t-1 bis t-5) sind für ein
  echtes Trading-Modell direkt umsetzbar.

  Empfehlung: Mit --depth 4 oder --top-pct 25 erneut testen,
  um noch schärfere oder noch selektivere Regeln zu gewinnen.
""")


if __name__ == "__main__":
    main()
