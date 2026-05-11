# Systematic Trend Following – Entwicklungshistorie v6 bis v8.3

**Projekt:** Trading Bot – Universum 260 US-Aktien  
**Zeitraum der Entwicklung:** 2026  
**Autor:** Lead Quant Software Engineer  
**Ziel:** Ablösung des Deep-Learning-Ansatzes (v5 / PyTorch / LSTM) durch ein vollständig transparentes, regelbasiertes Trend-Following & Momentum-System.

---

## Philosophischer Ausgangspunkt

Version 5 des Systems basierte auf einem neuronalen Netz (LSTM/Transformer), das aus 18 technischen Features Z-Scores berechnete und damit ein Ranking der 260 Aktien erzeugte. Das System war eine **Black Box**: Entscheidungen waren nicht erklärbar, die Performance hing von zufälligen Seeds ab, und das Live-Verhalten war schwer zu debuggen.

**Die neue Philosophie (ab v6):**

> *"Wir raten nicht, wir messen."*

Alle Regeln sind explizit, deterministisch und retrospektiv vollständig überprüfbar. Jeder Trade hat einen nachvollziehbaren Grund, jeder Parameter ist begründet messbar.

Die Infrastruktur (yfinance-Daten, Alpaca-Execution, Telegram-Monitoring, 260-Aktien-Universum in `features/sector_map.json`) bleibt unverändert. Nur das "Gehirn" wird ersetzt.

---

## v6.0 – Quantitative Signal Research (Event Study)

**Skript:** `research_signals.py`  
**Ansatz:** Event Study – statistische Auswertung von 22 binären Signalen auf den 7-Jahres-Tagesdaten des 260-Aktien-Universums.

### Was wir gemacht haben
Aus den bestehenden Features in `engineer.py` (SMA-Ratios, RSI, MACD, Bollinger Bands, ROC, Volume Ratio) wurden binäre Trigger-Signale (True/False) generiert. Für jeden Signal-Tag wurden die Forward Returns nach 5, 20 und 60 Handelstagen berechnet.

**Methodischer Fehler (erkannt und korrigiert):**  
Die starre Betrachtung von Forward Returns mittelt über alle Haltedauern und ignoriert extrem profitable "Fat Tail"-Ausreißer (z. B. NVDA während des KI-Booms). Außerdem fehlte ein simuliertes Risikomanagement, was fehlgeschlagene Breakouts die Statistik unverhältnismäßig nach unten zog.

**Erkenntnisse:**
- Breakout-Signale (`Close > SMA_200`, `Close > High_50d`) zeigen positive Mean Returns bei 20-Tage-Fenstern.
- Mean-Reversion-Signale (`RSI < 30`, `BB_pos < 0`) zeigen hohe Hit-Rates, aber begrenzte Upside.
- Reine Event-Studies sind unzureichend für Trendfolge-Systeme.

---

## v6.0 – Event-basierte Trade-Simulation

**Skript:** `research_trend_following.py`  
**Ansatz:** Anstelle starrer Forward-Return-Fenster: Simulierter Trade mit variablem Hold bis zum SMA-50-Bruch.

### Handelslogik
- **Entry:** `Breakout_50` (Close > High letzter 50 Tage) AND `Close > SMA_200`  
  Alternative: `Breakout_100`
- **Exit:** Close fällt unter SMA_50 → Verkauf am nächsten Open
- Überlappende Signale für denselben Ticker werden ignoriert

### Erkenntnisse
- **Payoff Ratio** für `Breakout_50`: ~2.0–2.5 (Gewinner ~2.5× größer als Verlierer)
- **Max Win** >+100% sichtbar (die KI-Wellen wie NVDA werden teilweise erfasst)
- **Hit-Rate:** ~35–40%
- Problem: Keine Gebühren, kein Portfolio-Management → unrealistische Baseline

---

## v6.0–v6.3 – Portfolio-Backtester (Iterative Entwicklung)

**Skript:** `backtest_v6.py`

### v6.0 – Basis-Backtester
**Parameter:** Startkapital 10.000€, max. 5 Positionen, Gebühr 20€/Order, Exit via SMA_50.

**Problem:** Portfolio "verstopft" ständig (96% Auslastung). Langweilige Trades belegen Slots und blockieren starke neue Ausbrüche (z. B. NVDA).

---

### v6.1 – Dynamische Rotation + ATR-Trailing-Stop
**Änderungen:**
- **Portfolio-Rotation:** Wenn neuer Kandidat >1.5× stärker als schwächste Position → Rotation
- **ATR-Trailing-Stop (3×ATR14):** Ersetzt den trägen SMA_50-Exit (Chandelier Exit)
- Trendstärke-Metrik: `(Close - SMA_200) / SMA_200`

**Problem:** System zu hyperaktiv (496 Trades in 6 Jahren). Payoff gesunken, Gebühren dominieren.

---

### v6.2 – Striktere Rotation + Pyramidisieren
**Änderungen:**
- ATR-Stop auf 4.0× ausgeweitet (mehr Atemraum)
- Rotation nur noch bei Faktor 2.0× UND Zielposition profitabel (keine Rotation aus Verlust-Trades)
- **Pyramidisierung:** Aufstocken bei +20% unrealisierten Gewinn

**Problem:** Katastrophale Einzelverluste bei Fake-Breakouts durch den 4.0× ATR-Stop. Pyramidisierung zu riskant ohne Stop-Anpassung.

---

### v6.3 – Asymmetrischer "Earned" Trailing Stop + Free-Ride
**Änderungen (Kernkonzept, bleibt bis v8.3 erhalten):**

```
Initialer Stop:   Kaufpreis - 2.0 × ATR_14  (eng, schützt vor Fake-Breakouts)
Earned Mode:      Wenn Höchstkurs > Kaufpreis + 2.0 × ATR_14 → Stop auf 3.5 × ATR
Free-Ride:        Beim Pyramidisieren: Stop wird sofort auf max(Stop, Avg_Entry_Price) gesetzt
Regel:            Stop darf niemals sinken
```

**Hardcoded (unveränderlich ab hier):**
- `INITIAL_CAPITAL = 10_000.0`
- `ORDER_FEE = 20.0`

**Erkenntnisse:** Motor funktioniert handwerklich korrekt. Schwäche liegt beim Entry-Signal.

---

## v6.3 – Signal-Optimierung (optimize_entries_v6.py)

**Skript:** `optimize_entries_v6.py`  
**Ansatz:** 5 verschiedene Entry-Signale mit dem v6.3-Motor testen, bei 2 Gebühren-Szenarien (20€ vs. 2€).

### Getestete Signale
| Signal | Logik |
|--------|-------|
| `Breakout_50` | Close > High_50_prev & Close > SMA_200 |
| `Breakout_100` | Close > High_100_prev & Close > SMA_200 |
| `RSI_Dip_Bull` | RSI_14 < 40 & Close > SMA_200 |
| `Double_Oversold` | RSI_14 < 30 & Close < BB_lower |
| `MACD_Crossover` | MACD-Hist dreht >0 & Close > SMA_200 |

### Erkenntnisse
- **Bester Gewinner bei 20€ Gebühren:** `Double_Oversold` (+119.84% Netto)
- **Bei 2€ Gebühren:** `Breakout_50` wieder konkurrenzfähig
- Die 20€ Fixgebühr ist ein massiver Faktor – sie vernichtet häufige, kleine Signale

---

## v6.5 – Kombinatorische Rule Discovery Engine

**Skript:** `discover_rules_v6.py`

### Architektur: Trigger-Filter-Matrix
Statt blinder `itertools.combinations` werden Signale strukturiert in zwei Pools aufgeteilt:

**TRIGGER Pool** (Events, die genau heute von False → True wechseln):
- `Trig_B50`: Breakout über 50-Tage-Hoch
- `Trig_B100`: Breakout über 100-Tage-Hoch
- `Trig_MACDcross`: MACD-Histogramm kreuzt 0 nach oben
- `Trig_RSI30`: RSI_14 kreuzt 30 nach oben (Erholungsbeginn)

**FILTER Pool** (Zustände, die heute True sein müssen):
- `Filt_SMA200`: Close > SMA_200
- `Filt_VolSpike`: Volume > SMA_Vol_20 × 1.5
- `Filt_TrendAlign`: SMA_20 > SMA_50 > SMA_200
- `Filt_RSIbull`: RSI_14 > 55
- `Filt_BBsqueeze`: BB-Breite / Close < 0.10

**Kombinatorik:** [Genau 1 Trigger] + [0, 1 oder 2 Filter]  
**Signal-Limits:** 40 ≤ Signale ≤ 3.500 (darunter zu selten, darüber zu gebühren-intensiv)

### Erkenntnisse
- Bestes Ergebnis: `RSI_30 | SMA200` (+33.99%)
- Mean-Reversion im Aufwärtstrend hat Edge
- Breakout-Signale (B50, B100) feuern zu häufig für 20€ Gebühren
- Die Kombinatorik alleine reicht nicht – der *Qualitätsbeweis* des Ausbruchs fehlt

---

## v7.0 – Alpha Research: Finding the Sweetspot

**Skript:** `find_sweetspot_v7.py`  
**Ansatz:** Standalone Trade-Level-Simulator (kein Portfolio-Management) zur Analyse der "Wellenqualität" von Breakout_50-Signalen.

### Getestete Intensitäts-Filter
| Variable | Werte |
|----------|-------|
| Amplitude (Tageskerze) | 1%, 3%, 5% |
| Volumen-Multiplikator | 1.5×, 2.0×, 3.0× |
| ADX (Trendstärke) | 20, 25, 30 |

**Zielvariable:** Profit Factor (Bruttogewinn / Bruttoverlust) – misst "Sauberkeit" der Wellen.

### Erkenntnisse
- **Bester Profit Factor (1.50):** Amplitude ≥5% + Volume ≥1.5× (ohne ADX-Filter)
- ADX-Filter verbessert PF kaum, reduziert aber Signalanzahl stark
- **Kernproblem:** Selbst mit PF 1.50 macht das System bei 20€ Gebühren und kleiner Positionsgröße keinen Netto-Gewinn
- **Neue Erkenntnis:** Das Problem liegt nicht am Signal-PF, sondern an der Kapitaleffizienz

---

## v7.0 – Predator 2-Slot Konzentration

**Skript:** `backtest_v7_final.py`  
**Ansatz:** Radikale Konzentration auf 2 statt 5 Slots → 5.000€ pro Trade → Gebührenanteil sinkt von ~2% auf ~0.8%

### Parameter
```python
MAX_POSITIONS   = 2
ATR_INIT        = 2.0
ATR_TRAIL       = 3.5
AMP_THRESHOLD   = 0.05   # 5% Tageskerze
VOL_MULTIPLIER  = 1.5
ROTATION_FACTOR = 1.5
STALL_DAYS      = 5      # Stall-Stop: Raus wenn nach 5 Tagen im Minus
```

### Ergebnis: -34.61% Gesamtrendite

**Diagnose:**
1. **5% Amplitude → "Erschöpfungs-Gap" Problem:** Wir kaufen am Tag des maximalen Intraday-Schubs – genau dann, wenn die Energie erschöpft ist. Der natürliche Pullback danach trifft direkt den ATR-Stop.
2. **Stall-Stop toxisch:** 5 Tage negative PnL schneidet legitime Konsolidierungen ab, die kurz danach weitergelaufen wären.
3. **Hit-Rate kollabiert:** ~30% – zu wenig für ein positives EV bei Payoff 1.6

**Gelernte Lektion:** *Kaufe die Stille vor dem Sturm, nicht den Sturm selbst.*

---

## v8.0 – Smart Money VCP (Volatility Contraction Pattern)

**Skript:** `backtest_v8_smart_money.py`  
**Kernidee:** Ersetze den "Explosions-Ausbruch" durch den "VCP-Ausbruch" – den Moment, wenn sich eine extrem enge Volatilitäts-Kontraktion aufbricht.

### Neues Entry-Signal (VCP-Breakout)
```
1. Breakout_50:    Close > High_50_prev  (Ausbruch passiert)
2. BB-Squeeze:     BB_Breite_GESTERN / Close_prev < 0.10  (Volatilität war fast tot)
3. Volume-Spike:   Volume > SMA_Vol_20 × 1.5  (Smart Money bricht den Squeeze)
4. Trend-Filter:   Close > SMA_200  (Langfristiger Aufwärtstrend)
```

**KEIN Amplitude-Filter:** Die 5%-Kerze-Regel wird komplett entfernt.  
**KEIN Stall-Stop:** Wird gelöscht (schneidet Konsolidierungen ab).

### Neuer Bärenmarkt-Filter (SPY)
Entry nur erlaubt wenn `SPY_Close > SPY_SMA_200 AND SPY_Close > SPY_SMA_50`.

### Ergebnis: -34.61% (nach Korrekturen der Positionsgröße)

**Diagnose:**
1. **Aggressive Rotation (45% aller Trades):** Rotation mit 1.5× ist zu permissiv
2. **SPY-Filter unzureichend:** Verhindert nicht alle schlechten Bärenphasen
3. **Hit-Rate 30%, Payoff 1.60 → EV negativ**

---

## v8.2 – 2D Grid-Search (Breadth & Rotation)

**Skript:** `optimize_v8_gridsearch.py`  
**Ansatz:** Systematische Optimierung von 2 Parametern gleichzeitig, statt Raten.

### Neue Konzepte

**Interne Marktbreite (kein SPY):**
```python
breadth = (close_piv > close_piv.rolling(200).mean()).mean(axis=1)
```
→ Anteil der 260 Aktien mit Close > SMA_200, täglich berechnet. Wert 0.0–1.0.

**"Diamond Hands" Rotations-Bremse:**
> Eine Position im "Earned Mode" (Stop bereits auf 3.5× ATR ausgeweitet) wird **niemals** durch Rotation verkauft. Nur "frische", unbewiesene Positionen können rotiert werden.

### Grid-Parameter
| Parameter | Werte |
|-----------|-------|
| `BREADTH_THRESHOLD` | 0.0, 0.20, 0.30, 0.40, 0.50 |
| `ROTATION_FACTOR` | 1.2, 1.5, 2.0, 2.5, 3.0, 999.0 |

→ **30 Kombinationen** total.

### Grid-Search Ergebnisse

**Top 5:**
| Rang | Breadth | Rotation | Rendite | Hit% | Payoff | EV% |
|------|---------|----------|---------|------|--------|-----|
| 1 | aus | 1.5× | +20.52% | 36.6% | 2.09 | +0.48% |
| 2 | ≥30% | 1.5× | +5.95% | 32.8% | 2.32 | +0.31% |
| 3 | aus | 2.0× | +1.34% | 38.5% | 1.74 | +0.23% |
| 4 | ≥40% | 2.0× | -6.02% | 33.3% | 2.07 | +0.09% |
| 5 | ≥30% | 2.0× | -9.72% | 33.6% | 2.01 | +0.05% |

**Erkenntnisse aus dem Grid:**

1. **Rotation 1.2× ist destruktiv** (Ø -40.6%): Zu viele Trades → Gebühren fressen alles
2. **Rotation 1.5× ist optimal** (Ø -7.6%, einziger mit positiven Läufen)
3. **Rotation AUS (999×) ist schlechter als 1.5×** (Ø -29.7%): Portfolios verstopfen
4. **Marktbreite-Filter schadet im Schnitt** – filtert zu viele Rebound-Wellen aus
5. **Breadth ≥50% ist das Schlechteste** (Ø -46.3%): Blockiert genau die besten Bouncephasen

---

## v8.3 – Champion Run (Deep Dive Analyse)

**Skript:** `backtest_v8_champion.py`  
**Parameter:** `BREADTH_THRESHOLD = 0.0`, `ROTATION_FACTOR = 1.5`, Diamond Hands aktiv.

### Ergebnisse
```
Zeitraum:         6.2 Jahre (2020-02 → 2026-04)
Gesamtrendite:   +20.52%  (CAGR +3.08%)
End-Kapital:      12.052€  (Start: 10.000€)
Max Drawdown:    -45.3%  (Trough: 2023-03-13, Peak: 2021-11-08)
Sharpe:           0.25
Gesamt-Trades:    145  (90 ATR | 54 Rotation | 1 Offen)
Gezahlte Geb.:    5.800€
Investitionsq.:   93.2%
```

### Trade-Statistik
```
Hit-Rate:         36.6%
Avg Win:         +7.74%
Avg Loss:        -3.71%
Payoff Ratio:     2.09
Profit Factor:    1.20
EV/Trade:        +0.48%
```

### Trade Lifecycle Analyse (neu in v8.3)

| Metrik | Wert | Interpretation |
|--------|------|----------------|
| Ø Haltedauer Winners | **34.0 Tage** | Trend wird korrekt geritten |
| Ø Haltedauer Losers | **9.8 Tage** | ATR-Stop schneidet schnell ab |
| Ø Peak-Unrealized (alle) | **+6.8%** | System sitzt auf Buchgewinn |
| Ø Peak-Unrealized (Winners) | **+14.7%** | Ausreichend für Earned Mode |
| Rotation Avg PnL | **-1.01%** | Rotationen feuern zu früh |
| Rotation Hit-Rate | **27.8%** | 72.2% der Rotations-Exits waren im Minus |
| Earned Mode Rate | **41.4%** | 41% aller Trades beweisen sich |
| Ø Return Earned | **+5.34%** | Earned-Trades verdienen das Geld |
| Ø Return Fresh | **-2.95%** | Nicht-Earned-Trades sind das Problem |

### Jahresrenditen
| Jahr | Rendite | Erklärung |
|------|---------|-----------|
| 2020 | -2.6% | Nur Teiljahr, COVID-Recovery |
| **2021** | **+33.7%** | Starker Bullenmarkt, Trend-Following ideal |
| **2022** | **-29.2%** | Bärenmarkt (kein Breadth-Filter aktiv) |
| 2023 | +9.6% | Erholung, Rebound-Wellen |
| 2024 | +13.4% | Moderater Bullenmarkt |
| 2025 | -7.4% | Marktvolatilität |
| 2026 | +12.8% | YTD (bis April) |

### Top Trades
| Ticker | Zeitraum | Dauer | Rendite | Status |
|--------|----------|-------|---------|--------|
| CRWD | Okt 2023 → Feb 2024 | 93d | +71.5% | ★ Earned |
| FDX | Dez 2025 → Mär 2026 | 52d | +28.8% | ★ Earned |
| ON | Nov 2020 → Jan 2021 | 53d | +27.6% | ★ Earned |
| NEM | Jul 2025 → Okt 2025 | 61d | +27.6% | ★ Earned |
| META | Apr 2023 → Aug 2023 | 76d | +21.5% | ★ Earned |

**Beobachtung:** Alle Top 5 Winner sind Earned-Mode-Trades. Der ATR-Earned-Mechanismus funktioniert korrekt.

---

## Gesamte Erkenntnisse & Lessons Learned

### 1. Die Gebühr ist der härteste Gegner
Bei 10.000€ Startkapital und 20€ Fixgebühr kostet ein Roundtrip (Kauf + Verkauf) 40€ = 0.4% des Gesamtkapitals. Selbst mit 2 Slots (5.000€/Trade) macht das 0.8% pro Trade. Bei 145 Trades über 6.2 Jahre wurden **5.800€ in Gebühren** bezahlt – das entspricht 58% des Startkapitals. Ohne Gebühren wäre die Bruttorendite weit positiv.

**Folgerung:** Jedes Signal muss einen Edge haben, der deutlich über 0.8% Round-Trip-Kosten liegt.

### 2. Qualität vor Quantität
- 800+ Signale in 7 Jahren sind *zu viele*, nicht zu wenige (Erkenntnis v6.5)
- VCP-Breakouts (BB-Squeeze) reduzieren die Rohsignale von >1.500 auf 1.390 – aber mit höherer Qualität
- Der Profit Factor stieg von ~1.0 (Rohe B50) auf 1.20 (Champion) und 1.50 (Sweetspot-Analyse ohne Gebühren)

### 3. Das Rotation-Dilemma
- **Zu wenig Rotation** (999×): Portfolio verstopft, beste Wellen werden verpasst
- **Zu viel Rotation** (1.2×): Gebühren-Spirale, 83 Rotationen/5 Jahre
- **Optimum: 1.5×** mit Diamond-Hands-Schutz: Bewiesene Gewinne werden gehalten, schwache Positionen können weichen

### 4. Der Earned-Mode ist der Kern-Alpha
**41% der Trades erreichen Earned Mode → Ø +5.34%**  
**59% der Trades erreichen Earned Mode nicht → Ø -2.95%**

Der gesamte positive Expected Value des Systems kommt aus den ~60 Trades, die sich "bewährt" haben. Die restlichen 85 Trades verbrennen Kapital. Die logische Weiterentwicklung: besser filtern, welche Trades das Potenzial haben, Earned Mode zu erreichen.

### 5. Marktumfeld ist entscheidend
2022 (-29.2%) zeigt: Das System ist ein Trendfolger und leidet in Bärenmärkten stark. Der externe SPY-Filter und der interne Breadth-Filter haben beide Nachteile:
- SPY-Filter: Zu binär (EIN Index für 260 Aktien)
- Breadth-Filter: Blockiert Rebound-Wellen in früher Recovery-Phase

**Offene Frage für v9:** Welcher Marktregime-Filter filtert genau die 2022er-Drawdown-Phase ohne die 2023er-Recovery zu blockieren?

### 6. Stall-Stop ist kontraproduktiv für VCP
Ein 5-Tage-Stall-Stop wurde in v7 eingeführt und in v8 bewusst entfernt. VCP-Ausbrüche nach Volatilitäts-Kompression folgen oft einem natürlichen Pullback (1–3 Wochen), der den Stall-Stop triggert, bevor die eigentliche Trendbewegung beginnt. Der ATR-Stop allein ist die bessere Lösung.

### 7. Positionsgröße ist mathematisch entscheidend
| Slots | Trade-Größe | Gebührenanteil Roundtrip |
|-------|-------------|--------------------------|
| 5 | 2.000€ | 2.0% |
| 2 | 5.000€ | 0.8% |
| 1 | 10.000€ | 0.4% |

Die Reduktion von 5 auf 2 Slots hat die Gebührenbelastung pro Trade von 2.0% auf 0.8% mehr als halbiert. Weiteres Konzentrieren birgt jedoch erhöhtes Klumpenrisiko.

---

## Offene Entwicklungspunkte (Ausblick v9)

1. **Earnings-Filter:** VCP-Ausbrüche nach Quartalsergebnissen haben höhere Hit-Rates (Katalysator ist bekannt). Implementierung eines Earnings-Fensters (+3d nach EPS) als Entry-Voraussetzung.

2. **ADX-Mindestfilter:** ADX > 20 zum Zeitpunkt des VCP-Ausbruchs als zusätzliche Filterbedingung. In der Sweetspot-Analyse (v7) hat ADX den Profit Factor zwar kaum verbessert, aber in Kombination mit BB-Squeeze könnte er die "Fake-Breakouts" in seitwärtslaufenden Märkten reduzieren.

3. **Besserer Regime-Filter:** Statt binärem SPY-Filter oder Breadth-Schwelle: Gleitender 20-Tage Breadth-Durchschnitt oder Breadth-Momentum (Veränderung der Marktbreite über 10 Tage).

4. **Sektor-Rotation:** Aktuell werden alle 260 Aktien gleichbehandelt. Eine Sektor-Gewichtung (nur Käufe in Sektoren mit positiver relativer Stärke) könnte die Qualität der Entry-Kandidaten erhöhen.

5. **Broker-Wechsel:** Bei einem Neo-Broker (2€ Gebühr statt 20€) würde die gesamte Strategie deutlich profitabler. Die Sweetspot-Analyse zeigte: Ohne Gebühren hat das System einen Profit Factor von 1.50.

---

## Dateiübersicht

| Datei | Version | Funktion |
|-------|---------|----------|
| `research_signals.py` | v6.0 | Event Study – 22 Signale, Forward Returns |
| `research_trend_following.py` | v6.0 | Trade-Simulation mit SMA50-Exit |
| `backtest_v6.py` | v6.3 | Portfolio-Backtester (5 Slots, Asymm. ATR-Stop) |
| `optimize_entries_v6.py` | v6.3 | Signal-Optimierung (5 Signale × 2 Gebühren) |
| `discover_rules_v6.py` | v6.5 | Kombinatorische Rule Discovery (Trigger-Filter-Matrix) |
| `find_sweetspot_v7.py` | v7.0 | Alpha Research – Profit Factor Optimierung |
| `backtest_v7_final.py` | v7.0 | Predator 2-Slot (5% Amp + Stall-Stop) |
| `backtest_v8_smart_money.py` | v8.0 | VCP-Backtester (BB-Squeeze, SPY-Filter) |
| `optimize_v8_gridsearch.py` | v8.2 | 2D Grid-Search (30 Kombo: Breadth × Rotation) |
| `backtest_v8_champion.py` | v8.3 | Champion Run + Deep-Dive Lifecycle-Analyse |

### Ergebnis-Dateien
| Datei | Inhalt |
|-------|--------|
| `rule_discovery_results.csv` | v6.0 Rule Discovery Ergebnisse |
| `rule_discovery_v65.csv` | v6.5 Trigger-Filter-Matrix Ergebnisse |
| `signal_comparison_v6.csv` | Signal-Optimierung Vergleich |
| `sweetspot_results_stall5.csv` | v7.0 Sweetspot Analyse |
| `v82_gridsearch_results.csv` | v8.2 Grid-Search alle 30 Kombinationen |
| `champion_trades.csv` | v8.3 alle 145 Champion-Trades mit Metadaten |
| `champion_equity.png` | v8.3 Equity-Kurve + Gantt-Chart + Jahresrenditen |

---

*Dokumentation erstellt: Mai 2026*
