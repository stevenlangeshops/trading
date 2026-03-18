"""
main.py
────────
Zentraler Einstiegspunkt für den Trading-Bot.

Kommandos:
  download    – Krypto-Daten via CCXT (Binance)
  stocks      – Aktien/ETF-Daten via yfinance
  features    – Feature-Engineering (einzeln oder --multi für alle Assets)
  train       – LSTM trainieren
  backtest    – Backtesting
  optimize    – Hyperparameter-Optimierung mit Optuna
  all         – Komplette Krypto-Pipeline (download→features→train→backtest)

Beispiele:
  # Krypto (wie bisher)
  python main.py all --symbol BTC/USDT --timeframe 1h

  # Aktien herunterladen
  python main.py stocks --timeframe 1h --years 2
  python main.py stocks --ticker AAPL --timeframe 1d --years 5

  # Multi-Asset Training
  python main.py features --multi --timeframe 1h --threshold 0.002
  python main.py train --ticker combined --timeframe 1h
  python main.py backtest --ticker AAPL --timeframe 1h

  # Hyperparameter optimieren
  python main.py optimize --ticker AAPL --timeframe 1h --trials 50
  python main.py optimize --multi --timeframe 1h --trials 30
"""

import argparse
import sys
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    colorize=True,
)
logger.add("logs/trading_bot_{time:YYYY-MM-DD}.log", rotation="1 day", retention="14 days")


# ── Sub-Commands ──────────────────────────────────────────────────────────────

def cmd_download(args):
    from data.download import fetch_ohlcv, save
    df = fetch_ohlcv(args.symbol, args.timeframe, args.limit, args.exchange)
    save(df, args.symbol, args.timeframe)


def cmd_stocks(args):
    from data.download_stocks import download_all, fetch_ticker, save_ticker
    if args.ticker:
        df = fetch_ticker(args.ticker, args.timeframe, args.years)
        if df is not None:
            save_ticker(df, args.ticker, args.timeframe)
            logger.success(f"Gespeichert: {args.ticker} ({len(df)} Kerzen)")
        else:
            logger.error(f"Keine Daten für {args.ticker}")
    else:
        download_all(args.timeframe, args.years)


def cmd_features(args):
    from features.engineer import run_pipeline, build_combined_dataset
    if args.multi:
        build_combined_dataset(
            timeframe  = args.timeframe,
            horizon    = args.horizon,
            threshold  = args.threshold,
            seq_len    = args.seq_len,
            label_mode = args.mode,
        )
    else:
        run_pipeline(
            symbol     = args.symbol,
            timeframe  = args.timeframe,
            horizon    = args.horizon,
            threshold  = args.threshold,
            seq_len    = args.seq_len,
            label_mode = args.mode,
        )


def cmd_train(args):
    from models.trainer import train
    train(
        symbol     = args.symbol,
        timeframe  = args.timeframe,
        mode       = args.mode,
        ticker     = getattr(args, "ticker", None),
        epochs     = args.epochs,
        patience   = args.patience,
        hidden_dim = args.hidden_dim,
        num_layers = getattr(args, "num_layers", 2),
        batch_size = args.batch_size,
        lr         = args.lr,
    )


def cmd_backtest(args):
    from strategy.backtest import run_backtest
    ticker = getattr(args, "ticker", None) or args.symbol.replace("/", "_")
    run_backtest(
        ticker           = ticker,
        timeframe        = args.timeframe,
        mode             = args.mode,
        init_cash        = args.cash,
        fees             = args.fees,
        test_on          = getattr(args, "test_on",          None),
        entry_threshold  = getattr(args, "entry_threshold", 0.99),
        exit_threshold   = getattr(args, "exit_threshold",  0.99),
        hold_days        = getattr(args, "hold_days",        24),
        stop_loss        = getattr(args, "stop_loss",        0.05),
    )


def cmd_optimize(args):
    from models.optimize import run_optimization
    ticker = getattr(args, "ticker", "AAPL")
    run_optimization(
        ticker    = ticker,
        timeframe = args.timeframe,
        trials    = args.trials,
        multi     = args.multi,
        jobs      = args.jobs,
    )


def cmd_all(args):
    logger.info("═" * 60)
    logger.info("SCHRITT 1/4: Daten herunterladen")
    logger.info("═" * 60)
    cmd_download(args)

    logger.info("═" * 60)
    logger.info("SCHRITT 2/4: Feature-Engineering")
    logger.info("═" * 60)
    cmd_features(args)

    logger.info("═" * 60)
    logger.info("SCHRITT 3/4: Modell trainieren")
    logger.info("═" * 60)
    cmd_train(args)

    logger.info("═" * 60)
    logger.info("SCHRITT 4/4: Backtesting")
    logger.info("═" * 60)
    cmd_backtest(args)

    logger.success("Pipeline abgeschlossen.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="Neural Network Trading Bot – trading_v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Gemeinsame Argumente
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--symbol",    default="BTC/USDT")
    common.add_argument("--timeframe", default="1h")
    common.add_argument("--mode",      choices=["cls","reg"], default="cls")

    # download (Krypto)
    p = sub.add_parser("download", parents=[common])
    p.add_argument("--limit",    type=int, default=5000)
    p.add_argument("--exchange", default="binance")

    # stocks (Aktien/ETFs via yfinance)
    p = sub.add_parser("stocks")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--years",     type=int, default=2)
    p.add_argument("--ticker",    default=None, help="Einzelner Ticker, sonst alle aus asset_list.txt")

    # features
    p = sub.add_parser("features", parents=[common])
    p.add_argument("--horizon",   type=int,   default=6)
    p.add_argument("--threshold", type=float, default=0.002)
    p.add_argument("--seq_len",   type=int,   default=48)
    p.add_argument("--multi",     action="store_true", help="Alle Assets kombiniert verarbeiten")

    # train
    p = sub.add_parser("train", parents=[common])
    p.add_argument("--ticker",     default=None, help="z.B. AAPL oder 'combined'")
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--patience",   type=int,   default=10)
    p.add_argument("--hidden_dim", type=int,   default=128)
    p.add_argument("--num_layers", type=int,   default=2)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)

    # backtest
    p = sub.add_parser("backtest", parents=[common])
    p.add_argument("--ticker",          default=None,  help="z.B. AAPL oder 'combined'")
    p.add_argument("--test_on",         default=None,  help="Asset für Out-of-Sample Test (z.B. AAPL bei combined Modell)")
    p.add_argument("--cash",            type=float, default=10_000.0)
    p.add_argument("--fees",            type=float, default=0.001)
    p.add_argument("--entry_threshold", type=float, default=0.99,
                   help="Kaufsignal ab diesem Wert. Standard=0.99 → automatisch auf 75. Perzentil")
    p.add_argument("--exit_threshold",  type=float, default=0.99,
                   help="Verkaufssignal unter diesem Wert. Standard=0.99 → automatisch auf 40. Perzentil")
    p.add_argument("--hold_days",        type=int,   default=24,
                   help="Max. Haltedauer in Perioden (= horizon aus Training)")
    p.add_argument("--stop_loss",        type=float, default=0.05,
                   help="Stop-Loss in Dezimal, z.B. 0.05 = 5%%")

    # optimize
    p = sub.add_parser("optimize")
    p.add_argument("--ticker",     default="AAPL")
    p.add_argument("--timeframe",  default="1h")
    p.add_argument("--trials",     type=int, default=50)
    p.add_argument("--multi",      action="store_true")
    p.add_argument("--jobs",       type=int, default=1)

    # all (Krypto-Pipeline komplett)
    p = sub.add_parser("all", parents=[common])
    p.add_argument("--limit",      type=int,   default=5000)
    p.add_argument("--exchange",   default="binance")
    p.add_argument("--horizon",    type=int,   default=6)
    p.add_argument("--threshold",  type=float, default=0.002)
    p.add_argument("--seq_len",    type=int,   default=48)
    p.add_argument("--multi",      action="store_true")
    p.add_argument("--ticker",     default=None)
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--patience",   type=int,   default=10)
    p.add_argument("--hidden_dim", type=int,   default=128)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--cash",       type=float, default=10_000.0)
    p.add_argument("--fees",       type=float, default=0.001)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    dispatch = {
        "download": cmd_download,
        "stocks":   cmd_stocks,
        "features": cmd_features,
        "train":    cmd_train,
        "backtest": cmd_backtest,
        "optimize": cmd_optimize,
        "all":      cmd_all,
    }
    dispatch[args.command](args)
