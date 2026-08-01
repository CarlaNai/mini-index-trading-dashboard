"""
project.py

Main functions of the CS50P final project - Mini Índice Trading Dashboard.

This file contains all the BUSINESS LOGIC: result calculations, data
validation, and performance metrics. Nothing here knows about databases
or screens - that separation is what makes these functions easy to test.
"""

import csv
from datetime import date, timedelta

import pandas as pd

WIN_POINT_VALUE = 0.20  # each point of the mini index is worth R$ 0.20 per contract
VALID_DIRECTIONS = ("buy", "sell")

# Brokerage fee, charged once per leg (entry and exit), per contract.
BROKERAGE_FEE_PER_CONTRACT_PER_LEG = 0.18

# Brazilian day-trade IRRF (income tax withheld at source): 1% of the
# day's aggregate result, and only when that daily result is positive -
# it is never charged per trade, and never on a losing day. This is a
# real, well-known rule for day trading in Brazil, not an approximation.
DAY_TRADE_TAX_RATE = 0.01


def calculate_fees(contracts, fee_per_contract_per_leg=BROKERAGE_FEE_PER_CONTRACT_PER_LEG):
    """
    Total brokerage fee for one full round-trip trade: charged once when
    the position is opened and once again when it is closed.
    """
    return round(contracts * fee_per_contract_per_leg * 2, 2)


def calculate_daily_net_results(df):
    """
    Group trades by day and compute, for each day: the gross result, the
    total brokerage fees, the result after fees, the estimated IRRF tax
    (1% of the day's result-after-fees, only if positive), and the final
    net result. Returns an empty-but-correctly-shaped DataFrame if df is
    empty, so callers never need a special case for "no trades yet".
    """
    columns = ["trade_date", "gross_result", "fees", "result_after_fees", "tax", "net_result"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    working = df.copy()
    working["fees"] = working["contracts"].apply(calculate_fees)
    working["result_after_fees"] = working["result_financial"] - working["fees"]

    daily = working.groupby("trade_date", as_index=False).agg(
        gross_result=("result_financial", "sum"),
        fees=("fees", "sum"),
        result_after_fees=("result_after_fees", "sum"),
    )
    daily["tax"] = daily["result_after_fees"].apply(
        lambda value: round(value * DAY_TRADE_TAX_RATE, 2) if value > 0 else 0.0
    )
    daily["net_result"] = (daily["result_after_fees"] - daily["tax"]).round(2)
    return daily[columns]


def calculate_net_summary(df):
    """
    Collapse calculate_daily_net_results into a single dictionary of
    totals - what the dashboard shows as the net-of-costs metric cards.
    """
    daily = calculate_daily_net_results(df)
    if daily.empty:
        return {
            "gross_result": 0.0,
            "total_fees": 0.0,
            "result_after_fees": 0.0,
            "estimated_tax": 0.0,
            "net_result": 0.0,
        }

    return {
        "gross_result": round(daily["gross_result"].sum(), 2),
        "total_fees": round(daily["fees"].sum(), 2),
        "result_after_fees": round(daily["result_after_fees"].sum(), 2),
        "estimated_tax": round(daily["tax"].sum(), 2),
        "net_result": round(daily["net_result"].sum(), 2),
    }


def calculate_result(direction, entry_price, exit_price, contracts, point_value=WIN_POINT_VALUE):
    """
    Calculate the result of a trade in points and in currency (R$).

    Returns a tuple: (result_points, result_financial)

    Rules:
    - Buy:  profits when price goes up (exit - entry)
    - Sell: profits when price goes down (entry - exit)
    """
    direction = direction.lower().strip()

    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"Invalid direction: '{direction}'. Use 'buy' or 'sell'.")
    if contracts <= 0:
        raise ValueError("Number of contracts must be greater than zero.")
    if entry_price <= 0 or exit_price <= 0:
        raise ValueError("Prices must be greater than zero.")

    if direction == "buy":
        result_points = exit_price - entry_price
    else:  # sell
        result_points = entry_price - exit_price

    result_financial = result_points * point_value * contracts

    return round(result_points, 2), round(result_financial, 2)


def create_trade(trade_date, direction, entry_price, exit_price, contracts,
                  entry_time=None, exit_time=None, setup=None,
                  stop_points=None, mae_points=None, mfe_points=None,
                  emotional_state=None, technical_notes=None, screenshot_path=None,
                  is_demo=False):
    """
    Build a dictionary representing a full trade, already with the
    result calculated. This dictionary is what gets saved to the database.

    is_demo marks whether this trade is real user data (False) or part
    of the fictional dataset used to preview the dashboard (True). It is
    a dedicated flag - not something hidden inside technical_notes - so
    that a real trade's notes are never confused with a demo marker.
    """
    result_points, result_financial = calculate_result(
        direction, entry_price, exit_price, contracts
    )

    return {
        "trade_date": trade_date,
        "entry_time": entry_time,
        "exit_time": exit_time,
        "direction": direction.lower().strip(),
        "setup": setup,
        "contracts": contracts,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_points": stop_points,
        "mae_points": mae_points,
        "mfe_points": mfe_points,
        "result_points": result_points,
        "result_financial": result_financial,
        "emotional_state": emotional_state,
        "technical_notes": technical_notes,
        "screenshot_path": screenshot_path,
        "is_demo": is_demo,
    }


def _parse_brl_number(value):
    """
    Convert a Brazilian-formatted number string to float.
    Brazilian format uses '.' as the thousands separator and ',' as the
    decimal separator - the opposite of the format Python expects -
    e.g. "177.448,25" means 177448.25, not 177.44825.
    """
    return float(value.strip().replace(".", "").replace(",", "."))


def _convert_br_date(date_text):
    """Convert 'DD/MM/YYYY' to the 'YYYY-MM-DD' format used everywhere else."""
    day, month, year = date_text.strip().split("/")
    return f"{year}-{month}-{day}"


def parse_broker_csv(file_content):
    """
    Parse the daily trade report exported by the brokerage into a list
    of trade dictionaries (built with create_trade, so every value is
    already validated and the result is already calculated the same way
    as any manually logged trade).

    file_content is the raw text of the CSV (already decoded - the file
    is exported using the Latin-1 encoding, not UTF-8, so the caller
    must decode it with encoding="latin-1" before calling this function).

    The report has a few metadata lines (account, name, date) before the
    actual header row, so this function looks for the line starting
    with "Ativo;" instead of assuming a fixed number of lines to skip.

    setup and technical_notes are intentionally left as None: the broker
    report has no concept of "strategy" or "trade rationale", so those
    fields are meant to be filled in by the person reviewing the import,
    not guessed by this function.
    """
    lines = file_content.splitlines()
    header_index = next((i for i, line in enumerate(lines) if line.startswith("Ativo;")), None)

    if header_index is None:
        raise ValueError("Could not find the expected header row ('Ativo;...') in this file.")

    reader = csv.DictReader(lines[header_index:], delimiter=";")
    trades = []

    for row in reader:
        if not row.get("Ativo"):
            continue  # skip blank trailing lines

        side = row["Lado"].strip().upper()
        direction = "buy" if side == "C" else "sell"

        buy_price = _parse_brl_number(row["Preço Compra"])
        sell_price = _parse_brl_number(row["Preço Venda"])
        entry_price = buy_price if direction == "buy" else sell_price
        exit_price = sell_price if direction == "buy" else buy_price

        contracts_bought = _parse_brl_number(row["Qtd Compra"])
        contracts_sold = _parse_brl_number(row["Qtd Venda"])
        contracts = int(max(contracts_bought, contracts_sold))

        entry_datetime = row["Abertura"].strip().split(" ")
        exit_datetime = row["Fechamento"].strip().split(" ")
        trade_date = _convert_br_date(entry_datetime[0])
        entry_time = entry_datetime[1] if len(entry_datetime) > 1 else None
        exit_time = exit_datetime[1] if len(exit_datetime) > 1 else None

        trade = create_trade(
            trade_date=trade_date,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            contracts=contracts,
            entry_time=entry_time,
            exit_time=exit_time,
            mfe_points=_parse_brl_number(row["MEP"]),
            mae_points=_parse_brl_number(row["MEN"]),
        )
        trades.append(trade)

    return trades


def get_trade(connection, trade_id):
    """
    Fetch a single trade by id, as a dictionary. Returns None if no
    trade with that id exists. Used to pre-fill the edit form.
    """
    cursor = connection.cursor()
    row = cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,)).fetchone()
    return dict(row) if row is not None else None


def update_trade(connection, trade_id, trade_date, direction, entry_price, exit_price, contracts,
                  entry_time=None, exit_time=None, setup=None,
                  stop_points=None, mae_points=None, mfe_points=None,
                  emotional_state=None, technical_notes=None, screenshot_path=None):
    """
    Update an existing trade. The result (points and financial) is
    recalculated from the new values, exactly like a new trade - so an
    edited trade is never left with a stale result from before the edit.
    """
    result_points, result_financial = calculate_result(direction, entry_price, exit_price, contracts)
    direction = direction.lower().strip()

    cursor = connection.cursor()
    cursor.execute("""
        UPDATE trades SET
            trade_date = ?, entry_time = ?, exit_time = ?, direction = ?, setup = ?,
            contracts = ?, entry_price = ?, exit_price = ?, stop_points = ?,
            mae_points = ?, mfe_points = ?, result_points = ?, result_financial = ?,
            emotional_state = ?, technical_notes = ?, screenshot_path = ?
        WHERE id = ?
    """, (
        trade_date, entry_time, exit_time, direction, setup,
        contracts, entry_price, exit_price, stop_points,
        mae_points, mfe_points, result_points, result_financial,
        emotional_state, technical_notes, screenshot_path,
        trade_id,
    ))
    connection.commit()
    return cursor.rowcount


def delete_trade(connection, trade_id):
    """Delete a single trade by id. Returns True if a row was actually removed."""
    cursor = connection.cursor()
    cursor.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
    connection.commit()
    return cursor.rowcount > 0


def save_trade(connection, trade):
    """
    Insert a trade (dictionary created by create_trade) into the database.
    Returns the id of the newly created trade.
    """
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO trades (
            trade_date, entry_time, exit_time, direction, setup, contracts,
            entry_price, exit_price, stop_points, mae_points, mfe_points,
            result_points, result_financial, emotional_state,
            technical_notes, screenshot_path, is_demo
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade["trade_date"], trade["entry_time"], trade["exit_time"],
        trade["direction"], trade["setup"], trade["contracts"],
        trade["entry_price"], trade["exit_price"],
        trade["stop_points"], trade["mae_points"], trade["mfe_points"],
        trade["result_points"], trade["result_financial"],
        trade["emotional_state"], trade["technical_notes"],
        trade["screenshot_path"], int(trade.get("is_demo", False)),
    ))
    connection.commit()
    return cursor.lastrowid


def load_trades(connection):
    """
    Read all trades from the database and return them as a pandas
    DataFrame (an in-memory table, easy to filter and calculate on top of).

    pandas' read_sql_query only has built-in support for sqlite3
    connections directly (anything else needs a SQLAlchemy engine), so
    when the connection is talking to Postgres, rows are fetched and
    assembled into a DataFrame by hand instead. The `dialect` attribute
    is a duck-typed marker set by database.PostgresConnection - absent
    on a plain sqlite3.Connection - so this stays the only place in this
    module that needs to know two backends exist.
    """
    if getattr(connection, "dialect", "sqlite") == "postgres":
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM trades ORDER BY trade_date, entry_time")
        rows = [dict(row) for row in cursor.fetchall()]
        df = pd.DataFrame(rows)
    else:
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY trade_date, entry_time", connection)
    if not df.empty:
        df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def filter_by_setup(df, setup):
    """Return only the trades of a specific setup (e.g. 'TA')."""
    if setup is None or setup == "all":
        return df
    return df[df["setup"] == setup]


def filter_by_period(df, start_date=None, end_date=None):
    """Return only the trades within a date range."""
    result = df.copy()
    if start_date is not None:
        result = result[result["trade_date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        result = result[result["trade_date"] <= pd.to_datetime(end_date)]
    return result


def filter_by_time_range(df, start_time=None, end_time=None):
    """
    Return only the trades whose entry_time falls within [start_time, end_time].
    Times are "HH:MM" or "HH:MM:SS" strings - zero-padded 24h format sorts
    correctly as plain text, so no time parsing is needed here.
    Trades with no entry_time recorded are excluded whenever a time filter
    is active, since there is nothing to compare.
    """
    if start_time is None and end_time is None:
        return df

    result = df[df["entry_time"].notna()].copy()
    if start_time is not None:
        result = result[result["entry_time"] >= start_time]
    if end_time is not None:
        result = result[result["entry_time"] <= end_time]
    return result


def classify_shift(entry_time):
    """
    Classify a trade as 'Manhã' or 'Tarde' from its entry_time.
    The cutoff is 12:00 - anything before noon is morning, from noon
    onward is afternoon. Returns None when entry_time is missing, so
    trades without a recorded time are never silently placed in a shift.
    """
    if entry_time is None or (isinstance(entry_time, float) and pd.isna(entry_time)):
        return None
    hour = int(str(entry_time).split(":")[0])
    return "Manhã" if hour < 12 else "Tarde"


def filter_by_shift(df, shift):
    """Return only the trades that happened in the given shift ('Manhã'/'Tarde')."""
    if shift is None or shift == "all":
        return df
    result = df.copy()
    result["shift"] = result["entry_time"].apply(classify_shift)
    return result[result["shift"] == shift].drop(columns="shift")


def seed_demo_trades(connection):
    """Insert a varied, repeat-safe demo dataset for previewing the dashboard.

    Each record is flagged with is_demo=True (a dedicated database column,
    not a hidden marker in a business field), so it can always be told
    apart from the user's real trades and removed with clear_demo_trades.
    Calling this function again after the first load is a no-op.
    """
    cursor = connection.cursor()
    existing = cursor.execute("SELECT 1 FROM trades WHERE is_demo = 1 LIMIT 1").fetchone()
    if existing:
        return 0

    results_points = [180, -95, 260, 120, -150, 340, 85, -70, 210, 155,
                      -110, 290, 65, 245, -180, 130, 310, -55, 190, 110,
                      -125, 365, 150, -90, 275, 95, 220, -145, 330, 175]
    setups = ("TA", "TC", "TRM", "FQ")
    start = date(2026, 6, 2)

    for index, points in enumerate(results_points):
        trade_date = start + timedelta(days=index + (index // 5) * 2)
        direction = "buy" if index % 3 else "sell"
        entry_price = 138000 + index * 38
        exit_price = entry_price + points if direction == "buy" else entry_price - points
        trade = create_trade(
            trade_date=str(trade_date),
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            contracts=(index % 3) + 1,
            setup=setups[index % len(setups)],
            stop_points=180,
            entry_time=f"{9 + (index % 5):02d}:00",
            is_demo=True,
        )
        save_trade(connection, trade)

    return len(results_points)


def clear_demo_trades(connection):
    """
    Delete every trade flagged as demo data, leaving the user's real
    trades untouched. Returns the number of rows removed.
    """
    cursor = connection.cursor()
    cursor.execute("DELETE FROM trades WHERE is_demo = 1")
    connection.commit()
    return cursor.rowcount


def calculate_daily_results(df):
    """
    Group trades by day, summing the financial result and counting how
    many trades happened that day. Used both by the daily bar chart and
    by the calendar view, so the two always agree with each other.
    """
    if df.empty:
        return pd.DataFrame(columns=["trade_date", "result_financial", "trade_count"])

    grouped = df.groupby("trade_date", as_index=False).agg(
        result_financial=("result_financial", "sum"),
        trade_count=("result_financial", "count"),
    )
    return grouped


def calculate_efficiency_breakdown(df):
    """
    Count how many trades were winners, losers, or exactly break-even.
    This is the data behind the win/loss/breakeven pie chart.
    """
    if df.empty:
        return {"winners": 0, "losers": 0, "breakeven": 0}

    return {
        "winners": int((df["result_financial"] > 0).sum()),
        "losers": int((df["result_financial"] < 0).sum()),
        "breakeven": int((df["result_financial"] == 0).sum()),
    }


def calculate_performance_by_hour(df):
    """
    Group trades by the hour of day they were entered (0-23) and compute
    the result and win rate for each hour. Trades with no entry_time are
    excluded, since there is no hour to group them by.
    """
    columns = ["hour", "result_financial", "trade_count", "win_rate"]
    timed = df[df["entry_time"].notna()]
    if timed.empty:
        return pd.DataFrame(columns=columns)

    working = timed.copy()
    working["hour"] = working["entry_time"].apply(lambda value: int(str(value).split(":")[0]))

    grouped = working.groupby("hour").agg(
        result_financial=("result_financial", "sum"),
        trade_count=("result_financial", "count"),
        wins=("result_financial", lambda s: (s > 0).sum()),
    ).reset_index()
    grouped["win_rate"] = round(grouped["wins"] / grouped["trade_count"] * 100, 1)
    return grouped[columns]


WEEKDAY_NAMES_PT = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]


def calculate_performance_by_weekday(df):
    """
    Group trades by weekday (Monday..Sunday, in Portuguese) and compute
    the result and win rate for each day. Weekdays with no trades are
    still included, with zeroed metrics, so a chart never silently skips
    a day of the week.
    """
    columns = ["weekday", "result_financial", "trade_count", "win_rate"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    working = df.copy()
    working["weekday"] = working["trade_date"].dt.weekday.apply(lambda i: WEEKDAY_NAMES_PT[i])

    grouped = working.groupby("weekday").agg(
        result_financial=("result_financial", "sum"),
        trade_count=("result_financial", "count"),
        wins=("result_financial", lambda s: (s > 0).sum()),
    ).reindex(WEEKDAY_NAMES_PT).fillna(0).reset_index()
    grouped["win_rate"] = grouped.apply(
        lambda row: round(row["wins"] / row["trade_count"] * 100, 1) if row["trade_count"] > 0 else 0.0,
        axis=1,
    )
    return grouped[columns]


def calculate_streaks(df):
    """
    Compute the current winning/losing streak and the longest ones ever
    recorded, in chronological order. Break-even trades (result exactly
    zero) reset the streak without counting as either a win or a loss.
    """
    if df.empty:
        return {"current_type": None, "current_length": 0, "max_win_streak": 0, "max_loss_streak": 0}

    ordered = df.sort_values(["trade_date", "entry_time"])
    max_win_streak = max_loss_streak = 0
    current_type = None
    current_length = 0
    running_type = None
    running_length = 0

    for result in ordered["result_financial"]:
        outcome = "win" if result > 0 else "loss" if result < 0 else None

        if outcome is None:
            running_type, running_length = None, 0
            continue

        if outcome == running_type:
            running_length += 1
        else:
            running_type, running_length = outcome, 1

        max_win_streak = max(max_win_streak, running_length if outcome == "win" else max_win_streak)
        max_loss_streak = max(max_loss_streak, running_length if outcome == "loss" else max_loss_streak)
        current_type, current_length = running_type, running_length

    return {
        "current_type": current_type,
        "current_length": current_length,
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
    }


def calculate_mfe_efficiency(df, thresholds=(100, 150, 200, 250, 300, 400, 500)):
    """
    For each points threshold, the percentage of trades whose favorable
    excursion (mfe_points) reached at least that many points. This is
    the same "how far could this trade have gone" view as the reference
    screenshots' Zona de Eficiência. Trades with no mfe_points recorded
    are excluded from the percentage base.
    """
    columns = ["threshold", "trade_count", "reached", "percentage"]
    timed = df[df["mfe_points"].notna()]
    if timed.empty:
        return pd.DataFrame(columns=columns)

    total = len(timed)
    rows = []
    for threshold in thresholds:
        reached = int((timed["mfe_points"] >= threshold).sum())
        rows.append({"threshold": threshold, "trade_count": total, "reached": reached,
                     "percentage": round(reached / total * 100, 1)})
    return pd.DataFrame(rows, columns=columns)


def calculate_mae_efficiency(df, thresholds=(-100, -150, -200, -250, -300)):
    """
    For each points threshold (negative), the percentage of trades whose
    adverse excursion (mae_points) went at least that far against the
    position before it resolved. Mirrors calculate_mfe_efficiency, but
    for the "how much heat did this trade take" view (Zona de Recuo).
    """
    columns = ["threshold", "trade_count", "reached", "percentage"]
    timed = df[df["mae_points"].notna()]
    if timed.empty:
        return pd.DataFrame(columns=columns)

    total = len(timed)
    rows = []
    for threshold in thresholds:
        reached = int((timed["mae_points"] <= threshold).sum())
        rows.append({"threshold": threshold, "trade_count": total, "reached": reached,
                     "percentage": round(reached / total * 100, 1)})
    return pd.DataFrame(rows, columns=columns)


def calculate_metrics(df):
    """
    Calculate performance metrics from a DataFrame of trades.
    Returns a dictionary with all the dashboard indicators.

    If the DataFrame is empty, returns zeroed metrics instead of
    crashing - this matters because the dashboard needs to work even
    before any trade has been logged.
    """
    if df.empty:
        return {
            "total_result": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "days_traded": 0,
            "max_drawdown": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "profit_factor": 0.0,
            "risk_reward": 0.0,
            "expectancy": 0.0,
        }

    total_result = df["result_financial"].sum()
    total_trades = len(df)

    winners = df[df["result_financial"] > 0]
    losers = df[df["result_financial"] < 0]

    winning_trades = len(winners)
    losing_trades = len(losers)

    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

    days_traded = df["trade_date"].nunique()

    average_win = winners["result_financial"].mean() if winning_trades > 0 else 0.0
    average_loss = losers["result_financial"].mean() if losing_trades > 0 else 0.0

    total_gains = winners["result_financial"].sum()
    total_losses = abs(losers["result_financial"].sum())

    # Profit factor: how much is gained for each real lost. Above 1 = profitable.
    profit_factor = (total_gains / total_losses) if total_losses > 0 else float("inf")

    # Risk x reward: relationship between the average win size and the average loss size.
    risk_reward = (abs(average_win) / abs(average_loss)) if average_loss != 0 else float("inf")

    # Expectancy: how much, on average, each trade tends to yield.
    expectancy = total_result / total_trades if total_trades > 0 else 0.0

    # Max drawdown: largest drop of the equity curve relative to its previous peak.
    equity_curve = df.sort_values(["trade_date", "entry_time"])["result_financial"].cumsum()
    running_peak = equity_curve.cummax()
    drawdown = equity_curve - running_peak
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0

    return {
        "total_result": round(total_result, 2),
        "win_rate": round(win_rate, 1),
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "days_traded": days_traded,
        "max_drawdown": round(max_drawdown, 2),
        "average_win": round(average_win, 2),
        "average_loss": round(average_loss, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else None,
        "risk_reward": round(risk_reward, 2) if risk_reward != float("inf") else None,
        "expectancy": round(expectancy, 2),
    }
