"""
project.py

Main functions of the CS50P final project - Mini Índice Trading Dashboard.

This file contains all the BUSINESS LOGIC: result calculations, data
validation, and performance metrics. Nothing here knows about databases
or screens - that separation is what makes these functions easy to test.
"""

import pandas as pd

WIN_POINT_VALUE = 0.20  # each point of the mini index is worth R$ 0.20 per contract
VALID_DIRECTIONS = ("buy", "sell")


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
                  emotional_state=None, technical_notes=None, screenshot_path=None):
    """
    Build a dictionary representing a full trade, already with the
    result calculated. This dictionary is what gets saved to the database.
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
    }


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
            technical_notes, screenshot_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade["trade_date"], trade["entry_time"], trade["exit_time"],
        trade["direction"], trade["setup"], trade["contracts"],
        trade["entry_price"], trade["exit_price"],
        trade["stop_points"], trade["mae_points"], trade["mfe_points"],
        trade["result_points"], trade["result_financial"],
        trade["emotional_state"], trade["technical_notes"],
        trade["screenshot_path"],
    ))
    connection.commit()
    return cursor.lastrowid


def load_trades(connection):
    """
    Read all trades from the database and return them as a pandas
    DataFrame (an in-memory table, easy to filter and calculate on top of).
    """
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
