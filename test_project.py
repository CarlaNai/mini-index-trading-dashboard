"""
test_project.py

Tests for the functions in project.py, using pytest.
Run with: pytest test_project.py -v
"""

import pytest
import pandas as pd
from project import calculate_result, create_trade, calculate_metrics, filter_by_setup


# ---------- calculate_result tests ----------

def test_calculate_result_buy_profit():
    # Bought at 140000, sold at 140500 -> gained 500 points
    points, financial = calculate_result("buy", 140000, 140500, contracts=1)
    assert points == 500
    assert financial == 100.0  # 500 points * R$0.20 * 1 contract


def test_calculate_result_buy_loss():
    # Bought at 140000, sold at 139800 -> lost 200 points
    points, financial = calculate_result("buy", 140000, 139800, contracts=2)
    assert points == -200
    assert financial == -80.0  # -200 * 0.20 * 2


def test_calculate_result_sell_profit():
    # Sold at 140000, bought back at 139500 -> gained 500 points
    points, financial = calculate_result("sell", 140000, 139500, contracts=1)
    assert points == 500
    assert financial == 100.0


def test_calculate_result_invalid_direction():
    with pytest.raises(ValueError):
        calculate_result("sideways", 140000, 140500, contracts=1)


def test_calculate_result_invalid_contracts():
    with pytest.raises(ValueError):
        calculate_result("buy", 140000, 140500, contracts=0)


def test_calculate_result_invalid_price():
    with pytest.raises(ValueError):
        calculate_result("buy", -100, 140500, contracts=1)


# ---------- create_trade tests ----------

def test_create_trade_basic_fields():
    trade = create_trade(
        trade_date="2026-07-18", direction="buy",
        entry_price=140000, exit_price=140500, contracts=1,
        setup="TA"
    )
    assert trade["trade_date"] == "2026-07-18"
    assert trade["direction"] == "buy"
    assert trade["setup"] == "TA"
    assert trade["result_points"] == 500
    assert trade["result_financial"] == 100.0


def test_create_trade_normalizes_direction():
    # Should accept "BUY" or " buy " and normalize to "buy"
    trade = create_trade(
        trade_date="2026-07-18", direction="  BUY  ",
        entry_price=140000, exit_price=140500, contracts=1
    )
    assert trade["direction"] == "buy"


# ---------- calculate_metrics tests ----------

def _sample_df():
    """Sample DataFrame: 3 winning trades, 2 losing trades."""
    data = [
        {"trade_date": pd.Timestamp("2026-07-01"), "entry_time": "09:00", "result_financial": 100.0},
        {"trade_date": pd.Timestamp("2026-07-01"), "entry_time": "10:00", "result_financial": -50.0},
        {"trade_date": pd.Timestamp("2026-07-02"), "entry_time": "09:00", "result_financial": 200.0},
        {"trade_date": pd.Timestamp("2026-07-03"), "entry_time": "09:00", "result_financial": -30.0},
        {"trade_date": pd.Timestamp("2026-07-03"), "entry_time": "10:00", "result_financial": 150.0},
    ]
    return pd.DataFrame(data)


def test_calculate_metrics_total_result():
    df = _sample_df()
    metrics = calculate_metrics(df)
    assert metrics["total_result"] == 370.0  # 100-50+200-30+150


def test_calculate_metrics_win_rate():
    df = _sample_df()
    metrics = calculate_metrics(df)
    # 3 winning trades out of 5 total = 60%
    assert metrics["win_rate"] == 60.0


def test_calculate_metrics_days_traded():
    df = _sample_df()
    metrics = calculate_metrics(df)
    # 3 distinct dates: 07/01, 07/02, 07/03
    assert metrics["days_traded"] == 3


def test_calculate_metrics_empty_dataframe():
    df = pd.DataFrame(columns=["trade_date", "entry_time", "result_financial"])
    metrics = calculate_metrics(df)
    assert metrics["total_result"] == 0.0
    assert metrics["win_rate"] == 0.0
    assert metrics["total_trades"] == 0


# ---------- filter_by_setup tests ----------

def test_filter_by_setup():
    df = pd.DataFrame([
        {"setup": "TA", "result_financial": 100.0},
        {"setup": "TC", "result_financial": 50.0},
        {"setup": "TA", "result_financial": -20.0},
    ])
    filtered = filter_by_setup(df, "TA")
    assert len(filtered) == 2
    assert all(filtered["setup"] == "TA")


def test_filter_by_setup_all_returns_everything():
    df = pd.DataFrame([
        {"setup": "TA", "result_financial": 100.0},
        {"setup": "TC", "result_financial": 50.0},
    ])
    filtered = filter_by_setup(df, "all")
    assert len(filtered) == 2
