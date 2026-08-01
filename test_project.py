"""
test_project.py

Tests for the functions in project.py, using pytest.
Run with: pytest test_project.py -v
"""

import pytest
import pandas as pd
from project import (
    calculate_result, create_trade, calculate_metrics, filter_by_setup,
    save_trade, seed_demo_trades, clear_demo_trades, calculate_daily_results,
    get_trade, update_trade, delete_trade,
    calculate_fees, calculate_daily_net_results, calculate_net_summary,
    parse_broker_csv, filter_by_time_range, classify_shift, filter_by_shift,
    calculate_efficiency_breakdown, calculate_performance_by_hour,
    calculate_performance_by_weekday, calculate_streaks,
    calculate_mfe_efficiency, calculate_mae_efficiency,
)
from database import initialize_database


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


# ---------- is_demo / seed_demo_trades / clear_demo_trades tests ----------
#
# These tests use a real, temporary, in-memory database (":memory:")
# instead of mocking anything - the goal is to prove that demo trades
# are correctly flagged, isolated, and removable without touching real
# trades, which is exactly the behavior a fake in-memory object could
# not verify.

def test_create_trade_defaults_to_not_demo():
    trade = create_trade(
        trade_date="2026-07-18", direction="buy",
        entry_price=140000, exit_price=140500, contracts=1
    )
    assert trade["is_demo"] is False


def test_seed_demo_trades_flags_rows_as_demo():
    connection = initialize_database(":memory:")
    seed_demo_trades(connection)

    cursor = connection.cursor()
    total_rows = cursor.execute("SELECT COUNT(*) AS count FROM trades").fetchone()["count"]
    demo_rows = cursor.execute("SELECT COUNT(*) AS count FROM trades WHERE is_demo = 1").fetchone()["count"]

    assert total_rows == demo_rows  # every row inserted by seed should be flagged
    assert total_rows == 30


def test_seed_demo_trades_is_idempotent():
    # Calling it twice should not duplicate the demo dataset.
    connection = initialize_database(":memory:")
    first_call = seed_demo_trades(connection)
    second_call = seed_demo_trades(connection)

    assert first_call == 30
    assert second_call == 0


def test_clear_demo_trades_removes_only_demo_rows():
    connection = initialize_database(":memory:")

    real_trade = create_trade(
        trade_date="2026-07-18", direction="buy",
        entry_price=140000, exit_price=140500, contracts=1, setup="TA"
    )
    save_trade(connection, real_trade)
    seed_demo_trades(connection)

    removed = clear_demo_trades(connection)

    cursor = connection.cursor()
    remaining = cursor.execute("SELECT COUNT(*) AS count FROM trades").fetchone()["count"]
    remaining_is_demo = cursor.execute("SELECT COUNT(*) AS count FROM trades WHERE is_demo = 1").fetchone()["count"]

    assert removed == 30
    assert remaining == 1  # only the real trade is left
    assert remaining_is_demo == 0


# ---------- calculate_daily_results tests ----------

def test_calculate_daily_results_groups_and_counts():
    df = pd.DataFrame([
        {"trade_date": pd.Timestamp("2026-07-01"), "result_financial": 100.0},
        {"trade_date": pd.Timestamp("2026-07-01"), "result_financial": -30.0},
        {"trade_date": pd.Timestamp("2026-07-02"), "result_financial": 50.0},
    ])
    daily = calculate_daily_results(df)

    day_one = daily[daily["trade_date"] == pd.Timestamp("2026-07-01")].iloc[0]
    assert day_one["result_financial"] == 70.0  # 100 - 30
    assert day_one["trade_count"] == 2

    day_two = daily[daily["trade_date"] == pd.Timestamp("2026-07-02")].iloc[0]
    assert day_two["result_financial"] == 50.0
    assert day_two["trade_count"] == 1


def test_calculate_daily_results_empty_dataframe():
    df = pd.DataFrame(columns=["trade_date", "result_financial"])
    daily = calculate_daily_results(df)
    assert daily.empty


# ---------- get_trade / update_trade / delete_trade tests ----------

def test_get_trade_returns_saved_trade():
    connection = initialize_database(":memory:")
    trade = create_trade(
        trade_date="2026-07-18", direction="buy",
        entry_price=140000, exit_price=140500, contracts=1, setup="TA"
    )
    trade_id = save_trade(connection, trade)

    fetched = get_trade(connection, trade_id)

    assert fetched is not None
    assert fetched["direction"] == "buy"
    assert fetched["setup"] == "TA"
    assert fetched["result_financial"] == 100.0


def test_get_trade_returns_none_for_missing_id():
    connection = initialize_database(":memory:")
    assert get_trade(connection, 9999) is None


def test_update_trade_recalculates_result():
    connection = initialize_database(":memory:")
    trade = create_trade(
        trade_date="2026-07-18", direction="buy",
        entry_price=140000, exit_price=140500, contracts=1, setup="TA"
    )
    trade_id = save_trade(connection, trade)

    # Correcting a typo: the real exit price was 141000, not 140500.
    update_trade(
        connection, trade_id,
        trade_date="2026-07-18", direction="buy",
        entry_price=140000, exit_price=141000, contracts=1, setup="TA"
    )

    updated = get_trade(connection, trade_id)
    assert updated["exit_price"] == 141000
    assert updated["result_points"] == 1000
    assert updated["result_financial"] == 200.0  # 1000 points * R$0.20 * 1 contract


def test_delete_trade_removes_only_target_row():
    connection = initialize_database(":memory:")
    trade_a = create_trade("2026-07-18", "buy", 140000, 140500, 1)
    trade_b = create_trade("2026-07-19", "sell", 140000, 139500, 1)
    id_a = save_trade(connection, trade_a)
    id_b = save_trade(connection, trade_b)

    deleted = delete_trade(connection, id_a)

    assert deleted is True
    assert get_trade(connection, id_a) is None
    assert get_trade(connection, id_b) is not None


def test_delete_trade_returns_false_for_missing_id():
    connection = initialize_database(":memory:")
    assert delete_trade(connection, 9999) is False


# ---------- calculate_fees / calculate_daily_net_results / calculate_net_summary ----------

def test_calculate_fees_charges_both_legs():
    # R$0.18 per contract, charged once on entry and once on exit.
    assert calculate_fees(contracts=1) == 0.36
    assert calculate_fees(contracts=20) == 7.20


def test_calculate_daily_net_results_taxes_only_positive_days():
    df = pd.DataFrame([
        # Profitable day: 07/01, gross 1000, 10 contracts total fees
        {"trade_date": pd.Timestamp("2026-07-01"), "contracts": 10, "result_financial": 1000.0},
        # Losing day: 07/02, gross -200
        {"trade_date": pd.Timestamp("2026-07-02"), "contracts": 5, "result_financial": -200.0},
    ])
    daily = calculate_daily_net_results(df)

    day_one = daily[daily["trade_date"] == pd.Timestamp("2026-07-01")].iloc[0]
    fees_day_one = calculate_fees(10)
    after_fees_day_one = 1000.0 - fees_day_one
    expected_tax_day_one = round(after_fees_day_one * 0.01, 2)
    assert day_one["fees"] == fees_day_one
    assert day_one["tax"] == expected_tax_day_one
    assert day_one["net_result"] == round(after_fees_day_one - expected_tax_day_one, 2)

    day_two = daily[daily["trade_date"] == pd.Timestamp("2026-07-02")].iloc[0]
    # A losing day must never be taxed, even after fees are subtracted.
    assert day_two["tax"] == 0.0


def test_calculate_net_summary_aggregates_all_days():
    df = pd.DataFrame([
        {"trade_date": pd.Timestamp("2026-07-01"), "contracts": 10, "result_financial": 1000.0},
        {"trade_date": pd.Timestamp("2026-07-02"), "contracts": 5, "result_financial": -200.0},
    ])
    summary = calculate_net_summary(df)
    assert summary["gross_result"] == 800.0  # 1000 - 200
    assert summary["total_fees"] == calculate_fees(10) + calculate_fees(5)
    assert summary["estimated_tax"] > 0  # only day one contributes tax


def test_calculate_net_summary_empty_dataframe():
    df = pd.DataFrame(columns=["trade_date", "contracts", "result_financial"])
    summary = calculate_net_summary(df)
    assert summary["net_result"] == 0.0


# ---------- parse_broker_csv ----------
#
# This uses the exact structure of a real report exported by the
# brokerage: metadata lines, a blank line, then a ";"-separated table.

SAMPLE_BROKER_CSV = (
    "Conta: 1001125\n"
    "Titular: MARCOS THIAGO CARDOSO DO NASCIMENTO\n"
    "Data: 28/07/2026\n"
    "\n"
    "Ativo;Abertura;Fechamento;Qtd Compra;Qtd Venda;Lado;Preço Compra;Preço Venda;"
    "Preço de Mercado;MEP;MEN;Res. Operação;Res. Operação (%);Drawdown;Total\n"
    "WINQ26;28/07/2026 09:01:02;28/07/2026 09:01:03;20;20;C;177.448,25;177.575,00;"
    "177.905,00;126,75;-43,25;507,00;126,75;0,00;507,00\n"
    "WINQ26;28/07/2026 09:42:58;28/07/2026 09:43:09;5;5;V;178.740,00;178.780,00;"
    "177.905,00;55,00;-30,00;40,00;40,00;-15,00;1.826,00\n"
)


def test_parse_broker_csv_row_count():
    trades = parse_broker_csv(SAMPLE_BROKER_CSV)
    assert len(trades) == 2


def test_parse_broker_csv_buy_side_mapping():
    trades = parse_broker_csv(SAMPLE_BROKER_CSV)
    buy_trade = trades[0]  # Lado = C

    assert buy_trade["direction"] == "buy"
    assert buy_trade["trade_date"] == "2026-07-28"
    assert buy_trade["entry_time"] == "09:01:02"
    assert buy_trade["exit_time"] == "09:01:03"
    assert buy_trade["contracts"] == 20
    assert buy_trade["entry_price"] == 177448.25
    assert buy_trade["exit_price"] == 177575.00
    # This is the number the brokerage itself reports for this row -
    # confirming our own calculate_result formula agrees with theirs.
    assert buy_trade["result_financial"] == 507.00


def test_parse_broker_csv_sell_side_mapping():
    trades = parse_broker_csv(SAMPLE_BROKER_CSV)
    sell_trade = trades[1]  # Lado = V

    assert sell_trade["direction"] == "sell"
    assert sell_trade["entry_price"] == 178780.00  # Preço Venda: the sell happened first
    assert sell_trade["exit_price"] == 178740.00   # Preço Compra: bought back to close
    assert sell_trade["result_financial"] == 40.00


def test_parse_broker_csv_leaves_setup_and_notes_empty():
    trades = parse_broker_csv(SAMPLE_BROKER_CSV)
    for trade in trades:
        assert trade["setup"] is None
        assert trade["technical_notes"] is None


def test_parse_broker_csv_missing_header_raises():
    with pytest.raises(ValueError):
        parse_broker_csv("this is not a valid broker report\njust some text\n")


# ---------- filter_by_time_range / classify_shift / filter_by_shift ----------

def _timed_df():
    return pd.DataFrame([
        {"entry_time": "09:15", "result_financial": 100.0},
        {"entry_time": "11:45", "result_financial": -20.0},
        {"entry_time": "14:30", "result_financial": 50.0},
        {"entry_time": None, "result_financial": 10.0},
    ])


def test_filter_by_time_range_keeps_only_within_bounds():
    df = _timed_df()
    filtered = filter_by_time_range(df, start_time="10:00", end_time="15:00")
    assert set(filtered["entry_time"]) == {"11:45", "14:30"}


def test_filter_by_time_range_no_bounds_returns_everything():
    df = _timed_df()
    filtered = filter_by_time_range(df)
    assert len(filtered) == len(df)


def test_classify_shift_morning_and_afternoon():
    assert classify_shift("09:15") == "Manhã"
    assert classify_shift("11:59") == "Manhã"
    assert classify_shift("12:00") == "Tarde"
    assert classify_shift("17:30") == "Tarde"
    assert classify_shift(None) is None


def test_filter_by_shift():
    df = _timed_df()
    morning = filter_by_shift(df, "Manhã")
    assert set(morning["entry_time"]) == {"09:15", "11:45"}


# ---------- calculate_efficiency_breakdown ----------

def test_calculate_efficiency_breakdown_counts_all_three_outcomes():
    df = pd.DataFrame([
        {"result_financial": 100.0},
        {"result_financial": -50.0},
        {"result_financial": 0.0},
        {"result_financial": 30.0},
    ])
    breakdown = calculate_efficiency_breakdown(df)
    assert breakdown == {"winners": 2, "losers": 1, "breakeven": 1}


# ---------- calculate_performance_by_hour / by_weekday ----------

def test_calculate_performance_by_hour_groups_correctly():
    df = pd.DataFrame([
        {"entry_time": "09:10", "result_financial": 100.0},
        {"entry_time": "09:45", "result_financial": -40.0},
        {"entry_time": "14:05", "result_financial": 60.0},
    ])
    by_hour = calculate_performance_by_hour(df)

    hour_nine = by_hour[by_hour["hour"] == 9].iloc[0]
    assert hour_nine["trade_count"] == 2
    assert hour_nine["result_financial"] == 60.0  # 100 - 40
    assert hour_nine["win_rate"] == 50.0


def test_calculate_performance_by_weekday_includes_all_days_even_empty():
    # A Monday and a Wednesday only - other weekdays must still appear, zeroed.
    df = pd.DataFrame([
        {"trade_date": pd.Timestamp("2026-07-06"), "result_financial": 100.0},  # Monday
        {"trade_date": pd.Timestamp("2026-07-08"), "result_financial": -20.0},  # Wednesday
    ])
    by_weekday = calculate_performance_by_weekday(df)

    assert len(by_weekday) == 7
    monday = by_weekday[by_weekday["weekday"] == "Segunda"].iloc[0]
    assert monday["result_financial"] == 100.0
    tuesday = by_weekday[by_weekday["weekday"] == "Terça"].iloc[0]
    assert tuesday["trade_count"] == 0


# ---------- calculate_streaks ----------

def test_calculate_streaks_tracks_current_and_max():
    df = pd.DataFrame([
        {"trade_date": pd.Timestamp("2026-07-01"), "entry_time": "09:00", "result_financial": 100.0},
        {"trade_date": pd.Timestamp("2026-07-01"), "entry_time": "09:30", "result_financial": 50.0},
        {"trade_date": pd.Timestamp("2026-07-01"), "entry_time": "10:00", "result_financial": -20.0},
        {"trade_date": pd.Timestamp("2026-07-02"), "entry_time": "09:00", "result_financial": -30.0},
        {"trade_date": pd.Timestamp("2026-07-02"), "entry_time": "09:30", "result_financial": -10.0},
    ])
    streaks = calculate_streaks(df)

    # Sequence is win, win, loss, loss, loss: 2 wins in a row, then 3 losses in a row.
    assert streaks["max_win_streak"] == 2
    assert streaks["max_loss_streak"] == 3
    assert streaks["current_type"] == "loss"
    assert streaks["current_length"] == 3


def test_calculate_streaks_breakeven_resets_streak():
    df = pd.DataFrame([
        {"trade_date": pd.Timestamp("2026-07-01"), "entry_time": "09:00", "result_financial": 100.0},
        {"trade_date": pd.Timestamp("2026-07-01"), "entry_time": "09:30", "result_financial": 0.0},
        {"trade_date": pd.Timestamp("2026-07-01"), "entry_time": "10:00", "result_financial": 50.0},
    ])
    streaks = calculate_streaks(df)
    # The breakeven trade resets the streak, so the current streak is
    # only 1 win long, not 2.
    assert streaks["current_type"] == "win"
    assert streaks["current_length"] == 1


# ---------- calculate_mfe_efficiency / calculate_mae_efficiency ----------

def test_calculate_mfe_efficiency_percentages():
    df = pd.DataFrame([
        {"mfe_points": 320.0},
        {"mfe_points": 180.0},
        {"mfe_points": 90.0},
        {"mfe_points": None},
    ])
    efficiency = calculate_mfe_efficiency(df, thresholds=(100, 200, 300))

    row_100 = efficiency[efficiency["threshold"] == 100].iloc[0]
    assert row_100["trade_count"] == 3  # the None row is excluded from the base
    assert row_100["reached"] == 2      # 320 and 180 both reached 100+
    assert row_100["percentage"] == pytest.approx(66.7, abs=0.1)


def test_calculate_mae_efficiency_percentages():
    df = pd.DataFrame([
        {"mae_points": -220.0},
        {"mae_points": -80.0},
        {"mae_points": None},
    ])
    efficiency = calculate_mae_efficiency(df, thresholds=(-100, -200))

    row_100 = efficiency[efficiency["threshold"] == -100].iloc[0]
    assert row_100["trade_count"] == 2
    assert row_100["reached"] == 1  # only -220 went past -100
    assert row_100["percentage"] == 50.0
