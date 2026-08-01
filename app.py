"""Streamlit interface for the Mini Índice Trading Dashboard."""

import calendar as calendar_module
from datetime import timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from database import get_connection, ensure_connection
from project import (
    calculate_metrics, create_trade, filter_by_setup, load_trades, save_trade,
    calculate_daily_results,
    get_trade, update_trade, delete_trade, parse_broker_csv, calculate_net_summary,
    filter_by_time_range, filter_by_shift, calculate_efficiency_breakdown,
    calculate_performance_by_hour, calculate_performance_by_weekday,
    calculate_mfe_efficiency, calculate_mae_efficiency,
)


st.set_page_config(
    page_title="Mini Índice | Painel", page_icon="◈", layout="wide",
)

SETUP_OPTIONS = ["TA", "TC", "TRM", "FQ"]
DIRECTION_LABELS = {"buy": "▲ Compra", "sell": "▼ Venda"}
EMOTIONAL_STATES = ["Confiante", "Calmo", "Focado", "Atento", "Cauteloso", "Neutro", "Ansioso", "Irritado", "Impulsivo", "Com medo"]
MIN_EMOTIONAL_STATES = 3


# ----------------------------------------------------------------------
# Small display helpers
# ----------------------------------------------------------------------

def format_currency(value):
    """Format Brazilian-real values consistently throughout the dashboard."""
    sign = "+" if value > 0 else ""
    return f"{sign}R$ {value:,.2f}"


def render_metric(column, label, value_text, delta_text=None, numeric_value=None, tone=None):
    """
    Render one metric card as custom HTML instead of st.metric.

    Color is decided either from an explicit `tone` ("gain"/"loss"/"neutral"),
    or automatically from the sign of `numeric_value` when tone is not given -
    positive numbers render green, negative numbers render red. This is what
    lets Max Drawdown and Média Perdedora show up in red like a real loss,
    while Resultado Total still turns green on a profitable period.
    """
    if tone is None:
        if numeric_value is None or numeric_value == 0:
            tone = "neutral"
        else:
            tone = "gain" if numeric_value > 0 else "loss"

    delta_html = f'<div class="metric-delta">{delta_text}</div>' if delta_text else ""
    column.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value metric-{tone}">{value_text}</div>
          {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def dark_chart(fig):
    """Apply the dashboard visual language to Plotly figures."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c5ced1", family="Inter, Arial, sans-serif"),
        margin=dict(l=12, r=12, t=28, b=12),
        hoverlabel=dict(bgcolor="#151b22", font_color="#eef7f5"),
        legend=dict(font=dict(color="#c5ced1")),
        height=330,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#29333a", tickfont=dict(color="#89979c"))
    # Convention requested for this dashboard: the y-axis is drawn on
    # the right side of the chart instead of Plotly's default left side.
    fig.update_yaxes(
        gridcolor="rgba(91, 111, 116, .18)", zerolinecolor="#405157", tickfont=dict(color="#89979c"),
        side="right",
    )
    return fig


def pnl_bar_chart(df, x_col, x_label):
    """
    Build a consistently-styled result-by-category bar chart (used for
    both "performance by hour" and "performance by weekday"). Bars are
    colored by sign, labeled with their value, and the hover tooltip
    surfaces trade count and win rate - the two numbers that give
    context to whether a result is worth trusting.
    """
    colors = df["result_financial"].apply(lambda v: "#22d6a0" if v >= 0 else "#fa5c78")
    fig = px.bar(
        df, x=x_col, y="result_financial",
        labels={x_col: x_label, "result_financial": "Resultado (R$)"},
        custom_data=["trade_count", "win_rate"],
    )
    fig.update_traces(
        marker_color=colors, marker_line_width=0,
        hovertemplate=(
            "<b>%{x}</b><br>Resultado: R$ %{y:,.2f}"
            "<br>Operações: %{customdata[0]}<br>Win rate: %{customdata[1]}%<extra></extra>"
        ),
        texttemplate="%{y:,.0f}", textposition="outside", textfont_size=10, textfont_color="#9ba9ad",
        cliponaxis=False,
    )
    fig.update_layout(showlegend=False, bargap=.4, uniformtext_minsize=8)
    fig.add_hline(y=0, line_width=1, line_color="#405157")
    return fig


def threshold_bar_chart(df, color, x_label):
    """Shared styling for the MFE/MAE threshold-efficiency bar charts."""
    fig = px.bar(df, x="threshold", y="percentage", labels={"threshold": x_label, "percentage": "% das operações"})
    fig.update_traces(
        marker_color=color, marker_line_width=0,
        texttemplate="%{y:.0f}%", textposition="outside", textfont_size=10, textfont_color="#9ba9ad",
        cliponaxis=False,
        hovertemplate=f"<b>%{{x}} pts</b><br>%{{y:.0f}}% das operações<extra></extra>",
    )
    fig.update_layout(bargap=.4, uniformtext_minsize=8)
    return fig


def empty_state(icon, title, message):
    """A consistent, friendlier placeholder for charts/sections with no data yet."""
    st.markdown(
        f"""
        <div class="empty-state">
          <div class="empty-state-icon">{icon}</div>
          <div class="empty-state-title">{title}</div>
          <div class="empty-state-message">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_calendar_html(year, month, daily_lookup):
    """
    Build a month calendar as an HTML/CSS grid: one cell per day, colored
    green for a profitable day, red for a losing day, and blank for days
    with no trades.

    daily_lookup is a dict of {date: (result_financial, trade_count)}.
    """
    weekday_names = ["Dom", "Seg", "Ter", "Qua", "Qui", "Sex", "Sáb"]
    header_cells = "".join(f'<div class="cal-weekday">{name}</div>' for name in weekday_names)

    # firstweekday=6 makes weeks start on Sunday (Brazilian convention),
    # matching the weekday header above.
    month_weeks = calendar_module.Calendar(firstweekday=6).monthdatescalendar(year, month)

    day_cells = ""
    for week in month_weeks:
        for day in week:
            in_month = day.month == month
            entry = daily_lookup.get(day)

            if not in_month:
                day_cells += '<div class="cal-day cal-day-outside"></div>'
            elif entry is None:
                day_cells += f'<div class="cal-day"><span class="cal-day-number">{day.day}</span></div>'
            else:
                result, trade_count = entry
                css_class = "cal-day-gain" if result >= 0 else "cal-day-loss"
                sign = "+" if result >= 0 else ""
                day_cells += (
                    f'<div class="cal-day {css_class}">'
                    f'<span class="cal-day-number">{day.day}</span>'
                    f'<span class="cal-day-result">{sign}{result:,.0f}</span>'
                    f'<span class="cal-day-count">{trade_count} op.</span>'
                    f"</div>"
                )

    return f'<div class="cal-grid">{header_cells}{day_cells}</div>'


@st.dialog("Registrar operação")
def new_trade_dialog(connection):
    """
    Full trade-entry form, opened as a modal from the main action bar.
    Keeping it in a dialog (instead of an always-open form) keeps the
    page itself focused on the data, while still making the form feel
    close at hand.
    """
    with st.form("new_trade_form", clear_on_submit=True):
        date_col, entry_time_col, exit_time_col = st.columns(3)
        with date_col:
            trade_date = st.date_input("Data")
        with entry_time_col:
            entry_time = st.time_input("Horário de entrada", value=None)
        with exit_time_col:
            exit_time = st.time_input("Horário de saída", value=None)

        direction_col, setup_col = st.columns(2)
        with direction_col:
            direction = st.selectbox("Direção", ["buy", "sell"], format_func=lambda item: DIRECTION_LABELS[item])
        with setup_col:
            setup = st.selectbox("Setup", SETUP_OPTIONS)

        entry_col, exit_col, contracts_col = st.columns(3)
        with entry_col:
            entry_price = st.number_input("Entrada (pts)", min_value=0.0, step=5.0, format="%.0f")
        with exit_col:
            exit_price = st.number_input("Saída (pts)", min_value=0.0, step=5.0, format="%.0f")
        with contracts_col:
            contracts = st.number_input("Contratos", min_value=1, step=1)

        stop_points = st.number_input("Stop (pontos)", min_value=0.0, step=10.0, format="%.0f")

        emotional_state = st.multiselect(
            "Estado emocional", EMOTIONAL_STATES,
            help=f"Selecione pelo menos {MIN_EMOTIONAL_STATES} estados que descrevem como você estava durante a operação.",
        )
        technical_notes = st.text_area(
            "O que você fez e por quê?",
            placeholder="O que viu no gráfico? Por que entrou? O que faria diferente?",
        )
        submitted = st.form_submit_button("Gravar operação", width='stretch')

        if submitted:
            if len(emotional_state) < MIN_EMOTIONAL_STATES:
                st.error(f"Selecione pelo menos {MIN_EMOTIONAL_STATES} estados emocionais.")
            else:
                try:
                    trade = create_trade(
                        str(trade_date), direction, entry_price, exit_price, contracts,
                        entry_time=entry_time.strftime("%H:%M:%S") if entry_time else None,
                        exit_time=exit_time.strftime("%H:%M:%S") if exit_time else None,
                        setup=setup, stop_points=stop_points,
                        emotional_state=", ".join(emotional_state),
                        technical_notes=technical_notes or None,
                    )
                    save_trade(connection, trade)
                    st.session_state["flash_success"] = "Operação registrada com sucesso."
                    st.rerun()
                except ValueError as error:
                    st.error(f"Não foi possível registrar: {error}")


@st.dialog("Excluir operação")
def confirm_delete_dialog(connection, trade_id, trade_label):
    """
    A real modal confirmation for a destructive action, instead of a
    checkbox-then-button pattern. This traps focus on the decision and
    makes accidental deletion much harder.
    """
    st.markdown(f"Tem certeza de que deseja excluir a operação **{trade_label}**?")
    st.caption("Essa ação não pode ser desfeita.")
    col_cancel, col_confirm = st.columns(2)
    if col_cancel.button("Cancelar", width='stretch'):
        st.rerun()
    if col_confirm.button("Excluir definitivamente", type="primary", width='stretch'):
        delete_trade(connection, trade_id)
        st.session_state["flash_success"] = "Operação excluída."
        st.rerun()


# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------

st.markdown(
    """
    <style>
      :root { --mint: #22d6a0; --cyan: #35c8df; --ink: #091012; --panel: #151b22; --line: #28343a; }
      .stApp {
        background:
          radial-gradient(circle at 76% 0%, rgba(17, 142, 111, .19), transparent 31rem),
          radial-gradient(circle at 14% 83%, rgba(81, 31, 85, .16), transparent 35rem),
          #091012;
        color: #e8eeef;
      }
      [data-testid="stHeader"] { background: transparent; }
      #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stAppToolbar"],
      [data-testid="stDecoration"], [data-testid="stStatusWidget"] { visibility: hidden; display: none; }
      h1, h2, h3 { font-family: Inter, Arial, sans-serif !important; letter-spacing: -.035em; color: #f2f6f6 !important; }
      h1 { font-size: 4.2rem !important; font-weight: 800 !important; margin-bottom: .15rem !important; line-height: 1.05 !important; }
      h2, h3 { font-size: 1rem !important; font-weight: 700 !important; }
      p, label { font-family: Inter, Arial, sans-serif !important; }

      /* Focus states for keyboard navigation (accessibility) */
      button:focus-visible, [role="tab"]:focus-visible, input:focus-visible,
      [data-baseweb="select"] *:focus-visible {
        outline: 2px solid var(--mint) !important; outline-offset: 2px;
      }

      [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(31, 39, 47, .93), rgba(19, 25, 31, .94));
        border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.1rem;
        min-height: 112px; box-shadow: 0 10px 28px rgba(0,0,0,.16);
      }
      [data-testid="stMetricLabel"] { color: #9ba9ad !important; font-size: .69rem !important; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
      [data-testid="stMetricValue"] { color: var(--mint) !important; font-size: 1.55rem !important; font-family: Consolas, monospace; }
      [data-testid="stMetricDelta"] { font-size: .73rem !important; }

      /* Custom metric cards (used instead of st.metric so the value
         color can depend on whether the number is a gain or a loss,
         which st.metric alone does not let us control per-card). */
      .metric-card {
        background: linear-gradient(135deg, rgba(31, 39, 47, .93), rgba(19, 25, 31, .94));
        border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.1rem;
        min-height: 112px; box-shadow: 0 10px 28px rgba(0,0,0,.16);
        display: flex; flex-direction: column; justify-content: center; gap: .3rem;
        transition: border-color .15s ease, transform .15s ease;
      }
      .metric-card:hover { border-color: #3a4850; transform: translateY(-1px); }
      .metric-label { color: #9ba9ad; font-size: .69rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
      .metric-value { font-size: 1.55rem; font-family: Consolas, monospace; font-weight: 700; }
      .metric-value.metric-gain { color: #22d6a0; }
      .metric-value.metric-loss { color: #fa5c78; }
      .metric-value.metric-neutral { color: #e8eeef; }
      .metric-delta { color: #8d9a9e; font-size: .73rem; }

      .block-container { max-width: 1680px; padding: 1.3rem 2.2rem 3rem; }
      .eyebrow { color: #38d6b1; font-size: .68rem; font-weight: 800; letter-spacing: .18em; text-transform: uppercase; margin-bottom: .3rem; }
      .subtitle { color: #8d9a9e; margin-top: 0; margin-bottom: 1.3rem; }
      .section-title { color: #dce6e5; font-size: 1rem; font-weight: 720; margin: 1.6rem 0 .7rem; }
      .chart-card {
        background: linear-gradient(135deg, rgba(25, 33, 40, .91), rgba(15, 21, 25, .92));
        border: 1px solid var(--line); border-radius: 12px; padding: .45rem .55rem .15rem; min-height: 100%;
      }
      .chart-card h3 { padding: .45rem .55rem 0; margin: 0; }
      [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(135deg, rgba(25, 33, 40, .92), rgba(15, 21, 25, .94));
        border-color: var(--line) !important; border-radius: 12px !important;
        box-shadow: 0 10px 28px rgba(0,0,0,.16);
      }
      [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stPlotlyChart"] { margin-top: -.5rem; }

      /* Compact stat rows shown next to the efficiency donut chart */
      .eff-stats { display: flex; flex-direction: column; justify-content: center; gap: .55rem; height: 100%; padding: .6rem .8rem; }
      .eff-row { display: flex; align-items: center; gap: .55rem; font-size: .86rem; }
      .eff-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
      .eff-dot-gain { background: #22d6a0; }
      .eff-dot-loss { background: #fa5c78; }
      .eff-dot-neutral { background: #5b6f74; }
      .eff-label { color: #b7c2c4; flex-grow: 1; }
      .eff-count { color: #eef6f4; font-family: Consolas, monospace; font-weight: 700; }
      .eff-share { color: #778489; font-size: .74rem; width: 2.6rem; text-align: right; }

      [data-testid="stForm"] { border: 1px solid #2a373c; background: rgba(9, 14, 17, .35); border-radius: 10px; padding: .8rem; }
      div[data-baseweb="select"] > div, div[data-baseweb="base-input"] > div, div[data-baseweb="input"] > div {
        background: #10161b !important; border-color: #2b383e !important; color: #e9f0ef !important;
      }
      div[data-baseweb="select"] *, div[data-baseweb="base-input"] input, div[data-baseweb="input"] input {
        color: #e9f0ef !important; -webkit-text-fill-color: #e9f0ef !important;
      }
      div[data-baseweb="select"] > div:hover, div[data-baseweb="base-input"] > div:hover { border-color: #31cfa4 !important; }

      /* Consistent height/border across native inputs, selects and
         buttons so they read as one coherent control system instead
         of mismatched browser/Streamlit defaults. */
      div[data-baseweb="select"] > div, div[data-baseweb="base-input"], div[data-baseweb="input"] {
        min-height: 2.6rem !important; border-radius: 8px !important;
      }
      .stButton > button, [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(90deg, #16bf88, #20d39c); color: #07110e; border: 0; border-radius: 8px;
        font-weight: 800; letter-spacing: .025em; transition: filter .12s ease; min-height: 2.6rem;
      }
      .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover { filter: brightness(1.08); color: #06110e; }
      .stButton > button[kind="secondary"] {
        background: transparent; color: #cfd9d8 !important; border: 1px solid #33424a;
      }
      .stButton > button[kind="secondary"]:hover { border-color: #536269; }

      [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
      .stAlert { border-radius: 10px; }
      hr { border-color: #263139 !important; }
      .demo-note { color: #819095; font-size: .72rem; line-height: 1.45; margin-top: .4rem; }

      /* Tabs: give the primary navigation clear affordance */
      [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: .3rem; border-bottom: 1px solid var(--line);
      }
      [data-testid="stTabs"] button[role="tab"] {
        color: #8d9a9e; font-weight: 700; font-size: .88rem; padding: .6rem 1rem;
      }
      [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--mint) !important; border-bottom: 2px solid var(--mint) !important;
      }

      /* Calendar view: 7-column grid, one row of weekday headers plus
         one cell per day. Colors mirror the equity/daily charts. */
      .cal-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 6px; margin-top: .5rem; }
      .cal-weekday { color: #778489; font-size: .68rem; font-weight: 700; text-transform: uppercase; text-align: center; padding-bottom: .3rem; }
      .cal-day {
        background: rgba(21, 27, 34, .55); border: 1px solid var(--line); border-radius: 8px;
        min-height: 72px; padding: .4rem .5rem; display: flex; flex-direction: column; gap: .15rem;
      }
      .cal-day-outside { background: transparent; border-color: transparent; }
      .cal-day-number { color: #9ba9ad; font-size: .72rem; font-weight: 700; }
      .cal-day-result { font-family: Consolas, monospace; font-size: .85rem; font-weight: 700; }
      .cal-day-count { color: #778489; font-size: .65rem; }
      .cal-day-gain { background: rgba(34, 214, 160, .14); border-color: rgba(34, 214, 160, .45); }
      .cal-day-gain .cal-day-result { color: #22d6a0; }
      .cal-day-loss { background: rgba(250, 92, 120, .14); border-color: rgba(250, 92, 120, .45); }
      .cal-day-loss .cal-day-result { color: #fa5c78; }
      .manage-hint { color: #819095; font-size: .82rem; margin: -.2rem 0 .8rem; }

      /* Friendlier empty state for charts/sections without data */
      .empty-state { text-align: center; padding: 2.2rem 1rem; color: #778489; }
      .empty-state-icon { font-size: 1.6rem; margin-bottom: .4rem; opacity: .7; }
      .empty-state-title { color: #b7c2c4; font-weight: 700; font-size: .88rem; margin-bottom: .2rem; }
      .empty-state-message { font-size: .78rem; line-height: 1.4; max-width: 30rem; margin: 0 auto; }

      @media (max-width: 800px) { .block-container { padding: 1.2rem 1rem 2rem; } h1 { font-size: 2.6rem !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)

if "connection" not in st.session_state:
    st.session_state.connection = get_connection()
st.session_state.connection = ensure_connection(st.session_state.connection)
connection = st.session_state.connection

# A one-shot success flash that survives the rerun triggered by the
# delete dialog (the dialog closes via rerun before this script body
# would otherwise get a chance to show a message).
if st.session_state.get("flash_success"):
    st.toast(st.session_state.pop("flash_success"), icon="✅")


# ----------------------------------------------------------------------
# Sidebar: data entry, grouped so only what's needed is open by default
# ----------------------------------------------------------------------

st.markdown('<div class="eyebrow">MINI ÍNDICE · PAINEL DE PERFORMANCE</div>', unsafe_allow_html=True)
st.title("Evolução das operações")
st.markdown('<p class="subtitle">Acompanhe seus resultados, consistência e desempenho por estratégia.</p>', unsafe_allow_html=True)

trades_df = load_trades(connection)

# Primary actions live in the main page body as a compact bar right
# under the title - registering a trade, importing a CSV, and
# exporting a backup are all one click away, without competing for
# space with the dashboard content below.
action_col1, action_col2, action_col3 = st.columns([1.2, 1.6, 1.2])

with action_col1:
    if st.button("＋ Registrar operação", width='stretch', type="primary"):
        new_trade_dialog(connection)

with action_col2:
    with st.popover("📂 Importar da corretora", width='stretch'):
        uploaded_csv = st.file_uploader(
            "Relatório diário (.csv)", type="csv",
            help="Exporte o relatório diário de operações da sua corretora e envie o arquivo aqui.",
        )
        if uploaded_csv is not None:
            # file_uploader keeps returning the same file across every
            # rerun (including the rerun triggered by "Cancelar
            # importação"), so without this guard the CSV would be
            # re-parsed and pending_import would be silently refilled
            # right after the user cancels. Only (re)process when this
            # is actually a different upload than the last one handled.
            if st.session_state.get("processed_upload_id") != uploaded_csv.file_id:
                try:
                    csv_text = uploaded_csv.getvalue().decode("latin-1")
                    parsed_trades = parse_broker_csv(csv_text)
                    st.session_state.pending_import = parsed_trades
                    st.session_state.processed_upload_id = uploaded_csv.file_id
                    st.success(f"{len(parsed_trades)} operações lidas. Revise na aba \"Gerenciar\", logo abaixo, antes de confirmar.")
                except ValueError as error:
                    st.error(f"Não foi possível ler este arquivo: {error}")

with action_col3:
    # A manual backup: the database itself is safe (Neon), but this
    # gives a plain file you can keep in your own Google Drive/OneDrive
    # for extra peace of mind, or open straight in Excel.
    export_csv = trades_df.to_csv(index=False, sep=";").encode("utf-8-sig")
    st.download_button(
        "⬇ Exportar CSV", data=export_csv,
        file_name=f"operacoes_{pd.Timestamp.now().strftime('%Y-%m-%d')}.csv",
        mime="text/csv", width='stretch', disabled=trades_df.empty,
        help="Baixa uma cópia de todas as operações - útil como backup manual." if not trades_df.empty else "Sem operações para exportar ainda.",
    )


if st.session_state.get("pending_import"):
    st.markdown('<div class="section-title">REVISAR IMPORTAÇÃO</div>', unsafe_allow_html=True)
    st.caption(f"{len(st.session_state.pending_import)} operações lidas do arquivo. Preencha o setup de cada uma e confirme.")

    import_df = pd.DataFrame(st.session_state.pending_import)
    import_df["setup"] = import_df["setup"].fillna("TA")
    import_df["technical_notes"] = import_df["technical_notes"].fillna("")

    display_columns = [
        "trade_date", "entry_time", "exit_time", "direction", "contracts",
        "entry_price", "exit_price", "result_financial", "setup", "technical_notes",
    ]
    edited_df = st.data_editor(
        import_df[display_columns],
        width='stretch',
        hide_index=True,
        disabled=[c for c in display_columns if c not in ("setup", "technical_notes")],
        column_config={
            "setup": st.column_config.SelectboxColumn("Setup", options=SETUP_OPTIONS, required=True),
            "technical_notes": st.column_config.TextColumn("Leitura técnica"),
            "result_financial": st.column_config.NumberColumn("Resultado (R$)", format="R$ %.2f"),
        },
        key="import_review_editor",
    )

    import_col1, import_col2 = st.columns(2)
    if import_col1.button("Confirmar importação", type="primary", width='stretch'):
        for original, (_, edited_row) in zip(st.session_state.pending_import, edited_df.iterrows()):
            original["setup"] = edited_row["setup"]
            original["technical_notes"] = edited_row["technical_notes"] or None
            save_trade(connection, original)
        st.session_state.pending_import = None
        st.session_state["flash_success"] = "Operações importadas com sucesso."
        st.rerun()

    if import_col2.button("Cancelar importação", width='stretch'):
        st.session_state.pending_import = None
        st.rerun()

    st.divider()

if trades_df.empty:
    empty_state(
        "◈", "Nenhuma operação registrada ainda",
        "Use o botão \"+ Registrar operação\" acima para começar seu diário.",
    )
else:
    # Demo data is no longer offered as a feature, but this keeps any
    # leftover demo rows from earlier testing permanently hidden,
    # instead of mixing fictional numbers into real performance.
    working_df = trades_df[trades_df["is_demo"] == 0]

    if working_df.empty:
        empty_state(
            "◈", "Nenhuma operação registrada ainda",
            "Use o botão \"+ Registrar operação\" acima para começar seu diário.",
        )
        st.stop()

    st.markdown('<div class="section-title">FILTROS</div>', unsafe_allow_html=True)
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    with filter_col1:
        available_setups = ["all"] + sorted(working_df["setup"].dropna().unique().tolist())
        selected_setup = st.selectbox("Estratégia", available_setups, format_func=lambda item: "Todas" if item == "all" else item)

    with filter_col2:
        min_date = working_df["trade_date"].min().date()
        max_date = working_df["trade_date"].max().date()
        date_range = st.date_input("Período", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    with filter_col3:
        hour_range = st.slider("Horário", 0, 23, (0, 23), format="%dh")

    with filter_col4:
        selected_shift = st.selectbox("Turno", ["all", "Manhã", "Tarde"], format_func=lambda item: "Todos" if item == "all" else item)

    filtered_df = filter_by_setup(working_df, selected_setup)

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df["trade_date"].dt.date >= start_date) & (filtered_df["trade_date"].dt.date <= end_date)
        ]

    start_hour, end_hour = hour_range
    if (start_hour, end_hour) != (0, 23):
        filtered_df = filter_by_time_range(filtered_df, start_time=f"{start_hour:02d}:00", end_time=f"{end_hour:02d}:59:59")

    filtered_df = filter_by_shift(filtered_df, selected_shift)

    if filtered_df.empty:
        empty_state(
            "🔎", "Nenhuma operação corresponde aos filtros",
            "Tente ampliar o período, o horário ou escolher \"Todas\" em estratégia e turno.",
        )
    else:
        metrics = calculate_metrics(filtered_df)

        # These four numbers are the "vital signs" of the journal, so they
        # stay visible above the tabs regardless of which section is open.
        row_one = st.columns(4)
        render_metric(row_one[0], "Resultado total", format_currency(metrics["total_result"]), numeric_value=metrics["total_result"])
        render_metric(row_one[1], "Win rate", f"{metrics['win_rate']:.1f}%", f"{metrics['winning_trades']} wins · {metrics['losing_trades']} losses", tone="neutral")
        render_metric(row_one[2], "Dias operados", metrics["days_traded"], f"{metrics['total_trades']} operações", tone="neutral")
        render_metric(row_one[3], "Max drawdown", format_currency(metrics["max_drawdown"]), tone="loss" if metrics["max_drawdown"] < 0 else "neutral")

        tab_overview, tab_advanced, tab_history, tab_manage = st.tabs(
            ["📊 Visão geral", "🔍 Análise avançada", "📅 Histórico", "⚙️ Gerenciar"]
        )

        # ------------------------------------------------------------
        # Tab 1 - Visão geral
        # ------------------------------------------------------------
        with tab_overview:
            equity_col, daily_col = st.columns(2)
            equity_df = filtered_df.sort_values(["trade_date", "entry_time"]).copy()
            equity_df["cumulative_result"] = equity_df["result_financial"].cumsum()
            fig_equity = px.line(
                equity_df, x="trade_date", y="cumulative_result", markers=True,
                labels={"trade_date": "Data", "cumulative_result": "Resultado acumulado (R$)"},
            )
            fig_equity.update_traces(
                line=dict(color="#24d3a0", width=3), marker=dict(color="#6be6c4", size=5),
                fill="tozeroy", fillcolor="rgba(36, 211, 160, .10)",
            )
            with equity_col:
                st.markdown('<div class="chart-card"><h3>Curva de capital</h3>', unsafe_allow_html=True)
                st.plotly_chart(dark_chart(fig_equity), width='stretch', config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)

            daily_df = calculate_daily_results(filtered_df)
            daily_df["resultado"] = daily_df["result_financial"].apply(lambda value: "Lucro" if value >= 0 else "Prejuízo")
            fig_daily = px.bar(
                daily_df, x="trade_date", y="result_financial", color="resultado",
                color_discrete_map={"Lucro": "#22d6a0", "Prejuízo": "#fa5c78"},
                labels={"trade_date": "Data", "result_financial": "Resultado diário (R$)"},
            )
            fig_daily.update_layout(showlegend=False, bargap=.25)
            with daily_col:
                st.markdown('<div class="chart-card"><h3>Resultado diário</h3>', unsafe_allow_html=True)
                st.plotly_chart(dark_chart(fig_daily), width='stretch', config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-title">MÉTRICAS COMPLEMENTARES</div>', unsafe_allow_html=True)
            row_two = st.columns(4)
            render_metric(row_two[0], "Média vencedora", format_currency(metrics["average_win"]), tone="gain")
            render_metric(row_two[1], "Média perdedora", format_currency(metrics["average_loss"]), tone="loss" if metrics["average_loss"] < 0 else "neutral")
            render_metric(row_two[2], "Risco × retorno", f"{metrics['risk_reward']:.2f}:1" if metrics["risk_reward"] is not None else "N/A", tone="neutral")
            render_metric(row_two[3], "Profit factor", f"{metrics['profit_factor']:.2f}" if metrics["profit_factor"] is not None else "N/A", tone="neutral")

            st.markdown('<div class="section-title">RESULTADO LÍQUIDO ESTIMADO</div>', unsafe_allow_html=True)
            st.caption("Taxa de corretagem: R$ 0,18 por contrato, cobrada na entrada e na saída. Imposto: 1% de IRRF sobre o resultado do dia, somente quando positivo (regra oficial de day trade).")
            net_summary = calculate_net_summary(filtered_df)
            row_three = st.columns(4)
            render_metric(row_three[0], "Resultado bruto", format_currency(net_summary["gross_result"]), numeric_value=net_summary["gross_result"])
            render_metric(row_three[1], "Taxas de corretagem", format_currency(-net_summary["total_fees"]) if net_summary["total_fees"] else "R$ 0,00", tone="loss" if net_summary["total_fees"] else "neutral")
            render_metric(row_three[2], "Imposto estimado (IRRF)", format_currency(-net_summary["estimated_tax"]) if net_summary["estimated_tax"] else "R$ 0,00", tone="loss" if net_summary["estimated_tax"] else "neutral")
            render_metric(row_three[3], "Resultado líquido", format_currency(net_summary["net_result"]), numeric_value=net_summary["net_result"])

        # ------------------------------------------------------------
        # Tab 2 - Análise avançada
        # ------------------------------------------------------------
        with tab_advanced:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<h3>Eficiência das operações</h3>', unsafe_allow_html=True)
            donut_col, stats_col = st.columns([1.4, 1], gap="medium")

            efficiency = calculate_efficiency_breakdown(filtered_df)
            total_ops = efficiency["winners"] + efficiency["losers"] + efficiency["breakeven"]

            with donut_col:
                pie_df = pd.DataFrame([
                    {"resultado": "Vencedoras", "quantidade": efficiency["winners"]},
                    {"resultado": "Perdedoras", "quantidade": efficiency["losers"]},
                    {"resultado": "Zeradas", "quantidade": efficiency["breakeven"]},
                ])
                fig_pie = px.pie(
                    pie_df, names="resultado", values="quantidade", hole=0.62,
                    color="resultado",
                    color_discrete_map={"Vencedoras": "#22d6a0", "Perdedoras": "#fa5c78", "Zeradas": "#5b6f74"},
                )
                fig_pie.update_traces(
                    textinfo="percent", textfont_size=13, marker=dict(line=dict(color="#0d1318", width=2)),
                )
                fig_pie.add_annotation(
                    text=f"{total_ops}<br><span style='font-size:11px;color:#8d9a9e'>operações</span>",
                    showarrow=False, font=dict(size=22, color="#eef6f4", family="Consolas, monospace"),
                )
                st.plotly_chart(dark_chart(fig_pie), width='stretch', config={"displayModeBar": False})

            with stats_col:
                # A compact, custom stat list next to the donut instead of
                # a second, half-empty chart card - every number here maps
                # directly to a slice of the donut on the left.
                st.markdown('<div class="eff-stats">', unsafe_allow_html=True)
                stat_rows = [
                    ("eff-dot-gain", "Vencedoras", efficiency["winners"]),
                    ("eff-dot-loss", "Perdedoras", efficiency["losers"]),
                    ("eff-dot-neutral", "Zeradas", efficiency["breakeven"]),
                ]
                for dot_class, label, count in stat_rows:
                    share = f"{(count / total_ops * 100):.0f}%" if total_ops else "0%"
                    st.markdown(
                        f"""
                        <div class="eff-row">
                          <span class="eff-dot {dot_class}"></span>
                          <span class="eff-label">{label}</span>
                          <span class="eff-count">{count}</span>
                          <span class="eff-share">{share}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-title">DESEMPENHO POR MOMENTO DO DIA</div>', unsafe_allow_html=True)

            by_hour = calculate_performance_by_hour(filtered_df)
            st.markdown('<div class="chart-card"><h3>Performance por horário de entrada</h3>', unsafe_allow_html=True)
            if by_hour.empty:
                empty_state("🕐", "Sem horários registrados", "Informe o horário de entrada ao registrar operações para ver esta análise.")
            else:
                st.plotly_chart(dark_chart(pnl_bar_chart(by_hour, "hour", "Hora do dia")), width='stretch', config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

            by_weekday = calculate_performance_by_weekday(filtered_df)
            st.markdown('<div class="chart-card"><h3>Performance por dia da semana</h3>', unsafe_allow_html=True)
            st.plotly_chart(dark_chart(pnl_bar_chart(by_weekday, "weekday", "Dia da semana")), width='stretch', config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="section-title">ZONAS DE EFICIÊNCIA (MFE × MAE)</div>', unsafe_allow_html=True)
            mfe_col, mae_col = st.columns(2)

            with mfe_col:
                mfe_efficiency = calculate_mfe_efficiency(filtered_df)
                st.markdown('<div class="chart-card"><h3>A favor (MFE)</h3>', unsafe_allow_html=True)
                st.caption("Quantos % das operações chegaram a favor em cada nível de pontos.")
                if mfe_efficiency.empty:
                    empty_state("📈", "Sem MFE registrado", "Preencha o campo de MFE (favorável máximo) para ver esta análise.")
                else:
                    st.plotly_chart(dark_chart(threshold_bar_chart(mfe_efficiency, "#22d6a0", "Pontos a favor (MFE)")), width='stretch', config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)

            with mae_col:
                mae_efficiency = calculate_mae_efficiency(filtered_df)
                st.markdown('<div class="chart-card"><h3>Contra (MAE)</h3>', unsafe_allow_html=True)
                st.caption("Quantos % das operações andaram contra em cada nível de pontos antes de resolver.")
                if mae_efficiency.empty:
                    empty_state("📉", "Sem MAE registrado", "Preencha o campo de MAE (adverso máximo) para ver esta análise.")
                else:
                    st.plotly_chart(dark_chart(threshold_bar_chart(mae_efficiency, "#fa5c78", "Pontos contra (MAE)")), width='stretch', config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)

        # ------------------------------------------------------------
        # Tab 3 - Histórico
        # ------------------------------------------------------------
        with tab_history:
            view_mode = st.radio("Modo de visualização", ["Calendário", "Tabela"], horizontal=True, label_visibility="collapsed")

            if view_mode == "Calendário":
                # Build the list of months that actually have trades, so the
                # selector never lets the user pick an empty month.
                month_periods = sorted(filtered_df["trade_date"].dt.to_period("M").unique())
                month_labels = [period.strftime("%B %Y").capitalize() for period in month_periods]

                if "calendar_month_index" not in st.session_state or st.session_state.calendar_month_index >= len(month_labels):
                    st.session_state.calendar_month_index = len(month_labels) - 1

                nav_prev, nav_label, nav_next = st.columns([1, 4, 1])
                if nav_prev.button("‹", width='stretch', disabled=st.session_state.calendar_month_index == 0):
                    st.session_state.calendar_month_index -= 1
                    st.rerun()
                nav_label.markdown(
                    f'<div style="text-align:center; padding-top:.4rem; font-weight:700; color:#dce6e5;">'
                    f'{month_labels[st.session_state.calendar_month_index]}</div>',
                    unsafe_allow_html=True,
                )
                if nav_next.button("›", width='stretch', disabled=st.session_state.calendar_month_index == len(month_labels) - 1):
                    st.session_state.calendar_month_index += 1
                    st.rerun()

                selected_period = month_periods[st.session_state.calendar_month_index]

                daily_results = calculate_daily_results(filtered_df)
                daily_lookup = {
                    row["trade_date"].date(): (row["result_financial"], row["trade_count"])
                    for _, row in daily_results.iterrows()
                }

                calendar_html = build_calendar_html(selected_period.year, selected_period.month, daily_lookup)
                st.markdown(calendar_html, unsafe_allow_html=True)
            else:
                st.dataframe(filtered_df, width='stretch', hide_index=True)

        # ------------------------------------------------------------
        # Tab 4 - Gerenciar (edit/delete)
        # ------------------------------------------------------------
        with tab_manage:
            st.markdown('<div class="section-title">EDITAR OU EXCLUIR OPERAÇÃO</div>', unsafe_allow_html=True)
            st.markdown('<p class="manage-hint">Escolha uma operação abaixo para revisar os dados, corrigir algo ou removê-la do diário.</p>', unsafe_allow_html=True)

            # Build a human-readable label for each trade in the current view,
            # so the person picks "which trade" without needing to know its id.
            editable_df = working_df.sort_values("trade_date", ascending=False)
            trade_options = {
                int(row["id"]): (
                    f"#{int(row['id'])} · {row['trade_date'].strftime('%d/%m/%Y')} · "
                    f"{row['setup'] or '—'} · {format_currency(row['result_financial'])}"
                )
                for _, row in editable_df.iterrows()
            }

            if not trade_options:
                empty_state("🗂️", "Nenhuma operação disponível", "Registre uma operação para poder editá-la ou excluí-la aqui.")
            else:
                selected_id = st.selectbox(
                    "Selecionar operação",
                    options=list(trade_options.keys()),
                    format_func=lambda trade_id: trade_options[trade_id],
                )
                trade_to_edit = get_trade(connection, selected_id)

                with st.form("edit_trade_form"):
                    date_col, entry_time_col, exit_time_col = st.columns(3)
                    with date_col:
                        edit_date = st.date_input("Data", value=pd.to_datetime(trade_to_edit["trade_date"]).date())
                    with entry_time_col:
                        existing_entry_time = trade_to_edit.get("entry_time")
                        edit_entry_time = st.time_input(
                            "Horário de entrada",
                            value=pd.to_datetime(existing_entry_time).time() if existing_entry_time else None,
                        )
                    with exit_time_col:
                        existing_exit_time = trade_to_edit.get("exit_time")
                        edit_exit_time = st.time_input(
                            "Horário de saída",
                            value=pd.to_datetime(existing_exit_time).time() if existing_exit_time else None,
                        )

                    direction_col, setup_col = st.columns(2)
                    with direction_col:
                        edit_direction = st.selectbox(
                            "Direção", ["buy", "sell"],
                            index=["buy", "sell"].index(trade_to_edit["direction"]),
                            format_func=lambda item: DIRECTION_LABELS[item],
                        )
                    with setup_col:
                        edit_setup = st.selectbox(
                            "Setup", SETUP_OPTIONS,
                            index=SETUP_OPTIONS.index(trade_to_edit["setup"]) if trade_to_edit["setup"] in SETUP_OPTIONS else 0,
                        )

                    entry_col, exit_col, contracts_col = st.columns(3)
                    with entry_col:
                        edit_entry_price = st.number_input("Entrada (pts)", min_value=0.0, step=5.0, value=float(trade_to_edit["entry_price"]), format="%.0f")
                    with exit_col:
                        edit_exit_price = st.number_input("Saída (pts)", min_value=0.0, step=5.0, value=float(trade_to_edit["exit_price"]), format="%.0f")
                    with contracts_col:
                        edit_contracts = st.number_input("Contratos", min_value=1, step=1, value=int(trade_to_edit["contracts"]))

                    edit_stop_points = st.number_input("Stop (pontos)", min_value=0.0, step=10.0, value=float(trade_to_edit["stop_points"] or 0.0), format="%.0f")

                    existing_states = [
                        state.strip() for state in (trade_to_edit.get("emotional_state") or "").split(",") if state.strip()
                    ]
                    edit_emotional_state = st.multiselect(
                        "Estado emocional", EMOTIONAL_STATES, default=existing_states,
                        help=f"Selecione pelo menos {MIN_EMOTIONAL_STATES} estados que descrevem como você estava durante a operação.",
                    )
                    edit_technical_notes = st.text_area(
                        "O que você fez e por quê?",
                        value=trade_to_edit["technical_notes"] or "",
                        placeholder="O que viu no gráfico? Por que entrou? O que faria diferente?",
                    )

                    save_edit = st.form_submit_button("Salvar alterações", width='stretch')

                    if save_edit:
                        if len(edit_emotional_state) < MIN_EMOTIONAL_STATES:
                            st.error(f"Selecione pelo menos {MIN_EMOTIONAL_STATES} estados emocionais.")
                        else:
                            try:
                                update_trade(
                                    connection, selected_id,
                                    trade_date=str(edit_date), direction=edit_direction,
                                    entry_time=edit_entry_time.strftime("%H:%M:%S") if edit_entry_time else None,
                                    exit_time=edit_exit_time.strftime("%H:%M:%S") if edit_exit_time else None,
                                    entry_price=edit_entry_price, exit_price=edit_exit_price,
                                    contracts=edit_contracts, setup=edit_setup, stop_points=edit_stop_points,
                                    emotional_state=", ".join(edit_emotional_state),
                                    technical_notes=edit_technical_notes or None,
                                )
                                st.session_state["flash_success"] = "Operação atualizada com sucesso."
                                st.rerun()
                            except ValueError as error:
                                st.error(f"Não foi possível salvar: {error}")

                if st.button("🗑️ Excluir esta operação", type="secondary"):
                    confirm_delete_dialog(connection, selected_id, trade_options[selected_id])
