# Mini Índice — Trading Performance Dashboard

A trade journal and performance dashboard for mini índice (WIN) day
trading, built with Streamlit. Logs trades, calculates net result
(already accounting for brokerage fees and Brazilian day-trade income
tax), and shows equity evolution, efficiency, and behavioral patterns
over time.

## Stack

- **Frontend/app**: [Streamlit](https://streamlit.io)
- **Database**: PostgreSQL ([Neon](https://neon.tech), free tier) in
  production; local SQLite for development — switched automatically,
  with no manual configuration needed (see `database.py`)
- **Charts**: Plotly
- **Data**: pandas
- **Tests**: pytest (`test_project.py`, covers all business logic in
  `project.py`)

## Structure

```
app.py            # UI (Streamlit) - presentation layer
project.py        # Pure business logic (calculations, no database/UI)
database.py       # Database connection (local SQLite / production Postgres)
test_project.py   # Automated tests for the business logic
requirements.txt  # Python dependencies
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

With no extra configuration, this uses a local SQLite file (`trades.db`),
created automatically on first run.

## Running the tests

```bash
pytest test_project.py
```

## Production deploy

The app is published on [Streamlit Community Cloud](https://streamlit.io/cloud),
connected to a free Postgres database on [Neon](https://neon.tech) for
data persistence (Streamlit Community Cloud does not keep local files
between restarts - hence the external Postgres database).

The production database connection is configured through a
`DATABASE_URL` environment variable, set in the app's **Secrets** on
Streamlit Cloud (never in the source code). Without that variable set,
the app falls back to local SQLite automatically - useful for
development.

## Security

- No database credentials live in the source code or in this
  repository.
- `.db` files and `.streamlit/secrets.toml` are ignored by Git (see
  `.gitignore`).
