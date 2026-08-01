"""
database.py

Single responsibility: create the database table and provide a
connection to it. No business logic lives here.

Two backends are supported:
  - SQLite (default): a local trades.db file. Used for local development
    and by every test in test_project.py. This path is untouched from
    the original implementation.
  - Postgres (via a DATABASE_URL, typically a free Neon database): used
    in production, when the app runs on Streamlit Community Cloud,
    whose local filesystem does not persist between restarts. A trade
    logged there would otherwise be lost the next time the app sleeps
    and wakes back up.

get_connection() is the single entry point the app should call - it
picks the right backend automatically depending on whether a
DATABASE_URL is configured, so nothing else in the codebase needs to
know or care which database is actually in use.
"""

import os
import sqlite3

DB_NAME = "trades.db"


def connect(db_name=DB_NAME):
    """
    Open a connection to the SQLite database.
    If the file does not exist yet, SQLite creates it automatically.
    """
    # check_same_thread=False is needed because Streamlit can handle
    # user interactions (like a button click) in a different thread
    # than the one that first opened this connection. This is safe in
    # our case: Streamlit serves one user per session, not many users
    # writing to the same connection at the same time.
    connection = sqlite3.connect(db_name, check_same_thread=False)
    # Makes query results behave like dictionaries (e.g. row["direction"])
    # instead of positional tuples (e.g. row[3]) - much easier to read.
    connection.row_factory = sqlite3.Row
    return connection


def create_table(connection):
    """
    Create the 'trades' table if it does not exist yet.
    'IF NOT EXISTS' avoids errors if this function is called more than
    once (for example, every time the app starts).
    """
    cursor = connection.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT NOT NULL,
            entry_time TEXT,
            exit_time TEXT,
            direction TEXT NOT NULL,
            setup TEXT,
            contracts INTEGER NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,
            stop_points REAL,
            mae_points REAL,
            mfe_points REAL,
            result_points REAL NOT NULL,
            result_financial REAL NOT NULL,
            emotional_state TEXT,
            technical_notes TEXT,
            screenshot_path TEXT,
            is_demo INTEGER NOT NULL DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    connection.commit()
    _ensure_is_demo_column(connection)


def _ensure_is_demo_column(connection):
    """
    Migration step: if the table already existed from an earlier version
    of this project (before is_demo existed), add the column now instead
    of crashing. This keeps a database created with an older version of
    database.py working with the current code, without losing any data.
    """
    cursor = connection.cursor()
    existing_columns = [row["name"] for row in cursor.execute("PRAGMA table_info(trades)")]
    if "is_demo" not in existing_columns:
        cursor.execute("ALTER TABLE trades ADD COLUMN is_demo INTEGER NOT NULL DEFAULT 0")
        connection.commit()


def initialize_database(db_name=DB_NAME):
    """
    Convenience function: connects and ensures the table exists.
    This is the function the rest of the project should call.
    """
    connection = connect(db_name)
    create_table(connection)
    return connection


# ----------------------------------------------------------------------
# Postgres backend (production / Streamlit Community Cloud)
# ----------------------------------------------------------------------

class PostgresCursor:
    """
    Thin adapter around a psycopg2 cursor so the SQL already written in
    project.py - using SQLite's '?' placeholders - works unchanged
    against Postgres. It also emulates sqlite3's cursor.lastrowid for
    inserts (Postgres has no such attribute; it uses "RETURNING id"
    instead), so save_trade() doesn't need a database-specific branch.
    """

    def __init__(self, real_cursor):
        self._cursor = real_cursor
        self.lastrowid = None

    def execute(self, query, params=()):
        translated = query.replace("?", "%s").strip()
        is_insert = translated.rstrip(";").upper().startswith("INSERT INTO TRADES")
        if is_insert:
            translated = translated.rstrip(";") + " RETURNING id"
        self._cursor.execute(translated, params)
        if is_insert:
            row = self._cursor.fetchone()
            self.lastrowid = row["id"] if row else None
        return self

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    @property
    def rowcount(self):
        return self._cursor.rowcount

    @property
    def description(self):
        return self._cursor.description


class PostgresConnection:
    """
    Adapter exposing the small slice of the sqlite3.Connection API that
    project.py and app.py actually use (.cursor(), .commit()), backed by
    a real psycopg2 connection to Postgres.
    """

    # Lets code that must behave differently per backend (load_trades,
    # in project.py) check this instead of importing database.py just
    # to do an isinstance() check - keeps project.py database-agnostic,
    # as its own module docstring intends.
    dialect = "postgres"

    def __init__(self, real_connection):
        self._connection = real_connection

    def cursor(self):
        import psycopg2.extras
        return PostgresCursor(self._connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor))

    def commit(self):
        self._connection.commit()

    def rollback(self):
        self._connection.rollback()

    def close(self):
        self._connection.close()


def _normalize_postgres_url(connection_string):
    """
    Neon (and some other providers) hand out URLs starting with
    'postgres://', which psycopg2 accepts, but making sure sslmode is
    present avoids a class of confusing connection failures on
    providers that require TLS but don't encode it in the URL.
    """
    if connection_string.startswith("postgres://"):
        connection_string = connection_string.replace("postgres://", "postgresql://", 1)
    if "sslmode=" not in connection_string:
        separator = "&" if "?" in connection_string else "?"
        connection_string = f"{connection_string}{separator}sslmode=require"
    return connection_string


def connect_postgres(connection_string):
    """Open a connection to a Postgres database (e.g. a free Neon database)."""
    import psycopg2
    real_connection = psycopg2.connect(_normalize_postgres_url(connection_string), connect_timeout=10)
    return PostgresConnection(real_connection)


def create_table_postgres(connection):
    """
    Postgres equivalent of create_table() - same shape, Postgres-flavored DDL.

    "CREATE TABLE IF NOT EXISTS" does not fully protect against two
    sessions creating the same table (and its implicit id sequence) at
    almost the same moment - which happens in practice whenever two
    people open the app within the same second or two. When that race
    is what caused the failure, the table exists either way by the time
    it's caught, so it's safe to treat as success; anything else still
    surfaces normally.
    """
    cursor = connection.cursor()
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                trade_date TEXT NOT NULL,
                entry_time TEXT,
                exit_time TEXT,
                direction TEXT NOT NULL,
                setup TEXT,
                contracts INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                stop_points REAL,
                mae_points REAL,
                mfe_points REAL,
                result_points REAL NOT NULL,
                result_financial REAL NOT NULL,
                emotional_state TEXT,
                technical_notes TEXT,
                screenshot_path TEXT,
                is_demo INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.commit()
    except Exception as error:
        connection.rollback()
        if "already exists" not in str(error).lower():
            raise
    _ensure_is_demo_column_postgres(connection)


def _ensure_is_demo_column_postgres(connection):
    """Postgres equivalent of _ensure_is_demo_column(), using information_schema instead of PRAGMA."""
    cursor = connection.cursor()
    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'trades'")
    existing_columns = [row["column_name"] for row in cursor.fetchall()]
    if "is_demo" not in existing_columns:
        try:
            cursor.execute("ALTER TABLE trades ADD COLUMN is_demo INTEGER NOT NULL DEFAULT 0")
            connection.commit()
        except Exception as error:
            connection.rollback()
            if "already exists" not in str(error).lower():
                raise


def initialize_database_postgres(connection_string):
    """Postgres equivalent of initialize_database()."""
    connection = connect_postgres(connection_string)
    create_table_postgres(connection)
    return connection


def _resolve_database_url():
    """
    Look for a configured Postgres connection string, checking Streamlit
    secrets first (how it's configured on Streamlit Community Cloud),
    then falling back to a plain environment variable (useful for local
    testing against a real Postgres database without touching secrets).
    Returns None if neither is set, which means "use local SQLite".
    """
    try:
        import streamlit as st
        secret_value = st.secrets.get("DATABASE_URL")
        if secret_value:
            return secret_value
    except Exception:
        pass
    return os.environ.get("DATABASE_URL")


def get_connection():
    """
    The single entry point the app should call. Picks Postgres when a
    DATABASE_URL is configured (production), or local SQLite otherwise
    (local development) - callers don't need to know which one they got.
    """
    database_url = _resolve_database_url()
    if database_url:
        return initialize_database_postgres(database_url)
    return initialize_database()


def ensure_connection(connection):
    """
    Verify a stored connection is still alive, reconnecting if not.
    Free-tier Postgres databases (like Neon) can suspend their compute
    after a few minutes of inactivity, which drops any connection that
    was already open - without this check, the first action after such
    a pause would fail with a raw database error instead of quietly
    reconnecting.
    """
    try:
        connection.cursor().execute("SELECT 1")
        return connection
    except Exception:
        return get_connection()
