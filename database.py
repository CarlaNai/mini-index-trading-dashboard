"""
database.py

Single responsibility: create the database table and provide a
connection to it. No business logic lives here.
"""

import sqlite3

DB_NAME = "trades.db"


def connect(db_name=DB_NAME):
    """
    Open a connection to the SQLite database.
    If the file does not exist yet, SQLite creates it automatically.
    """
    connection = sqlite3.connect(db_name)
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
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    connection.commit()


def initialize_database(db_name=DB_NAME):
    """
    Convenience function: connects and ensures the table exists.
    This is the function the rest of the project should call.
    """
    connection = connect(db_name)
    create_table(connection)
    return connection
