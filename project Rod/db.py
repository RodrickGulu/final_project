import sqlite3
from flask import g, current_app

# Connect to SQLite database
DATABASE = 'users.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

def close_db():
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with current_app.app_context():
        db = get_db()
        with current_app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()

def is_database_empty():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    count = cur.fetchone()[0]
    return count == 0

# Mock user database for demonstration purposes
def authenticate(username, password):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = cur.fetchone()
    return user

def add_user(full_name, username, password):
    db = get_db()
    cur = db.cursor()
    try:
        cur.execute("INSERT INTO users (full_names, username, password) VALUES (?, ?, ?)", (full_name, username, password))
        db.commit()
        print("User added successfully:", username)
    except Exception as e:
        print("Error adding user:", e)
