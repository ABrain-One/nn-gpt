# SQLite Database Access Guide

Your database `/home/tehreem/nn-gpt/db/ab.nn.db` is ~1.2 GB, which exceeds the limit of some free VS Code extensions. You can access it using the standard `sqlite3` command-line tool or the provided Python script.

## 1. Using the Command Line (`sqlite3`)

You can enter the interactive shell by running:
```bash
sqlite3 /home/tehreem/nn-gpt/db/ab.nn.db
```

### Useful Commands:
- `.tables` - List all tables.
- `.schema [table_name]` - Show the schema for a specific table.
- `SELECT * FROM nn LIMIT 10;` - Query data (always use `LIMIT` to avoid long outputs).
- `.quit` - Exit the shell.

## 2. Using the Python Query Tool (`db_query.py`)

I have created a script [db_query.py](file:///home/tehreem/nn-gpt/db_query.py) for easy querying and exporting.

### Examples:
- **List tables and counts:**
  ```bash
  python3 db_query.py --list
  ```
- **Run a custom query:**
  ```bash
  python3 db_query.py --query "SELECT name, framework FROM nn LIMIT 5"
  ```
- **Export to CSV:**
  ```bash
  python3 db_query.py --query "SELECT * FROM prun LIMIT 100" --export results.csv
  ```
