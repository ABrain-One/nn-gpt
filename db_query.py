import sqlite3
import pandas as pd
import argparse
import os

from ab.nn.util.Const import ab_root_path

DB_PATH = ab_root_path / "db/ab.nn.db"

def main():
    parser = argparse.ArgumentParser(description="Query and explore the SQLite database.")
    parser.add_argument("--list", action="store_true", help="List all tables and their row counts.")
    parser.add_argument("--query", type=str, help="Run a specific SQL query.")
    parser.add_argument("--export", type=str, help="Export query result to a CSV file.")
    
    args = parser.parse_args()

    print(f"Connecting to database: {DB_PATH}")
    if not os.path.exists(DB_PATH):
        print(f"Error: Database file NOT FOUND at {DB_PATH}")
        return

    try:
        # Using a very short timeout to detect locks/hangs immediately
        print("Opening connection (2s timeout)...")
        conn = sqlite3.connect(DB_PATH, timeout=2)
        print("Connection object created. Checking for life...")
        conn.execute("SELECT 1") 
        print("Database is responsive.")
    except Exception as e:
        print(f"\nERROR: Could not access database: {e}")
        print("This usually means the file is locked by another process or the filesystem is extremely slow.")
        return
    
    try:
        if args.list:
            print("Fetching table names...")
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            if not tables:
                print("No tables found in the database. It might be empty or corrupted.")
            else:
                print(f"{'Table Name':<20} | {'Row Count':<10}")
                print("-" * 35)
                for (table_name,) in tables:
                    try:
                        cursor.execute(f"SELECT count(*) FROM \"{table_name}\"")
                        count = cursor.fetchone()[0]
                        print(f"{table_name:<20} | {count:<10}")
                    except Exception as te:
                        print(f"Error reading table {table_name}: {te}")
        
        elif args.query:
            df = pd.read_sql_query(args.query, conn)
            print(df)
            
            if args.export:
                df.to_csv(args.export, index=False)
                print(f"\nResults exported to {args.export}")
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error executing command: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
