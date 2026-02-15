"""Database interface for accessing training results"""
import sqlite3
import pandas as pd
import os
from typing import Dict, Optional


class DatabaseInterface:
    """Interface for accessing the training database"""
    
    def __init__(self, db_path: str = 'db/ab.nn.db'):
        self.db_path = os.path.abspath(db_path)
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Ensure database file exists"""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database not found at {self.db_path}")
    
    def connect(self):
        """Create database connection"""
        return sqlite3.connect(self.db_path)
    
    def get_all_results(self, order_by: str = 'best_accuracy DESC') -> pd.DataFrame:
        """Get all scheduler results"""
        conn = self.connect()
        query = f"""
            SELECT id, model_name, scheduler_type, epoch, learning_rate, 
                   best_accuracy, final_accuracy, training_time, created_at
            FROM scheduler_results
            ORDER BY {order_by}
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_results_by_model(self, model_name: str) -> pd.DataFrame:
        """Get results for a specific model"""
        conn = self.connect()
        query = """
            SELECT id, model_name, scheduler_type, epoch, learning_rate,
                   best_accuracy, final_accuracy, training_time, created_at
            FROM scheduler_results
            WHERE model_name = ?
            ORDER BY best_accuracy DESC
        """
        df = pd.read_sql_query(query, conn, params=(model_name,))
        conn.close()
        return df
    
    def get_results_by_scheduler(self, scheduler_type: str) -> pd.DataFrame:
        """Get results for a specific scheduler type"""
        conn = self.connect()
        query = """
            SELECT id, model_name, scheduler_type, epoch, learning_rate,
                   best_accuracy, final_accuracy, training_time, created_at
            FROM scheduler_results
            WHERE scheduler_type = ?
            ORDER BY best_accuracy DESC
        """
        df = pd.read_sql_query(query, conn, params=(scheduler_type,))
        conn.close()
        return df
    
    def get_best_result(self) -> Dict:
        """Get the best performing result"""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, model_name, scheduler_type, epoch, learning_rate,
                   best_accuracy, final_accuracy, training_time, created_at
            FROM scheduler_results
            ORDER BY best_accuracy DESC
            LIMIT 1
        """)
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'model': result[1],
                'scheduler': result[2],
                'epochs': result[3],
                'learning_rate': result[4],
                'best_accuracy': result[5],
                'final_accuracy': result[6],
                'training_time_min': result[7],
                'created_at': result[8]
            }
        return None
    
    def add_result(self, model_name: str, scheduler_type: str, epoch: int,
                   learning_rate: float, best_accuracy: float,
                   final_accuracy: float, training_time: float):
        """Add a new training result"""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO scheduler_results
            (model_name, scheduler_type, epoch, learning_rate, best_accuracy, final_accuracy, training_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (model_name, scheduler_type, epoch, learning_rate, best_accuracy, final_accuracy, training_time))
        conn.commit()
        conn.close()
    
    def get_class_data(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """Get class-level accuracy data"""
        conn = self.connect()
        if model_name:
            query = """
                SELECT id, model_name, class_name, accuracy, precision, recall, f1_score, epoch, created_at
                FROM class_data
                WHERE model_name = ?
                ORDER BY accuracy DESC
            """
            df = pd.read_sql_query(query, conn, params=(model_name,))
        else:
            query = """
                SELECT id, model_name, class_name, accuracy, precision, recall, f1_score, epoch, created_at
                FROM class_data
                ORDER BY model_name, accuracy DESC
            """
            df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def add_class_data(self, model_name: str, class_name: str,
                       accuracy: Optional[float] = None,
                       precision: Optional[float] = None,
                       recall: Optional[float] = None,
                       f1_score: Optional[float] = None,
                       epoch: Optional[int] = None):
        """Add class-level accuracy data"""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO class_data
            (model_name, class_name, accuracy, precision, recall, f1_score, epoch)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (model_name, class_name, accuracy, precision, recall, f1_score, epoch))
        conn.commit()
        conn.close()
    
    def get_statistics(self) -> Dict:
        """Get overall statistics"""
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM scheduler_results")
        total_runs = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(best_accuracy) FROM scheduler_results")
        avg_accuracy = cursor.fetchone()[0]
        
        cursor.execute("SELECT MAX(best_accuracy) FROM scheduler_results")
        max_accuracy = cursor.fetchone()[0]
        
        cursor.execute("SELECT MIN(best_accuracy) FROM scheduler_results")
        min_accuracy = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(training_time) FROM scheduler_results")
        total_training_time = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT model_name) FROM scheduler_results")
        unique_models = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT scheduler_type) FROM scheduler_results")
        unique_schedulers = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_runs': total_runs,
            'unique_models': unique_models,
            'unique_schedulers': unique_schedulers,
            'avg_accuracy': round(avg_accuracy, 2) if avg_accuracy else None,
            'max_accuracy': max_accuracy,
            'min_accuracy': min_accuracy,
            'total_training_time_min': round(total_training_time, 2) if total_training_time else None
        }
    
    def print_summary(self):
        """Print database summary"""
        print("=" * 70)
        print("DATABASE SUMMARY")
        print("=" * 70)
        
        results = self.get_all_results()
        print(f"\nAll Results ({len(results)} rows):")
        print(results.to_string(index=False))
        
        stats = self.get_statistics()
        print(f"\n" + "=" * 70)
        print("STATISTICS")
        print("=" * 70)
        for key, value in stats.items():
            print(f"{key:.<40} {value}")
        
        best = self.get_best_result()
        if best:
            print(f"\n" + "=" * 70)
            print("BEST RESULT")
            print("=" * 70)
            for key, value in best.items():
                print(f"{key:.<40} {value}")
    
    def export_to_csv(self, output_file: str = 'training_results.csv'):
        """Export results to CSV"""
        df = self.get_all_results()
        df.to_csv(output_file, index=False)
        print(f"✓ Exported {len(df)} rows to {output_file}")


def main():
    """Demo usage"""
    db = DatabaseInterface()
    db.print_summary()


if __name__ == '__main__':
    main()
