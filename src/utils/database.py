"""
Database manager for storing conversations and Q&A history
Uses SQLite for simplicity (no setup required)
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import json

class ConversationDB:
    """Manage conversation history in SQLite database"""
    
    def __init__(self, db_path: str = "data/conversations.db"):
        """Initialize database connection"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.init_database()
    
    def init_database(self):
        """Create tables if they don't exist"""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                model_used TEXT NOT NULL,
                sources TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_feedback INTEGER,
                response_time REAL
            )
        """)
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_questions INTEGER DEFAULT 0,
                model_preference TEXT
            )
        """)
        
        # Popular queries tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_analytics (
                query_hash TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                count INTEGER DEFAULT 1,
                last_asked DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
        print("✅ Database initialized")
    
    def save_conversation(
        self,
        session_id: str,
        question: str,
        answer: str,
        model_used: str,
        sources: List[Dict] = None,
        response_time: float = None
    ) -> int:
        """
        Save a Q&A conversation
        
        Args:
            session_id: Unique session identifier
            question: User's question
            answer: Model's answer
            model_used: Which model generated the answer
            sources: List of source documents used
            response_time: Time taken to generate answer
            
        Returns:
            Conversation ID
        """
        cursor = self.conn.cursor()
        
        # Convert sources to JSON
        sources_json = json.dumps(sources) if sources else None
        
        # Insert conversation
        cursor.execute("""
            INSERT INTO conversations 
            (session_id, question, answer, model_used, sources, response_time)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, question, answer, model_used, sources_json, response_time))
        
        conversation_id = cursor.lastrowid
        
        # Update session
        cursor.execute("""
            INSERT INTO sessions (session_id, total_questions, model_preference)
            VALUES (?, 1, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                last_activity = CURRENT_TIMESTAMP,
                total_questions = total_questions + 1,
                model_preference = ?
        """, (session_id, model_used, model_used))
        
        # Update analytics
        query_hash = str(hash(question.lower().strip()))
        cursor.execute("""
            INSERT INTO query_analytics (query_hash, query_text, count)
            VALUES (?, ?, 1)
            ON CONFLICT(query_hash) DO UPDATE SET
                count = count + 1,
                last_asked = CURRENT_TIMESTAMP
        """, (query_hash, question))
        
        self.conn.commit()
        return conversation_id
    
    def get_session_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get conversation history for a session"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT id, question, answer, model_used, sources, timestamp, user_feedback
            FROM conversations
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit))
        
        results = []
        for row in cursor.fetchall():
            sources = json.loads(row[4]) if row[4] else []
            results.append({
                'id': row[0],
                'question': row[1],
                'answer': row[2],
                'model_used': row[3],
                'sources': sources,
                'timestamp': row[5],
                'feedback': row[6]
            })
        
        return results
    
    def search_conversations(
        self,
        search_term: str,
        limit: int = 20
    ) -> List[Dict]:
        """
        Search conversations by keyword
        
        Args:
            search_term: Term to search for
            limit: Maximum results
            
        Returns:
            List of matching conversations
        """
        cursor = self.conn.cursor()
        
        search_pattern = f"%{search_term}%"
        cursor.execute("""
            SELECT id, question, answer, model_used, timestamp
            FROM conversations
            WHERE question LIKE ? OR answer LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (search_pattern, search_pattern, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'question': row[1],
                'answer': row[2],
                'model_used': row[3],
                'timestamp': row[4]
            })
        
        return results
    
    def get_popular_queries(self, limit: int = 10) -> List[Dict]:
        """Get most frequently asked questions"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT query_text, count, last_asked
            FROM query_analytics
            ORDER BY count DESC
            LIMIT ?
        """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'query': row[0],
                'count': row[1],
                'last_asked': row[2]
            })
        
        return results
    
    def get_analytics(self) -> Dict:
        """Get overall analytics"""
        cursor = self.conn.cursor()
        
        # Total conversations
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_conversations = cursor.fetchone()[0]
        
        # Total sessions
        cursor.execute("SELECT COUNT(*) FROM sessions")
        total_sessions = cursor.fetchone()[0]
        
        # Model usage
        cursor.execute("""
            SELECT model_used, COUNT(*) as count
            FROM conversations
            GROUP BY model_used
        """)
        model_usage = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Average response time
        cursor.execute("""
            SELECT AVG(response_time)
            FROM conversations
            WHERE response_time IS NOT NULL
        """)
        avg_response_time = cursor.fetchone()[0]
        
        # Conversations per day (last 7 days)
        cursor.execute("""
            SELECT DATE(timestamp) as date, COUNT(*) as count
            FROM conversations
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        """)
        daily_activity = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        return {
            'total_conversations': total_conversations,
            'total_sessions': total_sessions,
            'model_usage': model_usage,
            'avg_response_time': avg_response_time,
            'daily_activity': daily_activity
        }
    
    def add_feedback(self, conversation_id: int, feedback: int):
        """
        Add user feedback to a conversation
        
        Args:
            conversation_id: ID of conversation
            feedback: 1 for positive, -1 for negative
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE conversations
            SET user_feedback = ?
            WHERE id = ?
        """, (feedback, conversation_id))
        self.conn.commit()
    
    def export_history(self, session_id: str = None, filename: str = None) -> str:
        """
        Export conversation history to JSON file
        
        Args:
            session_id: Specific session to export (None for all)
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        if session_id:
            conversations = self.get_session_history(session_id, limit=1000)
        else:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, session_id, question, answer, model_used, timestamp
                FROM conversations
                ORDER BY timestamp DESC
            """)
            conversations = [
                {
                    'id': row[0],
                    'session_id': row[1],
                    'question': row[2],
                    'answer': row[3],
                    'model_used': row[4],
                    'timestamp': row[5]
                }
                for row in cursor.fetchall()
            ]
        
        # Generate filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_export_{timestamp}.json"
        
        export_path = Path("data/exports") / filename
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to file
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump({
                'exported_at': datetime.now().isoformat(),
                'total_conversations': len(conversations),
                'conversations': conversations
            }, f, indent=2)
        
        return str(export_path)
    
    def clear_old_sessions(self, days: int = 30):
        """Delete sessions older than specified days"""
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM conversations
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        """, (days,))
        deleted = cursor.rowcount
        self.conn.commit()
        return deleted
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

# Usage example
if __name__ == "__main__":
    db = ConversationDB()
    
    # Save a test conversation
    conv_id = db.save_conversation(
        session_id="test_session_1",
        question="What is machine learning?",
        answer="Machine learning is a subset of AI...",
        model_used="ft:gpt-3.5-turbo-0125:personal:research-assistant:Cm91GJSD",
        sources=[{"title": "ML Paper", "page": 3}],
        response_time=1.5
    )
    print(f"Saved conversation ID: {conv_id}")
    
    # Get analytics
    analytics = db.get_analytics()
    print(f"\nAnalytics: {analytics}")
    
    # Search
    results = db.search_conversations("machine learning")
    print(f"\nSearch results: {len(results)}")
    
    db.close()