from flask_sqlalchemy import SQLAlchemy 
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    public_id = db.Column(db.String(36), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<User {self.public_id}>'
    
    def to_dict(self):
        """Convert user object to dictionary"""
        return {
            'id': self.id,
            'public_id': self.public_id,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None,
            'last_seen': self.last_seen.strftime('%Y-%m-%d %H:%M:%S') if self.last_seen else None
        }

class History(db.Model):
    __tablename__ = 'history'
    
    history_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.String(36), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    
    def __repr__(self):
        return f'<History {self.symbol} by {self.user_id}>'
    
    def to_dict(self):
        """Convert history object to dictionary"""
        return {
            'id': self.history_id,
            'user_id': self.user_id,
            'symbol': self.symbol
        }
    