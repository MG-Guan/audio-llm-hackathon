import os
from sqlalchemy import create_engine
from database import Base, SQLALCHEMY_DATABASE_URL, engine as global_engine, SessionLocal

def reset_database():
    """
    Delete and reconstruct the SQLite database.
    This is a debug helper function that should only be used in development.
    """
    try:
        # Close all existing connections
        global_engine.dispose()
        
        # Extract the database file path from the URL
        # SQLite URL format: sqlite:///path/to/database.db
        db_path = SQLALCHEMY_DATABASE_URL.replace('sqlite:///', '')
        
        # Create a temporary session to ensure no active transactions
        temp_session = SessionLocal()
        try:
            temp_session.close()
        except:
            pass
        
        # Delete the existing database file if it exists
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
        except PermissionError:
            return {"error": "Could not delete database file - it may be in use. Try restarting the server."}
            
        # Create a new engine and recreate all tables
        new_engine = create_engine(
            SQLALCHEMY_DATABASE_URL,
            connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(bind=new_engine)
        
        return {"message": "Database reset successfully"}
    except Exception as e:
        return {"error": f"Failed to reset database: {str(e)}"}