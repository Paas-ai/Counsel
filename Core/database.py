# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 21:39:58 2025

@author: Kumanan
"""

import logging
import json
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
from ..config import Config

logger = logging.getLogger(__name__)

def get_db():
    """
    Create database connection to PostgreSQL
    """
    try:
        conn = psycopg2.connect(
            dbname=Config.DB_NAME,
            user=Config.DB_USER,
            password=Config.DB_PASSWORD,
            host=Config.DB_HOST,
            port=Config.DB_PORT
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        raise

def init_db():
    """
    Initialize database tables
    """
    conn = get_db()
    try:
        with conn.cursor() as cur:
            # Enable UUID extension
            cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
            cur.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')

            # Create users table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    mobile_number VARCHAR(15) NOT NULL UNIQUE,
                    username VARCHAR(100),
                    sub VARCHAR(255) UNIQUE,
                    email VARCHAR(255),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP WITH TIME ZONE,
                    is_active BOOLEAN DEFAULT FALSE
                )
            ''')

            # Create user_tokens table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS user_tokens (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    user_id UUID NOT NULL,
                    access_token TEXT NOT NULL,
                    id_token TEXT,
                    refresh_token TEXT,
                    token_expiry TIMESTAMP WITH TIME ZONE NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')

            # Create oidc_auth_sessions table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS oidc_auth_sessions (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    mobile_number VARCHAR(15) NOT NULL,
                    state VARCHAR(255) NOT NULL UNIQUE,
                    nonce VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    FOREIGN KEY (mobile_number) REFERENCES users(mobile_number)
                )
            ''')
            
            # Create indices
            cur.execute('CREATE INDEX IF NOT EXISTS idx_users_mobile ON users(mobile_number)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_users_sub ON users(sub)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_tokens_user ON user_tokens(user_id)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_auth_sessions_mobile ON oidc_auth_sessions(mobile_number)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_auth_sessions_state ON oidc_auth_sessions(state)')
            
            conn.commit()
            logger.info("Database initialized successfully")

    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        raise
    finally:
        conn.close()

def update_processing_status(user_id: str, status_data: dict):
    """
    Update processing status in database
    """
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                'UPDATE users SET processing_status = %s WHERE id = %s',
                (json.dumps(status_data), user_id)
            )
            conn.commit()
            logger.info(f"Updated processing status for user {user_id}")
    except Exception as e:
        logger.error(f"Error updating processing status: {str(e)}")
        raise
    finally:
        conn.close()