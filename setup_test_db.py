# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 19:34:28 2025

@author: Kumanan
"""
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
from dotenv import load_dotenv
import sys

def setup_test_db():
    # Load environment variables from .env.test
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        print(f"Warning: .env file not found at {env_path}")
        return

    # Connection parameters for postgres database first
    params = {
        'host': os.getenv('TEST_DB_HOST', 'localhost'),
        'user': os.getenv('TEST_DB_USER', 'postgres'),
        'password': os.getenv('TEST_DB_PASSWORD'),
        'port': int(os.getenv('TEST_DB_PORT', 5432))
    }

    print("Attempting to connect to PostgreSQL...")
    print(f"Host: {params['host']}")
    print(f"Port: {params['port']}")
    print(f"User: {params['user']}")

    try:
        # First connect to default postgres database
        params['dbname'] = 'postgres'
        conn = psycopg2.connect(**params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Create test database if it doesn't exist
        db_name = os.getenv('TEST_DB_NAME', 'test_audio_chat')
        print(f"\nAttempting to create database: {db_name}")
        
        # Check if database exists
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = cur.fetchone()
        
        if not exists:
            cur.execute(f'CREATE DATABASE {db_name}')
            print(f"Database {db_name} created successfully")
        else:
            print(f"Database {db_name} already exists")

        cur.close()
        conn.close()

        # Connect to the new test database
        print("\nConnecting to test database...")
        params['dbname'] = db_name
        conn = psycopg2.connect(**params)
        cur = conn.cursor()

        # Create extensions if they don't exist
        cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
        cur.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')

        # Create test tables
        print("\nCreating tables...")
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

        conn.commit()
        print("\nTest database setup completed successfully!")

    except psycopg2.Error as e:
        print(f"\nError setting up test database: {str(e)}")
        if 'password authentication failed' in str(e):
            print("\nHint: Check your database password in .env file")
        elif 'Connection refused' in str(e):
            print("\nHint: Make sure PostgreSQL service is running")
        sys.exit(1)
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    setup_test_db()