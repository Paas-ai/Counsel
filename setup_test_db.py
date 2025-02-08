# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 19:34:28 2025

@author: Kumanan
"""

#!/usr/bin/env python

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
from dotenv import load_dotenv

load_dotenv()

def setup_test_db():
    # Connection parameters for creating database
    params = {
        'host': os.getenv('TEST_DB_HOST', 'localhost'),
        'user': os.getenv('TEST_DB_USER'),
        'password': os.getenv('TEST_DB_PASSWORD'),
        'port': int(os.getenv('TEST_DB_PORT', 5432))
    }

    try:
        # Connect to PostgreSQL server
        conn = psycopg2.connect(**params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # Create test database
        db_name = os.getenv('TEST_DB_NAME')
        cur.execute(f"DROP DATABASE IF EXISTS {db_name}")
        cur.execute(f"CREATE DATABASE {db_name}")

        cur.close()
        conn.close()

        # Connect to the new test database
        params['dbname'] = db_name
        conn = psycopg2.connect(**params)
        cur = conn.cursor()

        # Create test tables
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
        print("Test database setup completed successfully")

    except Exception as e:
        print(f"Error setting up test database: {str(e)}")
        raise
    finally:
        if 'cur' in locals():
            cur.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    setup_test_db()