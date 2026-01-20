import psycopg2
from psycopg2 import sql

class DBSetup():
    def __init__(self, db_name="pongML_db", user="postgres", password="postgres", host="localhost", port="5432"):
        self.db_name = db_name 
        self.user = user 
        self.password = password
        self.host = host
        self.port = port    

    def db_setup(self):
        conn = psycopg2.connect(
            dbname="postgres",
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )
        conn.autocommit = True
        cursor = conn.cursor()

        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (self.db_name,))
        exists = cursor.fetchone()

        if not exists:
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.db_name)))
            print(f"Created database {self.db_name}")
        else:
            print(f"Database {self.db_name} already exists")

        conn.close()

    def db_create_tbl(self):
        conn = psycopg2.connect(
            dbname=self.db_name,
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port
        )
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS game_data (
                id SERIAL PRIMARY KEY,
                game_id TEXT,
                winner TEXT,
                date DATE,
                ball_x REAL,
                ball_y REAL,
                agent_y1 REAL,
                agent_y2 REAL,
                opp_y1 REAL,
                opp_y2 REAL                
            )    
        """)
   
        conn.commit()
        conn.close()
        print("Created schema")

if __name__ == "__main__":
    db = DBSetup()
    db.db_setup()
    db.db_create_tbl()
