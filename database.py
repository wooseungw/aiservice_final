import sqlite3

def initialize_db():
    conn = sqlite3.connect('chroma.db')
    cur = conn.cursor()
    
    # tenants 테이블 생성
    cur.execute('''
        CREATE TABLE IF NOT EXISTS tenants (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL
        )
    ''')
    
    # default_tenant 테넌트 생성
    cur.execute('''
        INSERT OR IGNORE INTO tenants (name) VALUES ('default_tenant')
    ''')
    
    conn.commit()
    conn.close()


def check_tenant():
    conn = sqlite3.connect('chroma.db')
    cur = conn.cursor()
    
    cur.execute('SELECT * FROM tenants WHERE name = "default_tenant"')
    tenant = cur.fetchone()
    
    if tenant:
        print("default_tenant exists:", tenant)
    else:
        print("default_tenant does not exist")
    
    conn.close()