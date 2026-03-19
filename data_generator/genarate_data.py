# ════════════════════════════════════════════════════════════════════════════
# Synthetic Data Generator — Student Early Warning System
# Generates realistic student records for Vijaybhoomi University
# ════════════════════════════════════════════════════════════════════════════

import sqlite3
import numpy as np
from datetime import datetime, timedelta

DB_PATH = 'student_risk.db'

SCHOOLS = {
    'VSST':   ['B.Tech CSE', 'B.Tech AI/ML'],
    'VSOD':   ['B.Des Fashion', 'B.Des Product', 'B.Des Communication'],
    'VSOL':   ['BA LLB', 'BBA LLB', 'LLB'],
    'JAGSOM': ['BBA', 'MBA'],
    'TSM':    ['B.Mus Performance', 'B.Mus Production',
               'B.Mus Composition', 'B.Tech Sound Engineering']
}

FIRST_NAMES = [
    'Aarav','Vivaan','Aditya','Arjun','Sai','Arnav','Ayaan','Krishna',
    'Aadhya','Ananya','Diya','Isha','Avni','Sara','Priya','Anika',
    'Rohan','Kabir','Aryan','Vihaan','Saanvi','Myra','Kiara','Mira'
]
LAST_NAMES = [
    'Sharma','Patel','Kumar','Singh','Reddy','Nair','Chopra','Malhotra',
    'Shah','Gupta','Verma','Agarwal','Jain','Rao','Desai','Iyer'
]

def create_schema(conn):
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS attendance')
    c.execute('DROP TABLE IF EXISTS academic_records')
    c.execute('DROP TABLE IF EXISTS behavioral_incidents')
    c.execute('DROP TABLE IF EXISTS student_demographics')
    c.execute('DROP TABLE IF EXISTS counselor_referrals')
    c.execute('DROP TABLE IF EXISTS students')

    c.execute('''CREATE TABLE students (
        student_id INTEGER PRIMARY KEY,
        student_name TEXT NOT NULL,
        school TEXT NOT NULL,
        program TEXT NOT NULL,
        year INTEGER NOT NULL,
        enrollment_date TEXT NOT NULL)''')
    c.execute('''CREATE TABLE academic_records (
        student_id INTEGER PRIMARY KEY,
        cgpa REAL NOT NULL,
        previous_failures INTEGER DEFAULT 0,
        assignment_completion REAL DEFAULT 1.0)''')
    c.execute('''CREATE TABLE attendance (
        student_id INTEGER, date TEXT,
        status TEXT, unexcused INTEGER DEFAULT 0)''')
    c.execute('''CREATE TABLE behavioral_incidents (
        student_id INTEGER PRIMARY KEY,
        total_incidents INTEGER DEFAULT 0,
        days_since_last INTEGER DEFAULT 365)''')
    c.execute('''CREATE TABLE student_demographics (
        student_id INTEGER PRIMARY KEY,
        scholarship INTEGER DEFAULT 0,
        extracurricular INTEGER DEFAULT 0,
        hostel_resident INTEGER DEFAULT 0,
        part_time_job INTEGER DEFAULT 0)''')
    c.execute('''CREATE TABLE counselor_referrals (
        student_id INTEGER PRIMARY KEY,
        total_referrals INTEGER DEFAULT 0,
        last_referral_date TEXT)''')
    conn.commit()

def generate(num_students=200, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    c = conn.cursor()

    school_list  = list(SCHOOLS.keys())
    distribution = [60, 40, 40, 35, 25]
    np.random.seed(42)

    sid = 1
    for school, count in zip(school_list, distribution):
        programs = SCHOOLS[school]
        for _ in range(count):
            name    = f"{np.random.choice(FIRST_NAMES)} {np.random.choice(LAST_NAMES)}"
            program = np.random.choice(programs)
            year    = np.random.randint(1, 5)
            enroll  = datetime.now().strftime('%Y-%m-%d')

            cgpa    = round(np.clip(np.random.normal(7.0, 1.2), 3.0, 10.0), 2)
            fails   = min(int(np.random.poisson(0.3)), 5)
            comp    = round(np.clip(np.random.normal(0.80, 0.13), 0.3, 1.0), 2)
            att_r   = np.clip(np.random.normal(0.91, 0.06), 0.4, 1.0)
            inc     = min(int(np.random.poisson(0.3)), 10)
            days    = np.random.randint(1, 60) if inc > 0 else 365
            scholar = 1 if np.random.random() < 0.3 else 0
            extra   = 1 if np.random.random() < 0.6 else 0
            hostel  = 1 if np.random.random() < 0.4 else 0
            parttime= 1 if np.random.random() < 0.2 else 0
            refs    = min(int(np.random.poisson(0.2)), 5)

            c.execute("INSERT INTO students VALUES (?,?,?,?,?,?)",
                (sid, name, school, program, year, enroll))
            c.execute("INSERT INTO academic_records VALUES (?,?,?,?)",
                (sid, cgpa, fails, comp))

            unexc = 0
            for day in range(30):
                present = np.random.random() < att_r
                status  = 'Present' if present else 'Absent'
                u       = 0 if present else int(np.random.random() < 0.6)
                unexc  += u
                d = (datetime.now() - timedelta(days=29-day)).strftime('%Y-%m-%d')
                c.execute("INSERT INTO attendance VALUES (?,?,?,?)", (sid,d,status,u))

            c.execute("INSERT INTO behavioral_incidents VALUES (?,?,?)", (sid,inc,days))
            c.execute("INSERT INTO student_demographics VALUES (?,?,?,?,?)",
                (sid,scholar,extra,hostel,parttime))
            rd = (datetime.now()-timedelta(days=np.random.randint(1,45))).strftime('%Y-%m-%d') \
                 if refs > 0 else None
            c.execute("INSERT INTO counselor_referrals VALUES (?,?,?)", (sid,refs,rd))
            sid += 1

    conn.commit()
    conn.close()
    print(f"Generated {num_students} students across {len(SCHOOLS)} schools")
    print(f"Database saved to {db_path}")

if __name__ == '__main__':
    generate()
