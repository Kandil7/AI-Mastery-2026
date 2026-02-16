# Getting Started with Databases

This beginner-friendly guide will walk you through the fundamentals of databases, from basic concepts to your first hands-on experience. No prior database knowledge is required—just curiosity and a willingness to learn!

---

## Table of Contents

1. [What is a Database?](#1-what-is-a-database)
2. [Installing Database Systems](#2-installing-database-systems)
3. [Basic Command-Line Operations](#3-basic-command-line-operations)
4. [GUI Tools for Database Management](#4-gui-tools-for-database-management)
5. [Creating Your First Database and Table](#5-creating-your-first-database-and-table)
6. [CRUD Operations: The Foundation of Data Management](#6-crud-operations-the-foundation-of-data-management)
7. [First Steps Exercises](#7-first-steps-exercises)

---

## 1. What is a Database?

A database is like a highly organized digital filing cabinet that stores, retrieves, and manages data efficiently. Think of it as a structured collection of information that can be accessed, managed, and updated systematically.

### Analogy: The Library System

Imagine a library:
- **Books** = Data records
- **Shelves** = Tables (organized collections of similar records)
- **Catalog system** = Indexes (help you find books quickly)
- **Librarian** = Database Management System (DBMS) - the software that manages everything

Without a database, you'd have to search through piles of unorganized papers (like spreadsheets or text files), which becomes impossible as your data grows.

### Key Characteristics of Databases

| Feature | Description |
|---------|-------------|
| **Persistence** | Data survives after the application closes |
| **Structure** | Data is organized according to defined rules |
| **Concurrency** | Multiple users can access data simultaneously |
| **Integrity** | Rules ensure data accuracy and consistency |
| **Security** | Access controls protect sensitive information |

### Why Databases Matter for AI/ML

Databases are the backbone of AI applications because:
- They store training data, model parameters, and inference results
- They enable efficient data retrieval for real-time predictions
- They provide transactional integrity for critical operations
- They support complex queries for data analysis and feature engineering

---

## 2. Installing Database Systems

Let's get you set up with PostgreSQL (recommended for beginners) and MySQL.

### PostgreSQL Installation

#### Windows
1. Download PostgreSQL installer from [postgresql.org](https://www.postgresql.org/download/windows/)
2. Run the installer and follow the prompts
3. Set a password for the `postgres` user (remember this!)
4. Note the port number (default: 5432)
5. Check "pgAdmin" during installation for GUI tool

#### macOS
```bash
# Using Homebrew (recommended)
brew install postgresql

# Start PostgreSQL
brew services start postgresql

# Or using the official installer from postgresql.org
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### MySQL Installation

#### Windows
1. Download MySQL Community Server from [mysql.com](https://dev.mysql.com/downloads/mysql/)
2. Run the installer and choose "Developer Default"
3. Set root password during setup
4. Install MySQL Workbench for GUI

#### macOS
```bash
# Using Homebrew
brew install mysql

# Start MySQL
brew services start mysql
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install mysql-server
sudo systemctl start mysql
sudo systemctl enable mysql
```

### Verify Installation

After installation, test your setup:

**PostgreSQL:**
```bash
# Test connection
psql -U postgres -h localhost -p 5432

# If prompted for password, enter the one you set during installation
```

**MySQL:**
```bash
# Test connection
mysql -u root -p

# Enter your root password when prompted
```

If you see a welcome message and a command prompt (`postgres=#` or `mysql>`), your installation was successful!

---

## 3. Basic Command-Line Operations

### Connecting to Your Database

**PostgreSQL:**
```bash
# Connect as postgres user
psql -U postgres -h localhost -d postgres

# Connect to specific database
psql -U username -h localhost -d mydatabase
```

**MySQL:**
```bash
# Connect as root
mysql -u root -p

# Connect to specific database
mysql -u username -p mydatabase
```

### Essential Commands

| Command | PostgreSQL | MySQL | Purpose |
|---------|------------|-------|---------|
| Help | `\?` | `help` or `\h` | Show help |
| List databases | `\l` | `SHOW DATABASES;` | See available databases |
| Switch database | `\c dbname` | `USE dbname;` | Change current database |
| List tables | `\dt` | `SHOW TABLES;` | See tables in current database |
| Describe table | `\d tablename` | `DESCRIBE tablename;` | Show table structure |
| Exit | `\q` | `EXIT;` or `QUIT;` | Quit the client |

### Creating Your First Database

**PostgreSQL:**
```sql
-- Create a new database
CREATE DATABASE learning_db;

-- Connect to it
\c learning_db
```

**MySQL:**
```sql
-- Create a new database
CREATE DATABASE learning_db;

-- Use it
USE learning_db;
```

---

## 4. GUI Tools for Database Management

While command-line tools are powerful, GUIs make learning easier.

### pgAdmin (PostgreSQL)
- Included in PostgreSQL installer
- Web-based interface accessible at `http://localhost:5050`
- Features: Query editor, visual table designer, data browser

### MySQL Workbench
- Included in MySQL installer
- Desktop application with visual tools
- Features: SQL editor, ER diagram designer, data modeling

### DBeaver (Cross-platform)
- Free, open-source universal database tool
- Supports PostgreSQL, MySQL, SQLite, and many others
- Download from [dbeaver.io](https://dbeaver.io/)

**Pro Tip**: Use GUI tools for exploration and learning, but practice command-line operations for deeper understanding and automation.

---

## 5. Creating Your First Database and Table

Let's build a simple "Students" database to store student information.

### Step 1: Create the Database
```sql
-- PostgreSQL
CREATE DATABASE school_db;

-- MySQL  
CREATE DATABASE school_db;
```

### Step 2: Connect to the Database
```sql
-- PostgreSQL
\c school_db;

-- MySQL
USE school_db;
```

### Step 3: Create a Table
```sql
CREATE TABLE students (
    student_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    age INTEGER CHECK (age >= 0 AND age <= 150),
    enrollment_date DATE DEFAULT CURRENT_DATE,
    is_active BOOLEAN DEFAULT TRUE
);
```

**Explanation of the CREATE TABLE statement:**
- `SERIAL PRIMARY KEY`: Auto-incrementing unique identifier
- `VARCHAR(50) NOT NULL`: Text field with max 50 characters, required
- `UNIQUE`: Ensures no duplicate emails
- `CHECK`: Validates age is reasonable
- `DEFAULT`: Sets default values for new records

### Step 4: Verify Table Creation
```sql
-- PostgreSQL
\d students

-- MySQL
DESCRIBE students;
```

You should see the table structure with all the columns we defined.

---

## 6. CRUD Operations: The Foundation of Data Management

CRUD stands for **C**reate, **R**ead, **U**pdate, and **D**elete—the four basic operations for managing data.

### Create (INSERT)
Add new records to your table:

```sql
-- Insert a single student
INSERT INTO students (first_name, last_name, email, age)
VALUES ('Alice', 'Johnson', 'alice.johnson@example.com', 20);

-- Insert multiple students at once
INSERT INTO students (first_name, last_name, email, age)
VALUES 
    ('Bob', 'Smith', 'bob.smith@example.com', 21),
    ('Carol', 'Davis', 'carol.davis@example.com', 19),
    ('David', 'Wilson', 'david.wilson@example.com', 22);
```

### Read (SELECT)
Retrieve data from your table:

```sql
-- Get all students
SELECT * FROM students;

-- Get specific columns
SELECT first_name, last_name, email FROM students;

-- Filter results
SELECT * FROM students WHERE age > 20;

-- Sort results
SELECT * FROM students ORDER BY last_name ASC;

-- Count records
SELECT COUNT(*) FROM students;
```

### Update (UPDATE)
Modify existing records:

```sql
-- Update a specific student
UPDATE students 
SET age = 21 
WHERE student_id = 1;

-- Update multiple records
UPDATE students 
SET is_active = FALSE 
WHERE enrollment_date < '2023-01-01';

-- Update with calculation
UPDATE students 
SET age = age + 1 
WHERE student_id IN (1, 2);
```

### Delete (DELETE)
Remove records from your table:

```sql
-- Delete a specific student
DELETE FROM students WHERE student_id = 3;

-- Delete all inactive students
DELETE FROM students WHERE is_active = FALSE;

-- Be careful! This deletes ALL records:
-- DELETE FROM students;
```

### Practice Exercise: Student Management
Try these operations in order:
1. Insert 3 more students
2. Select all students ordered by age (descending)
3. Update one student's email
4. Delete the youngest student
5. Verify the count matches expectations

---

## 7. First Steps Exercises

Complete these exercises to solidify your understanding:

### Exercise 1: Book Catalog
Create a `books` table with columns: `book_id`, `title`, `author`, `isbn`, `publication_year`, `genre`, `price`. Add 5 books and practice CRUD operations.

### Exercise 2: Simple Blog
Create a `posts` table with: `post_id`, `title`, `content`, `author`, `created_at`, `is_published`. Insert 3 blog posts and practice filtering by publication status.

### Exercise 3: Data Validation
Experiment with constraints:
- Try inserting a student with age = -5 (should fail)
- Try inserting two students with the same email (should fail)
- What happens when you try to insert NULL into a NOT NULL column?

### Exercise 4: Query Practice
Write queries to:
- Find all students whose last name starts with 'S'
- Calculate average age of students
- Find the oldest student
- List students grouped by age

### Bonus Challenge
Install SQLite (lightweight, file-based database) and repeat the exercises. Compare the experience with PostgreSQL/MySQL.

---

## 🧠 Knowledge Check: Quick Quiz

Test your understanding with these multiple-choice questions:

1. **What does CRUD stand for?**
   - A) Create, Retrieve, Update, Delete
   - B) Create, Read, Update, Delete ✅
   - C) Connect, Read, Use, Delete
   - D) Control, Record, Update, Destroy

2. **Which SQL command creates a new database?**
   - A) `NEW DATABASE` 
   - B) `CREATE DB`
   - C) `CREATE DATABASE` ✅
   - D) `MAKE DATABASE`

3. **What does `SERIAL PRIMARY KEY` do in PostgreSQL?**
   - A) Creates a text field that auto-increments
   - B) Creates an auto-incrementing integer that's unique ✅
   - C) Creates a random UUID identifier
   - D) Creates a foreign key relationship

4. **Which constraint ensures no duplicate values in a column?**
   - A) `NOT NULL`
   - B) `CHECK`
   - C) `UNIQUE` ✅
   - D) `DEFAULT`

5. **What happens when you run `SELECT * FROM students WHERE age > 20;`?**
   - A) Returns all students regardless of age
   - B) Returns only students older than 20 ✅
   - C) Returns students younger than 20
   - D) Returns an error because age is not indexed

**Answers**: 1-B, 2-C, 3-B, 4-C, 5-B

---

## Next Steps

Congratulations! You've taken your first steps in database mastery. In the next document, we'll dive deep into data types—the building blocks of your database schema.

**Recommended next reading**: [`01_data_types_fundamentals.md`](01_data_types_fundamentals.md)

> 💡 **Pro Tip**: Save your practice queries in a `.sql` file. This builds your personal reference library and helps you review concepts later.