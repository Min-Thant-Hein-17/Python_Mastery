# Python Fundamentals

Python Fundamentals is a sub-repo about projects and practices of Python fundamentals, including varibles, lists, data types, OOP, functions and conditions, etc.

###############################

# 🐍 Python Roadmap

A structured, beginner-to-intermediate roadmap for learning Python — from setup to building real projects. Based on a visual roadmap I came across on Instagram (credit and link below).

> 📌 Reference: [Original Instagram Post by @mastercode.sagar](https://www.instagram.com/p/DbkFLsdk-e9/?utm_source=ig_web_copy_link&igsh=NTc4MTIwNjQ2YQ==)

---

## 📍 Roadmap Overview

| # | Topic | Focus |
|---|-------|-------|
| 1 | [Basics](#1-basics) | Build a strong foundation |
| 2 | [Control Flow](#2-control-flow) | Learn decision making |
| 3 | [Data Structures](#3-data-structures) | Organize and store data effectively |
| 4 | [Functions](#4-functions) | Write reusable code |
| 5 | [Modules & Packages](#5-modules--packages) | Use the power of Python |
| 6 | [OOP (Object Oriented Programming)](#6-oop-object-oriented-programming) | Think in real-world scenarios |
| 7 | [File Handling](#7-file-handling) | Work with files |
| 8 | [Advanced Topics](#8-advanced-topics) | Level up your skills |
| 9 | [Projects](#9-projects) | Build, practice, improve |

---

## 1. Basics

- Python Setup
- Syntax & Comments
- Variables & Data Types
- Operators
- Input / Output

### 🔧 Python Setup
1. **Install Python** — Download the latest version for your OS from [python.org](https://www.python.org/). Run the installer, and make sure to check **"Add Python to PATH"** before installing.
2. **Verify Installation** — Open a terminal and run:
   ```bash
   python --version   # or python -V
   ```
3. **Choose a Code Editor** — [VS Code](https://code.visualstudio.com/) is recommended for beginners: lightweight, powerful, and extensible.
4. **Run Your First Program** — Create `hello.py`:
   ```python
   print("Hello, Python!")
   ```
   Run it:
   ```bash
   python hello.py
   ```

**Tips:** Keep Python updated · Use virtual environments for projects · Practice, build, and have fun.

### ✍️ Syntax & Comments
- Python syntax is the set of rules that defines a valid Python program.
- Python is **case-sensitive**.
- Statements are written in a logical order.
- **Indentation** is used to define blocks of code (mandatory, usually 4 spaces).
- A compound statement starts with a colon `:`.
- A new line signifies the end of a statement.

```python
if x > 0:
    print("Positive")
else:
    print("Non-positive")
```

**Comments:**
```python
# Single-line comment
x = 10  # This is a comment
print(x)  # Output the value of x

"""
Multi-line comment.
It can span across
multiple lines.
"""
print("Hello, Python!")
```

### 🧮 Variables & Data Types
A variable is a container that stores a data value. In Python, you don't need to declare the variable type — just assign a value using `=`.

```python
name = "Santo"     # string
age = 20           # integer
price = 99.99       # float
is_student = True  # boolean
```

**Rules for naming variables:**
- Must start with a letter (a-z, A-Z) or underscore `_`
- Can contain letters, digits, and underscores
- Cannot start with a number
- Cannot use special characters (`!`, `@`, `#`, etc.)
- Cannot be a Python keyword

**Multiple assignment:**
```python
a, b, c = 10, 20, 30   # multiple values
x = y = z = 5           # same value
```

**Built-in data types:**

| Data Type | Description | Example |
|-----------|-------------|---------|
| `int` | Integer numbers | `x = 10` |
| `float` | Decimal numbers | `y = 3.14` |
| `str` | Sequence of characters (string) | `name = "Santo"` |
| `bool` | Boolean values (True or False) | `is_valid = True` |
| `list` | Ordered, changeable collection | `marks = [85, 90, 95]` |
| `tuple` | Ordered, unchangeable collection | `point = (10, 20)` |
| `set` | Unordered collection of unique items | `data = {1, 2, 3}` |
| `dict` | Key-value pair collection | `info = {"name": "Santo", "age": 20}` |

Use `type()` to check a variable's data type:
```python
print(type(x))  # <class 'int'>
```

### ➕ Operators
Operators are special symbols that perform operations on values and variables.

- **Arithmetic:** `+` `-` `*` `/` `//` `%` `**`
- **Comparison:** `==` `!=` `>` `<` `>=` `<=`
- **Assignment:** `=` `+=` `-=` `*=` `/=` `%=` `**=` `//=`
- **Logical:** `and` `or` `not`
- **Identity:** `is` `is not`
- **Membership:** `in` `not in`

```python
a = 10
b = 20
print(a + b)          # 30
print(a < b)           # True
print(a != b)          # True
print(a and b)         # 20 (b is truthy)
print(a in [5, 10, 15]) # True
```

### ⌨️ Input & Output
- `input()` takes user input and **always returns a string** — cast it (`int()`, `float()`, `bool()`) if you need another type.
- `print()` displays output; supports `sep` and `end` for custom formatting.

```python
name = input("Enter your name: ")
age = int(input("Enter your age: "))
print("Hello,", name + "!")
print("Next year, you will be", age + 1, "years old.")
```

```python
print("A", "B", "C", sep="-")   # A-B-C
print("Done", end="***")         # Done***
```

---

## 2. Control Flow
- `if-else`
- `for` loop
- `while` loop
- `break`, `continue`, `pass`

## 3. Data Structures
- List
- Tuple
- Set
- Dictionary

## 4. Functions
- Defining Functions
- Parameters & Arguments
- Return Statement
- Scope (Local & Global)

## 5. Modules & Packages
- Importing Modules
- Built-in Modules
- Creating Your Own Module
- pip & External Packages

## 6. OOP (Object Oriented Programming)
- Classes & Objects
- Inheritance
- Polymorphism
- Encapsulation

## 7. File Handling
- Read / Write Files
- File Modes
- `with open()`

## 8. Advanced Topics
- Exception Handling
- List Comprehension
- Lambda Functions
- Regular Expressions

## 9. Projects
- Build small projects
- Web Scraping
- Automation
- APIs

---

## 📚 Reference

This roadmap is based on the visual notes shared by **@mastercode.sagar** on Instagram:
🔗 [https://www.instagram.com/p/DbkFLsdk-e9/](https://www.instagram.com/p/DbkFLsdk-e9/?utm_source=ig_web_copy_link&igsh=NTc4MTIwNjQ2YQ==)

## ✅ Progress Tracker

- [ ] Basics
- [ ] Control Flow
- [ ] Data Structures
- [ ] Functions
- [ ] Modules & Packages
- [ ] OOP
- [ ] File Handling
- [ ] Advanced Topics
- [ ] Projects

---

*Made while learning Python — contributions and suggestions welcome!* 🚀


####################################




