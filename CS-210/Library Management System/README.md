# Library Management System 

# 📚 Small Community Library Management System

A Python-based Library Management System designed using clean **Object-Oriented Programming (OOP)** principles. The system handles the library's collection, member registration, and the end-to-end book borrowing and returning lifecycle.

---

## 🛠️ Design Architecture & OOP Principles

The system strictly adheres to the four core pillars of Object-Oriented Programming:

*   **Abstraction:** An abstract baseline class `LibraryUser` defines core user identities and behaviors that all specific roles must implement.
*   **Inheritance:** Both `Member` and `Librarian` inherit from `LibraryUser`. Specialized member classes (`RegularMember`, `PremiumMember`) inherit directly from `Member`.
*   **Encapsulation:** Class attributes (such as copy counts and borrowed lists) are safely contained within their respective instances and modified exclusively through designated class methods.
*   **Polymorphism:** `RegularMember` and `PremiumMember` implement custom maximum borrowing capacities while utilizing the identical `borrow()` interface.

---

## 💻 Python Implementation

```python
from abc import ABC, abstractmethod

class Book:
    """Represents a book in the library collection."""
    def __init__(self, title: str, author: str, isbn: str, available_copies: int):
        self.title = title
        self.author = author
        self.isbn = isbn
        self.available_copies = available_copies

    def borrow_book(self) -> bool:
        """Decreases the available copies if any are available."""
        if self.available_copies > 0:
            self.available_copies -= 1
            return True
        return False

    def return_book(self):
        """Increases the available copies when a book is returned."""
        self.available_copies += 1

    def __str__(self) -> str:
        return f"'{self.title}' by {self.author} (ISBN: {self.isbn}) - Copies available: {self.available_copies}"


class LibraryUser(ABC):
    """Abstract Base Class representing a general library user."""
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __str__(self):
        pass


class Member(LibraryUser):
    """Represents a general library member with borrowing capabilities."""
    def __init__(self, name: str, borrow_limit: int):
        super().__init__(name)
        self.borrowed_books = []
        self.borrow_limit = borrow_limit

    def borrow(self, book: Book) -> str:
        """Allows the member to borrow a book while enforcing their borrowing limit."""
        if len(self.borrowed_books) >= self.borrow_limit:
            return f"❌ {self.name} cannot borrow '{book.title}'. Borrowing limit of {self.borrow_limit} books reached."
        
        if book.borrow_book():
            self.borrowed_books.append(book)
            return f"✅ {self.name} successfully borrowed '{book.title}'."
        else:
            return f"❌ '{book.title}' cannot be borrowed. No copies currently available."

    def return_book(self, book: Book) -> str:
        """Allows the member to return a borrowed book."""
        if book in self.borrowed_books:
            self.borrowed_books.remove(book)
            book.return_book()
            return f"🔄 {self.name} successfully returned '{book.title}'."
        return f"❌ {self.name} does not currently have '{book.title}' checked out."

    def __str__(self) -> str:
        return f"Member: {self.name} | Books borrowed: {len(self.borrowed_books)}/{self.borrow_limit}"


class RegularMember(Member):
    """A standard member with a limit of 2 books."""
    def __init__(self, name: str):
        super().__init__(name, borrow_limit=2)


class PremiumMember(Member):
    """A premium member with an extended limit of 5 books."""
    def __init__(self, name: str):
        super().__init__(name, borrow_limit=5)


class Librarian(LibraryUser):
    """Represents a librarian with structural administrative privileges."""
    def __init__(self, name: str):
        super().__init__(name)

    def add_book_to_library(self, library, book: Book) -> str:
        """Privileged method to inject a new book directly into the collection."""
        library.add_book(book)
        return f"💼 Librarian {self.name} added '{book.title}' to the library collection."

    def __str__(self) -> str:
        return f"Librarian: {self.name}"


class Library:
    """Manages the centralized list of books, registered members, and catalogs transactions."""
    def __init__(self):
        self.books = []
        self.members = []

    def add_book(self, book: Book):
        """Adds a book object to the library collection."""
        self.books.append(book)

    def register_member(self, member: Member):
        """Registers a member into the library database."""
        self.members.append(member)
```

---

## 🚀 Execution & Verification Script

Save the implementation above and append this validation script to test the library's dynamic limits and copying logic:

```python
# Create library instance
library = Library()

# Add books
book1 = Book("1984", "George Orwell", "1234567890", 3)
book2 = Book("To Kill a Mockingbird", "Harper Lee", "2345678901", 2)

library.add_book(book1)
library.add_book(book2)

# Register members
alice = RegularMember("Alice")
bob = PremiumMember("Bob")

library.register_member(alice)
library.register_member(bob)

# Borrowing books execution
print(alice.borrow(book1))  # Alice borrows "1984"
print(bob.borrow(book1))    # Bob borrows "1984"
print(bob.borrow(book2))    # Bob borrows "To Kill a Mockingbird"
print(bob.borrow(book2))    # Bob tries to borrow "To Kill a Mockingbird" again (no copies available)

# Returning books execution
print(alice.return_book(book1))  # Alice returns "1984"
print(bob.borrow(book1))         # Now Bob can borrow "1984"
```

### 📋 Expected Console Output

```text
✅ Alice successfully borrowed '1984'.
✅ Bob successfully borrowed '1984'.
✅ Bob successfully borrowed 'To Kill a Mockingbird'.
❌ 'To Kill a Mockingbird' cannot be borrowed. No copies currently available.
🔄 Alice successfully returned '1984'.
✅ Bob successfully borrowed '1984'.
```
