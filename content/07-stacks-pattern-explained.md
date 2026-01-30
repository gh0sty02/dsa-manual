# Stacks Pattern

**Difficulty:** Easy to Medium
**Prerequisites:** Basic data structures, Arrays
**Estimated Reading Time:** 22 minutes

## Introduction

The Stack pattern is one of the most fundamental and versatile patterns in computer science. A stack is a Last-In-First-Out (LIFO) data structure where elements are added and removed from the same end (the top). This simple concept is incredibly powerful for solving problems involving nested structures, backtracking, expression evaluation, and maintaining state.

**Why it matters:** Stacks are everywhere in computing - from the call stack that manages function calls in your programs, to undo functionality in applications, to parsing nested structures like HTML/XML. Understanding stacks is essential for system design, compiler construction, and algorithm design. Companies frequently test stack knowledge because it demonstrates understanding of how programs actually execute and how to manage state efficiently.

**Real-world analogy:** Think of a stack of plates in a cafeteria. You can only add a new plate to the top of the stack (push), and you can only remove the top plate (pop). You can't take a plate from the middle without removing all the plates above it first. The last plate you put on top is the first one that comes off. This is exactly how a stack data structure works - Last In, First Out!

## Core Concepts

### Key Principles

1. **LIFO (Last In, First Out):** The most recently added element is the first to be removed

2. **Single access point:** All operations happen at one end (the top)

3. **Push operation:** Add element to top - O(1)

4. **Pop operation:** Remove element from top - O(1)

5. **Peek/Top operation:** View top element without removing - O(1)

6. **State tracking:** Stacks naturally maintain history/state

### Essential Terms

- **Stack:** LIFO data structure
- **Push:** Add element to top
- **Pop:** Remove element from top
- **Peek/Top:** View top element without removing
- **Empty:** Stack has no elements
- **Overflow:** Attempting to push to full stack (fixed size)
- **Underflow:** Attempting to pop from empty stack

### Visual Overview

```
Stack Operations:

Initial: []

Push 1:  [1]      ← Top
         
Push 2:  [2]      ← Top
         [1]
         
Push 3:  [3]      ← Top
         [2]
         [1]
         
Pop:     [2]      ← Top (removed 3)
         [1]
         
Peek:    [2]      ← Top (view only, don't remove)
         [1]
         
Pop:     [1]      ← Top (removed 2)

Pop:     []       (removed 1, stack empty)

Visual Representation:
│   │
│   │  ← Top
│ 3 │  Push 3
│ 2 │  Push 2
│ 1 │  Push 1
└───┘
```

## How It Works

### Basic Stack Operations

1. **Push(element):**
   - Add element to top of stack
   - Increment top pointer
   - O(1) time complexity

2. **Pop():**
   - Remove and return top element
   - Decrement top pointer
   - O(1) time complexity

3. **Peek()/Top():**
   - Return top element without removing
   - O(1) time complexity

4. **isEmpty():**
   - Check if stack has no elements
   - O(1) time complexity

### Step-by-Step Example: Valid Parentheses

Problem: Check if string of parentheses is valid: "({[]})"

```
String: "({[]})"

Step 1: Process '('
Stack: ['(']
Action: Opening bracket, push to stack

Step 2: Process '{'
Stack: ['(', '{']
Action: Opening bracket, push to stack

Step 3: Process '['
Stack: ['(', '{', '[']
Action: Opening bracket, push to stack

Step 4: Process ']'
Stack: ['(', '{']
Action: Closing bracket, matches top '[', pop

Step 5: Process '}'
Stack: ['(']
Action: Closing bracket, matches top '{', pop

Step 6: Process ')'
Stack: []
Action: Closing bracket, matches top '(', pop

Result: Stack is empty → Valid! ✓

Example of Invalid:
String: "({[}])"

Steps 1-3: Stack: ['(', '{', '[']

Step 4: Process '}'
Top of stack is '[', but we have '}'
'[' and '}' don't match
Result: Invalid! ✗
```

## How to Identify This Pattern

Recognizing when to use a Stack is crucial for efficient problem solving. Here are the key indicators:

### Primary Indicators ✓

**Need to process nested structures**
- Matching parentheses/brackets
- HTML/XML tag validation
- Nested function calls
- Keywords: "nested", "matching", "balanced", "valid"
- Example: "Check if parentheses are balanced"

**Last-In-First-Out processing required**
- Most recent element needs processing first
- Reverse order processing
- Keywords: "LIFO", "reverse", "most recent"
- Example: "Process in reverse order"

**Need to backtrack or undo**
- Undo/redo functionality
- Backtracking in algorithms
- State restoration
- Keywords: "undo", "backtrack", "restore", "previous state"
- Example: "Implement undo functionality"

**Expression evaluation**
- Postfix/prefix notation
- Calculator implementation
- Operator precedence
- Keywords: "evaluate", "expression", "calculate", "RPN"
- Example: "Evaluate reverse Polish notation"

**Need to track previous elements**
- Remember history
- Compare with previous
- Keywords: "previous", "compare with last", "history"
- Example: "Compare with previous temperature"

**Parsing or compiling**
- Syntax checking
- Grammar validation
- Keywords: "parse", "compile", "syntax", "grammar"
- Example: "Decode string with nested brackets"

### Common Problem Phrases 🔑

Watch for these exact phrases:
- "Valid parentheses"
- "Balanced brackets"
- "Evaluate expression"
- "Reverse Polish notation"
- "Min stack" / "Max stack"
- "Decode string"
- "Remove adjacent duplicates"
- "Simplify path"
- "Basic calculator"
- "Implement undo/redo"

### Quick Decision Checklist ✅

Ask yourself:

1. **Do I need to match opening/closing pairs?** → Stack
2. **Is LIFO order required?** → Stack
3. **Need to track most recent element?** → Stack
4. **Evaluating an expression?** → Stack
5. **Backtracking involved?** → Stack
6. **Processing nested structures?** → Stack

If YES to any of these, Stack is likely the right choice!

### When NOT to Use Stack ✗

**Need random access to elements**
- Accessing middle elements
- → Use Array or List

**FIFO (First-In-First-Out) required**
- Process oldest first
- → Use Queue

**Need to find minimum/maximum efficiently**
- Unless it's "Min Stack" problem
- → Use Heap for general case

**Sorted order required**
- Maintaining sorted elements
- → Use Heap or Balanced BST

### Visual Recognition 👁️

**Stack Pattern Looks Like:**
```
Processing: a b c d
Stack builds: d
              c
              b
              a
              
Then unwinds: a ← First out
              b
              c
              d ← Last out
```

**Matching Pairs:**
```
Input:  ( { [ ] } )
        ↓   ↓ ↑ ↑ ↑ ↑
Stack:  ( { [ 
Pop when closing: ] matches [, } matches {, ) matches (
```

### Algorithm Signature 📝

**Parentheses Matching:**
```python
stack = []
for char in string:
    if char in opening:
        stack.append(char)
    else:  # closing bracket
        if not stack or not matches(stack[-1], char):
            return False
        stack.pop()
return len(stack) == 0
```

**Expression Evaluation:**
```python
stack = []
for token in tokens:
    if token is operand:
        stack.append(token)
    else:  # operator
        right = stack.pop()
        left = stack.pop()
        result = apply(operator, left, right)
        stack.append(result)
```

### Example Pattern Matching 💡

**Problem: "Check if string has valid parentheses"**

Analysis:
- ✓ Need to match opening/closing pairs
- ✓ Nested structure (parentheses inside parentheses)
- ✓ LIFO processing (match most recent opening)

**Verdict: USE STACK** ✓

**Problem: "Evaluate Reverse Polish Notation"**

Analysis:
- ✓ Expression evaluation
- ✓ Need to track operands
- ✓ Classic stack problem

**Verdict: USE STACK** ✓

**Problem: "Find next greater element"**

Analysis:
- ✓ Compare with previous elements
- ? Could use stack...
- ✓ But Monotonic Stack is better

**Verdict: USE MONOTONIC STACK** ✓

**Problem: "Implement a queue"**

Analysis:
- ✗ FIFO required (not LIFO)
- ✗ Wrong data structure

**Verdict: USE QUEUE** (Not Stack) ✗

### Pattern vs Problem Type 📊

| Problem Type | Use Stack? | Alternative |
|--------------|------------|-------------|
| Valid parentheses | ✅ YES | - |
| Expression evaluation | ✅ YES | - |
| Decode string | ✅ YES | - |
| Undo/Redo | ✅ YES | - |
| DFS traversal | ✅ YES | Recursion |
| Next greater element | ⚠️ Monotonic Stack | - |
| Process in order | ❌ NO | Queue |
| Random access | ❌ NO | Array |

### Keywords Cheat Sheet 📝

**STRONG "Stack" Keywords:**
- parentheses
- brackets
- balanced
- valid
- nested
- evaluate expression

**MODERATE Keywords:**
- undo
- backtrack
- reverse
- LIFO
- most recent
- decode

**ANTI-Keywords (probably NOT basic Stack):**
- next greater (Monotonic Stack)
- queue
- FIFO
- sorted
- minimum/maximum (unless "Min Stack")

### Red Flags 🚩

These suggest basic STACK might NOT be right:
- "Next greater/smaller" → Monotonic Stack
- "FIFO" mentioned → Queue
- Random access needed → Array
- Need sorted order → Heap/BST

### Green Flags 🟢

STRONG indicators for STACK:
- "Valid parentheses"
- "Balanced brackets"
- "Evaluate expression"
- "Reverse Polish Notation"
- "Nested" structures
- "Decode" with brackets
- "Undo/Redo"
- "DFS" traversal

## Implementation

### Problem 1: Valid Parentheses (LeetCode #20)

```python
def isValid(s: str) -> bool:
    """
    Determine if string has valid parentheses.
    
    Args:
        s: String containing parentheses characters
    
    Returns:
        True if valid, False otherwise
    
    Time Complexity: O(n) - single pass through string
    Space Complexity: O(n) - stack can hold all opening brackets
    """
    stack = []
    
    # Mapping of closing to opening brackets
    mapping = {
        ')': '(',
        '}': '{',
        ']': '['
    }
    
    for char in s:
        if char in mapping:
            # Closing bracket
            # Pop from stack (or use dummy if empty)
            top_element = stack.pop() if stack else '#'
            
            # Check if it matches
            if mapping[char] != top_element:
                return False
        else:
            # Opening bracket
            stack.append(char)
    
    # Valid if stack is empty
    return not stack


# Usage Examples
print(isValid("()"))        # True
print(isValid("()[]{}"))    # True
print(isValid("(]"))        # False
print(isValid("([)]"))      # False
print(isValid("{[]}"))      # True
```

### Problem 2: Min Stack (LeetCode #155)

```python
class MinStack:
    """
    Stack with O(1) getMin operation.
    
    Approach: Maintain two stacks - main stack and min stack.
    Min stack always has minimum at top.
    
    Space Complexity: O(n) - two stacks
    """
    
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val: int) -> None:
        """
        Push element to stack.
        
        Time: O(1)
        """
        self.stack.append(val)
        
        # Update min stack
        # Top of min_stack is current minimum
        if not self.min_stack:
            self.min_stack.append(val)
        else:
            # New min is min of current val and current min
            self.min_stack.append(min(val, self.min_stack[-1]))
    
    def pop(self) -> None:
        """
        Remove top element.
        
        Time: O(1)
        """
        self.stack.pop()
        self.min_stack.pop()
    
    def top(self) -> int:
        """
        Get top element.
        
        Time: O(1)
        """
        return self.stack[-1]
    
    def getMin(self) -> int:
        """
        Get minimum element in O(1).
        
        Time: O(1)
        """
        return self.min_stack[-1]


# Usage Example
minStack = MinStack()
minStack.push(-2)
minStack.push(0)
minStack.push(-3)
print(minStack.getMin())  # -3
minStack.pop()
print(minStack.top())     # 0
print(minStack.getMin())  # -2
```

### Problem 3: Evaluate Reverse Polish Notation (LeetCode #150)

```python
from typing import List

def evalRPN(tokens: List[str]) -> int:
    """
    Evaluate expression in Reverse Polish Notation.
    
    Args:
        tokens: List of operands and operators
    
    Returns:
        Result of evaluation
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(n) - stack size
    
    Example:
        ["2", "1", "+", "3", "*"] = (2 + 1) * 3 = 9
    """
    stack = []
    operators = {'+', '-', '*', '/'}
    
    for token in tokens:
        if token in operators:
            # Pop two operands
            right = stack.pop()
            left = stack.pop()
            
            # Apply operator
            if token == '+':
                result = left + right
            elif token == '-':
                result = left - right
            elif token == '*':
                result = left * right
            else:  # division
                # Python division truncates toward negative infinity
                # We want truncation toward zero
                result = int(left / right)
            
            stack.append(result)
        else:
            # Operand
            stack.append(int(token))
    
    # Final result is only element in stack
    return stack[0]


# Usage Examples
print(evalRPN(["2", "1", "+", "3", "*"]))  # 9
print(evalRPN(["4", "13", "5", "/", "+"]))  # 6
print(evalRPN(["10", "6", "9", "3", "/", "-11", "*", "/", "*", "17", "+", "5", "+"]))  # 22
```

### Problem 4: Decode String (LeetCode #394)

```python
def decodeString(s: str) -> str:
    """
    Decode string with pattern k[encoded_string].
    
    Args:
        s: Encoded string
    
    Returns:
        Decoded string
    
    Time Complexity: O(n) where n is length of decoded string
    Space Complexity: O(n)
    
    Example:
        "3[a]2[bc]" → "aaabcbc"
        "3[a2[c]]" → "accaccacc"
    """
    stack = []
    current_string = ""
    current_num = 0
    
    for char in s:
        if char.isdigit():
            # Build the number (could be multi-digit)
            current_num = current_num * 10 + int(char)
        elif char == '[':
            # Push current state to stack
            stack.append((current_string, current_num))
            # Reset for new context
            current_string = ""
            current_num = 0
        elif char == ']':
            # Pop previous context
            prev_string, num = stack.pop()
            # Repeat current_string num times and prepend prev_string
            current_string = prev_string + current_string * num
        else:
            # Regular character
            current_string += char
    
    return current_string


# Usage Examples
print(decodeString("3[a]2[bc]"))    # "aaabcbc"
print(decodeString("3[a2[c]]"))     # "accaccacc"
print(decodeString("2[abc]3[cd]ef")) # "abcabccdcdcdef"
```

### Problem 5: Remove All Adjacent Duplicates In String (LeetCode #1047)

```python
def removeDuplicates(s: str) -> str:
    """
    Remove all adjacent duplicate characters.
    
    Args:
        s: Input string
    
    Returns:
        String after removing all adjacent duplicates
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Example:
        "abbaca" → "ca"
        Process: abbaca → aaca → ca
    """
    stack = []
    
    for char in s:
        if stack and stack[-1] == char:
            # Current char matches top, remove both
            stack.pop()
        else:
            # No match, add to stack
            stack.append(char)
    
    return ''.join(stack)


# Usage Examples
print(removeDuplicates("abbaca"))   # "ca"
print(removeDuplicates("azxxzy"))   # "ay"
```

### Problem 6: Simplify Path (LeetCode #71)

```python
def simplifyPath(path: str) -> str:
    """
    Simplify Unix-style file path.
    
    Args:
        path: Absolute path
    
    Returns:
        Simplified canonical path
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Example:
        "/a/./b/../../c/" → "/c"
    """
    stack = []
    
    # Split by '/' and process each component
    components = path.split('/')
    
    for component in components:
        if component == '..' and stack:
            # Go up one directory
            stack.pop()
        elif component and component != '.' and component != '..':
            # Valid directory name
            stack.append(component)
        # Ignore: empty string, '.', or '..' when stack is empty
    
    # Build result path
    return '/' + '/'.join(stack)


# Usage Examples
print(simplifyPath("/home/"))               # "/home"
print(simplifyPath("/../"))                 # "/"
print(simplifyPath("/home//foo/"))          # "/home/foo"
print(simplifyPath("/a/./b/../../c/"))      # "/c"
```

### Problem 7: Basic Calculator II (LeetCode #227)

```python
def calculate(s: str) -> int:
    """
    Calculate result of expression with +, -, *, /.
    
    Args:
        s: Mathematical expression
    
    Returns:
        Result of calculation
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Example:
        "3+2*2" → 7
        " 3/2 " → 1
    """
    stack = []
    num = 0
    operator = '+'  # Start with + for first number
    
    for i, char in enumerate(s):
        if char.isdigit():
            num = num * 10 + int(char)
        
        # Process operator (or end of string)
        if char in '+-*/' or i == len(s) - 1:
            if char != ' ' or i == len(s) - 1:
                if operator == '+':
                    stack.append(num)
                elif operator == '-':
                    stack.append(-num)
                elif operator == '*':
                    stack.append(stack.pop() * num)
                elif operator == '/':
                    # Truncate toward zero
                    stack.append(int(stack.pop() / num))
                
                if i != len(s) - 1:
                    operator = char
                    num = 0
    
    return sum(stack)


# Usage Examples
print(calculate("3+2*2"))      # 7
print(calculate(" 3/2 "))      # 1
print(calculate(" 3+5 / 2 "))  # 5
```

## Complexity Analysis

### Time Complexity

**Push Operation:** O(1)
- Simply append to end of array/list

**Pop Operation:** O(1)
- Remove from end of array/list

**Peek Operation:** O(1)
- Access last element

**Problem Solutions:**
- **Valid Parentheses:** O(n) - single pass through string
- **Expression Evaluation:** O(n) - process each token once
- **Decode String:** O(n) - where n is decoded string length

### Space Complexity

**Stack Storage:** O(n)
- Worst case: all elements pushed to stack

**Min Stack:** O(n)
- Two stacks, each can hold all elements

**Expression Problems:** O(n)
- Stack size proportional to input

### Comparison with Alternatives

| Approach | Push/Pop | Space | Use Case |
|----------|----------|-------|----------|
| Stack (Array-based) | O(1) | O(n) | Most cases |
| Stack (Linked List) | O(1) | O(n) + pointer overhead | Dynamic size |
| Queue | O(1) | O(n) | FIFO needed |
| Recursion | - | O(n) call stack | DFS, backtracking |

## Examples

### Example 1: Valid Parentheses Detailed

```
Input: "({[]})"

Process each character:

char='(': opening bracket
Stack: ['(']

char='{': opening bracket  
Stack: ['(', '{']

char='[': opening bracket
Stack: ['(', '{', '[']

char=']': closing bracket
Top='[', matches ']' ✓
Pop '[']
Stack: ['(', '{']

char='}': closing bracket
Top='{', matches '}' ✓
Pop '{'
Stack: ['(']

char=')': closing bracket
Top='(', matches ')' ✓
Pop '('
Stack: []

Stack is empty → Valid! ✓
```

### Example 2: RPN Evaluation Detailed

```
Input: ["2", "1", "+", "3", "*"]

Process each token:

token="2": number
Stack: [2]

token="1": number
Stack: [2, 1]

token="+": operator
Pop 1, Pop 2
2 + 1 = 3
Stack: [3]

token="3": number
Stack: [3, 3]

token="*": operator
Pop 3, Pop 3
3 * 3 = 9
Stack: [9]

Result: 9
Expression: (2 + 1) * 3 = 9
```

### Example 3: Decode String Detailed

```
Input: "3[a2[c]]"

Process each character:

char='3': digit
current_num = 3

char='[': opening bracket
Push ("", 3) to stack
Reset: current_string="", current_num=0
Stack: [("", 3)]

char='a': letter
current_string = "a"

char='2': digit
current_num = 2

char='[': opening bracket
Push ("a", 2) to stack
Reset: current_string="", current_num=0
Stack: [("", 3), ("a", 2)]

char='c': letter
current_string = "c"

char=']': closing bracket
Pop ("a", 2)
current_string = "a" + "c" * 2 = "acc"
Stack: [("", 3)]

char=']': closing bracket
Pop ("", 3)
current_string = "" + "acc" * 3 = "accaccacc"
Stack: []

Result: "accaccacc"
```

## Edge Cases

### 1. Empty Stack
**Scenario:** Pop or peek from empty stack
**Challenge:** Will cause error
**Solution:**
```python
if stack:
    top = stack.pop()
else:
    # Handle empty case
```

### 2. Empty Input
**Scenario:** s = ""
**Challenge:** What to return?
**Solution:**
```python
if not s:
    return True  # or appropriate default
```

### 3. Only Opening Brackets
**Scenario:** s = "((("
**Challenge:** Stack not empty at end
**Solution:**
```python
return len(stack) == 0  # Will be False
```

### 4. Only Closing Brackets
**Scenario:** s = ")))"
**Challenge:** No matching opening
**Solution:**
```python
if char in closing and not stack:
    return False
```

### 5. Multi-digit Numbers
**Scenario:** "123[a]"
**Challenge:** Need to build complete number
**Solution:**
```python
num = num * 10 + int(char)  # Build number
```

### 6. Division by Zero
**Scenario:** ["1", "0", "/"]
**Challenge:** Division by zero
**Solution:**
```python
if operator == '/' and num == 0:
    # Handle error
```

## Common Pitfalls

### ❌ Pitfall 1: Not Checking Empty Stack Before Pop
**What happens:** Runtime error
**Why it's wrong:**
```python
# Wrong
char = mapping[closing_bracket]
top = stack.pop()  # Error if stack empty!
```
**Correct approach:**
```python
if stack:
    top = stack.pop()
else:
    return False  # or handle appropriately
```

### ❌ Pitfall 2: Forgetting to Check Stack Empty at End
**What happens:** False positives for invalid input
**Why it's wrong:**
```python
# Wrong - doesn't check if extra opening brackets remain
for char in s:
    # ... process
return True  # Wrong if stack not empty!
```
**Correct approach:**
```python
return len(stack) == 0  # or: return not stack
```

### ❌ Pitfall 3: Using Wrong Division for RPN
**What happens:** Incorrect results for negative division
**Why it's wrong:**
```python
# Wrong - Python's // truncates toward -infinity
result = left // right
```
**Correct approach:**
```python
# Correct - truncate toward zero
result = int(left / right)
```

### ❌ Pitfall 4: Building Multi-digit Numbers Incorrectly
**What happens:** "123" becomes 1, 2, 3 instead of 123
**Why it's wrong:**
```python
# Wrong
num = int(char)  # Resets each time!
```
**Correct approach:**
```python
num = num * 10 + int(char)  # Builds number
```

### ❌ Pitfall 5: Not Resetting State After Processing
**What happens:** Previous state affects current processing
**Why it's wrong:**
```python
# Wrong - num not reset after using
if char == '[':
    stack.append(num)
    # Forgot to reset num!
```
**Correct approach:**
```python
if char == '[':
    stack.append(num)
    num = 0  # Reset!
```

## Variations and Extensions

### Variation 1: Using List as Stack
**Description:** Python lists have stack operations
**Implementation:**
```python
stack = []
stack.append(x)  # Push
x = stack.pop()  # Pop
top = stack[-1]  # Peek
```

### Variation 2: Using collections.deque
**Description:** More efficient for some operations
**Implementation:**
```python
from collections import deque
stack = deque()
stack.append(x)  # Push
x = stack.pop()  # Pop
```

### Variation 3: Fixed-Size Stack
**Description:** Array with fixed capacity
**Implementation:**
```python
class FixedStack:
    def __init__(self, capacity):
        self.data = [None] * capacity
        self.top = -1
        self.capacity = capacity
```

### Variation 4: Recursive DFS (Implicit Stack)
**Description:** Recursion uses call stack
**When to use:** Tree/graph traversal
**Note:** Stack overflow risk with deep recursion

## Practice Problems

### Beginner
1. **Valid Parentheses (LeetCode #20)** - Classic stack problem
2. **Baseball Game (LeetCode #682)** - Stack with operations
3. **Remove Outermost Parentheses (LeetCode #1021)** - Stack variant
4. **Build Array With Stack Operations (LeetCode #1441)** - Simulation

### Intermediate
1. **Min Stack (LeetCode #155)** - Constant time minimum
2. **Evaluate Reverse Polish Notation (LeetCode #150)** - Expression evaluation
3. **Decode String (LeetCode #394)** - Nested structures
4. **Remove All Adjacent Duplicates (LeetCode #1047)** - String manipulation
5. **Simplify Path (LeetCode #71)** - Path processing
6. **Score of Parentheses (LeetCode #856)** - Nested scoring
7. **Remove K Digits (LeetCode #402)** - Stack optimization

### Advanced
1. **Basic Calculator (LeetCode #224)** - Full calculator
2. **Basic Calculator II (LeetCode #227)** - With operators
3. **Asteroid Collision (LeetCode #735)** - Complex simulation
4. **Exclusive Time of Functions (LeetCode #636)** - Function call simulation
5. **Validate Stack Sequences (LeetCode #946)** - Stack validation
6. **Largest Rectangle in Histogram (LeetCode #84)** - With monotonic stack

## Real-World Applications

### Industry Use Cases

1. **Function Call Stack:** Every program's execution uses a stack
   - Managing function calls and returns
   - Local variable storage
   - Return address tracking

2. **Undo/Redo Functionality:** Text editors, graphics software
   - Each action pushed to undo stack
   - Redo stack for undone actions
   - Examples: Word, Photoshop, IDEs

3. **Expression Evaluation:** Calculators, compilers
   - Infix to postfix conversion
   - Expression parsing
   - Syntax tree construction

4. **Browser History:** Back/forward navigation
   - Back button uses stack
   - Forward button when going back
   - Tab/window state management

5. **Parsing:** Compilers, interpreters
   - Syntax checking
   - Nested structure validation
   - HTML/XML parsing

### Popular Implementations

- **JVM/CLR:** Call stack for method invocation
- **Web Browsers:** Navigation history, JavaScript call stack
- **Compilers:** Symbol tables, syntax trees
- **Operating Systems:** Process stack, kernel stack
- **React:** Component rendering stack

### Practical Scenarios

- **Text editor:** Undo (Ctrl+Z) functionality
- **Calculator app:** Expression evaluation
- **Code editor:** Bracket matching, syntax highlighting
- **File system:** Path simplification
- **Game development:** Backtracking in puzzles

## Related Topics

### Prerequisites to Review
- **Arrays/Lists** - Understanding sequential storage
- **LIFO concept** - Last-In-First-Out principle
- **Recursion** - Related to call stack

### Next Steps
- **Monotonic Stack** - Advanced stack technique
- **Queue** - FIFO data structure
- **Recursion** - Implicit stack usage
- **DFS** - Stack-based graph traversal

### Similar Concepts
- **Queue** - FIFO instead of LIFO
- **Deque** - Double-ended queue
- **Call Stack** - Program execution stack
- **Recursion** - Uses call stack implicitly

### Further Reading
- [Stack - GeeksforGeeks](https://www.geeksforgeeks.org/stack-data-structure/)
- [LeetCode Stack Problems](https://leetcode.com/tag/stack/)
- [Stack Visualization](https://visualgo.net/en/list)
- Introduction to Algorithms (CLRS) - Stack and Queue chapter
