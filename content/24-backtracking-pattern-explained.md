# Backtracking Pattern

**Difficulty:** Intermediate to Advanced
**Prerequisites:** Recursion, Tree traversal, Basic combinatorics
**Estimated Reading Time:** 50 minutes

## Introduction

Backtracking is a systematic algorithmic technique for exploring all possible solutions to a problem by incrementally building candidates and abandoning ("backtracking from") candidates as soon as it determines they cannot lead to a valid solution. It's essentially an optimized brute force approach that prunes the search space.

**Why it matters:** Backtracking is the go-to approach for constraint satisfaction problems, puzzles, and combinatorial search problems. It appears in interview questions at every major tech company and forms the foundation for understanding more complex algorithms like branch-and-bound and alpha-beta pruning. Real-world applications include solving Sudoku, generating permutations and combinations, maze solving, and constraint satisfaction in AI systems.

**Real-world analogy:** Imagine exploring a maze. You walk down a path, and at each junction, you choose a direction. If you hit a dead end, you don't start over from the entrance—instead, you backtrack to the last junction and try a different path. You're systematically exploring all possibilities while avoiding paths you've already tried. That's backtracking! You build your path step by step, and when you realize a path won't work, you undo your last steps and try something else.

## Core Concepts

### Key Principles

1. **Incremental Solution Building:** Build the solution one piece at a time, making a choice at each step.

2. **Constraint Checking:** After each choice, check if the partial solution violates any constraints. If it does, abandon that path immediately.

3. **Backtrack on Failure:** When a partial solution cannot be completed, undo the last choice (backtrack) and try the next alternative.

4. **Recursion as the Engine:** Backtracking is typically implemented using recursion, where each recursive call represents a deeper level of choice-making.

5. **State Management:** Carefully maintain and restore state as you explore and backtrack.

### Essential Terms

- **Decision Tree:** A tree where each node represents a state, and edges represent choices. Backtracking traverses this tree.
- **Candidate Solution:** A partial or complete solution being built incrementally.
- **Constraint:** A condition that must be satisfied for a solution to be valid.
- **Pruning:** The act of abandoning a branch of the search tree early when it's clear it won't lead to a solution.
- **State:** The current configuration of the partial solution, including choices made so far.
- **Backtrack:** The act of undoing the last choice and returning to a previous state to try alternatives.

### Visual Overview

```
Decision Tree for Permutations of [1,2,3]:

                          []
                /          |          \
              [1]         [2]         [3]
             /  \        /  \        /  \
          [1,2][1,3]  [2,1][2,3]  [3,1][3,2]
           |     |      |     |      |     |
        [1,2,3][1,3,2][2,1,3][2,3,1][3,1,2][3,2,1]

At each level, we make a choice.
When we reach a leaf, we have a complete solution.
We backtrack to explore other branches.
```

## How It Works

### The Backtracking Algorithm Template

**Step 1: Base Case Check**
- If the current state represents a complete solution, save it and return.
- This is the recursion termination condition.

**Step 2: Iterate Through Choices**
- For each possible choice at the current step:
  - Make the choice (modify the state)
  - Check if the choice is valid
  - If valid, recurse to the next step
  - Unmake the choice (restore the state) - this is the "backtrack"

**Step 3: Pruning**
- Before recursing, check if the current partial solution can possibly lead to a valid complete solution.
- If not, skip the recursion (prune that branch).

### Detailed Walkthrough Example

**Problem:** Generate all subsets of [1, 2, 3]
**Output:** [[], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]]

```
Starting state: current = [], index = 0

Step 1: At index 0 (considering element 1)
  Option 1: Don't include 1
    current = [] → Recurse with index = 1
      At index 1 (considering element 2)
        Option 1: Don't include 2
          current = [] → Recurse with index = 2
            At index 2 (considering element 3)
              Option 1: Don't include 3
                current = [] → Recurse with index = 3
                  Base case reached! Save [] ✓
              Option 2: Include 3
                current = [3] → Recurse with index = 3
                  Base case reached! Save [3] ✓
                Backtrack: remove 3 from current
        Option 2: Include 2
          current = [2] → Recurse with index = 2
            At index 2 (considering element 3)
              Option 1: Don't include 3
                current = [2] → Recurse with index = 3
                  Base case reached! Save [2] ✓
              Option 2: Include 3
                current = [2,3] → Recurse with index = 3
                  Base case reached! Save [2,3] ✓
                Backtrack: remove 3 from current
            Backtrack: remove 2 from current
  
  Option 2: Include 1
    current = [1] → Recurse with index = 1
      At index 1 (considering element 2)
        Option 1: Don't include 2
          current = [1] → Recurse with index = 2
            At index 2 (considering element 3)
              Option 1: Don't include 3
                current = [1] → Recurse with index = 3
                  Base case reached! Save [1] ✓
              Option 2: Include 3
                current = [1,3] → Recurse with index = 3
                  Base case reached! Save [1,3] ✓
                Backtrack: remove 3 from current
        Option 2: Include 2
          current = [1,2] → Recurse with index = 2
            At index 2 (considering element 3)
              Option 1: Don't include 3
                current = [1,2] → Recurse with index = 3
                  Base case reached! Save [1,2] ✓
              Option 2: Include 3
                current = [1,2,3] → Recurse with index = 3
                  Base case reached! Save [1,2,3] ✓
                Backtrack: remove 3 from current
            Backtrack: remove 2 from current
      Backtrack: remove 1 from current

Final result: [[], [3], [2], [2,3], [1], [1,3], [1,2], [1,2,3]]
```

## Implementation

### Python Implementation - Backtracking Template

```python
from typing import List, Set

def backtracking_template(choices: List, constraints):
    """
    Generic backtracking template.
    
    This is the foundational structure for all backtracking problems.
    Customize the functions for specific problems.
    """
    def backtrack(state, choices_remaining):
        # Base case: complete solution
        if is_complete(state):
            result.append(state.copy())  # Save a copy!
            return
        
        # Try all possible choices
        for choice in get_choices(choices_remaining):
            if is_valid(state, choice, constraints):
                # Make choice
                make_choice(state, choice)
                
                # Recurse
                backtrack(state, update_choices(choices_remaining, choice))
                
                # Backtrack (undo choice)
                undo_choice(state, choice)
    
    result = []
    backtrack(initial_state, choices)
    return result


def subsets(nums: List[int]) -> List[List[int]]:
    """
    Generate all possible subsets (power set) of a list.
    
    This is the classic backtracking problem that demonstrates
    the pattern clearly.
    
    Args:
        nums: List of integers
        
    Returns:
        List of all possible subsets
        
    Time Complexity: O(n * 2ⁿ) - 2ⁿ subsets, each taking O(n) to copy
    Space Complexity: O(n) for recursion depth
    
    Example:
        >>> subsets([1, 2, 3])
        [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
    """
    result = []
    
    def backtrack(start: int, current: List[int]):
        # Every state is a valid subset, so save it
        result.append(current[:])  # Save a copy of current state
        
        # Try adding each remaining number
        for i in range(start, len(nums)):
            # Make choice: include nums[i]
            current.append(nums[i])
            
            # Recurse: explore subsets that include nums[i]
            backtrack(i + 1, current)
            
            # Backtrack: remove nums[i] to try other options
            current.pop()
    
    backtrack(0, [])
    return result


def permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all permutations of a list.
    
    Args:
        nums: List of distinct integers
        
    Returns:
        All possible permutations
        
    Time Complexity: O(n * n!) - n! permutations, each taking O(n) to build
    Space Complexity: O(n) for recursion depth
    
    Example:
        >>> permutations([1, 2, 3])
        [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
    """
    result = []
    
    def backtrack(current: List[int], remaining: Set[int]):
        # Base case: all numbers used
        if not remaining:
            result.append(current[:])
            return
        
        # Try each remaining number
        for num in list(remaining):  # Convert to list to avoid modification during iteration
            # Make choice: add num to permutation
            current.append(num)
            remaining.remove(num)
            
            # Recurse with updated state
            backtrack(current, remaining)
            
            # Backtrack: undo the choice
            current.pop()
            remaining.add(num)
    
    backtrack([], set(nums))
    return result


def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    """
    Find all unique combinations that sum to target.
    Numbers can be used multiple times.
    
    Args:
        candidates: List of distinct positive integers
        target: Target sum
        
    Returns:
        All unique combinations that sum to target
        
    Time Complexity: O(n^(target/min)) - exponential
    Space Complexity: O(target/min) for recursion depth
    
    Example:
        >>> combination_sum([2,3,6,7], 7)
        [[2,2,3], [7]]
    """
    result = []
    candidates.sort()  # Sort to enable pruning
    
    def backtrack(start: int, current: List[int], current_sum: int):
        # Base case: found target sum
        if current_sum == target:
            result.append(current[:])
            return
        
        # Pruning: if sum exceeds target, no point continuing
        if current_sum > target:
            return
        
        # Try each candidate starting from 'start'
        for i in range(start, len(candidates)):
            # Pruning: if current candidate is too large, all remaining are too
            if current_sum + candidates[i] > target:
                break
            
            # Make choice: include this candidate
            current.append(candidates[i])
            
            # Recurse: can reuse same number, so pass 'i' not 'i+1'
            backtrack(i, current, current_sum + candidates[i])
            
            # Backtrack
            current.pop()
    
    backtrack(0, [], 0)
    return result


def word_search(board: List[List[str]], word: str) -> bool:
    """
    Determine if word exists in the grid.
    
    The word can be constructed from letters of sequentially adjacent cells,
    where adjacent cells are horizontally or vertically neighboring.
    
    Args:
        board: 2D grid of characters
        word: Word to search for
        
    Returns:
        True if word exists in grid
        
    Time Complexity: O(m * n * 4^L) where L is word length
    Space Complexity: O(L) for recursion depth
    
    Example:
        >>> board = [['A','B','C','E'],['S','F','C','S'],['A','D','E','E']]
        >>> word_search(board, "ABCCED")
        True
    """
    rows, cols = len(board), len(board[0])
    
    def backtrack(row: int, col: int, index: int) -> bool:
        # Base case: found all characters
        if index == len(word):
            return True
        
        # Check bounds and character match
        if (row < 0 or row >= rows or col < 0 or col >= cols or
            board[row][col] != word[index]):
            return False
        
        # Mark cell as visited (make choice)
        temp = board[row][col]
        board[row][col] = '#'  # Mark as visited
        
        # Explore all 4 directions
        found = (backtrack(row + 1, col, index + 1) or
                 backtrack(row - 1, col, index + 1) or
                 backtrack(row, col + 1, index + 1) or
                 backtrack(row, col - 1, index + 1))
        
        # Backtrack: restore cell
        board[row][col] = temp
        
        return found
    
    # Try starting from each cell
    for i in range(rows):
        for j in range(cols):
            if backtrack(i, j, 0):
                return True
    
    return False


def solve_sudoku(board: List[List[str]]) -> None:
    """
    Solve Sudoku puzzle in-place.
    
    Args:
        board: 9x9 Sudoku board with empty cells marked as '.'
        
    Returns:
        None (modifies board in-place)
        
    Time Complexity: O(9^m) where m is number of empty cells
    Space Complexity: O(m) for recursion depth
    
    Example:
        >>> board = [["5","3",".",".","7",".",".",".","."],
                     ["6",".",".","1","9","5",".",".","."], ...]
        >>> solve_sudoku(board)
        # Board is now solved
    """
    def is_valid(row: int, col: int, num: str) -> bool:
        """Check if placing num at (row, col) is valid."""
        # Check row
        for j in range(9):
            if board[row][j] == num:
                return False
        
        # Check column
        for i in range(9):
            if board[i][col] == num:
                return False
        
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False
        
        return True
    
    def backtrack() -> bool:
        """Try to fill the board."""
        # Find next empty cell
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    # Try each digit 1-9
                    for num in '123456789':
                        if is_valid(i, j, num):
                            # Make choice
                            board[i][j] = num
                            
                            # Recurse
                            if backtrack():
                                return True
                            
                            # Backtrack
                            board[i][j] = '.'
                    
                    # No valid digit found, backtrack
                    return False
        
        # No empty cells, puzzle solved!
        return True
    
    backtrack()


def max_unique_substrings(s: str) -> int:
    """
    Split string into maximum number of unique substrings.
    
    Args:
        s: Input string
        
    Returns:
        Maximum number of unique substrings possible
        
    Time Complexity: O(2^n) - exponential
    Space Complexity: O(n) for recursion and set
    
    Example:
        >>> max_unique_substrings("ababccc")
        5  # ["a", "b", "ab", "c", "cc"]
    """
    def backtrack(start: int, seen: Set[str]) -> int:
        # Base case: reached end of string
        if start == len(s):
            return 0
        
        max_count = 0
        
        # Try all possible substrings starting at 'start'
        for end in range(start + 1, len(s) + 1):
            substring = s[start:end]
            
            # Only use if not seen before
            if substring not in seen:
                # Make choice
                seen.add(substring)
                
                # Recurse and get count
                count = 1 + backtrack(end, seen)
                max_count = max(max_count, count)
                
                # Backtrack
                seen.remove(substring)
        
        return max_count
    
    return backtrack(0, set())


# Example usage
if __name__ == "__main__":
    print("=== Subsets ===")
    print(subsets([1, 2, 3]))
    print()
    
    print("=== Permutations ===")
    print(permutations([1, 2, 3]))
    print()
    
    print("=== Combination Sum ===")
    print(combination_sum([2, 3, 6, 7], 7))
    print()
    
    print("=== Word Search ===")
    board = [['A','B','C','E'],['S','F','C','S'],['A','D','E','E']]
    print(word_search(board, "ABCCED"))
    print()
    
    print("=== Max Unique Substrings ===")
    print(max_unique_substrings("ababccc"))
```

### Code Explanation

**Key Design Patterns:**

1. **State Management:** Each function maintains state (current solution being built) and carefully saves/restores it during backtracking.

2. **Choice and Unchoice:** The pattern of "make choice → recurse → undo choice" is universal in backtracking.

3. **Base Case:** All functions check for completion condition first.

4. **Pruning:** Functions like `combination_sum` demonstrate early termination when a path can't succeed.

5. **Pass by Reference vs Copy:** When saving results, we always copy (`current[:]`) to avoid reference issues.

## Complexity Analysis

### Time Complexity

**Subsets:**
- **Time:** O(n * 2ⁿ)
- **Why?** There are 2ⁿ possible subsets (each element can be included or excluded). Each subset takes O(n) time to copy into the result.

**Permutations:**
- **Time:** O(n * n!)
- **Why?** There are n! permutations. Each permutation takes O(n) time to build and copy.

**Combination Sum:**
- **Time:** O(n^(target/min))
- **Why?** In the worst case, if the minimum candidate is 1, we could have a recursion tree of depth 'target'. At each level, we can branch n ways, leading to exponential complexity.

**Word Search:**
- **Time:** O(m * n * 4^L) where m×n is board size, L is word length
- **Why?** We try starting from each cell (m×n). From each cell, we can go 4 directions, and we do this for up to L characters.

**Sudoku Solver:**
- **Time:** O(9^m) where m is number of empty cells
- **Why?** For each empty cell, we try up to 9 digits. This creates a tree with branching factor 9 and depth m.

**Max Unique Substrings:**
- **Time:** O(2^n)
- **Why?** At each position, we can split or continue, leading to 2^n possible ways to partition the string.

### Space Complexity

All backtracking solutions have **O(depth)** space complexity for the recursion stack, where depth is the maximum recursion depth.

- **Subsets:** O(n) - maximum depth is n
- **Permutations:** O(n) - maximum depth is n
- **Combination Sum:** O(target/min) - depth depends on how many numbers fit in target
- **Word Search:** O(L) - depth is word length
- **Sudoku:** O(m) - depth is number of empty cells
- **Max Unique Substrings:** O(n) - depth is string length

### Comparison with Alternatives

| Approach | Time Complexity | Space | When to Use |
|----------|----------------|-------|-------------|
| **Backtracking** | Exponential (2ⁿ, n!, etc.) | O(depth) | When you need all solutions or optimal solution from constrained search space |
| **Dynamic Programming** | Polynomial (often O(n²)) | O(n) or O(n²) | When problem has optimal substructure and overlapping subproblems |
| **Greedy** | Linear/Polynomial | O(1) or O(n) | When local optimal leads to global optimal |
| **Brute Force** | Exponential (no pruning) | O(1) or O(n) | Never - backtracking is always better |
| **Branch and Bound** | Exponential (optimized) | O(depth) | For optimization problems with bounds |

## Examples

### Example 1: Subsets of [1,2,3]

**Input:** nums = [1,2,3]
**Output:** [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]

**Detailed Trace:**

```
Call Stack Visualization:

backtrack(start=0, current=[])
├─ Save [] (before any additions)
├─ i=0: current=[1]
│  ├─ backtrack(start=1, current=[1])
│  │  ├─ Save [1]
│  │  ├─ i=1: current=[1,2]
│  │  │  ├─ backtrack(start=2, current=[1,2])
│  │  │  │  ├─ Save [1,2]
│  │  │  │  ├─ i=2: current=[1,2,3]
│  │  │  │  │  ├─ backtrack(start=3, current=[1,2,3])
│  │  │  │  │  │  └─ Save [1,2,3] (start >= len, base case)
│  │  │  │  │  └─ pop() → current=[1,2]
│  │  │  │  └─ return
│  │  │  └─ pop() → current=[1]
│  │  ├─ i=2: current=[1,3]
│  │  │  ├─ backtrack(start=3, current=[1,3])
│  │  │  │  └─ Save [1,3]
│  │  │  └─ pop() → current=[1]
│  │  └─ return
│  └─ pop() → current=[]
├─ i=1: current=[2]
│  ├─ backtrack(start=2, current=[2])
│  │  ├─ Save [2]
│  │  ├─ i=2: current=[2,3]
│  │  │  ├─ backtrack(start=3, current=[2,3])
│  │  │  │  └─ Save [2,3]
│  │  │  └─ pop() → current=[2]
│  │  └─ return
│  └─ pop() → current=[]
├─ i=2: current=[3]
│  ├─ backtrack(start=3, current=[3])
│  │  └─ Save [3]
│  └─ pop() → current=[]
└─ return

Result: [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
```

### Example 2: Combination Sum - candidates=[2,3,6,7], target=7

**Input:** candidates = [2,3,6,7], target = 7
**Output:** [[2,2,3], [7]]

**Step-by-step:**

```
After sorting: [2, 3, 6, 7]

backtrack(start=0, current=[], sum=0)
├─ Try 2: current=[2], sum=2
│  ├─ backtrack(start=0, current=[2], sum=2)
│  │  ├─ Try 2: current=[2,2], sum=4
│  │  │  ├─ backtrack(start=0, current=[2,2], sum=4)
│  │  │  │  ├─ Try 2: current=[2,2,2], sum=6
│  │  │  │  │  ├─ backtrack(start=0, current=[2,2,2], sum=6)
│  │  │  │  │  │  ├─ Try 2: sum would be 8 > 7, BREAK pruning
│  │  │  │  │  │  └─ return (no solutions)
│  │  │  │  │  └─ pop() → current=[2,2]
│  │  │  │  ├─ Try 3: current=[2,2,3], sum=7 ✓ FOUND!
│  │  │  │  │  └─ Save [2,2,3]
│  │  │  │  └─ pop() → current=[2,2]
│  │  │  └─ pop() → current=[2]
│  │  ├─ Try 3: current=[2,3], sum=5
│  │  │  ├─ backtrack(start=1, current=[2,3], sum=5)
│  │  │  │  ├─ Try 3: sum would be 8 > 7, BREAK
│  │  │  │  └─ return
│  │  │  └─ pop() → current=[2]
│  │  └─ (6 and 7 would exceed target, pruned)
│  └─ pop() → current=[]
├─ Try 3: current=[3], sum=3
│  └─ (similar exploration, no valid solutions)
├─ Try 6: current=[6], sum=6
│  └─ (only 7 would work but 6+7 > 7)
├─ Try 7: current=[7], sum=7 ✓ FOUND!
│  └─ Save [7]
└─ return

Result: [[2,2,3], [7]]
```

### Example 3: Word Search - board with word "ABCCED"

**Input:** 
```
board = [['A','B','C','E'],
         ['S','F','C','S'],
         ['A','D','E','E']]
word = "ABCCED"
```
**Output:** True

**Trace:**

```
Starting searches from each cell...

Try (0,0) 'A':
  A matches word[0] ✓
  Mark (0,0) as visited (#)
  
  Try neighbors for 'B' (word[1]):
    Down (1,0)='S' ✗
    Right (0,1)='B' ✓
      Mark (0,1) as visited
      
      Try neighbors for 'C' (word[2]):
        Down (1,1)='F' ✗
        Right (0,2)='C' ✓
          Mark (0,2) as visited
          
          Try neighbors for 'C' (word[3]):
            Down (1,2)='C' ✓
              Mark (1,2) as visited
              
              Try neighbors for 'E' (word[4]):
                Down (2,2)='E' ✓
                  Mark (2,2) as visited
                  
                  Try neighbors for 'D' (word[5]):
                    Right (2,3)='E' ✗
                    Left (2,1)='D' ✓
                      Mark (2,1) as visited
                      
                      index=6 == len(word) ✓
                      FOUND! Return True

Backtracking happens but we already found the word.
Final answer: True
```

### Example 4: Sudoku Solver (Simplified Example)

**Input:** Partial 4x4 board (simplified from 9x9)
```
board = [['5','3','.','.'],
         ['6','.','.','.''],
         ['.','9','8','.'],
         ['.','.','.','6']]
```

**Trace (first few steps):**

```
Find first empty cell: (0,2)

Try digits 1-4 for (0,2):
  Try '1': Is valid? Check row, column, 2x2 box
    Row 0 has 5,3 → '1' OK
    Col 2 has 8 → '1' OK
    Box (0,0) has 5,3,6 → '1' OK ✓
    Place '1' at (0,2)
    
    Find next empty: (0,3)
    Try digits 1-4 for (0,3):
      Try '1': Already in row 0 ✗
      Try '2': 
        Row check OK, Col check OK, Box check OK ✓
        Place '2' at (0,3)
        
        Find next empty: (1,1)
        Try digits 1-4 for (1,1):
          ... and so on ...
          
If at any point no digit works, backtrack:
  Remove last placed digit
  Try next digit
  Continue...
  
Eventually find complete valid solution or determine no solution exists.
```

## Edge Cases

### 1. Empty Input
**Scenario:** Empty array/string/board
**Challenge:** May need special handling for base case
**Solution:** Return appropriate empty result
**Code example:**
```python
if not nums:
    return [[]]  # For subsets, empty set of empty set
    # or return [] for most other problems
```

### 2. Single Element
**Scenario:** Input with only one element
**Challenge:** Simplest non-trivial case
**Solution:** Should work naturally with recursion base case
**Code example:**
```python
# For subsets([1]):
# Should return [[], [1]]
# Base case handles this naturally
```

### 3. All Duplicates
**Scenario:** Input like [1,1,1,1]
**Challenge:** Need to avoid duplicate solutions
**Solution:** Skip duplicates during choice-making or use set
**Code example:**
```python
# For permutations with duplicates:
for i in range(len(nums)):
    if i > 0 and nums[i] == nums[i-1]:
        continue  # Skip duplicate
    # ... rest of logic
```

### 4. No Valid Solution
**Scenario:** Constraints cannot be satisfied
**Challenge:** Need to return empty result, not error
**Solution:** Backtracking naturally handles this by exhausting all paths
**Code example:**
```python
# If no path leads to solution:
# result list remains empty
# return []
```

### 5. Multiple Valid Solutions
**Scenario:** Many solutions exist
**Challenge:** May want first solution or all solutions
**Solution:** For first, return immediately; for all, collect in list
**Code example:**
```python
# For first solution only:
def backtrack(...):
    if is_complete():
        return True  # Stop immediately
    
# For all solutions:
def backtrack(...):
    if is_complete():
        results.append(current.copy())
        return  # Continue searching
```

### 6. Large Search Space
**Scenario:** Exponential number of possibilities
**Challenge:** May timeout
**Solution:** Aggressive pruning, early termination, or switch algorithm
**Code example:**
```python
# Add pruning:
if current_sum > target:
    return  # Don't recurse further
if lower_bound > best_so_far:
    return  # Branch and bound optimization
```

### 7. State Restoration Issues
**Scenario:** Forgetting to backtrack properly
**Challenge:** State gets corrupted
**Solution:** Always undo changes or use copies
**Code example:**
```python
# WRONG:
current.append(choice)
backtrack(current)
# Forgot to pop()!

# CORRECT:
current.append(choice)
backtrack(current)
current.pop()  # Always restore state
```

## Common Pitfalls

### ❌ Pitfall 1: Not Copying When Saving Solutions
**What happens:** All results reference the same list object
**Why it's wrong:** When you backtrack and modify the list, all "saved" solutions change
**Correct approach:**
```python
# WRONG:
if is_complete():
    result.append(current)  # Saves reference!

# CORRECT:
if is_complete():
    result.append(current[:])  # or current.copy()
    # This saves a copy, not the reference
```

### ❌ Pitfall 2: Modifying Loop Variable
**What happens:** Iterating over something you're modifying
**Why it's wrong:** Can skip elements or cause errors
**Correct approach:**
```python
# WRONG:
for num in remaining:
    remaining.remove(num)  # Modifying while iterating!
    ...

# CORRECT:
for num in list(remaining):  # Iterate over copy
    remaining.remove(num)
    ...
    remaining.add(num)  # Restore
```

### ❌ Pitfall 3: Forgetting to Backtrack
**What happens:** State isn't restored, future paths use corrupted state
**Why it's wrong:** Breaks the fundamental backtracking pattern
**Correct approach:**
```python
# WRONG:
board[i][j] = num
if backtrack():
    return True
# Forgot to restore board[i][j]!

# CORRECT:
board[i][j] = num
if backtrack():
    return True
board[i][j] = '.'  # Always restore
```

### ❌ Pitfall 4: Wrong Base Case
**What happens:** Infinite recursion or missing solutions
**Why it's wrong:** Base case defines when to stop and save
**Correct approach:**
```python
# WRONG:
if len(current) == n:  # But n might be wrong
    result.append(current[:])

# CORRECT:
if start == len(nums):  # Use appropriate stopping condition
    result.append(current[:])
    return
```

### ❌ Pitfall 5: Not Pruning When Possible
**What happens:** Exploring paths that can't possibly lead to solution
**Why it's wrong:** Wastes time on futile branches
**Correct approach:**
```python
# WRONG: No pruning
for i in range(start, len(candidates)):
    current.append(candidates[i])
    backtrack(i, current, current_sum + candidates[i])
    current.pop()

# CORRECT: Prune when sum exceeds target
for i in range(start, len(candidates)):
    if current_sum + candidates[i] > target:
        break  # No point trying larger candidates
    current.append(candidates[i])
    backtrack(i, current, current_sum + candidates[i])
    current.pop()
```

### ❌ Pitfall 6: Incorrect Parameter Passing
**What happens:** Passing wrong index or not updating properly
**Why it's wrong:** Revisits same choices or skips valid ones
**Correct approach:**
```python
# For subsets (can't reuse elements):
backtrack(i + 1, current)  # Start from next index

# For combination sum (can reuse):
backtrack(i, current, new_sum)  # Can use same index again

# For permutations (need all remaining):
backtrack(current, remaining - {num})  # Exclude current choice
```

### ❌ Pitfall 7: Not Handling Duplicates
**What happens:** Generates duplicate solutions
**Why it's wrong:** Violates problem constraints
**Correct approach:**
```python
# When input has duplicates:
nums.sort()  # Sort first
for i in range(start, len(nums)):
    # Skip duplicates
    if i > start and nums[i] == nums[i-1]:
        continue
    current.append(nums[i])
    backtrack(i + 1, current)
    current.pop()
```

## Variations and Extensions

### Variation 1: Subsets with Duplicates
**Description:** Generate subsets when input contains duplicate elements
**When to use:** LeetCode #90
**Key differences:** Must skip duplicate elements at same recursion level
**Implementation:**
```python
def subsets_with_dup(nums: List[int]) -> List[List[int]]:
    """
    Generate subsets from array with duplicates.
    
    Time: O(n * 2ⁿ)
    Space: O(n)
    """
    nums.sort()  # Sort to group duplicates
    result = []
    
    def backtrack(start, current):
        result.append(current[:])
        
        for i in range(start, len(nums)):
            # Skip duplicates at same level
            if i > start and nums[i] == nums[i-1]:
                continue
            
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()
    
    backtrack(0, [])
    return result
```

### Variation 2: Permutations with Duplicates
**Description:** Generate permutations when input has duplicate elements
**When to use:** LeetCode #47
**Key differences:** Use frequency map to track usage
**Implementation:**
```python
def permute_unique(nums: List[int]) -> List[List[int]]:
    """
    Generate unique permutations from array with duplicates.
    
    Time: O(n * n!)
    Space: O(n)
    """
    from collections import Counter
    
    result = []
    counter = Counter(nums)
    
    def backtrack(current):
        if len(current) == len(nums):
            result.append(current[:])
            return
        
        for num in counter:
            if counter[num] > 0:
                current.append(num)
                counter[num] -= 1
                
                backtrack(current)
                
                current.pop()
                counter[num] += 1
    
    backtrack([])
    return result
```

### Variation 3: Combination Sum II (No Reuse)
**Description:** Combination sum where each number used once
**When to use:** LeetCode #40
**Key differences:** Pass `i+1` instead of `i`, handle duplicates
**Implementation:**
```python
def combination_sum2(candidates: List[int], target: int) -> List[List[int]]:
    """
    Find combinations that sum to target (each number used once).
    
    Time: O(2^n)
    Space: O(n)
    """
    candidates.sort()
    result = []
    
    def backtrack(start, current, current_sum):
        if current_sum == target:
            result.append(current[:])
            return
        
        if current_sum > target:
            return
        
        for i in range(start, len(candidates)):
            # Skip duplicates at same level
            if i > start and candidates[i] == candidates[i-1]:
                continue
            
            if current_sum + candidates[i] > target:
                break
            
            current.append(candidates[i])
            backtrack(i + 1, current, current_sum + candidates[i])  # i+1, not i
            current.pop()
    
    backtrack(0, [], 0)
    return result
```

### Variation 4: N-Queens
**Description:** Place N queens on N×N board so none attack each other
**When to use:** Classic backtracking problem, LeetCode #51
**Key differences:** 2D board state, complex validity checking
**Implementation:**
```python
def solve_n_queens(n: int) -> List[List[str]]:
    """
    Solve N-Queens puzzle.
    
    Time: O(n!)
    Space: O(n²)
    """
    def is_valid(row, col):
        # Check column
        for i in range(row):
            if board[i][col] == 'Q':
                return False
        
        # Check diagonal (top-left to current)
        i, j = row - 1, col - 1
        while i >= 0 and j >= 0:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j -= 1
        
        # Check anti-diagonal (top-right to current)
        i, j = row - 1, col + 1
        while i >= 0 and j < n:
            if board[i][j] == 'Q':
                return False
            i -= 1
            j += 1
        
        return True
    
    def backtrack(row):
        if row == n:
            result.append([''.join(r) for r in board])
            return
        
        for col in range(n):
            if is_valid(row, col):
                board[row][col] = 'Q'
                backtrack(row + 1)
                board[row][col] = '.'
    
    result = []
    board = [['.' for _ in range(n)] for _ in range(n)]
    backtrack(0)
    return result
```

### Variation 5: Palindrome Partitioning
**Description:** Partition string into all palindromic substrings
**When to use:** LeetCode #131
**Key differences:** String partitioning with palindrome constraint
**Implementation:**
```python
def partition_palindrome(s: str) -> List[List[str]]:
    """
    Find all palindrome partitions.
    
    Time: O(n * 2^n)
    Space: O(n)
    """
    def is_palindrome(sub):
        return sub == sub[::-1]
    
    def backtrack(start, current):
        if start == len(s):
            result.append(current[:])
            return
        
        for end in range(start + 1, len(s) + 1):
            substring = s[start:end]
            if is_palindrome(substring):
                current.append(substring)
                backtrack(end, current)
                current.pop()
    
    result = []
    backtrack(0, [])
    return result
```

### Variation 6: Generate Parentheses
**Description:** Generate all valid combinations of n pairs of parentheses
**When to use:** LeetCode #22
**Key differences:** Track count of open/close parentheses
**Implementation:**
```python
def generate_parentheses(n: int) -> List[str]:
    """
    Generate all valid parentheses combinations.
    
    Time: O(4^n / √n) - Catalan number
    Space: O(n)
    """
    def backtrack(current, open_count, close_count):
        if len(current) == 2 * n:
            result.append(current)
            return
        
        if open_count < n:
            backtrack(current + '(', open_count + 1, close_count)
        
        if close_count < open_count:
            backtrack(current + ')', open_count, close_count + 1)
    
    result = []
    backtrack('', 0, 0)
    return result
```

## Practice Problems

### Beginner
1. **Subsets** - Generate all subsets of a set
   - LeetCode #78

2. **Permutations** - Generate all permutations
   - LeetCode #46

3. **Combination Sum III** - Find k numbers that add up to n
   - LeetCode #216

4. **Letter Combinations of Phone Number** - Generate all letter combinations
   - LeetCode #17

### Intermediate
1. **Combination Sum** - Numbers can be reused
   - LeetCode #39

2. **Combination Sum II** - Each number used once
   - LeetCode #40

3. **Word Search** - Find word in 2D grid
   - LeetCode #79

4. **Palindrome Partitioning** - All palindrome partitions
   - LeetCode #131

5. **Factor Combinations** - Find all factor combinations of a number
   - LeetCode #254 (Premium)

6. **Split String into Max Unique Substrings** - Maximize unique substrings
   - LeetCode #1593

7. **Generate Parentheses** - Valid parenthesis combinations
   - LeetCode #22

8. **Subsets II** - Subsets with duplicates
   - LeetCode #90

9. **Permutations II** - Permutations with duplicates
   - LeetCode #47

### Advanced
1. **Sudoku Solver** - Solve 9×9 Sudoku
   - LeetCode #37

2. **N-Queens** - Place N queens on N×N board
   - LeetCode #51

3. **N-Queens II** - Count number of solutions
   - LeetCode #52

4. **Word Search II** - Find all words from dictionary in grid
   - LeetCode #212

5. **Remove Invalid Parentheses** - Minimum removals to make valid
   - LeetCode #301

6. **Expression Add Operators** - Add operators to make target
   - LeetCode #282

7. **Regular Expression Matching** - Advanced pattern matching
   - LeetCode #10

## Real-World Applications

### Industry Use Cases

1. **Constraint Satisfaction Problems:** Scheduling (airline crews, university courses), resource allocation, timetabling.

2. **Game AI:** Chess engines use backtracking with alpha-beta pruning to explore game trees.

3. **Circuit Design:** VLSI design uses backtracking for wire routing and component placement.

4. **Natural Language Processing:** Parsing ambiguous grammar rules to find valid parse trees.

5. **Configuration Management:** Finding valid product configurations with compatibility constraints.

### Popular Implementations

- **Sudoku Solvers:** Mobile games and puzzle apps
- **Constraint Solvers (Z3, MiniZinc):** Use advanced backtracking for verification and optimization
- **Prolog Programming Language:** Built on backtracking as core execution model
- **SAT Solvers:** Modern SAT solvers use CDCL (Conflict-Driven Clause Learning), an evolution of backtracking

### Practical Scenarios

- **Network Configuration:** Finding valid network topologies under constraints
- **Puzzle Solvers:** Crossword generators, maze solvers
- **Resource Scheduling:** Employee shift scheduling, meeting room allocation
- **Manufacturing:** Finding valid assembly sequences
- **Logistics:** Vehicle routing with constraints

## Related Topics

### Prerequisites to Review
- **Recursion** - Backtracking is built on recursive calls
- **Tree Traversal (DFS)** - Backtracking is essentially DFS on implicit decision tree
- **Stack Data Structure** - Understanding call stack helps understand backtracking
- **Basic Combinatorics** - Helps understand complexity of search space

### Next Steps
- **Branch and Bound** - Optimization-focused backtracking variant
- **Dynamic Programming** - Often more efficient alternative when applicable
- **Constraint Propagation** - Technique to reduce search space in CSP
- **Memoization** - Combine with backtracking for overlapping subproblems
- **Alpha-Beta Pruning** - Specialized backtracking for game trees

### Similar Concepts
- **Depth-First Search (DFS)** - Backtracking is DFS with constraint checking
- **Divide and Conquer** - Both break problem into subproblems
- **Exhaustive Search** - Backtracking is optimized exhaustive search
- **State Space Search** - General framework that includes backtracking

### Further Reading
- "Algorithm Design Manual" by Skiena - Chapter on Backtracking
- "Introduction to Algorithms" (CLRS) - Dynamic Programming and Greedy chapters
- [Backtracking - GeeksforGeeks](https://www.geeksforgeeks.org/backtracking-algorithms/)
- [LeetCode Backtracking Patterns](https://leetcode.com/discuss/general-discussion/1072437/backtracking-questions-solutions)
- "The Art of Computer Programming Vol 4" by Knuth - Combinatorial algorithms
- "Programming Challenges" by Skiena & Revilla - Practice problems
