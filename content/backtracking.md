# Backtracking

**Difficulty:** Intermediate  
**Prerequisites:** Recursion, Arrays, Basic tree concepts  
**Estimated Reading Time:** 25 minutes

---

## Introduction

Backtracking is a systematic algorithmic technique for exploring all possible solutions to a problem by incrementally building candidates and abandoning a candidate ("backtracking") as soon as it determines that the candidate cannot lead to a valid solution.

**Why it matters:** Backtracking is the foundation for solving constraint satisfaction problems, puzzles, and combinatorial optimization challenges. It's used in everything from Sudoku solvers and chess engines to scheduling systems and AI planning algorithms. Many NP-complete problems that appear unsolvable can be tackled effectively with backtracking.

**Real-world analogy:** Imagine navigating a maze. You walk down a path, and when you hit a dead end, you don't give up entirely. Instead, you retrace your steps back to the last intersection where you had a choice, then try a different path. You keep doing this—exploring new paths and backtracking from dead ends—until you find the exit. That's exactly how backtracking works: try a possibility, and if it doesn't work out, undo your choice and try something else.

---

## Core Concepts

### Key Principles

1. **Incremental construction:** Build solutions one piece at a time, making choices at each step.

2. **Constraint checking:** After each choice, verify if we're still on a valid path. If not, backtrack immediately rather than wasting time exploring invalid branches.

3. **State restoration:** When backtracking, we must undo our previous choice to restore the state before trying the next option.

4. **Exhaustive search with pruning:** Explore all possibilities systematically, but prune (cut off) branches that can't lead to valid solutions.

### Essential Terms

- **Candidate solution:** A partial or complete solution being built incrementally
- **Valid solution:** A candidate that satisfies all constraints
- **Decision tree:** A tree where each node represents a choice, and paths from root to leaf represent potential solutions
- **Pruning:** Eliminating branches of the decision tree that cannot lead to valid solutions
- **State:** The current configuration of the partial solution
- **Backtrack:** Undo the last choice and try a different option

### Visual Overview

```
Decision Tree for Finding All Subsets of [1,2,3]:

                         []
                    /          \
                 [1]            []
               /    \          /   \
            [1,2]  [1]      [2]    []
           /   \   /  \    /  \   /  \
        [1,2,3][1,2][1,3][1] [2,3][2][3][]

At each node, we make a choice: include the next element or skip it.
When we reach a leaf, we have a complete solution.
We backtrack to explore other branches.
```

---

## How It Works

Backtracking follows a general template that applies to most problems:

### Algorithm Steps

1. **Choose:** Select an unexplored option from the current state
2. **Explore:** Recursively explore the consequences of that choice
3. **Unchoose (Backtrack):** Undo the choice to restore the previous state
4. **Repeat:** Try the next option until all possibilities are exhausted

### Detailed Process

```
State Trace for Generating Permutations of [1,2,3]:

Start: path=[], choices=[1,2,3]

Choose 1: path=[1], choices=[2,3]
  Choose 2: path=[1,2], choices=[3]
    Choose 3: path=[1,2,3], choices=[]
    ✓ Found solution: [1,2,3]
    Backtrack: path=[1,2], choices=[3]
  Backtrack: path=[1], choices=[2,3]

  Choose 3: path=[1,3], choices=[2]
    Choose 2: path=[1,3,2], choices=[]
    ✓ Found solution: [1,3,2]
    Backtrack: path=[1,3], choices=[2]
  Backtrack: path=[1], choices=[2,3]
Backtrack: path=[], choices=[1,2,3]

Choose 2: path=[2], choices=[1,3]
  Choose 1: path=[2,1], choices=[3]
    Choose 3: path=[2,1,3], choices=[]
    ✓ Found solution: [2,1,3]
    ... (continues for all permutations)
```

### Visual Walkthrough

```
Example: N-Queens (4x4 board)

Initial state (empty board):
. . . .
. . . .
. . . .
. . . .

Step 1: Place queen in row 0, column 1:
. Q . .
. . . .
. . . .
. . . .

Step 2: Try row 1, column 3 (column 0,1,2 would be attacked):
. Q . .
. . . Q
. . . .
. . . .

Step 3: Try row 2, column 0 (attacked by row 0!)
❌ BACKTRACK - no valid position in row 2

Backtrack to row 1, try next option...
(This continues until a valid 4-queens solution is found)
```

---

## Implementation

### Python Implementation

```python
from typing import List, Set, Callable, Any, Optional

def backtrack_template(
    result: List[Any],
    path: List[Any],
    choices: List[Any],
    is_valid: Callable[[List[Any], Any], bool],
    is_complete: Callable[[List[Any]], bool]
) -> None:
    """
    Generic backtracking template that can solve various problems.

    Args:
        result: List to store all valid solutions
        path: Current partial solution being built
        choices: Available options to choose from
        is_valid: Function to check if adding a choice keeps solution valid
        is_complete: Function to check if current path is a complete solution

    Time Complexity: O(b^d) where b is branching factor, d is depth
    Space Complexity: O(d) for recursion stack
    """
    # Base case: if we have a complete solution, save it
    if is_complete(path):
        result.append(path[:])  # Make a copy of the current path
        return

    # Explore all choices
    for choice in choices:
        # Prune: skip invalid choices early
        if not is_valid(path, choice):
            continue

        # Choose: add this option to current path
        path.append(choice)

        # Explore: recursively build on this choice
        backtrack_template(result, path, choices, is_valid, is_complete)

        # Unchoose (Backtrack): remove the choice to try next option
        path.pop()


def generate_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all permutations of a list using backtracking.

    Args:
        nums: List of unique integers

    Returns:
        List of all possible permutations

    Time Complexity: O(n! * n) - n! permutations, each takes O(n) to build
    Space Complexity: O(n) for recursion depth

    Example:
        >>> generate_permutations([1, 2, 3])
        [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
    """
    result = []

    def backtrack(path: List[int], used: Set[int]) -> None:
        # Base case: permutation is complete
        if len(path) == len(nums):
            result.append(path[:])
            return

        # Try each number that hasn't been used yet
        for num in nums:
            if num in used:
                continue  # Skip already used numbers

            # Choose: mark this number as used
            path.append(num)
            used.add(num)

            # Explore: continue building permutation
            backtrack(path, used)

            # Unchoose: backtrack for next iteration
            path.pop()
            used.remove(num)

    backtrack([], set())
    return result


def solve_n_queens(n: int) -> List[List[str]]:
    """
    Solve the N-Queens problem using backtracking.

    Place n queens on an n×n chessboard so no two queens attack each other.

    Args:
        n: Size of the board and number of queens

    Returns:
        List of all valid board configurations

    Time Complexity: O(n!) - trying n positions in first row, n-2 in second, etc.
    Space Complexity: O(n²) for the board

    Example:
        >>> solve_n_queens(4)
        [[".Q..","...Q","Q...","..Q."], ["..Q.","Q...","...Q",".Q.."]]
    """
    result = []
    board = [['.'] * n for _ in range(n)]  # Initialize empty board

    # Sets to track attacked columns and diagonals
    cols = set()  # Columns with queens
    diag1 = set()  # Diagonals (row - col is constant)
    diag2 = set()  # Anti-diagonals (row + col is constant)

    def backtrack(row: int) -> None:
        # Base case: all queens placed successfully
        if row == n:
            # Convert board to required string format
            result.append([''.join(row) for row in board])
            return

        # Try placing queen in each column of current row
        for col in range(n):
            # Check if this position is under attack
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue  # Skip invalid positions (pruning!)

            # Choose: place queen at (row, col)
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)

            # Explore: move to next row
            backtrack(row + 1)

            # Unchoose: remove queen (backtrack)
            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    backtrack(0)
    return result


def generate_subsets(nums: List[int]) -> List[List[int]]:
    """
    Generate all possible subsets (the power set) using backtracking.

    Args:
        nums: List of unique integers

    Returns:
        List of all possible subsets

    Time Complexity: O(2^n * n) - 2^n subsets, each takes O(n) to copy
    Space Complexity: O(n) for recursion depth

    Example:
        >>> generate_subsets([1, 2, 3])
        [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
    """
    result = []

    def backtrack(start: int, path: List[int]) -> None:
        # Every partial solution is a valid subset
        result.append(path[:])

        # Try adding each remaining element
        for i in range(start, len(nums)):
            # Choose: include nums[i]
            path.append(nums[i])

            # Explore: continue from next index to avoid duplicates
            backtrack(i + 1, path)

            # Unchoose: backtrack
            path.pop()

    backtrack(0, [])
    return result


def combination_sum(candidates: List[int], target: int) -> List[List[int]]:
    """
    Find all unique combinations where candidates sum to target.
    Same number can be used multiple times.

    Args:
        candidates: List of distinct positive integers
        target: Target sum

    Returns:
        List of all unique combinations that sum to target

    Time Complexity: O(n^(target/min)) where n is number of candidates
    Space Complexity: O(target/min) for recursion depth

    Example:
        >>> combination_sum([2,3,6,7], 7)
        [[2,2,3], [7]]
    """
    result = []

    def backtrack(start: int, path: List[int], remaining: int) -> None:
        # Base case: found valid combination
        if remaining == 0:
            result.append(path[:])
            return

        # Base case: exceeded target
        if remaining < 0:
            return

        # Try each candidate starting from 'start'
        for i in range(start, len(candidates)):
            # Choose: include candidates[i]
            path.append(candidates[i])

            # Explore: can reuse same element, so pass 'i' not 'i+1'
            backtrack(i, path, remaining - candidates[i])

            # Unchoose: backtrack
            path.pop()

    backtrack(0, [], target)
    return result
```

**Usage Example:**

```python
# Generate permutations
perms = generate_permutations([1, 2, 3])
print(f"Permutations: {perms}")
# Output: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]

# Solve 4-Queens
solutions = solve_n_queens(4)
print(f"4-Queens solutions: {len(solutions)} found")
for solution in solutions:
    for row in solution:
        print(row)
    print()
# Output: 2 valid board configurations

# Generate subsets
subsets = generate_subsets([1, 2, 3])
print(f"Subsets: {subsets}")
# Output: [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]

# Combination sum
combos = combination_sum([2, 3, 6, 7], 7)
print(f"Combinations that sum to 7: {combos}")
# Output: [[2,2,3], [7]]
```

### Code Explanation

**Key Design Decisions:**

1. **The Choose-Explore-Unchoose Pattern:** This is the heart of backtracking. We make a choice (add to path/mark as used), explore its consequences recursively, then undo the choice (pop/remove) before trying the next option. This ensures we try all possibilities without interference between branches.

2. **Early Pruning:** In the N-Queens solution, we check `if col in cols or (row - col) in diag1...` BEFORE recursing. This prevents exploring invalid branches entirely, saving enormous amounts of computation.

3. **State Tracking with Sets:** Using sets for `cols`, `diag1`, and `diag2` in N-Queens gives us O(1) lookup to check if a position is under attack. The diagonal formulas `row - col` and `row + col` are mathematical tricks: all positions on the same diagonal share the same value.

4. **Path Copying:** When we find a solution, we do `result.append(path[:])`. This creates a copy because `path` continues to be modified. Without copying, all solutions would end up empty!

5. **Start Index in Subsets:** The `backtrack(i + 1, path)` in subsets ensures we don't create duplicate subsets. We only consider elements after the current one, so we never generate [2, 1] after already generating [1, 2].

---

## Complexity Analysis

### Time Complexity

- **Best Case:** Depends on problem; could be O(1) if first path succeeds
- **Average Case:** O(b^d) where b is branching factor, d is depth
  - For permutations: O(n! × n) - n! permutations to generate, each takes O(n) to construct
  - For subsets: O(2^n × n) - 2^n subsets, each takes O(n) to copy
  - For N-Queens: O(n!) - approximately n choices in row 1, n-2 in row 2, etc.
- **Worst Case:** Same as average case - must explore entire decision tree

**Why?** Backtracking explores a decision tree where:

- Each level represents a choice point
- The branching factor (b) is how many options we have at each choice
- The depth (d) is how many choices we need to make

In the worst case, we explore every path in the tree: b × b × b... (d times) = b^d nodes. Pruning can dramatically reduce this in practice, but doesn't change the worst-case complexity.

For permutations specifically: we have n choices for position 1, then n-1 for position 2, etc., giving us n × (n-1) × (n-2) × ... × 1 = n! leaf nodes. Each permutation takes O(n) time to copy, so total is O(n! × n).

### Space Complexity

- **Recursion Stack:** O(d) where d is the maximum depth
  - For permutations: O(n) - we go n levels deep
  - For subsets: O(n) - maximum recursion depth is n
  - For N-Queens: O(n) - one queen per row
- **Additional Space:** Depends on implementation
  - State tracking (sets, visited arrays): typically O(n) to O(n²)
  - Output storage: varies by problem (O(n! × n) for permutations, O(2^n × n) for subsets)

**Why?** Each recursive call adds a frame to the call stack. The maximum depth equals the number of choices we make before reaching a solution or dead end. We also need space to track our current state (which elements we've used, which positions are attacked, etc.).

### Comparison with Alternatives

| Approach                    | Time (Avg)  | Time (Worst) | Space | When to Use                                                 |
| --------------------------- | ----------- | ------------ | ----- | ----------------------------------------------------------- |
| **Backtracking**            | O(b^d)      | O(b^d)       | O(d)  | Constraint satisfaction, exploring all solutions            |
| **Branch and Bound**        | O(b^d)      | O(b^d)       | O(d)  | Optimization problems (find best solution)                  |
| **Dynamic Programming**     | O(n²)-O(n³) | O(n²)-O(n³)  | O(n²) | Overlapping subproblems (DP not applicable to permutations) |
| **Greedy**                  | O(n log n)  | O(n log n)   | O(1)  | Problems with greedy choice property                        |
| **Brute Force (iterative)** | O(n!)       | O(n!)        | O(n)  | When decision tree structure isn't clear                    |

**Note:** While DP is faster when applicable, many backtracking problems (like permutations or N-Queens) don't have overlapping subproblems, making DP unusable.

---

## Examples

### Example 1: Generating Permutations of [1, 2, 3]

**Input:** `[1, 2, 3]`  
**Output:** `[[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]`

**Step-by-Step Trace:**

```
Initial call: backtrack(path=[], used={})

Level 1: Choose 1
  path=[1], used={1}

  Level 2: Choose 2
    path=[1,2], used={1,2}

    Level 3: Choose 3
      path=[1,2,3], used={1,2,3}
      ✓ SOLUTION FOUND: [1,2,3]
      Backtrack: path=[1,2], used={1,2}

    Backtrack: path=[1], used={1}

  Level 2: Choose 3
    path=[1,3], used={1,3}

    Level 3: Choose 2
      path=[1,3,2], used={1,2,3}
      ✓ SOLUTION FOUND: [1,3,2]
      Backtrack: path=[1,3], used={1,3}

    Backtrack: path=[1], used={1}

  Backtrack: path=[], used={}

Level 1: Choose 2
  path=[2], used={2}

  Level 2: Choose 1
    path=[2,1], used={1,2}

    Level 3: Choose 3
      path=[2,1,3], used={1,2,3}
      ✓ SOLUTION FOUND: [2,1,3]

... (continues for [2,3,1], [3,1,2], [3,2,1])

Total: 6 permutations found
```

### Example 2: 4-Queens Problem

**Input:** `n = 4`  
**Output:** 2 valid board configurations

**Detailed Trace:**

```
Initial: Place first queen in row 0

Try row 0, col 0:
Q . . .    ← Placed queen at (0,0)
. . . .    cols={0}, diag1={0}, diag2={0}
. . . .
. . . .

Row 1: Can't use col 0 (same column)
       Can't use col 1 (diagonal attacked: row-col = 1-1 = 0, already in diag1)
       Try col 2:
Q . . .
. . Q .    ← Placed queen at (1,2)
. . . .    cols={0,2}, diag1={0,-1}, diag2={0,3}
. . . .

Row 2: Can't use col 0 (same column)
       Can't use col 1 (diagonal attacked)
       Can't use col 2 (same column)
       Can't use col 3 (diagonal attacked)
       ❌ NO VALID POSITION - BACKTRACK

Remove queen from (1,2), try row 1, col 3:
Q . . .
. . . Q    ← Placed queen at (1,3)
. . . .    cols={0,3}, diag1={0,-2}, diag2={0,4}
. . . .

Row 2: Can't use col 0, 1, 2, 3 - all attacked!
       ❌ BACKTRACK AGAIN

Remove queen from (0,0), try row 0, col 1:
. Q . .    ← Placed queen at (0,1)
. . . .    cols={1}, diag1={-1}, diag2={1}
. . . .
. . . .

Row 1: Try col 3:
. Q . .
. . . Q    ← Placed queen at (1,3)
. . . .    cols={1,3}, diag1={-1,-2}, diag2={1,4}
. . . .

Row 2: Try col 0:
. Q . .
. . . Q
Q . . .    ← Placed queen at (2,0)
. . . .    cols={0,1,3}, diag1={-1,-2,2}, diag2={1,4,2}

Row 3: Try col 2:
. Q . .
. . . Q
Q . . .
. . Q .    ← Placed queen at (3,2)
           ✓ SOLUTION FOUND!

Continue exploring to find second solution...
Final solution 2:
. . Q .
Q . . .
. . . Q
. Q . .
```

### Example 3: Subsets of [1, 2, 3]

**Input:** `[1, 2, 3]`  
**Output:** `[[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]`

**Step-by-Step Execution:**

```
backtrack(start=0, path=[])
  Add [] to result ✓

  i=0: Choose 1
    backtrack(start=1, path=[1])
      Add [1] to result ✓

      i=1: Choose 2
        backtrack(start=2, path=[1,2])
          Add [1,2] to result ✓

          i=2: Choose 3
            backtrack(start=3, path=[1,2,3])
              Add [1,2,3] to result ✓
            Backtrack to [1,2]
        Backtrack to [1]

      i=2: Choose 3 (skip 2)
        backtrack(start=3, path=[1,3])
          Add [1,3] to result ✓
        Backtrack to [1]
    Backtrack to []

  i=1: Choose 2 (skip 1)
    backtrack(start=2, path=[2])
      Add [2] to result ✓

      i=2: Choose 3
        backtrack(start=3, path=[2,3])
          Add [2,3] to result ✓
        Backtrack to [2]
    Backtrack to []

  i=2: Choose 3 (skip 1 and 2)
    backtrack(start=3, path=[3])
      Add [3] to result ✓
    Backtrack to []

Result: [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
```

### Example 4: Combination Sum for Target 7

**Input:** `candidates = [2, 3, 6, 7], target = 7`  
**Output:** `[[2,2,3], [7]]`

**Complete Trace:**

```
backtrack(start=0, path=[], remaining=7)

  i=0: Choose 2 (remaining: 7-2=5)
    backtrack(start=0, path=[2], remaining=5)

      i=0: Choose 2 again (remaining: 5-2=3)
        backtrack(start=0, path=[2,2], remaining=3)

          i=0: Choose 2 again (remaining: 3-2=1)
            backtrack(start=0, path=[2,2,2], remaining=1)
              i=0: Choose 2 (remaining: 1-2=-1) ❌ negative, return
              i=1: Choose 3 (remaining: 1-3=-2) ❌ negative, return
              i=2,3: Skip (too large)
            Backtrack to [2,2]

          i=1: Choose 3 (remaining: 3-3=0)
            backtrack(start=1, path=[2,2,3], remaining=0)
              ✓ SOLUTION FOUND: [2,2,3]
            Backtrack to [2,2]

          i=2,3: Skip (too large)
        Backtrack to [2]

      i=1: Choose 3 (remaining: 5-3=2)
        backtrack(start=1, path=[2,3], remaining=2)
          i=1: Choose 3 (remaining: 2-3=-1) ❌ negative
          i=2,3: Skip
        Backtrack to [2]

      i=2,3: Skip (too large)
    Backtrack to []

  i=1: Choose 3 (remaining: 7-3=4)
    backtrack(start=1, path=[3], remaining=4)
      i=1: Choose 3 (remaining: 4-3=1)
        ... no valid combinations
      i=2,3: Skip
    Backtrack to []

  i=2: Choose 6 (remaining: 7-6=1)
    backtrack(start=2, path=[6], remaining=1)
      All candidates too large
    Backtrack to []

  i=3: Choose 7 (remaining: 7-7=0)
    backtrack(start=3, path=[7], remaining=0)
      ✓ SOLUTION FOUND: [7]
    Backtrack to []

Final Result: [[2,2,3], [7]]
```

---

## Edge Cases

### 1. Empty Input

**Scenario:** Input array/list is empty `[]`  
**Challenge:** What does "all permutations of nothing" mean?  
**Solution:** Return a list containing one empty solution `[[]]`

- For permutations: `[[]]` (one way to arrange nothing)
- For subsets: `[[]]` (the empty set is always a subset)

```python
# Edge case handling
def generate_permutations(nums: List[int]) -> List[List[int]]:
    if not nums:
        return [[]]  # One empty permutation
    # ... rest of code
```

### 2. Single Element

**Scenario:** Input contains only one element `[5]`  
**Challenge:** Trivial case that should terminate quickly  
**Solution:**

- Permutations: `[[5]]` (only one arrangement)
- Subsets: `[[], [5]]` (empty set and the element itself)
- N-Queens with n=1: `[["Q"]]` (trivial solution)

```python
# These cases hit the base case immediately
generate_permutations([5])  # Returns [[5]]
generate_subsets([5])       # Returns [[], [5]]
```

### 3. Duplicates in Input

**Scenario:** Input contains duplicate elements `[1, 2, 2]`  
**Challenge:** May generate duplicate solutions if not handled  
**Solution:** Sort input and skip duplicates at the same level

```python
def permutations_with_duplicates(nums: List[int]) -> List[List[int]]:
    """Handle duplicate elements in input."""
    result = []
    nums.sort()  # Sort to group duplicates

    def backtrack(path: List[int], used: List[bool]) -> None:
        if len(path) == len(nums):
            result.append(path[:])
            return

        for i in range(len(nums)):
            # Skip used elements
            if used[i]:
                continue
            # Skip duplicates at same recursion level
            # If current element equals previous AND previous wasn't used
            # then we already explored this branch
            if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                continue

            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False

    backtrack([], [False] * len(nums))
    return result

# Example: [1,2,2] → [[1,2,2], [2,1,2], [2,2,1]]
# Without duplicate handling → [[1,2,2], [1,2,2], [2,1,2], [2,1,2], [2,2,1], [2,2,1]]
```

### 4. No Valid Solution Exists

**Scenario:** Constraints make solution impossible (e.g., 2-Queens or 3-Queens)  
**Challenge:** Must exhaust entire search space before concluding  
**Solution:** Return empty result list `[]`

```python
solve_n_queens(2)  # Returns [] - impossible to place 2 non-attacking queens
solve_n_queens(3)  # Returns [] - also impossible
```

### 5. Very Large Input

**Scenario:** Large n causing exponential explosion (e.g., `generate_permutations(range(15))`)  
**Challenge:** Will take astronomically long (15! = 1.3 trillion permutations!)  
**Solution:**

- Set reasonable limits
- Add early termination conditions
- Consider iterative deepening or approximation algorithms

```python
def generate_permutations_limited(nums: List[int], max_results: int = 10000) -> List[List[int]]:
    """Generate permutations with a result limit."""
    result = []

    def backtrack(path: List[int], used: Set[int]) -> bool:
        if len(result) >= max_results:
            return True  # Signal to stop early

        if len(path) == len(nums):
            result.append(path[:])
            return len(result) >= max_results

        for num in nums:
            if num in used:
                continue
            path.append(num)
            used.add(num)
            if backtrack(path, used):  # Check for early termination
                return True
            path.pop()
            used.remove(num)
        return False

    backtrack([], set())
    return result
```

### 6. Invalid Board Configurations

**Scenario:** For N-Queens, starting with pre-placed queens  
**Challenge:** Must respect existing queens while placing new ones  
**Solution:** Initialize tracking sets with existing queen positions

```python
def solve_n_queens_with_initial(
    n: int,
    initial_queens: List[tuple]
) -> List[List[str]]:
    """Solve N-Queens with some queens already placed."""
    # Initialize board and tracking sets
    board = [['.'] * n for _ in range(n)]
    cols, diag1, diag2 = set(), set(), set()

    # Place initial queens and update tracking
    for row, col in initial_queens:
        board[row][col] = 'Q'
        cols.add(col)
        diag1.add(row - col)
        diag2.add(row + col)

    # Start backtracking from first row without a queen
    # ... rest of implementation
```

---

## Common Pitfalls

### ❌ Pitfall 1: Forgetting to Backtrack (Not Undoing Choices)

**What happens:** Solutions interfere with each other, producing incorrect results

```python
# WRONG - Missing backtrack step
def wrong_permutations(nums):
    result = []
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for num in nums:
            if num in used:
                continue
            path.append(num)
            used.add(num)
            backtrack(path, used)
            # ❌ MISSING: path.pop() and used.remove(num)
    backtrack([], set())
    return result

# This will only generate ONE permutation because
# path and used keep growing and never reset!
```

**Why it's wrong:** Without removing the choice after exploring it, subsequent branches operate on contaminated state. The path keeps growing, and `used` never allows revisiting numbers.

**Correct approach:**

```python
# CORRECT - Always undo your choice
path.append(num)
used.add(num)
backtrack(path, used)
path.pop()        # ✓ Undo the choice
used.remove(num)  # ✓ Mark as available again
```

### ❌ Pitfall 2: Not Making a Copy When Saving Solutions

**What happens:** All solutions end up being empty or identical

```python
# WRONG - Storing reference instead of copy
def wrong_subsets(nums):
    result = []
    path = []

    def backtrack(start):
        result.append(path)  # ❌ Storing reference!
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1)
            path.pop()

    backtrack(0)
    return result

# All elements in result point to the SAME path object
# After backtracking completes, path is empty
# So result looks like: [[], [], [], [], [], [], [], []]
```

**Why it's wrong:** Python lists are mutable objects passed by reference. When you append `path` to `result`, you're storing a reference to the same list object that keeps getting modified.

**Correct approach:**

```python
# CORRECT - Make a copy of the path
result.append(path[:])   # ✓ Creates a new list with same elements
```

### ❌ Pitfall 3: Wrong Base Case Logic

**What happens:** Infinite recursion or missing solutions

```python
# WRONG - Base case never triggers
def wrong_permutations(nums):
    result = []
    def backtrack(path, used):
        # ❌ Using > instead of ==
        if len(path) > len(nums):  # This is NEVER true!
            result.append(path[:])
            return
        # ... rest of code

# WRONG - Missing return statement
def also_wrong(nums):
    result = []
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            # ❌ MISSING return - keeps recursing!
        # ... rest of code
```

**Why it's wrong:** First example: `len(path)` grows by 1 each level, and we stop when it equals `len(nums)`, so it can never be greater. Second example: Without returning, we continue recursing even after finding a solution, leading to errors or duplicates.

**Correct approach:**

```python
# CORRECT - Exact match with return
if len(path) == len(nums):
    result.append(path[:])
    return  # ✓ Stop recursing
```

### ❌ Pitfall 4: Inefficient Constraint Checking

**What happens:** Algorithm is correct but unnecessarily slow

```python
# SLOW - Checking entire board every time
def slow_n_queens(n):
    def is_safe(board, row, col):
        # ❌ O(n²) check for each placement!
        for r in range(n):
            for c in range(n):
                if board[r][c] == 'Q':
                    # Check if attacks (row, col)
                    if r == row or c == col:
                        return False
                    if abs(r - row) == abs(c - col):
                        return False
        return True
    # ... rest of code
```

**Why it's wrong:** This works correctly but checks every single cell on the board for every placement attempt. For a 100×100 board, that's 10,000 checks per placement!

**Correct approach:**

```python
# FAST - O(1) lookups using sets
cols = set()
diag1 = set()  # row - col
diag2 = set()  # row + col

def is_safe(row, col):
    # ✓ O(1) set lookups!
    return (col not in cols and
            (row - col) not in diag1 and
            (row + col) not in diag2)
```

### ❌ Pitfall 5: Not Pruning Invalid Branches Early

**What happens:** Explore branches that can never lead to valid solutions

```python
# INEFFICIENT - No early pruning
def no_pruning_permutations(nums):
    result = []
    def backtrack(path):
        if len(path) == len(nums):
            # ❌ Only check validity at the END
            if len(set(path)) == len(path):  # No duplicates?
                result.append(path[:])
            return

        # Trying ALL numbers, even ones already used!
        for num in nums:
            path.append(num)
            backtrack(path)
            path.pop()
```

**Why it's wrong:** This explores invalid paths deeply before discovering they're invalid. For `nums = [1,2,3,4,5]`, it would explore [1,1,1,1,1], [1,1,1,1,2], etc., wasting enormous amounts of time on paths that can never succeed.

**Correct approach:**

```python
# EFFICIENT - Prune early
def backtrack(path, used):
    if len(path) == len(nums):
        result.append(path[:])
        return

    for num in nums:
        if num in used:
            continue  # ✓ Skip invalid choice immediately!
        # Only explore valid branches
```

### ❌ Pitfall 6: Modifying Input or Shared State Incorrectly

**What happens:** Side effects cause unexpected behavior in subsequent operations

```python
# WRONG - Modifying input array
def wrong_subsets(nums):
    result = []
    def backtrack(remaining):
        result.append(remaining[:])
        if not remaining:
            return

        # ❌ Modifying and recursing on same list
        for i in range(len(remaining)):
            val = remaining.pop(i)
            backtrack(remaining)
            remaining.insert(i, val)  # Trying to restore

    backtrack(nums)
    return result
```

**Why it's wrong:** Modifying the input array directly can cause subtle bugs, especially with insertion/removal which might not restore the exact original state. Also, it violates the principle that functions shouldn't modify their inputs.

**Correct approach:**

```python
# CORRECT - Use index-based approach, don't modify input
def backtrack(start, path):
    result.append(path[:])
    for i in range(start, len(nums)):
        backtrack(i + 1, path + [nums[i]])  # ✓ Create new list
```

---

## Variations and Extensions

### Variation 1: Iterative Backtracking (Using Explicit Stack)

**Description:** Convert recursive backtracking to iterative form using a stack

**When to use:** When dealing with very deep recursion that might cause stack overflow, or when you need more control over the execution flow

**Key differences:** Uses explicit stack data structure instead of call stack; manually manages state

**Implementation:**

```python
def permutations_iterative(nums: List[int]) -> List[List[int]]:
    """Generate permutations iteratively using explicit stack."""
    if not nums:
        return [[]]

    result = []
    # Stack contains: (current_path, available_choices)
    stack = [([], nums)]

    while stack:
        path, choices = stack.pop()

        # If we've made all choices, save solution
        if len(path) == len(nums):
            result.append(path)
            continue

        # Push all possible next choices onto stack
        for i, num in enumerate(choices):
            new_path = path + [num]
            new_choices = choices[:i] + choices[i+1:]
            stack.append((new_path, new_choices))

    return result
```

### Variation 2: Branch and Bound (for Optimization)

**Description:** Extension of backtracking that prunes branches based on bounds rather than just constraints

**When to use:** When you want the _best_ solution (maximum/minimum) rather than all solutions

**Key differences:** Maintains a "best so far" value and prunes branches that provably can't beat it

**Implementation:**

```python
def knapsack_branch_bound(
    weights: List[int],
    values: List[int],
    capacity: int
) -> int:
    """
    Solve 0/1 knapsack using branch and bound.
    Find maximum value without exceeding capacity.
    """
    n = len(weights)
    max_value = [0]  # Track best value found

    def bound(i: int, current_weight: int, current_value: int) -> float:
        """Calculate upper bound of potential value from this state."""
        if current_weight >= capacity:
            return 0

        bound_value = current_value
        total_weight = current_weight

        # Greedily add fractional items for upper bound
        for j in range(i, n):
            if total_weight + weights[j] <= capacity:
                total_weight += weights[j]
                bound_value += values[j]
            else:
                # Add fraction of remaining item
                fraction = (capacity - total_weight) / weights[j]
                bound_value += fraction * values[j]
                break

        return bound_value

    def backtrack(i: int, current_weight: int, current_value: int) -> None:
        if i == n:
            max_value[0] = max(max_value[0], current_value)
            return

        # Calculate bound for this branch
        if bound(i, current_weight, current_value) <= max_value[0]:
            return  # ✓ Prune: can't improve on best solution

        # Include current item if possible
        if current_weight + weights[i] <= capacity:
            backtrack(
                i + 1,
                current_weight + weights[i],
                current_value + values[i]
            )

        # Exclude current item
        backtrack(i + 1, current_weight, current_value)

    backtrack(0, 0, 0)
    return max_value[0]
```

### Variation 3: Constraint Propagation

**Description:** Actively propagate constraints to reduce search space before exploring

**When to use:** Complex constraint satisfaction problems like Sudoku where constraints interact

**Key differences:** After each choice, immediately propagate implications to narrow possibilities

**Conceptual Implementation:**

```python
def sudoku_with_propagation(board: List[List[str]]) -> bool:
    """
    Solve Sudoku with constraint propagation.
    After placing a number, immediately eliminate it from related cells.
    """
    # For each cell, track possible values
    possibilities = [
        [set('123456789') if board[i][j] == '.' else set()
         for j in range(9)]
        for i in range(9)
    ]

    def propagate(row: int, col: int, value: str) -> bool:
        """Remove value from all related cells' possibilities."""
        # Remove from row, column, and box
        for i in range(9):
            possibilities[row][i].discard(value)
            possibilities[i][col].discard(value)

        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                possibilities[i][j].discard(value)

        # Check if any cell has no possibilities left
        return all(
            len(possibilities[i][j]) > 0 or board[i][j] != '.'
            for i in range(9) for j in range(9)
        )

    # ... backtracking with propagation
```

### Variation 4: Forward Checking

**Description:** Before making a choice, verify it doesn't eliminate all options for future variables

**When to use:** Problems where choices significantly constrain future possibilities

**Key differences:** More aggressive pruning by simulating future implications

**Implementation:**

```python
def n_queens_forward_checking(n: int) -> List[List[str]]:
    """
    N-Queens with forward checking.
    Check if placement leaves valid options for remaining rows.
    """
    result = []
    board = [['.'] * n for _ in range(n)]

    def is_valid_with_future(
        row: int,
        col: int,
        cols: Set,
        diag1: Set,
        diag2: Set
    ) -> bool:
        """Check if this placement leaves options for future rows."""
        # Standard attack check
        if col in cols or (row - col) in diag1 or (row + col) in diag2:
            return False

        # Forward check: will future rows have valid positions?
        temp_cols = cols | {col}
        temp_diag1 = diag1 | {row - col}
        temp_diag2 = diag2 | {row + col}

        for future_row in range(row + 1, n):
            has_option = False
            for future_col in range(n):
                if (future_col not in temp_cols and
                    (future_row - future_col) not in temp_diag1 and
                    (future_row + future_col) not in temp_diag2):
                    has_option = True
                    break

            if not has_option:
                return False  # ✓ Prune: no valid position in future row

        return True

    # ... rest of implementation
```

### Variation 5: Memoization for Overlapping Subproblems

**Description:** Cache results of repeated subproblems to avoid recomputation

**When to use:** When the same state can be reached through different paths

**Key differences:** Dramatically speeds up certain problems with overlapping states

**Implementation:**

```python
def combination_sum_memo(
    candidates: List[int],
    target: int
) -> List[List[int]]:
    """
    Find all combinations that sum to target (elements can be reused).
    Uses memoization for repeated (start, remaining) states.
    """
    memo = {}

    def backtrack(start: int, remaining: int) -> List[List[int]]:
        # Check cache first
        if (start, remaining) in memo:
            return memo[(start, remaining)]

        if remaining == 0:
            return [[]]
        if remaining < 0 or start >= len(candidates):
            return []

        result = []

        # Include current candidate (can reuse)
        for combo in backtrack(start, remaining - candidates[start]):
            result.append([candidates[start]] + combo)

        # Exclude current candidate (move to next)
        result.extend(backtrack(start + 1, remaining))

        # Cache result
        memo[(start, remaining)] = result
        return result

    return backtrack(0, target)
```

### Variation 6: Randomized Backtracking

**Description:** Try choices in random order rather than fixed order

**When to use:** When you need diverse solutions or want to find _a_ solution quickly (not all solutions)

**Key differences:** May find solutions faster on average, useful for sampling solution space

**Implementation:**

```python
import random

def random_permutation(nums: List[int]) -> Optional[List[int]]:
    """Find a single random permutation using randomized backtracking."""
    def backtrack(path: List[int], remaining: List[int]) -> Optional[List[int]]:
        if not remaining:
            return path

        # Shuffle remaining choices
        random.shuffle(remaining)

        for i, num in enumerate(remaining):
            result = backtrack(
                path + [num],
                remaining[:i] + remaining[i+1:]
            )
            if result:
                return result  # Return first valid solution found

        return None

    return backtrack([], nums)
```

---

## Practice Problems

### Beginner

1. **Generate All Binary Strings of Length N** - Create all possible binary strings of given length
   - Similar to subsets but with fixed choices {0, 1}

2. **Letter Combinations of a Phone Number** - Given a digit string, return all possible letter combinations
   - LeetCode #17
   - Practice the basic choose-explore-unchoose pattern

3. **Generate Parentheses** - Generate all combinations of well-formed parentheses
   - LeetCode #22
   - Learn to add constraints (valid parentheses rules)

4. **Subsets** - Return all possible subsets of a set
   - LeetCode #78
   - Classic backtracking problem, great starting point

### Intermediate

1. **Permutations** - Generate all permutations of distinct integers
   - LeetCode #46
   - Add state tracking with "used" set

2. **Permutations II** - Generate permutations with duplicate elements
   - LeetCode #47
   - Handle duplicates correctly to avoid duplicate results

3. **Combination Sum** - Find all combinations that sum to target (elements reusable)
   - LeetCode #39
   - Practice pruning invalid branches early

4. **Combination Sum II** - Same but each element can only be used once
   - LeetCode #40
   - More complex duplicate handling

5. **Palindrome Partitioning** - Partition string into all possible palindromic substrings
   - LeetCode #131
   - Combine backtracking with palindrome checking

6. **Word Search** - Find if word exists in 2D board
   - LeetCode #79
   - 2D backtracking with visited tracking

7. **Restore IP Addresses** - Generate all valid IP addresses from a string
   - LeetCode #93
   - Multiple constraints to validate

### Advanced

1. **N-Queens** - Place N queens on N×N board with no attacks
   - LeetCode #51
   - Efficient constraint tracking with sets

2. **N-Queens II** - Count the number of N-Queens solutions
   - LeetCode #52
   - Optimization: just count, don't store solutions

3. **Sudoku Solver** - Fill a Sudoku board following all rules
   - LeetCode #37
   - Complex constraint satisfaction

4. **Word Search II** - Find all words from dictionary in a 2D board
   - LeetCode #212
   - Combine backtracking with Trie data structure

5. **Regular Expression Matching** - Implement regex matching with '.' and '\*'
   - LeetCode #10
   - Can be solved with backtracking (also DP)

6. **Wildcard Matching** - Match string with wildcards '?' and '\*'
   - LeetCode #44
   - Similar to regex but different rules

7. **Remove Invalid Parentheses** - Remove minimum parentheses to make string valid
   - LeetCode #301
   - BFS + backtracking hybrid

8. **Expression Add Operators** - Add operators (+, -, \*) to make expression equal target
   - LeetCode #282
   - Track partial results and handle operator precedence

9. **Matchsticks to Square** - Arrange matchsticks to form a square
   - LeetCode #473
   - Partition problem with backtracking

10. **Split Array into Fibonacci Sequence** - Split string into Fibonacci-like sequence
    - LeetCode #842
    - Constraint checking with large numbers

---

## Real-World Applications

### Industry Use Cases

1. **Artificial Intelligence & Game Playing:** Chess engines, Go programs, and other game AI use backtracking (often with alpha-beta pruning) to explore possible move sequences. When Deep Blue evaluates chess positions, it's using a sophisticated form of backtracking to search the game tree.

2. **Scheduling & Resource Allocation:** Airlines use backtracking algorithms to assign crews to flights, ensuring all constraints (pilot certifications, rest requirements, union rules) are satisfied. Universities use it for course scheduling, ensuring no professor teaches two classes simultaneously and all requirements are met.

3. **Configuration Management:** Software configuration tools use backtracking to find valid combinations of software packages that satisfy all dependencies and avoid conflicts. Package managers like apt and npm solve complex dependency graphs using constraint satisfaction.

4. **Circuit Design & Layout:** Electronic Design Automation (EDA) tools use backtracking to route wires on circuit boards and chips, ensuring no short circuits while meeting timing and space constraints.

5. **Cryptography & Cryptanalysis:** Breaking certain encryption schemes involves searching through possible keys or configurations using backtracking with constraint propagation.

6. **Natural Language Processing:** Parsing sentences and resolving ambiguities in grammar often uses backtracking to try different parse trees until finding one that fits all linguistic rules.

### Popular Implementations

- **Constraint Satisfaction Problem Solvers:** Libraries like Google OR-Tools and Python's `python-constraint` implement sophisticated backtracking with constraint propagation for solving CSPs.

- **Prolog Programming Language:** Prolog's entire execution model is based on backtracking. When a query fails, Prolog automatically backtracks to try alternative rules.

- **SAT Solvers:** Boolean satisfiability solvers (like MiniSat, Z3) use DPLL algorithm, which is essentially backtracking with unit propagation and other optimizations. These power formal verification tools.

- **Regex Engines:** Many regular expression engines (especially older ones) use backtracking to match patterns. This can lead to catastrophic backtracking on certain patterns.

- **Sudoku Solvers:** Nearly every Sudoku solver app uses backtracking, often enhanced with constraint propagation to reduce the search space.

### Practical Scenarios

- **Job Interview Preparation:** Backtracking problems are extremely common in technical interviews at major tech companies. Understanding the template helps you solve unfamiliar problems on the spot.

- **Puzzle Games:** Implementing solvers for puzzles like Sudoku, Kakuro, Crosswords, or logic grid puzzles all use backtracking.

- **Database Query Optimization:** Query planners explore different execution plans using backtracking to find efficient ways to join tables and apply filters.

- **Route Planning with Constraints:** Finding routes that satisfy multiple constraints (avoid tolls, prefer highways, visit specific waypoints in order) uses backtracking-style search.

- **Test Case Generation:** Generating comprehensive test cases that cover all combinations of inputs while respecting constraints uses backtracking.

---

## Related Topics

### Prerequisites to Review

- **Recursion** - Backtracking is fundamentally recursive; you must be comfortable with recursive thinking, base cases, and how the call stack works. Understanding tail recursion and recursion trees is especially helpful.

- **Arrays and Lists** - Most backtracking problems operate on collections. You need to be comfortable with array manipulation, slicing, and understanding reference vs. value semantics.

- **Sets and Hash Tables** - Efficient backtracking often requires O(1) lookups to check constraints (like "is this number already used?"). Understanding set operations is crucial.

- **Tree Concepts** - Viewing backtracking as traversing a decision tree helps immensely. Understanding tree depth, nodes, leaves, and branches provides a mental model.

### Next Steps

- **Dynamic Programming** - Many problems can be solved with either backtracking or DP. Learning DP helps you recognize when a problem has overlapping subproblems and optimal substructure, making DP more efficient than backtracking.

- **Branch and Bound** - A natural extension of backtracking for optimization problems. Instead of finding all solutions, you find the best one by maintaining bounds and pruning suboptimal branches.

- **Constraint Satisfaction Problems (CSP)** - Formalize the types of problems backtracking solves. Learn about arc consistency, constraint propagation, and variable ordering heuristics.

- **Graph Algorithms** - Many graph problems (like Hamiltonian path, graph coloring) use backtracking. Understanding DFS, BFS, and graph representations enhances your backtracking toolkit.

- **Greedy Algorithms** - Learn when a greedy approach can replace exponential backtracking with a polynomial algorithm. Understanding the trade-offs helps you choose the right tool.

### Similar Concepts

- **Depth-First Search (DFS)** - Backtracking is essentially DFS on an implicit decision tree. Understanding DFS on explicit graphs helps you visualize backtracking's exploration strategy.

- **Breadth-First Search (BFS)** - Sometimes BFS can solve problems that backtracking can, especially when you want the shortest/smallest solution. Comparing the two helps you choose appropriately.

- **Divide and Conquer** - Both break problems into smaller pieces, but divide and conquer solves independent subproblems while backtracking explores dependent choices.

- **Brute Force** - Backtracking is intelligent brute force—it tries all possibilities but prunes impossible branches early. Understanding naive brute force helps appreciate backtracking's efficiency gains.

### Further Reading

- **"Introduction to Algorithms" (CLRS)** - Chapter on backtracking and constraint satisfaction
  - Comprehensive academic treatment with complexity analysis

- **LeetCode Backtracking Explore Card** - https://leetcode.com/explore/learn/card/recursion-ii/
  - Interactive problems with hints and solutions

- **"Algorithm Design Manual" by Skiena** - Section 7.1 on backtracking
  - Practical perspective with real-world examples

- **Stanford CS106B Lectures on Backtracking**
  - Excellent video lectures explaining backtracking intuitively

- **"Artificial Intelligence: A Modern Approach" (Russell & Norvig)** - Chapter on constraint satisfaction
  - How backtracking fits into broader AI search strategies

---

**End of Guide**

This comprehensive guide covers backtracking from foundational concepts through advanced variations, with working code examples, detailed complexity analysis, and extensive practice problems. The step-by-step traces and visual walkthroughs should help you build strong intuition for this powerful algorithmic technique!
