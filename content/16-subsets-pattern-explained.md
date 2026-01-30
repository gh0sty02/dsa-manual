# Subsets Pattern

**Difficulty:** Intermediate to Advanced
**Prerequisites:** Recursion, Backtracking, Arrays, Strings, Trees
**Estimated Reading Time:** 55 minutes

## Introduction

The Subsets Pattern, also known as the Combinatorial Search Pattern, is a technique for generating all possible combinations, permutations, or variations of a given set of elements. It uses backtracking to systematically explore the solution space by making choices, exploring consequences, and undoing choices (backtracking) when necessary.

**Why it matters:** Combinatorial problems appear everywhere in computer science and real life: generating test cases, exploring game states, solving puzzles, optimizing schedules, and finding all possible configurations. The subsets pattern provides a systematic framework for exhaustively exploring possibilities without missing any or counting duplicates. Mastering this pattern unlocks your ability to solve complex problems that require exploring exponential solution spaces efficiently.

**Real-world analogy:** Think of planning a road trip where you can visit multiple cities, but you want to see all possible itineraries. You start from home, choose one city to visit, then from there choose another city (or return home), and so on. At each stop, you have multiple choices. To explore ALL possible trips, you'd visit one complete path, then "backtrack" to the last decision point and try a different choice. This systematic exploration of branching possibilities is exactly how the subsets pattern works!

## Core Concepts

### Key Principles

1. **Recursive Exploration:** Build solutions incrementally by making one choice at a time, recursively exploring consequences of that choice.

2. **Backtracking:** After exploring one path completely, undo the last choice (backtrack) and try alternative choices. This allows exhaustive exploration without exponential memory.

3. **State Space Tree:** Visualize the solution space as a tree where:
   - Each node represents a partial solution
   - Each branch represents a choice
   - Leaves represent complete solutions

4. **Choice/Explore/Unchoose Pattern:**
   - **Choose:** Make a decision (add element to current solution)
   - **Explore:** Recursively explore consequences
   - **Unchoose:** Undo the decision (backtrack) to try alternatives

### Essential Terms

- **Subset:** A set containing some (or all, or none) elements from original set
- **Power Set:** Set of all possible subsets (size = 2^n for n elements)
- **Permutation:** Arrangement where order matters (ABC ≠ BAC)
- **Combination:** Selection where order doesn't matter (ABC = BAC = CAB)
- **Backtracking:** Algorithmic technique for finding solutions by trial and error
- **Decision Tree:** Tree showing all possible decision paths
- **Pruning:** Skipping branches that can't lead to valid solutions
- **State:** Current partial solution being built
- **Base Case:** Condition when a complete solution is found

### Visual Overview

```
Generating Subsets of [1, 2, 3]:

Decision Tree (Include/Exclude each element):

                        []
                    /        \
              include 1    exclude 1
                [1]            []
              /    \          /    \
           +2      -2      +2      -2
          [1,2]   [1]    [2]      []
          / \     / \    / \      / \
        +3  -3  +3  -3 +3  -3   +3  -3
      [1,2,3][1,2][1,3][1][2,3][2][3][]

Subsets: [[], [1], [2], [1,2], [3], [1,3], [2,3], [1,2,3]]

Permutations of [1, 2, 3]:

Level 0:              [ ]
                   /   |   \
Level 1:        [1]   [2]   [3]
               / \    / \    / \
Level 2:   [1,2][1,3][2,1][2,3][3,1][3,2]
            |    |    |    |    |    |
Level 3: [1,2,3][1,3,2][2,1,3][2,3,1][3,1,2][3,2,1]

Permutations: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]

Backtracking Visualization:

State: []
  ├─ Choose 1 → [1]
  │   ├─ Choose 2 → [1,2]
  │   │   ├─ Choose 3 → [1,2,3] ✓ (solution found)
  │   │   └─ Backtrack to [1,2]
  │   ├─ Backtrack to [1]
  │   ├─ Choose 3 → [1,3]
  │   │   ├─ Choose 2 → [1,3,2] ✓ (solution found)
  │   │   └─ Backtrack to [1,3]
  │   └─ Backtrack to [1]
  ├─ Backtrack to []
  ├─ Choose 2 → [2]
  ...
```

## How It Works

### Subsets Generation - Step by Step

**Problem:** Generate all subsets of [1, 2, 3].

**Algorithm (Include/Exclude approach):**

1. Start with empty subset
2. For each element, recursively:
   - Include the element, explore further
   - Backtrack (remove element)
   - Exclude the element, explore further
3. Collect all complete subsets

**Detailed Walkthrough:**

```
Input: nums = [1, 2, 3]
Result: []  (will collect all subsets)
Current: [] (current subset being built)

Recursion Call Tree:

backtrack(index=0, current=[]):
  ├─ Add [] to result                    Result: [[]]
  │
  ├─ Choose nums[0]=1, current=[1]
  │  └─ backtrack(index=1, current=[1]):
  │      ├─ Add [1] to result            Result: [[], [1]]
  │      │
  │      ├─ Choose nums[1]=2, current=[1,2]
  │      │  └─ backtrack(index=2, current=[1,2]):
  │      │      ├─ Add [1,2] to result   Result: [[], [1], [1,2]]
  │      │      │
  │      │      ├─ Choose nums[2]=3, current=[1,2,3]
  │      │      │  └─ backtrack(index=3, current=[1,2,3]):
  │      │      │      └─ Add [1,2,3]    Result: [[], [1], [1,2], [1,2,3]]
  │      │      │      └─ index=3, return
  │      │      │
  │      │      └─ Backtrack: current=[1,2]
  │      │      └─ index=3, return
  │      │
  │      └─ Backtrack: current=[1]
  │      │
  │      └─ Choose nums[2]=3, current=[1,3]
  │         └─ backtrack(index=3, current=[1,3]):
  │             └─ Add [1,3]             Result: [..., [1,3]]
  │             └─ return
  │
  └─ Backtrack: current=[]
  │
  └─ Choose nums[1]=2, current=[2]
     └─ backtrack(index=2, current=[2]):
         ├─ Add [2]                      Result: [..., [2]]
         │
         ├─ Choose nums[2]=3, current=[2,3]
         │  └─ backtrack(index=3, current=[2,3]):
         │      └─ Add [2,3]             Result: [..., [2,3]]
         │
         └─ Backtrack: current=[2]

  └─ Choose nums[2]=3, current=[3]
     └─ backtrack(index=3, current=[3]):
         └─ Add [3]                      Result: [..., [3]]

Final Result: [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
```

### Permutations Generation - Step by Step

**Problem:** Generate all permutations of [1, 2, 3].

**Algorithm:**

1. Use a "used" array to track which elements are in current permutation
2. At each level, try adding each unused element
3. When permutation is complete (length = n), add to results
4. Backtrack by marking element as unused

**Detailed Walkthrough:**

```
Input: nums = [1, 2, 3]
Current: []
Used: [False, False, False]

Level 0 (choose first position):
  ├─ Choose 1: current=[1], used=[T,F,F]
  │  Level 1 (choose second position):
  │    ├─ Choose 2: current=[1,2], used=[T,T,F]
  │    │  Level 2 (choose third position):
  │    │    └─ Choose 3: current=[1,2,3], used=[T,T,T]
  │    │        → Add [1,2,3] to result ✓
  │    │        → Backtrack: current=[1,2], used=[T,T,F]
  │    │
  │    └─ Choose 3: current=[1,3], used=[T,F,T]
  │       Level 2:
  │         └─ Choose 2: current=[1,3,2], used=[T,T,T]
  │             → Add [1,3,2] to result ✓
  │             → Backtrack: current=[1,3], used=[T,F,T]
  │
  │  → Backtrack: current=[1], used=[T,F,F]
  │  → Backtrack: current=[], used=[F,F,F]
  │
  ├─ Choose 2: current=[2], used=[F,T,F]
  │  Level 1:
  │    ├─ Choose 1: current=[2,1], used=[T,T,F]
  │    │  Level 2:
  │    │    └─ Choose 3: current=[2,1,3], used=[T,T,T]
  │    │        → Add [2,1,3] to result ✓
  │    │
  │    └─ Choose 3: current=[2,3], used=[F,T,T]
  │       Level 2:
  │         └─ Choose 1: current=[2,3,1], used=[T,T,T]
  │             → Add [2,3,1] to result ✓
  │
  └─ Choose 3: current=[3], used=[F,F,T]
     Level 1:
       ├─ Choose 1: current=[3,1], used=[T,F,T]
       │  Level 2:
       │    └─ Choose 2: current=[3,1,2], used=[T,T,T]
       │        → Add [3,1,2] to result ✓
       │
       └─ Choose 2: current=[3,2], used=[F,T,T]
          Level 2:
            └─ Choose 1: current=[3,2,1], used=[T,T,T]
                → Add [3,2,1] to result ✓

Result: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
```

## Implementation

### Subsets (Basic)

```python
from typing import List

def subsets(nums: List[int]) -> List[List[int]]:
    """
    Generate all subsets (power set) of given array.
    
    Uses backtracking to include/exclude each element.
    
    Args:
        nums: Array of unique integers
        
    Returns:
        All possible subsets
        
    Time Complexity: O(n * 2^n)
        - 2^n subsets to generate
        - Each subset takes O(n) to copy
    Space Complexity: O(n)
        - Recursion depth is O(n)
        - Not counting output space
    """
    result = []
    
    def backtrack(index: int, current: List[int]) -> None:
        """
        Generate subsets starting from index.
        
        Args:
            index: Current position in nums
            current: Current subset being built
        """
        # Base case: processed all elements
        # Add current subset to results
        result.append(current[:])  # Copy current state
        
        # Try adding each remaining element
        for i in range(index, len(nums)):
            # Choose: include nums[i]
            current.append(nums[i])
            
            # Explore: recurse with next index
            backtrack(i + 1, current)
            
            # Unchoose: backtrack by removing nums[i]
            current.pop()
    
    backtrack(0, [])
    return result


# Iterative approach (bit manipulation)
def subsets_iterative(nums: List[int]) -> List[List[int]]:
    """
    Generate subsets using bit manipulation.
    
    For n elements, there are 2^n subsets.
    Each subset can be represented by n-bit binary number.
    
    Time Complexity: O(n * 2^n)
    Space Complexity: O(1) excluding output
    """
    n = len(nums)
    result = []
    
    # Generate all 2^n possible subsets
    for i in range(1 << n):  # 1 << n is 2^n
        subset = []
        for j in range(n):
            # Check if j-th bit is set
            if i & (1 << j):
                subset.append(nums[j])
        result.append(subset)
    
    return result


# Example usage
nums = [1, 2, 3]
print(subsets(nums))
# Output: [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
```

### Subsets with Duplicates

```python
def subsetsWithDup(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique subsets when input contains duplicates.
    
    Strategy: Sort first, then skip duplicate elements at same level.
    
    Args:
        nums: Array that may contain duplicates
        
    Returns:
        All unique subsets
        
    Time Complexity: O(n * 2^n)
    Space Complexity: O(n)
    """
    result = []
    nums.sort()  # Sort to group duplicates together
    
    def backtrack(index: int, current: List[int]) -> None:
        result.append(current[:])
        
        for i in range(index, len(nums)):
            # Skip duplicates at same recursion level
            # i > index ensures we only skip duplicates in same level,
            # not duplicates in different recursion branches
            if i > index and nums[i] == nums[i - 1]:
                continue
            
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()
    
    backtrack(0, [])
    return result


# Example usage
nums = [1, 2, 2]
print(subsetsWithDup(nums))
# Output: [[], [1], [1,2], [1,2,2], [2], [2,2]]
```

### Permutations

```python
def permute(nums: List[int]) -> List[List[int]]:
    """
    Generate all permutations of array.
    
    Uses backtracking with 'used' array to track selected elements.
    
    Args:
        nums: Array of unique integers
        
    Returns:
        All permutations
        
    Time Complexity: O(n! * n)
        - n! permutations
        - Each takes O(n) to build
    Space Complexity: O(n) for recursion and used array
    """
    result = []
    used = [False] * len(nums)
    
    def backtrack(current: List[int]) -> None:
        """Build permutations recursively."""
        # Base case: permutation complete
        if len(current) == len(nums):
            result.append(current[:])
            return
        
        # Try each unused element in current position
        for i in range(len(nums)):
            if used[i]:
                continue  # Already used in this permutation
            
            # Choose
            current.append(nums[i])
            used[i] = True
            
            # Explore
            backtrack(current)
            
            # Unchoose
            current.pop()
            used[i] = False
    
    backtrack([])
    return result


# Alternative: Swap-based approach (modifies input)
def permute_swap(nums: List[int]) -> List[List[int]]:
    """
    Generate permutations using swapping.
    
    More space-efficient but modifies input.
    """
    result = []
    
    def backtrack(start: int) -> None:
        if start == len(nums):
            result.append(nums[:])
            return
        
        for i in range(start, len(nums)):
            # Swap
            nums[start], nums[i] = nums[i], nums[start]
            
            # Recurse
            backtrack(start + 1)
            
            # Swap back (backtrack)
            nums[start], nums[i] = nums[i], nums[start]
    
    backtrack(0)
    return result


# Example usage
nums = [1, 2, 3]
print(permute(nums))
# Output: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
```

### Letter Case Permutation

```python
def letterCasePermutation(s: str) -> List[str]:
    """
    Generate all strings by changing case of letters.
    
    For each letter, we have two choices: lowercase or uppercase.
    Digits remain unchanged.
    
    Args:
        s: String containing letters and digits
        
    Returns:
        All case permutations
        
    Time Complexity: O(2^n * n) where n is number of letters
        - 2^n permutations for n letters
        - Each takes O(length of string) to build
    Space Complexity: O(n) for recursion
    """
    result = []
    
    def backtrack(index: int, current: List[str]) -> None:
        """
        Generate permutations starting from index.
        
        Args:
            index: Current position in string
            current: Current string being built
        """
        if index == len(s):
            result.append(''.join(current))
            return
        
        char = s[index]
        
        if char.isdigit():
            # Digit: only one choice
            current.append(char)
            backtrack(index + 1, current)
            current.pop()
        else:
            # Letter: try both cases
            # Lowercase
            current.append(char.lower())
            backtrack(index + 1, current)
            current.pop()
            
            # Uppercase
            current.append(char.upper())
            backtrack(index + 1, current)
            current.pop()
    
    backtrack(0, [])
    return result


# Example usage
s = "a1b2"
print(letterCasePermutation(s))
# Output: ["a1b2", "a1B2", "A1b2", "A1B2"]
```

### Generate Balanced Parentheses

```python
def generateParenthesis(n: int) -> List[str]:
    """
    Generate all valid combinations of n pairs of parentheses.
    
    Valid means:
    - Every opening parenthesis has matching closing one
    - At any point, # of closing ≤ # of opening
    
    Args:
        n: Number of pairs of parentheses
        
    Returns:
        All valid combinations
        
    Time Complexity: O(4^n / sqrt(n))
        - This is the n-th Catalan number
    Space Complexity: O(n) for recursion depth
    """
    result = []
    
    def backtrack(current: str, open_count: int, close_count: int) -> None:
        """
        Generate valid parentheses combinations.
        
        Args:
            current: Current string being built
            open_count: Number of opening parentheses used
            close_count: Number of closing parentheses used
        """
        # Base case: used all parentheses
        if len(current) == 2 * n:
            result.append(current)
            return
        
        # Can add opening parenthesis if haven't used all
        if open_count < n:
            backtrack(current + '(', open_count + 1, close_count)
        
        # Can add closing parenthesis if it wouldn't exceed opening count
        if close_count < open_count:
            backtrack(current + ')', open_count, close_count + 1)
    
    backtrack('', 0, 0)
    return result


# Example usage
n = 3
print(generateParenthesis(n))
# Output: ["((()))", "(()())", "(())()", "()(())", "()()()"]
```

### Unique Generalized Abbreviations

```python
def generateAbbreviations(word: str) -> List[str]:
    """
    Generate all unique abbreviations of a word.
    
    For each character, we can either:
    - Keep it as is
    - Abbreviate it (replace with number of chars)
    
    Args:
        word: Input string
        
    Returns:
        All possible abbreviations
        
    Time Complexity: O(2^n * n)
    Space Complexity: O(n)
    """
    result = []
    
    def backtrack(index: int, current: str, count: int) -> None:
        """
        Generate abbreviations starting from index.
        
        Args:
            index: Current position in word
            current: Current abbreviation being built
            count: Number of consecutive abbreviated characters
        """
        if index == len(word):
            # Add count if there are abbreviated chars at end
            if count > 0:
                current += str(count)
            result.append(current)
            return
        
        # Option 1: Abbreviate current character
        backtrack(index + 1, current, count + 1)
        
        # Option 2: Keep current character
        # If we had abbreviated chars before, add count first
        new_current = current + (str(count) if count > 0 else '') + word[index]
        backtrack(index + 1, new_current, 0)
    
    backtrack(0, '', 0)
    return result


# Example usage
word = "word"
print(generateAbbreviations(word))
# Output: ["word", "wor1", "wo1d", "wo2", "w1rd", "w1r1", "w2d", "w3",
#          "1ord", "1or1", "1o1d", "1o2", "2rd", "2r1", "3d", "4"]
```

### Code Explanation

**Key Design Patterns:**

1. **Choose-Explore-Unchoose:** The fundamental backtracking pattern:
   ```python
   current.append(choice)  # Choose
   backtrack(...)           # Explore
   current.pop()            # Unchoose
   ```

2. **Base Case Recognition:** Know when to stop recursing:
   - Subsets: When processed all elements
   - Permutations: When length equals input length
   - Parentheses: When used all pairs

3. **Pruning Invalid Paths:** Skip branches that can't lead to solutions:
   - Parentheses: Don't add ')' if it would exceed '('
   - Duplicates: Skip same value at same recursion level

4. **State Representation:** Choose efficient state format:
   - List for building solutions
   - Boolean array for tracking used elements
   - Counters for constraints (open/close parentheses)

5. **Result Collection:** Always copy state when adding to results:
   ```python
   result.append(current[:])  # Copy, don't reference
   ```

## Complexity Analysis

### Time Complexity

**Subsets:** O(n × 2^n)
- Generate 2^n subsets (each element can be included or excluded)
- Each subset takes O(n) time to copy to results
- Total: O(n × 2^n)

**Permutations:** O(n! × n)
- Generate n! permutations
- Each permutation takes O(n) time to build
- Total: O(n! × n)

**Combinations (k from n):** O(C(n,k) × k)
- Generate C(n,k) = n!/(k!(n-k)!) combinations
- Each takes O(k) time to copy
- Total: O(C(n,k) × k)

**Balanced Parentheses:** O(4^n / √n)
- This is the nth Catalan number: C(n) = (2n)! / ((n+1)! × n!)
- Grows exponentially but slower than 2^n

**Why Exponential?**
These problems inherently have exponential solution spaces:
- n elements → 2^n subsets
- n elements → n! permutations
- n pairs → Catalan(n) balanced parentheses

Backtracking is optimal because we must generate all solutions.

### Space Complexity

**Recursion Stack:** O(n)
- Maximum depth is typically n (one choice per element)
- Exception: Permutations can be optimized to O(1) with swapping

**Auxiliary Space:**
- Subsets: O(n) for current subset
- Permutations: O(n) for used array or current permutation
- Generally O(n) for tracking state

**Total Space:** O(n) not counting output
- Output space is exponential but not counted in complexity

### Comparison with Alternatives

| Problem | Backtracking | Iterative | Bit Manipulation |
|---------|--------------|-----------|------------------|
| Subsets | O(n×2^n), O(n) | O(n×2^n), O(1) | O(n×2^n), O(1) |
| Permutations | O(n!×n), O(n) | O(n!×n), O(n) | Not applicable |
| Combinations | O(C(n,k)×k), O(k) | O(C(n,k)×k), O(k) | Possible but complex |

**When to Use Each:**
- **Backtracking:** Most flexible, handles constraints easily, cleaner code
- **Iterative:** Better for simple cases, avoids recursion overhead
- **Bit Manipulation:** Efficient for subsets, but limited to include/exclude decisions

## Examples

### Example 1: Generate All Subsets

**Problem:** Find all subsets of [1,2,3].

**Input:** nums = [1, 2, 3]

**Solution Trace:**

```
backtrack(0, []):
  Add [] → result = [[]]
  
  i=0: Choose 1
    backtrack(1, [1]):
      Add [1] → result = [[], [1]]
      
      i=1: Choose 2
        backtrack(2, [1,2]):
          Add [1,2] → result = [[], [1], [1,2]]
          
          i=2: Choose 3
            backtrack(3, [1,2,3]):
              Add [1,2,3] → result = [[], [1], [1,2], [1,2,3]]
              return
          
          Backtrack: [1,2] → [1,2] (pop 3)
          return
      
      Backtrack: [1,2] → [1] (pop 2)
      
      i=2: Choose 3
        backtrack(3, [1,3]):
          Add [1,3] → result = [..., [1,3]]
          return
      
      Backtrack: [1,3] → [1] (pop 3)
      return
  
  Backtrack: [1] → [] (pop 1)
  
  i=1: Choose 2
    backtrack(2, [2]):
      Add [2] → result = [..., [2]]
      
      i=2: Choose 3
        backtrack(3, [2,3]):
          Add [2,3] → result = [..., [2,3]]
          return
      
      Backtrack: [2,3] → [2] (pop 3)
      return
  
  Backtrack: [2] → [] (pop 2)
  
  i=2: Choose 3
    backtrack(3, [3]):
      Add [3] → result = [..., [3]]
      return

Final: [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
```

### Example 2: Subsets with Duplicates

**Problem:** Generate unique subsets from [1, 2, 2].

**Input:** nums = [1, 2, 2]

**Key Insight:** After sorting, skip duplicates at same recursion level.

```
After sorting: [1, 2, 2]

backtrack(0, []):
  Add [] → result = [[]]
  
  i=0: nums[0]=1
    Add [1] → result = [[], [1]]
    
    i=1: nums[1]=2
      Add [1,2] → result = [[], [1], [1,2]]
      
      i=2: nums[2]=2
        Add [1,2,2] → result = [[], [1], [1,2], [1,2,2]]
    
    i=2: nums[2]=2
      Skip! (i > index and nums[2] == nums[1])
  
  i=1: nums[1]=2
    Add [2] → result = [..., [2]]
    
    i=2: nums[2]=2
      Add [2,2] → result = [..., [2,2]]
  
  i=2: nums[2]=2
    Skip! (i > index and nums[2] == nums[1])

Final: [[], [1], [1,2], [1,2,2], [2], [2,2]]
```

### Example 3: All Permutations

**Problem:** Generate all permutations of [1, 2, 3].

**Solution Trace (abbreviated):**

```
used = [F, F, F]

backtrack([]):
  i=0: Choose 1, used=[T,F,F]
    backtrack([1]):
      i=1: Choose 2, used=[T,T,F]
        backtrack([1,2]):
          i=2: Choose 3, used=[T,T,T]
            backtrack([1,2,3]): Complete! Add [1,2,3]
      
      i=2: Choose 3, used=[T,F,T]
        backtrack([1,3]):
          i=1: Choose 2, used=[T,T,T]
            backtrack([1,3,2]): Complete! Add [1,3,2]
  
  i=1: Choose 2, used=[F,T,F]
    backtrack([2]):
      i=0: Choose 1, used=[T,T,F]
        backtrack([2,1]):
          i=2: Choose 3, used=[T,T,T]
            backtrack([2,1,3]): Complete! Add [2,1,3]
      
      i=2: Choose 3, used=[F,T,T]
        backtrack([2,3]):
          i=0: Choose 1, used=[T,T,T]
            backtrack([2,3,1]): Complete! Add [2,3,1]
  
  i=2: Choose 3, used=[F,F,T]
    backtrack([3]):
      i=0: Choose 1, used=[T,F,T]
        backtrack([3,1]):
          i=1: Choose 2, used=[T,T,T]
            backtrack([3,1,2]): Complete! Add [3,1,2]
      
      i=1: Choose 2, used=[F,T,T]
        backtrack([3,2]):
          i=0: Choose 1, used=[T,T,T]
            backtrack([3,2,1]): Complete! Add [3,2,1]

Result: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
```

### Example 4: Balanced Parentheses

**Problem:** Generate all valid combinations of 3 pairs of parentheses.

**Input:** n = 3

**Solution Trace (partial):**

```
backtrack('', open=0, close=0):
  Add '(' → backtrack('(', open=1, close=0):
    Add '(' → backtrack('((', open=2, close=0):
      Add '(' → backtrack('(((', open=3, close=0):
        Add ')' → backtrack('((()', open=3, close=1):
          Add ')' → backtrack('((())', open=3, close=2):
            Add ')' → backtrack('((()))', open=3, close=3):
              Complete! Add "((()))"
      
      Add ')' → backtrack('(()', open=2, close=1):
        Add '(' → backtrack('(()(', open=3, close=1):
          Add ')' → backtrack('(()(),'backtrack, open=3, close=2):
            Add ')' → backtrack('(()())', open=3, close=3):
              Complete! Add "(()())"
        
        Add ')' → backtrack('(())', open=2, close=2):
          Add '(' → backtrack('(())(', open=3, close=2):
            Add ')' → backtrack('(())()', open=3, close=3):
              Complete! Add "(())()"
    
    Add ')' → backtrack('()', open=1, close=1):
      Add '(' → backtrack('()(', open=2, close=1):
        Add '(' → backtrack('()((', open=3, close=1):
          Add ')' → ... → Complete! Add "()(())"
        
        Add ')' → backtrack('()()', open=2, close=2):
          Add '(' → ... → Complete! Add "()()()"

Result: ["((()))", "(()())", "(())()", "()(())", "()()()"]
```

## Edge Cases

### 1. Empty Input

**Scenario:** Empty array or empty string.

**Challenge:** What is the subset/permutation of nothing?

**Solution:**

```python
def subsets(nums):
    if not nums:
        return [[]]  # Empty set has one subset: empty subset
    # Regular logic...

def permute(nums):
    if not nums:
        return [[]]  # One permutation of nothing: empty permutation
```

### 2. Single Element

**Scenario:** Array with one element [5].

**Challenge:** Base case for recursion.

**Solution:**
```python
# Subsets of [5]: [[], [5]]
# Permutations of [5]: [[5]]
# Handled naturally by the algorithm
```

### 3. All Identical Elements

**Scenario:** Array like [2, 2, 2, 2].

**Challenge:** Avoiding duplicate subsets/permutations.

**Solution:**

```python
# Must use subsetsWithDup approach
nums = [2, 2, 2, 2]
nums.sort()  # Already sorted

# Subsets: [[], [2], [2,2], [2,2,2], [2,2,2,2]]
# Skip duplicates at same recursion level
```

### 4. Very Large n (Exponential Explosion)

**Scenario:** n = 20 for subsets means 2^20 ≈ 1 million subsets.

**Challenge:** Memory and time constraints.

**Solution:**
```python
# No way around exponential time for these problems
# Consider:
# 1. Limit input size
# 2. Use generators to yield solutions one at a time
# 3. Add constraints to prune search space

def subsets_generator(nums):
    """Yield subsets one at a time."""
    def backtrack(index, current):
        yield current[:]
        for i in range(index, len(nums)):
            current.append(nums[i])
            yield from backtrack(i + 1, current)
            current.pop()
    
    yield from backtrack(0, [])
```

### 5. Invalid Constraints in Permutations

**Scenario:** Asking for permutations of length k > n.

**Challenge:** Impossible to create.

**Solution:**

```python
def permutations_k(nums, k):
    """Generate permutations of length k."""
    if k > len(nums):
        return []  # Impossible
    
    result = []
    used = [False] * len(nums)
    
    def backtrack(current):
        if len(current) == k:  # Stop at length k
            result.append(current[:])
            return
        
        for i in range(len(nums)):
            if used[i]:
                continue
            current.append(nums[i])
            used[i] = True
            backtrack(current)
            current.pop()
            used[i] = False
    
    backtrack([])
    return result
```

### 6. Parentheses with n = 0

**Scenario:** Generate parentheses with 0 pairs.

**Challenge:** Edge case for Catalan numbers.

**Solution:**

```python
def generateParenthesis(n):
    if n == 0:
        return [""]  # Empty string is valid with 0 pairs
    # Regular logic...
```

## Common Pitfalls

### ❌ Pitfall 1: Forgetting to Copy State When Adding to Results

**What happens:** All results reference same list, end up with duplicates or empty lists.

```python
# WRONG - References same list
def subsets_wrong(nums):
    result = []
    current = []
    
    def backtrack(index):
        result.append(current)  # BUG! All reference same list
        for i in range(index, len(nums)):
            current.append(nums[i])
            backtrack(i + 1)
            current.pop()
    
    backtrack(0)
    return result  # [[3], [3], [3], ...] all same reference!
```

**Correct approach:**

```python
# CORRECT - Copy the list
def subsets_correct(nums):
    result = []
    
    def backtrack(index, current):
        result.append(current[:])  # Copy with [:]
        # Or: result.append(list(current))
        # Or: result.append(current.copy())
        
        for i in range(index, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()
    
    backtrack(0, [])
    return result
```

### ❌ Pitfall 2: Not Backtracking (Undoing Choices)

**What happens:** State carries over between branches, wrong results.

```python
# WRONG - No backtracking
def permute_wrong(nums):
    result = []
    used = [False] * len(nums)
    current = []
    
    def backtrack():
        if len(current) == len(nums):
            result.append(current[:])
            return
        
        for i in range(len(nums)):
            if used[i]:
                continue
            current.append(nums[i])
            used[i] = True
            backtrack()
            # Missing: current.pop() and used[i] = False
    
    backtrack()
    return result  # Wrong permutations!
```

**Correct approach:**

```python
# CORRECT - Proper backtracking
def permute_correct(nums):
    result = []
    used = [False] * len(nums)
    
    def backtrack(current):
        if len(current) == len(nums):
            result.append(current[:])
            return
        
        for i in range(len(nums)):
            if used[i]:
                continue
            current.append(nums[i])
            used[i] = True
            backtrack(current)
            current.pop()        # Backtrack!
            used[i] = False      # Backtrack!
    
    backtrack([])
    return result
```

### ❌ Pitfall 3: Wrong Skip Condition for Duplicates

**What happens:** Either skip valid solutions or include duplicates.

```python
# WRONG - Incorrect skip logic
def subsetsWithDup_wrong(nums):
    nums.sort()
    result = []
    
    def backtrack(index, current):
        result.append(current[:])
        
        for i in range(index, len(nums)):
            # WRONG: if nums[i] == nums[i-1]: continue
            # This skips all duplicates, not just same-level ones
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()
    
    backtrack(0, [])
    return result  # Missing valid subsets!
```

**Correct approach:**

```python
# CORRECT - Skip only same-level duplicates
def subsetsWithDup_correct(nums):
    nums.sort()
    result = []
    
    def backtrack(index, current):
        result.append(current[:])
        
        for i in range(index, len(nums)):
            # Skip if same as previous at THIS level
            # i > index (not i > 0) is crucial!
            if i > index and nums[i] == nums[i - 1]:
                continue
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()
    
    backtrack(0, [])
    return result
```

### ❌ Pitfall 4: Using Wrong Index Range

**What happens:** Either infinite recursion or missing solutions.

```python
# WRONG - Using i instead of i+1
def subsets_wrong(nums):
    result = []
    
    def backtrack(index, current):
        result.append(current[:])
        
        for i in range(index, len(nums)):
            current.append(nums[i])
            backtrack(i, current)  # BUG! Should be i+1
            current.pop()
    
    backtrack(0, [])
    return result  # Infinite recursion or wrong results!
```

**Correct approach:**

```python
# CORRECT - Use i+1 for next index
def subsets_correct(nums):
    result = []
    
    def backtrack(index, current):
        result.append(current[:])
        
        for i in range(index, len(nums)):
            current.append(nums[i])
            backtrack(i + 1, current)  # Correct!
            current.pop()
    
    backtrack(0, [])
    return result
```

### ❌ Pitfall 5: Modifying Input Array

**What happens:** Unexpected side effects, input is changed.

```python
# WRONG - Modifying input
def permute_wrong(nums):
    result = []
    
    def backtrack(start):
        if start == len(nums):
            result.append(nums)  # BUG! Appending reference to input
            return
        
        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]
    
    backtrack(0)
    return result  # Input is modified! All results reference same list!
```

**Correct approach:**

```python
# CORRECT - Copy when adding to results
def permute_correct(nums):
    result = []
    
    def backtrack(start):
        if start == len(nums):
            result.append(nums[:])  # Copy!
            return
        
        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]
    
    backtrack(0)
    return result
```

## Variations and Extensions

### Variation 1: Combinations (Choose k from n)

**Description:** Generate all ways to choose k elements from n elements.

**When to use:** Lottery combinations, team selection, subset problems with size constraint.

**Implementation:**

```python
def combine(n: int, k: int) -> List[List[int]]:
    """
    Generate all combinations of k numbers from 1 to n.
    
    Time Complexity: O(C(n,k) * k)
    """
    result = []
    
    def backtrack(start: int, current: List[int]) -> None:
        # Pruning: if remaining elements + current < k, can't form combination
        if len(current) + (n - start + 1) < k:
            return
        
        if len(current) == k:
            result.append(current[:])
            return
        
        for i in range(start, n + 1):
            current.append(i)
            backtrack(i + 1, current)
            current.pop()
    
    backtrack(1, [])
    return result
```

### Variation 2: Combination Sum (With Repetition)

**Description:** Find all combinations that sum to target, can reuse elements.

**When to use:** Coin change, making change, partition problems.

**Implementation:**

```python
def combinationSum(candidates: List[int], target: int) -> List[List[int]]:
    """
    Find all unique combinations that sum to target.
    Same number can be used unlimited times.
    """
    result = []
    candidates.sort()  # For pruning
    
    def backtrack(start: int, current: List[int], remaining: int) -> None:
        if remaining == 0:
            result.append(current[:])
            return
        
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break  # Pruning: remaining candidates are too large
            
            current.append(candidates[i])
            backtrack(i, current, remaining - candidates[i])  # i, not i+1!
            current.pop()
    
    backtrack(0, [], target)
    return result
```

### Variation 3: N-Queens Problem

**Description:** Place n queens on n×n chessboard so none attack each other.

**When to use:** Constraint satisfaction, board game problems.

**Implementation:**

```python
def solveNQueens(n: int) -> List[List[str]]:
    """
    Find all valid n-queens configurations.
    
    Time Complexity: O(n!)
    """
    result = []
    board = [['.'] * n for _ in range(n)]
    cols = set()
    diag1 = set()  # row - col
    diag2 = set()  # row + col
    
    def backtrack(row: int) -> None:
        if row == n:
            result.append([''.join(row) for row in board])
            return
        
        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue
            
            # Choose
            board[row][col] = 'Q'
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            
            # Explore
            backtrack(row + 1)
            
            # Unchoose
            board[row][col] = '.'
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)
    
    backtrack(0)
    return result
```

### Variation 4: Sudoku Solver

**Description:** Fill 9×9 Sudoku grid following all constraints.

**When to use:** Puzzle solving, constraint satisfaction.

### Variation 5: Word Search II (Trie + Backtracking)

**Description:** Find all words from dictionary in 2D board.

**When to use:** Word games, pattern matching in grids.

## Practice Problems

### Beginner

1. **Subsets** - Generate all subsets
   - LeetCode #78

2. **Permutations** - Generate all permutations
   - LeetCode #46

3. **Combinations** - Choose k from n
   - LeetCode #77

4. **Letter Case Permutation** - Toggle letter cases
   - LeetCode #784

### Intermediate

1. **Subsets II** - Subsets with duplicates
   - LeetCode #90

2. **Permutations II** - Permutations with duplicates
   - LeetCode #47

3. **Combination Sum** - Find combinations summing to target
   - LeetCode #39

4. **Combination Sum II** - With duplicates, no repetition
   - LeetCode #40

5. **Generate Parentheses** - Valid parentheses combinations
   - LeetCode #22

6. **Palindrome Partitioning** - Partition string into palindromes
   - LeetCode #131

7. **Letter Combinations of Phone Number** - Map digits to letters
   - LeetCode #17

### Advanced

1. **N-Queens** - Place n queens on board
   - LeetCode #51

2. **Sudoku Solver** - Solve Sudoku puzzle
   - LeetCode #37

3. **Word Search II** - Find dictionary words in board
   - LeetCode #212

4. **Expression Add Operators** - Insert operators to reach target
   - LeetCode #282

5. **Generalized Abbreviation** - Generate all abbreviations
   - LeetCode #320 (Premium)

6. **Different Ways to Add Parentheses** - Evaluate expressions
   - LeetCode #241

7. **Beautiful Arrangement** - Permutations with divisibility constraints
   - LeetCode #526

## Real-World Applications

### Industry Use Cases

1. **Test Case Generation:** Software testing tools use combinatorial algorithms to generate comprehensive test suites covering all possible input combinations, ensuring thorough code coverage.

2. **Cryptography:** Password cracking tools use permutation generation to try all possible character combinations. Combinatorial algorithms also design cryptographic protocols.

3. **Scheduling Systems:** Employee shift schedulers, university course timetabling, and meeting room allocation use backtracking to find valid schedules satisfying all constraints.

4. **Game AI:** Chess engines, puzzle solvers (Sudoku, crosswords), and game tree search use backtracking to explore possible moves and find optimal strategies.

5. **Bioinformatics:** DNA sequence analysis generates all possible mutations, RNA folding algorithms explore conformations, and protein structure prediction uses combinatorial search.

### Popular Implementations

- **SAT Solvers:** Boolean satisfiability problem solvers use advanced backtracking
  - Used in hardware verification, software verification, AI planning

- **Constraint Programming Libraries (Google OR-Tools):** Solve scheduling and optimization
  - Powers Google's logistics, workforce management

- **Game Engines (Unity, Unreal):** Procedural content generation
  - Level design, item combination systems

- **Bioinformatics Tools (BLAST, ClustalW):** Sequence alignment
  - DNA/protein analysis pipelines

### Practical Scenarios

- **E-commerce Product Recommendations:** Generate all compatible product bundles
- **Travel Planning:** Find all possible itineraries within constraints
- **Circuit Design:** Test all possible logic gate configurations
- **Menu Planning:** Generate meal combinations meeting dietary requirements
- **Fantasy Sports:** Create optimal team lineups from player pool

## Related Topics

### Prerequisites to Review

- **Recursion** - Fundamental technique for backtracking
- **Arrays and Lists** - Data structures for building solutions
- **Strings** - Many backtracking problems involve strings
- **Trees** - Understanding recursive tree traversal helps
- **Time Complexity** - Understanding exponential growth

### Next Steps

- **Dynamic Programming** - Optimizing overlapping subproblems (subset sum, knapsack)
- **Branch and Bound** - Optimized backtracking with pruning
- **Constraint Satisfaction Problems** - Systematic approach to backtracking
- **Graph Algorithms** - DFS is a form of backtracking
- **Bit Manipulation** - Efficient subset generation using bits

### Similar Concepts

- **Depth-First Search** - Backtracking is DFS with state management
- **Recursion Trees** - Visualizing backtracking as tree traversal
- **State Space Search** - General framework for exploration
- **Greedy Algorithms** - Alternative to exhaustive search (sometimes)

### Further Reading

- "Introduction to Algorithms" (CLRS) - Chapter on backtracking
- "Algorithm Design Manual" by Skiena - Backtracking chapter with examples
- "Cracking the Coding Interview" - Recursion and backtracking section
- [Backtracking Visualization](https://www.cs.usfca.edu/~galles/visualization/RecursiveBacktrack.html)
- [LeetCode Backtracking Problems](https://leetcode.com/tag/backtracking/)
- "Competitive Programming 4" - Advanced backtracking techniques
