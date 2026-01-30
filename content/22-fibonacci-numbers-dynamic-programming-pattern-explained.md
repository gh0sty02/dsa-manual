# Fibonacci Numbers (Dynamic Programming) Pattern

**Difficulty:** Beginner to Intermediate
**Prerequisites:** Recursion, Basic DP concepts, Arrays
**Estimated Reading Time:** 25 minutes

## Introduction

The Fibonacci Numbers pattern is the gateway to understanding dynamic programming. It represents a class of problems where the solution to a problem depends on solutions to smaller instances of the same problem, with overlapping subproblems that can be cached for efficiency. This pattern teaches the fundamental DP transformation: turning exponential-time recursive solutions into linear or polynomial-time solutions through memoization or tabulation.

**Why it matters:** This pattern is the "Hello World" of dynamic programming, but don't let its simplicity fool you. The techniques learned here—identifying recurrence relations, recognizing overlapping subproblems, and choosing between top-down and bottom-up approaches—are foundational skills that apply to virtually every DP problem you'll encounter. Companies use variations of this pattern in areas from financial modeling to game development to pathfinding algorithms.

**Real-world analogy:** Imagine climbing a staircase where you can take either 1 or 2 steps at a time. To figure out how many ways you can reach step N, you realize: "I can get to step N either from step N-1 (taking 1 step) or from step N-2 (taking 2 steps)." So the number of ways to reach N is the sum of ways to reach N-1 and N-2. This is exactly the Fibonacci recurrence! Instead of recalculating the same staircase positions over and over, you write down the answer for each step as you go—that's dynamic programming in action.

## Core Concepts

### Key Principles

1. **Recurrence relation:** The solution can be expressed as a function of solutions to smaller subproblems. For Fibonacci: F(n) = F(n-1) + F(n-2).

2. **Overlapping subproblems:** The same subproblems are solved multiple times in naive recursion. F(5) needs F(4) and F(3), but F(4) also needs F(3)—we compute F(3) twice!

3. **Optimal substructure:** The optimal solution contains optimal solutions to subproblems. Knowing F(n-1) and F(n-2) is sufficient to compute F(n).

4. **Memoization vs Tabulation:**
   - **Top-down (Memoization):** Recursive with caching
   - **Bottom-up (Tabulation):** Iterative table building
   - Both convert O(2^n) to O(n)

### Essential Terms

- **Fibonacci sequence:** 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55...
- **Base cases:** F(0) = 0, F(1) = 1 (or sometimes F(1) = F(2) = 1)
- **State:** The input parameter(s) that define a subproblem (e.g., n in F(n))
- **Transition:** How to compute current state from previous states
- **Memoization:** Caching results of function calls
- **Tabulation:** Building solution bottom-up in a table
- **State space:** Set of all possible states (0 to n)

### Visual Overview

```
Fibonacci Sequence:
n:     0  1  2  3  4  5  6   7   8   9   10  ...
F(n):  0  1  1  2  3  5  8  13  21  34  55  ...

Naive Recursion Tree for F(5):
                        F(5)
                    /          \
                F(4)              F(3)
              /      \          /      \
          F(3)      F(2)      F(2)    F(1)
         /   \      /   \     /   \
      F(2)  F(1) F(1) F(0) F(1) F(0)
      /  \
   F(1) F(0)

Problem: F(3) computed 2 times, F(2) computed 3 times, F(1) computed 5 times!
Total calls: 15 for F(5) → O(2^n) complexity

DP Solution (Memoization):
                        F(5)
                    /          \
                F(4)              F(3) ← cached!
              /      \          
          F(3)      F(2) ← cached!
         /   \      
      F(2)  F(1) ← cached!
      /  \
   F(1) F(0) ← all cached after first computation

Total calls: 6 for F(5) → O(n) complexity

DP Solution (Tabulation):
Build table: [0, 1, ?, ?, ?, ?]
            [0, 1, 1, ?, ?, ?]  ← F(2) = F(1) + F(0) = 1
            [0, 1, 1, 2, ?, ?]  ← F(3) = F(2) + F(1) = 2
            [0, 1, 1, 2, 3, ?]  ← F(4) = F(3) + F(2) = 3
            [0, 1, 1, 2, 3, 5]  ← F(5) = F(4) + F(3) = 5
```

**Key Insight:** By storing results of subproblems, we avoid redundant computation and reduce time complexity from exponential to linear.

## How It Works

### Algorithm Steps (Bottom-Up Tabulation)

1. **Create array/table** to store results
   - Size n+1 to hold F(0) through F(n)
   
2. **Initialize base cases**
   - F(0) = 0
   - F(1) = 1
   
3. **Iterate from 2 to n**
   - For each i: F(i) = F(i-1) + F(i-2)
   - Use previously computed values
   
4. **Return F(n)**
   - Final value in table

### Algorithm Steps (Top-Down Memoization)

1. **Create cache/memo dictionary**
   - Store computed values
   
2. **Base cases in recursive function**
   - If n ≤ 1, return n
   
3. **Check cache before computing**
   - If F(n) already in cache, return it
   
4. **Compute recursively**
   - F(n) = fib(n-1) + fib(n-2)
   - Store result in cache
   
5. **Return cached result**

### Visual Walkthrough: Computing F(6)

**Bottom-up approach:**

```
Step 0: Initialize
dp = [0, 1, ?, ?, ?, ?, ?]
     F(0) F(1) F(2) F(3) F(4) F(5) F(6)

Step 1: Compute F(2)
F(2) = F(1) + F(0) = 1 + 0 = 1
dp = [0, 1, 1, ?, ?, ?, ?]

Step 2: Compute F(3)
F(3) = F(2) + F(1) = 1 + 1 = 2
dp = [0, 1, 1, 2, ?, ?, ?]

Step 3: Compute F(4)
F(4) = F(3) + F(2) = 2 + 1 = 3
dp = [0, 1, 1, 2, 3, ?, ?]

Step 4: Compute F(5)
F(5) = F(4) + F(3) = 3 + 2 = 5
dp = [0, 1, 1, 2, 3, 5, ?]

Step 5: Compute F(6)
F(6) = F(5) + F(4) = 5 + 3 = 8
dp = [0, 1, 1, 2, 3, 5, 8]

Result: F(6) = 8
```

**State Transition Table:**

| i | F(i-2) | F(i-1) | F(i) = F(i-1) + F(i-2) |
|---|--------|--------|------------------------|
| 0 | - | - | 0 (base) |
| 1 | - | - | 1 (base) |
| 2 | 0 | 1 | 1 |
| 3 | 1 | 1 | 2 |
| 4 | 1 | 2 | 3 |
| 5 | 2 | 3 | 5 |
| 6 | 3 | 5 | 8 |

**Top-down with memoization:**

```
Call fib(6):
├─ Check memo: {} (empty)
├─ Call fib(5):
│  ├─ Check memo: {} (empty)
│  ├─ Call fib(4):
│  │  ├─ Check memo: {} (empty)
│  │  ├─ Call fib(3):
│  │  │  ├─ Check memo: {} (empty)
│  │  │  ├─ Call fib(2):
│  │  │  │  ├─ Check memo: {} (empty)
│  │  │  │  ├─ Call fib(1): return 1 (base)
│  │  │  │  ├─ Call fib(0): return 0 (base)
│  │  │  │  ├─ Compute: 1 + 0 = 1
│  │  │  │  └─ Store memo[2] = 1, return 1
│  │  │  ├─ Call fib(1): return 1 (base)
│  │  │  ├─ Compute: 1 + 1 = 2
│  │  │  └─ Store memo[3] = 2, return 2
│  │  ├─ Call fib(2): return 1 (from memo!) ✓
│  │  ├─ Compute: 2 + 1 = 3
│  │  └─ Store memo[4] = 3, return 3
│  ├─ Call fib(3): return 2 (from memo!) ✓
│  ├─ Compute: 3 + 2 = 5
│  └─ Store memo[5] = 5, return 5
├─ Call fib(4): return 3 (from memo!) ✓
├─ Compute: 5 + 3 = 8
└─ Store memo[6] = 8, return 8

Memo after execution: {2:1, 3:2, 4:3, 5:5, 6:8}
Function calls: 9 (vs 25 for naive recursion)
```

## Implementation

### Python Implementation

```python
from typing import List, Dict
import functools

def fibonacci_naive(n: int) -> int:
    """
    Naive recursive Fibonacci - VERY SLOW!
    
    This is for educational purposes only.
    
    Args:
        n: Position in Fibonacci sequence
        
    Returns:
        nth Fibonacci number
        
    Time Complexity: O(2^n) - exponential!
    Space Complexity: O(n) - recursion stack
    
    Example:
        >>> fibonacci_naive(10)
        55
    """
    if n <= 1:
        return n
    return fibonacci_naive(n - 1) + fibonacci_naive(n - 2)


def fibonacci_memoization(n: int) -> int:
    """
    Fibonacci using top-down DP (memoization).
    
    Args:
        n: Position in Fibonacci sequence
        
    Returns:
        nth Fibonacci number
        
    Time Complexity: O(n)
    Space Complexity: O(n) for memo + O(n) recursion stack
    
    Example:
        >>> fibonacci_memoization(10)
        55
    """
    memo: Dict[int, int] = {}
    
    def fib_helper(n: int) -> int:
        # Base cases
        if n <= 1:
            return n
        
        # Check cache
        if n in memo:
            return memo[n]
        
        # Compute and store
        memo[n] = fib_helper(n - 1) + fib_helper(n - 2)
        return memo[n]
    
    return fib_helper(n)


def fibonacci_tabulation(n: int) -> int:
    """
    Fibonacci using bottom-up DP (tabulation).
    
    This is the standard, most efficient approach.
    
    Args:
        n: Position in Fibonacci sequence
        
    Returns:
        nth Fibonacci number
        
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Example:
        >>> fibonacci_tabulation(10)
        55
    """
    if n <= 1:
        return n
    
    # Create DP table
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 1
    
    # Fill table
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]


def fibonacci_space_optimized(n: int) -> int:
    """
    Fibonacci with O(1) space - only store last two values.
    
    Key insight: We only need previous two values, not entire array.
    
    Args:
        n: Position in Fibonacci sequence
        
    Returns:
        nth Fibonacci number
        
    Time Complexity: O(n)
    Space Complexity: O(1) - constant space!
    
    Example:
        >>> fibonacci_space_optimized(10)
        55
    """
    if n <= 1:
        return n
    
    prev2 = 0  # F(0)
    prev1 = 1  # F(1)
    
    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1


def fibonacci_decorator(n: int) -> int:
    """
    Fibonacci using @functools.lru_cache decorator.
    
    Python's built-in memoization - very clean!
    
    Args:
        n: Position in Fibonacci sequence
        
    Returns:
        nth Fibonacci number
        
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Example:
        >>> fibonacci_decorator(10)
        55
    """
    @functools.lru_cache(maxsize=None)
    def fib(n: int) -> int:
        if n <= 1:
            return n
        return fib(n - 1) + fib(n - 2)
    
    return fib(n)


def staircase_ways(n: int) -> int:
    """
    Count ways to climb n stairs taking 1 or 2 steps at a time.
    This is exactly the Fibonacci problem!
    
    Args:
        n: Number of stairs
        
    Returns:
        Number of distinct ways to reach top
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Example:
        >>> staircase_ways(4)
        5  # [1,1,1,1], [1,1,2], [1,2,1], [2,1,1], [2,2]
        
    Recurrence: ways(n) = ways(n-1) + ways(n-2)
    - ways(n-1): reach n by taking 1 step from stair n-1
    - ways(n-2): reach n by taking 2 steps from stair n-2
    """
    if n <= 2:
        return n
    
    prev2 = 1  # ways(1) = 1
    prev1 = 2  # ways(2) = 2
    
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1


def staircase_ways_k_steps(n: int, k: int) -> int:
    """
    Count ways to climb n stairs taking 1, 2, ..., k steps at a time.
    Extension of Fibonacci pattern.
    
    Args:
        n: Number of stairs
        k: Maximum steps allowed at once
        
    Returns:
        Number of distinct ways to reach top
        
    Time Complexity: O(n × k)
    Space Complexity: O(n)
    
    Example:
        >>> staircase_ways_k_steps(4, 3)
        7  # Can take 1, 2, or 3 steps
        
    Recurrence: ways(n) = sum(ways(n-i) for i in 1..min(k,n))
    """
    if n <= 1:
        return 1
    
    dp = [0] * (n + 1)
    dp[0] = 1  # One way to stay at ground (do nothing)
    
    for i in range(1, n + 1):
        # Sum all ways from previous k positions
        for step in range(1, min(k, i) + 1):
            dp[i] += dp[i - step]
    
    return dp[i]


def number_factors(n: int) -> int:
    """
    Count ways to express n as sum of 1, 3, and 4.
    Another Fibonacci-like problem.
    
    Args:
        n: Target number
        
    Returns:
        Number of ways to express n
        
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Example:
        >>> number_factors(4)
        4  # [1,1,1,1], [1,3], [3,1], [4]
        >>> number_factors(5)
        6  # [1,1,1,1,1], [1,1,3], [1,3,1], [3,1,1], [1,4], [4,1]
        
    Recurrence: ways(n) = ways(n-1) + ways(n-3) + ways(n-4)
    """
    if n <= 2:
        return 1  # Only using 1's
    if n == 3:
        return 2  # [1,1,1] or [3]
    
    dp = [0] * (n + 1)
    dp[0] = 1  # Empty sum
    dp[1] = 1  # [1]
    dp[2] = 1  # [1,1]
    dp[3] = 2  # [1,1,1], [3]
    
    for i in range(4, n + 1):
        dp[i] = dp[i - 1] + dp[i - 3] + dp[i - 4]
    
    return dp[n]


def min_jumps_to_end(nums: List[int]) -> int:
    """
    Minimum jumps to reach end of array.
    Each element is max jump length from that position.
    
    This uses Fibonacci-like DP but finds minimum instead of count.
    
    Args:
        nums: Array where nums[i] is max jump from position i
        
    Returns:
        Minimum jumps to reach end, or -1 if impossible
        
    Time Complexity: O(n²)
    Space Complexity: O(n)
    
    Example:
        >>> min_jumps_to_end([2, 3, 1, 1, 4])
        2  # Jump 1 step to index 1, then 3 steps to end
        >>> min_jumps_to_end([3, 2, 1, 0, 4])
        -1  # Cannot reach end (stuck at index 3)
        
    Recurrence: jumps(i) = 1 + min(jumps(j) for all j that can reach i)
    """
    n = len(nums)
    if n <= 1:
        return 0
    
    # dp[i] = minimum jumps to reach index i
    dp = [float('inf')] * n
    dp[0] = 0  # Already at start
    
    for i in range(1, n):
        # Check all positions that can jump to i
        for j in range(i):
            if j + nums[j] >= i:
                dp[i] = min(dp[i], dp[j] + 1)
    
    return dp[n - 1] if dp[n - 1] != float('inf') else -1


def min_cost_climbing_stairs(cost: List[int]) -> int:
    """
    Minimum cost to reach top of stairs.
    Can start from step 0 or 1, can climb 1 or 2 steps at a time.
    
    Args:
        cost: Cost of stepping on each stair
        
    Returns:
        Minimum cost to reach top (beyond last stair)
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Example:
        >>> min_cost_climbing_stairs([10, 15, 20])
        15  # Start at index 1, pay 15, jump to top
        >>> min_cost_climbing_stairs([1, 100, 1, 1, 1, 100, 1, 1, 100, 1])
        6  # Path: 0→2→3→4→6→7→9→top, cost: 1+1+1+1+1+1=6
        
    Recurrence: minCost(i) = cost[i] + min(minCost(i-1), minCost(i-2))
    """
    n = len(cost)
    if n <= 1:
        return 0 if n == 0 else cost[0]
    
    prev2 = cost[0]
    prev1 = cost[1]
    
    for i in range(2, n):
        current = cost[i] + min(prev1, prev2)
        prev2 = prev1
        prev1 = current
    
    # Can step from n-1 or n-2 to reach top
    return min(prev1, prev2)


def house_robber(nums: List[int]) -> int:
    """
    House Robber: Rob houses to maximize money without robbing adjacent.
    Classic Fibonacci-like DP problem.
    
    Args:
        nums: Money in each house
        
    Returns:
        Maximum money that can be robbed
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Example:
        >>> house_robber([1, 2, 3, 1])
        4  # Rob houses 0 and 2: 1 + 3 = 4
        >>> house_robber([2, 7, 9, 3, 1])
        12  # Rob houses 0, 2, 4: 2 + 9 + 1 = 12
        
    Recurrence: rob(i) = max(rob(i-1), nums[i] + rob(i-2))
    - rob(i-1): skip current house
    - nums[i] + rob(i-2): rob current, skip previous
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    
    prev2 = nums[0]
    prev1 = max(nums[0], nums[1])
    
    for i in range(2, len(nums)):
        current = max(prev1, nums[i] + prev2)
        prev2 = prev1
        prev1 = current
    
    return prev1


# Usage Examples
if __name__ == "__main__":
    # Example 1: Basic Fibonacci
    n = 10
    print(f"Fibonacci({n}):")
    print(f"  Tabulation: {fibonacci_tabulation(n)}")
    print(f"  Memoization: {fibonacci_memoization(n)}")
    print(f"  Space-optimized: {fibonacci_space_optimized(n)}")
    # Output: 55
    
    # Example 2: Staircase problem
    stairs = 5
    print(f"\nWays to climb {stairs} stairs (1 or 2 steps):")
    print(f"  {staircase_ways(stairs)} ways")
    # Output: 8 ways
    
    # Example 3: Staircase with k steps
    print(f"\nWays to climb {stairs} stairs (1, 2, or 3 steps):")
    print(f"  {staircase_ways_k_steps(stairs, 3)} ways")
    # Output: 13 ways
    
    # Example 4: Number factors
    num = 5
    print(f"\nWays to express {num} as sum of 1, 3, 4:")
    print(f"  {number_factors(num)} ways")
    # Output: 6 ways
    
    # Example 5: Minimum jumps
    array = [2, 3, 1, 1, 4]
    print(f"\nMinimum jumps to end of {array}:")
    print(f"  {min_jumps_to_end(array)} jumps")
    # Output: 2 jumps
    
    # Example 6: Min cost climbing stairs
    costs = [10, 15, 20]
    print(f"\nMin cost to climb stairs {costs}:")
    print(f"  {min_cost_climbing_stairs(costs)}")
    # Output: 15
    
    # Example 7: House robber
    houses = [2, 7, 9, 3, 1]
    print(f"\nMax money from robbing {houses}:")
    print(f"  ${house_robber(houses)}")
    # Output: $12
```

### Code Explanation

**Key Design Decisions:**

1. **Why three variables (prev2, prev1, current) in space-optimized version?**
   - We only need last two Fibonacci numbers to compute next
   - No need to store entire array
   - Reduces space from O(n) to O(1)

2. **Why start loop from index 2 in tabulation?**
   - Base cases F(0) and F(1) are already set
   - F(2) is first value we need to compute
   - Pattern: dp[i] depends on dp[i-1] and dp[i-2]

3. **When to use memoization vs tabulation?**
   - **Memoization:** When not all subproblems needed (sparse computation)
   - **Tabulation:** When all subproblems needed, slightly faster (no recursion overhead)
   - **For Fibonacci:** Both compute all subproblems, so tabulation slightly better

4. **Why @lru_cache is convenient?**
   - Python's built-in memoization decorator
   - Automatically handles caching
   - Clean, Pythonic code
   - Good for interview coding speed

5. **How staircase relates to Fibonacci?**
   - To reach stair n, you can come from stair n-1 or n-2
   - Same recurrence: ways(n) = ways(n-1) + ways(n-2)
   - Just different base cases (ways(1)=1, ways(2)=2 instead of F(0)=0, F(1)=1)

## Complexity Analysis

### Time Complexity

**Naive Recursion:**
- **Recurrence:** T(n) = T(n-1) + T(n-2) + O(1)
- **Solution:** T(n) = O(φ^n) where φ ≈ 1.618 (golden ratio)
- **Simplified:** **O(2^n)** - exponential!
- **Why:** Each call spawns 2 more calls, forming binary tree of depth n

**Memoization:**
- **First call:** O(n) - computes each F(i) once for i = 0 to n
- **Cached calls:** O(1) - direct lookup
- **Overall:** **O(n)** - linear time
- **Why:** Each unique subproblem computed exactly once

**Tabulation:**
- **Loop:** O(n) iterations
- **Each iteration:** O(1) work
- **Overall:** **O(n)** - linear time
- **Why:** Single pass through array, constant work per element

**Space-optimized:**
- **Loop:** O(n) iterations
- **Each iteration:** O(1) work
- **Overall:** **O(n)** - linear time

### Space Complexity

**Naive Recursion:**
- **Recursion stack:** O(n) - maximum depth of recursion tree
- **No memoization:** O(1) auxiliary space
- **Overall:** **O(n)**

**Memoization:**
- **Memo dict:** O(n) - stores F(0) through F(n)
- **Recursion stack:** O(n) - call stack depth
- **Overall:** **O(n)**

**Tabulation:**
- **DP array:** O(n) - stores all values
- **Overall:** **O(n)**

**Space-optimized:**
- **Variables:** O(1) - only prev2, prev1, current
- **Overall:** **O(1)** - constant space!

### Comparison with Alternatives

| Approach | Time | Space | When to Use |
|----------|------|-------|-------------|
| **Naive Recursion** | O(2^n) | O(n) | Never! Only for learning |
| **Memoization** | O(n) | O(n) | When not all subproblems needed; clearer code |
| **Tabulation** | O(n) | O(n) | Standard DP; when all subproblems needed |
| **Space-Optimized** | O(n) | O(1) | Production code; memory-constrained |
| **Matrix Exponentiation** | O(log n) | O(1) | Very large n (n > 10^6) |
| **Closed Form (Binet's)** | O(1) | O(1) | Mathematical, floating point issues |

**When each approach shines:**
- **n < 20:** Any approach works
- **20 ≤ n < 10^6:** Space-optimized DP
- **n ≥ 10^6:** Matrix exponentiation or closed form
- **Interview:** Start with memoization/tabulation, optimize to O(1) space if asked

## Examples

### Example 1: Classic Fibonacci F(7)

**Problem:** Compute F(7)

**Solution (Tabulation):**
```python
n = 7
dp = [0, 1, 0, 0, 0, 0, 0, 0]

i=2: dp[2] = dp[1] + dp[0] = 1 + 0 = 1
     dp = [0, 1, 1, 0, 0, 0, 0, 0]

i=3: dp[3] = dp[2] + dp[1] = 1 + 1 = 2
     dp = [0, 1, 1, 2, 0, 0, 0, 0]

i=4: dp[4] = dp[3] + dp[2] = 2 + 1 = 3
     dp = [0, 1, 1, 2, 3, 0, 0, 0]

i=5: dp[5] = dp[4] + dp[3] = 3 + 2 = 5
     dp = [0, 1, 1, 2, 3, 5, 0, 0]

i=6: dp[6] = dp[5] + dp[4] = 5 + 3 = 8
     dp = [0, 1, 1, 2, 3, 5, 8, 0]

i=7: dp[7] = dp[6] + dp[5] = 8 + 5 = 13
     dp = [0, 1, 1, 2, 3, 5, 8, 13]

Result: F(7) = 13
```

**Solution (Space-Optimized):**
```python
n = 7
prev2 = 0, prev1 = 1

i=2: current = 0 + 1 = 1, prev2 = 1, prev1 = 1
i=3: current = 1 + 1 = 2, prev2 = 1, prev1 = 2
i=4: current = 2 + 1 = 3, prev2 = 2, prev1 = 3
i=5: current = 3 + 2 = 5, prev2 = 3, prev1 = 5
i=6: current = 5 + 3 = 8, prev2 = 5, prev1 = 8
i=7: current = 8 + 5 = 13, prev2 = 8, prev1 = 13

Result: F(7) = 13
```

### Example 2: Staircase - 5 Steps

**Problem:** How many ways to climb 5 stairs (1 or 2 steps at a time)?

**Solution:**
```python
n = 5
Base: ways(1) = 1, ways(2) = 2

dp = [0, 1, 2, 0, 0, 0]

i=3: dp[3] = dp[2] + dp[1] = 2 + 1 = 3
     Ways: [1,1,1], [1,2], [2,1]
     dp = [0, 1, 2, 3, 0, 0]

i=4: dp[4] = dp[3] + dp[2] = 3 + 2 = 5
     Ways: [1,1,1,1], [1,1,2], [1,2,1], [2,1,1], [2,2]
     dp = [0, 1, 2, 3, 5, 0]

i=5: dp[5] = dp[4] + dp[3] = 5 + 3 = 8
     Ways: [1,1,1,1,1], [1,1,1,2], [1,1,2,1], [1,2,1,1],
           [2,1,1,1], [1,2,2], [2,1,2], [2,2,1]
     dp = [0, 1, 2, 3, 5, 8]

Result: 8 ways
```

**Visualization:**
```
Climbing 5 stairs:

From stair 4: take 1 step → 5 ways reach 4, each adds 1 step to reach 5
    [1,1,1,1]+1, [1,1,2]+1, [1,2,1]+1, [2,1,1]+1, [2,2]+1

From stair 3: take 2 steps → 3 ways reach 3, each adds 2 steps to reach 5
    [1,1,1]+2, [1,2]+2, [2,1]+2

Total: 5 + 3 = 8 ways
```

### Example 3: Number Factors - Express 5

**Problem:** Express 5 using 1, 3, 4. How many ways?

**Solution:**
```python
n = 5
dp[0] = 1  # empty
dp[1] = 1  # [1]
dp[2] = 1  # [1,1]
dp[3] = 2  # [1,1,1], [3]

i=4: dp[4] = dp[3] + dp[1] + dp[0]
           = 2 + 1 + 1 = 4
     Ways: [1,1,1,1], [1,3], [3,1], [4]

i=5: dp[5] = dp[4] + dp[2] + dp[1]
           = 4 + 1 + 1 = 6
     Ways: [1,1,1,1,1], [1,1,3], [1,3,1], [3,1,1], [1,4], [4,1]

Result: 6 ways
```

**Breaking down dp[5]:**
```
From dp[4]: Take each of 4 ways, add 1
  [1,1,1,1]+1 = [1,1,1,1,1]
  [1,3]+1     = [1,3,1]
  [3,1]+1     = [3,1,1]
  [4]+1       = [4,1]

From dp[2]: Take the way, add 3
  [1,1]+3     = [1,1,3]

From dp[1]: Take the way, add 4
  [1]+4       = [1,4]

Total: 6 ways
```

### Example 4: House Robber - [2,7,9,3,1]

**Problem:** Rob houses to maximize money without robbing adjacent.

**Solution:**
```python
houses = [2, 7, 9, 3, 1]

dp[0] = 2  # Rob house 0

dp[1] = max(dp[0], houses[1])
      = max(2, 7) = 7  # Skip house 0, rob house 1

dp[2] = max(dp[1], houses[2] + dp[0])
      = max(7, 9 + 2) = 11  # Rob houses 0 and 2

dp[3] = max(dp[2], houses[3] + dp[1])
      = max(11, 3 + 7) = 11  # Keep previous max

dp[4] = max(dp[3], houses[4] + dp[2])
      = max(11, 1 + 11) = 12  # Rob houses 0, 2, 4

Result: Maximum money = $12
Path: Rob houses [0, 2, 4] → 2 + 9 + 1 = 12
```

**Decision tree:**
```
At house 0: rob (2) or skip (0) → rob (2)
At house 1: rob (7) + skip(0) or skip + rob(2) → rob (7)
At house 2: rob (9) + rob(2) or skip + rob(7) → 11 (rob 0,2)
At house 3: rob (3) + rob(7) or skip + rob(11) → 11 (keep prev)
At house 4: rob (1) + rob(11) or skip + rob(11) → 12 (rob 0,2,4)
```

## Edge Cases

### 1. n = 0
**Scenario:** Fibonacci of 0

**Challenge:** What is F(0)?

**Solution:** By definition, F(0) = 0

```python
def fibonacci(n):
    if n <= 1:
        return n  # Handles both 0 and 1
```

### 2. n = 1
**Scenario:** Fibonacci of 1

**Challenge:** Base case

**Solution:** F(1) = 1

```python
# Already handled by n <= 1 check
```

### 3. Negative n
**Scenario:** Fibonacci of negative number

**Challenge:** Undefined in standard sequence

**Solution:** Either return error or use extended Fibonacci

```python
def fibonacci(n):
    if n < 0:
        raise ValueError("Fibonacci not defined for negative numbers")
    if n <= 1:
        return n
    # ... rest of code
```

**Extended Fibonacci (optional):**
```python
# F(-n) = (-1)^(n+1) * F(n)
# F(-1) = 1, F(-2) = -1, F(-3) = 2, F(-4) = -3, ...
```

### 4. Very Large n
**Scenario:** n = 1000000

**Challenge:** Result exceeds integer limits; computation time

**Solution:** Use modulo arithmetic or return string

```python
def fibonacci_large(n, mod=10**9+7):
    """Compute F(n) mod 'mod' for large n."""
    if n <= 1:
        return n
    
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        current = (prev1 + prev2) % mod
        prev2, prev1 = prev1, current
    
    return prev1
```

### 5. Empty Array for Problems
**Scenario:** house_robber([])

**Challenge:** No houses to rob

**Solution:** Return 0

```python
def house_robber(nums):
    if not nums:
        return 0
    # ... rest of code
```

### 6. Single Element Array
**Scenario:** house_robber([5])

**Challenge:** Only one house

**Solution:** Rob it

```python
def house_robber(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    # ... rest of code
```

### 7. All Zeros
**Scenario:** min_cost_climbing_stairs([0,0,0,0])

**Challenge:** All costs are zero

**Solution:** Algorithm works, returns 0

```python
# No special handling needed
# min(0, 0) = 0 throughout
```

### 8. Stairs with k=1
**Scenario:** staircase_ways_k_steps(5, 1) - can only take 1 step

**Challenge:** Only one way (all 1's)

**Solution:** Works correctly, returns 1

```python
# When k=1, only sum previous 1 position
# Result: dp[i] = dp[i-1] for all i
# Only one path: [1,1,1,1,1]
```

## Common Pitfalls

### ❌ Pitfall 1: Not Handling Base Cases

**What happens:**
```python
# WRONG: Missing base case check
def fibonacci_wrong(n):
    dp = [0] * (n + 1)  # Crashes when n=0!
    dp[0] = 0
    dp[1] = 1  # IndexError if n=0
    
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]
```

**Why it's wrong:** When n=0, array has size 1, but we try to access dp[1].

**Correct approach:**
```python
def fibonacci_correct(n):
    if n <= 1:
        return n  # Handle base cases
    
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 1
    # ... rest of code
```

### ❌ Pitfall 2: Incorrect Variable Update Order

**What happens:**
```python
# WRONG: Updating in wrong order
def fibonacci_wrong(n):
    if n <= 1:
        return n
    
    prev2 = 0
    prev1 = 1
    
    for i in range(2, n + 1):
        prev1 = prev1 + prev2  # Overwrites prev1!
        prev2 = prev1  # Now prev2 = new value, not old prev1!
        # Lost the original prev1 value
    
    return prev1
```

**Why it's wrong:** Overwriting prev1 before using it to update prev2.

**Correct approach:**
```python
def fibonacci_correct(n):
    # ... base cases ...
    
    for i in range(2, n + 1):
        current = prev1 + prev2  # Compute first
        prev2 = prev1  # Then update
        prev1 = current
    
    return prev1
```

**Alternative (Python swap):**
```python
for i in range(2, n + 1):
    prev2, prev1 = prev1, prev1 + prev2
```

### ❌ Pitfall 3: Off-by-One in Loop Range

**What happens:**
```python
# WRONG: Loop goes to n-1 instead of n
def fibonacci_wrong(n):
    if n <= 1:
        return n
    
    dp = [0] * (n + 1)
    dp[0] = 0
    dp[1] = 1
    
    for i in range(2, n):  # Should be n+1!
        dp[i] = dp[i-1] + dp[i-2]
    
    return dp[n]  # dp[n] was never computed!
```

**Why it's wrong:** Loop stops at n-1, so dp[n] remains 0.

**Correct approach:**
```python
for i in range(2, n + 1):  # Include n
    dp[i] = dp[i-1] + dp[i-2]
```

### ❌ Pitfall 4: Using Naive Recursion for Large n

**What happens:**
```python
# WRONG: Will timeout for n > 35
def fibonacci_naive(n):
    if n <= 1:
        return n
    return fibonacci_naive(n-1) + fibonacci_naive(n-2)

result = fibonacci_naive(50)  # Takes forever!
```

**Why it's wrong:** O(2^n) complexity is prohibitive for n > 40.

**Correct approach:** Use memoization or tabulation.

### ❌ Pitfall 5: Forgetting to Cache in Memoization

**What happens:**
```python
# WRONG: Computing but not storing
def fibonacci_wrong(n, memo={}):
    if n <= 1:
        return n
    if n in memo:
        return memo[n]
    
    # Compute but DON'T store!
    return fibonacci_wrong(n-1, memo) + fibonacci_wrong(n-2, memo)
```

**Why it's wrong:** No benefit from memoization if we don't cache results.

**Correct approach:**
```python
def fibonacci_correct(n, memo={}):
    if n <= 1:
        return n
    if n in memo:
        return memo[n]
    
    memo[n] = fibonacci_correct(n-1, memo) + fibonacci_correct(n-2, memo)
    return memo[n]
```

### ❌ Pitfall 6: Mutable Default Argument

**What happens:**
```python
# WRONG: Mutable default persists across calls
def fibonacci_wrong(n, memo={}):
    # memo is shared across all calls!
    if n <= 1:
        return n
    if n in memo:
        return memo[n]
    
    memo[n] = fibonacci_wrong(n-1, memo) + fibonacci_wrong(n-2, memo)
    return memo[n]

# First call
fibonacci_wrong(5)  # memo gets populated

# Second call
fibonacci_wrong(3)  # memo still has old values! Might cause issues
```

**Why it's problematic:** memo dictionary persists between function calls.

**Correct approach:**
```python
def fibonacci_correct(n):
    memo = {}  # Create fresh memo
    
    def helper(n):
        if n <= 1:
            return n
        if n in memo:
            return memo[n]
        
        memo[n] = helper(n-1) + helper(n-2)
        return memo[n]
    
    return helper(n)

# Or use None as default
def fibonacci_correct(n, memo=None):
    if memo is None:
        memo = {}
    # ... rest of code
```

### ❌ Pitfall 7: Wrong Recurrence for Problem Variant

**What happens:**
```python
# WRONG: Using Fibonacci recurrence for 3-step staircase
def staircase_3_steps_wrong(n):
    if n <= 1:
        return n
    if n == 2:
        return 2
    
    prev2 = 1
    prev1 = 2
    
    # WRONG: This is still Fibonacci!
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1
```

**Why it's wrong:** For 3 steps, recurrence should be f(n) = f(n-1) + f(n-2) + f(n-3).

**Correct approach:**
```python
def staircase_3_steps_correct(n):
    if n <= 2:
        return n if n > 0 else 1
    if n == 3:
        return 4
    
    prev3 = 1
    prev2 = 2
    prev1 = 4
    
    for i in range(4, n + 1):
        current = prev1 + prev2 + prev3
        prev3 = prev2
        prev2 = prev1
        prev1 = current
    
    return prev1
```

## Variations and Extensions

### Variation 1: Tribonacci Numbers

**Description:** Like Fibonacci but sum of previous three terms.

**Recurrence:** T(n) = T(n-1) + T(n-2) + T(n-3)

**When to use:** Three-way decisions, more complex state transitions.

**Implementation:**
```python
def tribonacci(n: int) -> int:
    """
    Tribonacci: T(n) = T(n-1) + T(n-2) + T(n-3)
    T(0) = 0, T(1) = 1, T(2) = 1
    
    Time: O(n)
    Space: O(1)
    
    Example:
        >>> tribonacci(4)
        4  # 0, 1, 1, 2, 4
    """
    if n == 0:
        return 0
    if n <= 2:
        return 1
    
    prev3, prev2, prev1 = 0, 1, 1
    
    for _ in range(3, n + 1):
        current = prev1 + prev2 + prev3
        prev3, prev2, prev1 = prev2, prev1, current
    
    return prev1
```

### Variation 2: Fibonacci with Different Base Cases

**Description:** Start sequence with different initial values.

**Example:** Lucas numbers: L(0)=2, L(1)=1, then L(n) = L(n-1) + L(n-2)

**Implementation:**
```python
def lucas_numbers(n: int) -> int:
    """
    Lucas numbers: 2, 1, 3, 4, 7, 11, 18, ...
    Same recurrence as Fibonacci, different base.
    
    Time: O(n)
    Space: O(1)
    """
    if n == 0:
        return 2
    if n == 1:
        return 1
    
    prev2, prev1 = 2, 1
    
    for _ in range(2, n + 1):
        current = prev1 + prev2
        prev2, prev1 = prev1, current
    
    return prev1
```

### Variation 3: Counting Paths in Grid

**Description:** Count paths from top-left to bottom-right (only right/down moves).

**Recurrence:** paths(i,j) = paths(i-1,j) + paths(i,j-1)

**Implementation:**
```python
def unique_paths(m: int, n: int) -> int:
    """
    Count unique paths in m×n grid.
    Each cell: sum of paths from cell above and cell to left.
    
    Time: O(m × n)
    Space: O(n) with space optimization
    
    Example:
        >>> unique_paths(3, 2)
        3  # [right,down], [down,right], or variations
    """
    # Space-optimized: only keep one row
    dp = [1] * n  # Top row all 1's
    
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    
    return dp[n - 1]
```

### Variation 4: Decode Ways

**Description:** Count ways to decode string (like Fibonacci with constraints).

**Recurrence:** decode(i) = decode(i-1) + decode(i-2) if valid

**Implementation:**
```python
def num_decodings(s: str) -> int:
    """
    Count ways to decode string where 1='A', 2='B', ..., 26='Z'.
    
    Fibonacci-like but with validity constraints.
    
    Time: O(n)
    Space: O(1)
    
    Example:
        >>> num_decodings("226")
        3  # "BZ"(2,26), "VF"(22,6), "BBF"(2,2,6)
    """
    if not s or s[0] == '0':
        return 0
    
    n = len(s)
    prev2, prev1 = 1, 1
    
    for i in range(1, n):
        current = 0
        
        # Single digit
        if s[i] != '0':
            current += prev1
        
        # Two digits
        two_digit = int(s[i-1:i+1])
        if 10 <= two_digit <= 26:
            current += prev2
        
        prev2, prev1 = prev1, current
    
    return prev1
```

### Variation 5: Matrix Exponentiation (Fast Fibonacci)

**Description:** Compute Fibonacci in O(log n) using matrix multiplication.

**Key insight:** [F(n+1), F(n)] = [F(n), F(n-1)] × [[1,1],[1,0]]

**When to use:** Very large n (n > 10^6).

**Implementation:**
```python
def fibonacci_matrix(n: int) -> int:
    """
    Fibonacci using matrix exponentiation - O(log n).
    
    [[F(n+1), F(n)]] = [[1,1],[1,0]]^n
    
    Time: O(log n)
    Space: O(1)
    """
    if n <= 1:
        return n
    
    def matrix_mult(A, B):
        """Multiply two 2×2 matrices."""
        return [
            [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
            [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]
        ]
    
    def matrix_pow(M, n):
        """Compute M^n using binary exponentiation."""
        if n == 1:
            return M
        if n % 2 == 0:
            half = matrix_pow(M, n // 2)
            return matrix_mult(half, half)
        else:
            return matrix_mult(M, matrix_pow(M, n - 1))
    
    base = [[1, 1], [1, 0]]
    result = matrix_pow(base, n)
    
    return result[0][1]  # F(n)
```

## Practice Problems

### Beginner

1. **Fibonacci Number** - Compute nth Fibonacci number
   - LeetCode #509 (Fibonacci Number)
   - Foundation problem, try all 4 approaches

2. **Climbing Stairs** - Count ways to climb n stairs
   - LeetCode #70 (Climbing Stairs)
   - Direct Fibonacci application

3. **Min Cost Climbing Stairs** - Minimize cost to reach top
   - LeetCode #746 (Min Cost Climbing Stairs)
   - Fibonacci with cost optimization

4. **Tribonacci Number** - Sum of previous three terms
   - LeetCode #1137 (N-th Tribonacci Number)
   - Extension to three terms

### Intermediate

1. **House Robber** - Rob houses without adjacent
   - LeetCode #198 (House Robber)
   - Classic Fibonacci-like DP

2. **House Robber II** - Houses in circle
   - LeetCode #213 (House Robber II)
   - Extension with circular constraint

3. **Decode Ways** - Count ways to decode string
   - LeetCode #91 (Decode Ways)
   - Fibonacci with validity checks

4. **Unique Paths** - Count paths in grid
   - LeetCode #62 (Unique Paths)
   - 2D Fibonacci pattern

5. **Unique Paths II** - With obstacles
   - LeetCode #63 (Unique Paths II)
   - Grid paths with constraints

6. **Jump Game** - Can reach end?
   - LeetCode #55 (Jump Game)
   - Greedy or DP approach

7. **Jump Game II** - Minimum jumps
   - LeetCode #45 (Jump Game II)
   - DP or greedy optimization

### Advanced

1. **Knight Dialer** - Count phone numbers knight can dial
   - LeetCode #935 (Knight Dialer)
   - Multiple Fibonacci sequences

2. **Fibonacci-like Sequence** - Count sequences with constraints
   - Custom recurrence relations
   - Practice designing state transitions

3. **Domino and Tromino Tiling** - Tile board with dominoes
   - LeetCode #790 (Domino and Tromino Tiling)
   - Complex Fibonacci variant

4. **Ugly Number II** - nth ugly number
   - LeetCode #264 (Ugly Number II)
   - Multiple Fibonacci-like sequences merged

5. **Integer Break** - Break integer to maximize product
   - LeetCode #343 (Integer Break)
   - DP with multiple choices

6. **Perfect Squares** - Minimum perfect squares summing to n
   - LeetCode #279 (Perfect Squares)
   - Fibonacci-like with variable step sizes

## Real-World Applications

### Industry Use Cases

1. **Financial Modeling - Population Growth**
   - **How it's used:** Model rabbit populations, bacterial growth
   - **Why it's effective:** Natural Fibonacci-like growth patterns
   - **Historical:** Original Fibonacci problem was about rabbit breeding

2. **Computer Graphics - Spiral Generation**
   - **How it's used:** Generate golden spirals using Fibonacci ratios
   - **Why it's effective:** Aesthetically pleasing proportions
   - **Applications:** Logo design, UI layout, nature visualization

3. **Game Development - Pathfinding**
   - **How it's used:** Count paths through game levels
   - **Why it's effective:** Grid-based movement creates Fibonacci-like recurrences
   - **Example:** Tower defense games, puzzle games

4. **Network Routing - Path Counting**
   - **How it's used:** Count distinct routes between nodes
   - **Why it's effective:** Multiple paths combine Fibonacci-style
   - **Applications:** Redundancy planning, load balancing

5. **Resource Allocation**
   - **How it's used:** Optimal server selection, task scheduling
   - **Why it's effective:** Binary decisions at each step
   - **Example:** House robber pattern for non-adjacent selection

6. **DNA Sequence Analysis**
   - **How it's used:** Count possible RNA structures
   - **Why it's effective:** Base pairing creates Fibonacci-like counts
   - **Research:** Bioinformatics, protein folding

### Popular Implementations

- **Python `functools.lru_cache`:** Built-in memoization for DP
- **NumPy:** Fast array operations for tabulation
- **Dynamic Programming Libraries:** Many frameworks use Fibonacci as teaching example
- **Algorithm Visualizers:** Show DP optimization graphically
- **Financial Software:** Fibonacci retracements in trading
- **Architecture:** Golden ratio (Fibonacci-based) in building design

### Practical Scenarios

- **Trading Algorithms:** Fibonacci retracement levels for support/resistance
- **UI/UX Design:** Spacing using Fibonacci ratios
- **Music Theory:** Fibonacci in rhythm patterns and composition
- **Botany:** Fibonacci in plant spirals (sunflower seeds, pinecones)
- **Error Correction:** Fibonacci in coding theory
- **Database Indexing:** B-tree splitting uses Fibonacci-like calculations
- **Compiler Optimization:** Fibonacci for register allocation timing
- **Cache Replacement:** Fibonacci-based eviction policies

## Related Topics

### Prerequisites to Review

- **Recursion** - Understanding recursive thinking
- **Basic Arrays** - For tabulation approach
- **Hash Maps/Dictionaries** - For memoization
- **Big-O Notation** - Understanding complexity improvements

### Next Steps

- **0/1 Knapsack** - More complex DP pattern
- **Longest Common Subsequence** - String DP
- **Matrix Chain Multiplication** - Advanced DP
- **Edit Distance** - Classic DP problem
- **Palindromic Subsequence** - String DP variants

### Similar Concepts

- **Linear Recurrence Relations** - General form of Fibonacci
- **Generating Functions** - Mathematical approach to sequences
- **Divide and Conquer** - Related to recursion
- **Greedy Algorithms** - Alternative to DP for some problems
- **Graph Algorithms** - Path counting uses similar DP

### Further Reading

- **"Introduction to Algorithms" (CLRS)** - Chapter 15 on Dynamic Programming
  - Rigorous treatment of DP fundamentals
  
- **"Dynamic Programming for Coding Interviews"** - Meenakshi & Kamal Rawat
  - Practical DP patterns starting with Fibonacci
  
- **LeetCode Dynamic Programming Study Plan:**
  - https://leetcode.com/study-plan/dynamic-programming/
  - Structured progression starting with Fibonacci
  
- **GeeksforGeeks Fibonacci:**
  - https://www.geeksforgeeks.org/program-for-nth-fibonacci-number/
  - Multiple approaches with complexity analysis
  
- **"The Algorithm Design Manual" by Skiena**
  - Chapter on DP with Fibonacci examples
  
- **Mathematical Analysis:**
  - "Concrete Mathematics" by Graham, Knuth, Patashnik
  - Deep dive into recurrence relations and generating functions
  
- **Golden Ratio:**
  - Study φ (phi) = (1 + √5) / 2 ≈ 1.618
  - Lim[F(n+1)/F(n)] = φ as n → ∞
