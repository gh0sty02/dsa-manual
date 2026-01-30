# 0/1 Knapsack (Dynamic Programming) Pattern

**Difficulty:** Intermediate to Advanced
**Prerequisites:** Dynamic Programming basics, Arrays, Recursion, Memoization
**Estimated Reading Time:** 30 minutes

## Introduction

The 0/1 Knapsack pattern is a fundamental dynamic programming technique for solving optimization problems where you must choose a subset of items to maximize (or minimize) some objective while satisfying constraints. The "0/1" means each item can either be included (1) or excluded (0) - you cannot take fractions of items.

**Why it matters:** This pattern is the foundation for countless real-world optimization problems: resource allocation, portfolio selection, cutting stock problems, and budget planning. Understanding 0/1 knapsack unlocks the ability to solve an entire class of subset selection problems using dynamic programming. It's also a cornerstone interview topic at top tech companies.

**Real-world analogy:** Imagine you're packing for a hiking trip. Your backpack can hold 20 kg, and you have various items: tent (5kg, crucial), sleeping bag (3kg, important), extra clothes (2kg, nice), camera (1kg, optional), books (4kg, luxury), etc. You can't take half a tent or 60% of a sleeping bag - each item is all-or-nothing (0 or 1). Your goal: pack items that maximize value while staying under 20kg. That's the 0/1 knapsack problem!

## Core Concepts

### Key Principles

1. **Optimal substructure:** The optimal solution to the problem contains optimal solutions to subproblems. If we know the best way to pack weight W-w without item i, we can decide whether to include item i.

2. **Overlapping subproblems:** The same subproblems are solved multiple times in naive recursion. We use memoization/tabulation to avoid redundant computation.

3. **Decision at each item:** For each item, we have exactly two choices:
   - Include it (if it fits): get its value + optimal solution for remaining capacity
   - Exclude it: take optimal solution without this item
   - Choose whichever gives better value

4. **Bottom-up or top-down:** Can solve using either:
   - Bottom-up (tabulation): Build table from smaller subproblems
   - Top-down (memoization): Recursion with caching

### Essential Terms

- **0/1 constraint:** Each item can be taken 0 or 1 times (not fractional, not multiple times)
- **Capacity (W):** Maximum weight/size constraint
- **Value:** Benefit gained from including an item
- **Weight:** Cost/size of including an item
- **DP table dp[i][w]:** Maximum value achievable using first i items with capacity w
- **State:** Combination of (items considered, remaining capacity)
- **Recurrence relation:** Formula expressing solution in terms of subproblems

### Visual Overview

```
Problem: Items with weights and values, capacity W
Items: {w1:v1, w2:v2, w3:v3, ...}

Decision Tree (Naive Recursion):
                    Item 0: include or exclude?
                   /                        \
            Include (val + f(1,W-w0))    Exclude (f(1,W))
               /              \              /           \
         Item 1: I or E?   Item 1: I or E?  Item 1: I or E?  ...
         
Problem: 2^n paths to explore! (exponential)

DP Table (Efficient Solution):
         w=0  w=1  w=2  w=3  w=4  w=5  ← capacity
    i=0   0    0    0    0    0    0
    i=1   0   v1   v1   v1   v1   v1   (item 1: weight=1, value=v1)
    i=2   0   v1   v1   v2   v2  v1+v2 (item 2: weight=3, value=v2)
    i=3   0   v1   v3  max  max  max   (item 3: weight=2, value=v3)
    ...

Each cell: max value with first i items and capacity w
Build bottom-up: O(n × W) time and space
```

**Key Insight:** Instead of exploring 2^n combinations, we build a table of size n × W, solving each subproblem once.

### Recurrence Relation

```
dp[i][w] = maximum value using first i items with capacity w

Base cases:
dp[0][w] = 0 for all w (no items → no value)
dp[i][0] = 0 for all i (no capacity → no value)

Recurrence:
If weight[i] > w:
    dp[i][w] = dp[i-1][w]  (can't include item i, too heavy)
Else:
    dp[i][w] = max(
        dp[i-1][w],              (exclude item i)
        value[i] + dp[i-1][w - weight[i]]  (include item i)
    )
```

## How It Works

### Algorithm Steps (Bottom-Up Tabulation)

1. **Create DP table** of size (n+1) × (W+1)
   - dp[i][w] represents max value with first i items and capacity w
   
2. **Initialize base cases**
   - dp[0][w] = 0 for all w (no items)
   - dp[i][0] = 0 for all i (no capacity)
   
3. **Fill table row by row** (for each item)
   - For each capacity w:
     - If current item's weight > w: can't include it
       - dp[i][w] = dp[i-1][w]
     - Else: choose better of include vs exclude
       - dp[i][w] = max(exclude, include)
       
4. **Result in dp[n][W]**
   - Bottom-right cell contains maximum achievable value

5. **Optional: Backtrack to find items selected**
   - Start from dp[n][W]
   - If dp[i][w] != dp[i-1][w], item i was included
   - Move to dp[i-1][w - weight[i]]

### Visual Walkthrough: Small Example

**Problem:**
- Capacity: W = 7
- Items: 
  - Item 1: weight=1, value=1
  - Item 2: weight=3, value=4
  - Item 3: weight=4, value=5
  - Item 4: weight=5, value=7

```
Building DP Table Step-by-Step:

Initial state (base cases):
       w=0  w=1  w=2  w=3  w=4  w=5  w=6  w=7
  i=0   0    0    0    0    0    0    0    0
  i=1   0    
  i=2   0    
  i=3   0    
  i=4   0    

Step 1: Fill row for item 1 (weight=1, value=1)
For w=0: can't fit → dp[1][0] = dp[0][0] = 0
For w=1: can fit → dp[1][1] = max(dp[0][1], 1+dp[0][0]) = max(0, 1) = 1
For w=2: can fit → dp[1][2] = max(dp[0][2], 1+dp[0][1]) = max(0, 1) = 1
...all w≥1: value = 1

       w=0  w=1  w=2  w=3  w=4  w=5  w=6  w=7
  i=0   0    0    0    0    0    0    0    0
  i=1   0    1    1    1    1    1    1    1
  i=2   0    
  i=3   0    
  i=4   0    

Step 2: Fill row for item 2 (weight=3, value=4)
For w=0,1,2: can't fit → copy from previous row
For w=3: can fit → max(dp[1][3], 4+dp[1][0]) = max(1, 4) = 4
For w=4: can fit → max(dp[1][4], 4+dp[1][1]) = max(1, 5) = 5
For w=5: can fit → max(dp[1][5], 4+dp[1][2]) = max(1, 5) = 5
For w=6: can fit → max(dp[1][6], 4+dp[1][3]) = max(1, 5) = 5
For w=7: can fit → max(dp[1][7], 4+dp[1][4]) = max(1, 5) = 5

       w=0  w=1  w=2  w=3  w=4  w=5  w=6  w=7
  i=0   0    0    0    0    0    0    0    0
  i=1   0    1    1    1    1    1    1    1
  i=2   0    1    1    4    5    5    5    5
  i=3   0    
  i=4   0    

Step 3: Fill row for item 3 (weight=4, value=5)
For w=0,1,2,3: can't fit → copy from previous row
For w=4: can fit → max(dp[2][4], 5+dp[2][0]) = max(5, 5) = 5
For w=5: can fit → max(dp[2][5], 5+dp[2][1]) = max(5, 6) = 6
For w=6: can fit → max(dp[2][6], 5+dp[2][2]) = max(5, 6) = 6
For w=7: can fit → max(dp[2][7], 5+dp[2][3]) = max(5, 9) = 9

       w=0  w=1  w=2  w=3  w=4  w=5  w=6  w=7
  i=0   0    0    0    0    0    0    0    0
  i=1   0    1    1    1    1    1    1    1
  i=2   0    1    1    4    5    5    5    5
  i=3   0    1    1    4    5    6    6    9
  i=4   0    

Step 4: Fill row for item 4 (weight=5, value=7)
For w=0,1,2,3,4: can't fit → copy from previous row
For w=5: can fit → max(dp[3][5], 7+dp[3][0]) = max(6, 7) = 7
For w=6: can fit → max(dp[3][6], 7+dp[3][1]) = max(6, 8) = 8
For w=7: can fit → max(dp[3][7], 7+dp[3][2]) = max(9, 8) = 9

Final table:
       w=0  w=1  w=2  w=3  w=4  w=5  w=6  w=7
  i=0   0    0    0    0    0    0    0    0
  i=1   0    1    1    1    1    1    1    1
  i=2   0    1    1    4    5    5    5    5
  i=3   0    1    1    4    5    6    6    9
  i=4   0    1    1    4    5    7    8    9

Answer: dp[4][7] = 9

Backtracking to find items:
At dp[4][7]=9: 9 ≠ dp[3][7]=9, so item 4 NOT included
At dp[3][7]=9: 9 ≠ dp[2][7]=5, so item 3 included! (value=5)
  Move to dp[2][7-4]=dp[2][3]=4
At dp[2][3]=4: 4 ≠ dp[1][3]=1, so item 2 included! (value=4)
  Move to dp[1][3-3]=dp[1][0]=0
Done.

Items selected: {Item 2, Item 3} with total value 4+5=9
```

**State Transition Visualization:**

```
Decision at dp[3][7] (item 3, capacity 7):
    Item 3: weight=4, value=5
    
    Option 1: Exclude item 3
        → dp[2][7] = 5
        
    Option 2: Include item 3
        → 5 + dp[2][7-4] = 5 + dp[2][3] = 5 + 4 = 9 ✓
        
    Choose max(5, 9) = 9
```

## Implementation

### Python Implementation

```python
from typing import List, Tuple

def knapsack_01_recursive(weights: List[int], values: List[int], capacity: int) -> int:
    """
    0/1 Knapsack using naive recursion (exponential time).
    
    This is for understanding only - very slow!
    
    Args:
        weights: Weight of each item
        values: Value of each item
        capacity: Maximum weight capacity
        
    Returns:
        Maximum value achievable
        
    Time Complexity: O(2^n) - explores all subsets
    Space Complexity: O(n) - recursion stack
    
    Example:
        >>> knapsack_01_recursive([1,3,4,5], [1,4,5,7], 7)
        9
    """
    def helper(i: int, remaining_capacity: int) -> int:
        """
        Returns max value using items from index i onwards
        with given remaining capacity.
        """
        # Base case: no items left or no capacity
        if i >= len(weights) or remaining_capacity <= 0:
            return 0
        
        # If current item doesn't fit, skip it
        if weights[i] > remaining_capacity:
            return helper(i + 1, remaining_capacity)
        
        # Choose max of including or excluding current item
        include = values[i] + helper(i + 1, remaining_capacity - weights[i])
        exclude = helper(i + 1, remaining_capacity)
        
        return max(include, exclude)
    
    return helper(0, capacity)


def knapsack_01_memoization(weights: List[int], values: List[int], capacity: int) -> int:
    """
    0/1 Knapsack using top-down DP (memoization).
    
    Args:
        weights: Weight of each item
        values: Value of each item
        capacity: Maximum weight capacity
        
    Returns:
        Maximum value achievable
        
    Time Complexity: O(n × W) where W is capacity
    Space Complexity: O(n × W) for memoization + O(n) recursion stack
    
    Example:
        >>> knapsack_01_memoization([1,3,4,5], [1,4,5,7], 7)
        9
    """
    n = len(weights)
    # Memoization table: memo[i][w] = max value for items i..n-1 with capacity w
    memo = {}
    
    def helper(i: int, remaining_capacity: int) -> int:
        # Base case
        if i >= n or remaining_capacity <= 0:
            return 0
        
        # Check memo
        if (i, remaining_capacity) in memo:
            return memo[(i, remaining_capacity)]
        
        # If current item doesn't fit
        if weights[i] > remaining_capacity:
            result = helper(i + 1, remaining_capacity)
        else:
            # Choose max of include vs exclude
            include = values[i] + helper(i + 1, remaining_capacity - weights[i])
            exclude = helper(i + 1, remaining_capacity)
            result = max(include, exclude)
        
        memo[(i, remaining_capacity)] = result
        return result
    
    return helper(0, capacity)


def knapsack_01_tabulation(weights: List[int], values: List[int], capacity: int) -> int:
    """
    0/1 Knapsack using bottom-up DP (tabulation).
    
    This is the standard, most efficient approach.
    
    Args:
        weights: Weight of each item
        values: Value of each item
        capacity: Maximum weight capacity
        
    Returns:
        Maximum value achievable
        
    Time Complexity: O(n × W)
    Space Complexity: O(n × W)
    
    Example:
        >>> knapsack_01_tabulation([1,3,4,5], [1,4,5,7], 7)
        9
    """
    n = len(weights)
    
    # Create DP table: dp[i][w] = max value with first i items and capacity w
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # Fill table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            # Current item is at index i-1 (0-indexed)
            current_weight = weights[i - 1]
            current_value = values[i - 1]
            
            if current_weight > w:
                # Can't include current item
                dp[i][w] = dp[i - 1][w]
            else:
                # Choose max of include vs exclude
                exclude = dp[i - 1][w]
                include = current_value + dp[i - 1][w - current_weight]
                dp[i][w] = max(exclude, include)
    
    return dp[n][capacity]


def knapsack_01_with_items(weights: List[int], values: List[int], 
                           capacity: int) -> Tuple[int, List[int]]:
    """
    0/1 Knapsack that returns both max value and selected items.
    
    Args:
        weights: Weight of each item
        values: Value of each item
        capacity: Maximum weight capacity
        
    Returns:
        (max_value, list of selected item indices)
        
    Time Complexity: O(n × W)
    Space Complexity: O(n × W)
    
    Example:
        >>> knapsack_01_with_items([1,3,4,5], [1,4,5,7], 7)
        (9, [1, 2])  # Items at indices 1 and 2 (values 4 and 5)
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    # Build DP table
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            current_weight = weights[i - 1]
            current_value = values[i - 1]
            
            if current_weight > w:
                dp[i][w] = dp[i - 1][w]
            else:
                dp[i][w] = max(dp[i - 1][w], 
                              current_value + dp[i - 1][w - current_weight])
    
    # Backtrack to find selected items
    selected_items = []
    i, w = n, capacity
    
    while i > 0 and w > 0:
        # If value changed from previous row, item i-1 was included
        if dp[i][w] != dp[i - 1][w]:
            selected_items.append(i - 1)  # 0-indexed item
            w -= weights[i - 1]  # Reduce capacity
        i -= 1
    
    selected_items.reverse()  # Put in original order
    
    return dp[n][capacity], selected_items


def knapsack_01_space_optimized(weights: List[int], values: List[int], 
                                capacity: int) -> int:
    """
    0/1 Knapsack with space optimization (only O(W) space).
    
    Key insight: We only need previous row to compute current row.
    
    Args:
        weights: Weight of each item
        values: Value of each item
        capacity: Maximum weight capacity
        
    Returns:
        Maximum value achievable
        
    Time Complexity: O(n × W)
    Space Complexity: O(W) - only one row!
    
    Example:
        >>> knapsack_01_space_optimized([1,3,4,5], [1,4,5,7], 7)
        9
    """
    n = len(weights)
    
    # Only need one row: current and previous
    # Use two arrays or process backwards in single array
    dp = [0] * (capacity + 1)
    
    for i in range(n):
        # Process backwards to avoid overwriting values we need
        for w in range(capacity, weights[i] - 1, -1):
            # dp[w] currently holds dp[i-1][w] (previous row)
            # Update to dp[i][w]
            dp[w] = max(dp[w], values[i] + dp[w - weights[i]])
    
    return dp[capacity]


def equal_subset_sum_partition(nums: List[int]) -> bool:
    """
    Check if array can be partitioned into two subsets with equal sum.
    This is a variant of 0/1 knapsack.
    
    Args:
        nums: Array of positive integers
        
    Returns:
        True if equal partition exists
        
    Time Complexity: O(n × sum)
    Space Complexity: O(sum)
    
    Example:
        >>> equal_subset_sum_partition([1, 5, 11, 5])
        True  # [1, 5, 5] and [11]
        >>> equal_subset_sum_partition([1, 2, 3, 5])
        False
    """
    total_sum = sum(nums)
    
    # If sum is odd, can't partition equally
    if total_sum % 2 != 0:
        return False
    
    target = total_sum // 2
    
    # Can we find subset that sums to target?
    # This is 0/1 knapsack where weight = value
    dp = [False] * (target + 1)
    dp[0] = True  # Empty subset sums to 0
    
    for num in nums:
        # Process backwards to avoid using same element twice
        for s in range(target, num - 1, -1):
            dp[s] = dp[s] or dp[s - num]
    
    return dp[target]


def subset_sum(nums: List[int], target: int) -> bool:
    """
    Check if there's a subset that sums to target.
    
    Args:
        nums: Array of integers
        target: Target sum
        
    Returns:
        True if subset exists
        
    Time Complexity: O(n × target)
    Space Complexity: O(target)
    
    Example:
        >>> subset_sum([3, 34, 4, 12, 5, 2], 9)
        True  # 4 + 5 = 9
    """
    # dp[s] = True if we can make sum s
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        for s in range(target, num - 1, -1):
            dp[s] = dp[s] or dp[s - num]
    
    return dp[target]


def count_subset_sum(nums: List[int], target: int) -> int:
    """
    Count number of subsets that sum to target.
    
    Args:
        nums: Array of positive integers
        target: Target sum
        
    Returns:
        Number of subsets with given sum
        
    Time Complexity: O(n × target)
    Space Complexity: O(target)
    
    Example:
        >>> count_subset_sum([1, 1, 2, 3], 4)
        3  # {1,3}, {1,3}, {1,1,2}
    """
    # dp[s] = number of ways to make sum s
    dp = [0] * (target + 1)
    dp[0] = 1  # One way to make 0: empty subset
    
    for num in nums:
        for s in range(target, num - 1, -1):
            dp[s] += dp[s - num]
    
    return dp[target]


def target_sum(nums: List[int], target: int) -> int:
    """
    Assign + or - to each number to reach target. Count ways.
    
    This is a disguised subset sum problem:
    If P is subset with +, N is subset with -, then:
    P - N = target
    P + N = sum(nums)
    => P = (target + sum) / 2
    
    Problem reduces to: count subsets that sum to (target + sum) / 2
    
    Args:
        nums: Array of integers
        target: Target sum
        
    Returns:
        Number of ways to assign signs
        
    Time Complexity: O(n × sum)
    Space Complexity: O(sum)
    
    Example:
        >>> target_sum([1, 1, 1, 1, 1], 3)
        5  # +1+1+1+1-1, +1+1+1-1+1, etc.
    """
    total = sum(nums)
    
    # Check if solution exists
    if (target + total) % 2 != 0 or abs(target) > total:
        return 0
    
    subset_sum_target = (target + total) // 2
    
    # Count subsets that sum to subset_sum_target
    dp = [0] * (subset_sum_target + 1)
    dp[0] = 1
    
    for num in nums:
        for s in range(subset_sum_target, num - 1, -1):
            dp[s] += dp[s - num]
    
    return dp[subset_sum_target]


# Usage Examples
if __name__ == "__main__":
    # Example 1: Basic 0/1 knapsack
    weights = [1, 3, 4, 5]
    values = [1, 4, 5, 7]
    capacity = 7
    
    max_value = knapsack_01_tabulation(weights, values, capacity)
    print(f"Maximum value: {max_value}")
    # Output: 9
    
    # Example 2: With items
    max_value, items = knapsack_01_with_items(weights, values, capacity)
    print(f"Max value: {max_value}, Items: {items}")
    # Output: Max value: 9, Items: [1, 2]
    
    # Example 3: Equal partition
    nums = [1, 5, 11, 5]
    can_partition = equal_subset_sum_partition(nums)
    print(f"Can partition {nums}: {can_partition}")
    # Output: True
    
    # Example 4: Subset sum
    has_subset = subset_sum([3, 34, 4, 12, 5, 2], 9)
    print(f"Has subset summing to 9: {has_subset}")
    # Output: True
    
    # Example 5: Count subsets
    count = count_subset_sum([1, 1, 2, 3], 4)
    print(f"Number of subsets summing to 4: {count}")
    # Output: 3
    
    # Example 6: Target sum
    ways = target_sum([1, 1, 1, 1, 1], 3)
    print(f"Ways to reach target 3: {ways}")
    # Output: 5
```

### Code Explanation

**Key Design Decisions:**

1. **Why dp[i][w] instead of dp[i][capacity - w]?**
   - dp[i][w] is more intuitive: "max value with capacity w"
   - Makes recurrence relation clearer
   - Easier to understand base cases

2. **Why process backwards in space-optimized version?**
   - If we process forward, we might use updated values from current iteration
   - Processing backwards ensures we use values from "previous row"
   - Example: computing dp[5] needs dp[2] from previous iteration, not current

3. **Why 1-indexed loop but 0-indexed arrays?**
   - DP table has n+1 rows (0 to n) where row 0 means "no items"
   - Arrays are 0-indexed, so item i is at weights[i-1]
   - This separation makes base cases cleaner

4. **How backtracking works to find items?**
   - If dp[i][w] != dp[i-1][w], then item i-1 was included
   - When item included, we used capacity, so move to dp[i-1][w - weight[i-1]]
   - When item excluded, just move to dp[i-1][w]

5. **Why subset sum uses OR operation?**
   - dp[s] means "can we achieve sum s?"
   - dp[s] = dp[s] OR dp[s - num] means:
     - Either we could already make sum s (dp[s])
     - Or we can make sum s by adding num to a subset that makes s - num

## Complexity Analysis

### Time Complexity

**Standard 0/1 Knapsack:**
- **Building DP table:** O(n × W) where n is number of items, W is capacity
- **Two nested loops:** Outer loop n items, inner loop W capacities
- **Each cell:** O(1) computation
- **Overall:** **O(n × W)**

**With backtracking for items:**
- **DP table:** O(n × W)
- **Backtracking:** O(n) - at most n items selected
- **Overall:** **O(n × W)**

**Space-optimized version:**
- **Time:** Still O(n × W) - same number of computations
- **Space:** O(W) instead of O(n × W)

### Space Complexity

**Standard tabulation:**
- **DP table:** O(n × W) - (n+1) × (W+1) array
- **Overall:** **O(n × W)**

**Space-optimized:**
- **Single row:** O(W)
- **Overall:** **O(W)**

**Memoization:**
- **Memo dict:** O(n × W) worst case
- **Recursion stack:** O(n)
- **Overall:** **O(n × W)**

### Comparison with Alternatives

| Approach | Time | Space | When to Use |
|----------|------|-------|-------------|
| **Brute Force** | O(2^n) | O(n) | Never! Only for n ≤ 20 |
| **Greedy** | O(n log n) | O(1) | ❌ Doesn't work for 0/1! |
| **DP Tabulation** | O(n × W) | O(n × W) | Standard choice |
| **DP Space-Opt** | O(n × W) | O(W) | When memory limited |
| **DP Memoization** | O(n × W) | O(n × W) | When easier to think recursively |
| **Branch & Bound** | O(2^n) avg | O(n) | When W is huge but optimal pruning possible |

**Is O(n × W) "polynomial"?**
- Sort of! It's **pseudo-polynomial**
- Polynomial in n (number of items)
- But exponential in the number of bits needed to represent W
- If W = 10^9, this is not practical!
- For very large W, use approximation algorithms

**When DP Knapsack Wins:**
- W is reasonable (< 10^6 typically)
- Need exact optimal solution
- Can afford O(n × W) space
- n is moderate (< 10^3 typically)

**When DP Knapsack Struggles:**
- W is huge (10^9+): space and time prohibitive
- Need approximate solution fast: use greedy approximation
- n is huge but W small: still works if W is manageable

## Examples

### Example 1: Classic 0/1 Knapsack

**Problem:** weights=[2,1,3,2], values=[12,10,20,15], capacity=5

**Solution:**
```python
weights = [2, 1, 3, 2]
values = [12, 10, 20, 15]
capacity = 5

Building DP table:
       w=0  w=1  w=2  w=3  w=4  w=5
  i=0   0    0    0    0    0    0
  i=1   0    0   12   12   12   12  (item 0: w=2, v=12)
  i=2   0   10   12   22   22   22  (item 1: w=1, v=10)
  i=3   0   10   12   22   30   32  (item 2: w=3, v=20)
  i=4   0   10   15   22   30   37  (item 3: w=2, v=15)

Key transitions:
dp[2][3] = max(dp[1][3], 10 + dp[1][2]) = max(12, 22) = 22
  (include item 1: value 10 + value at capacity 2 with item 0)
  
dp[3][5] = max(dp[2][5], 20 + dp[2][2]) = max(22, 32) = 32
  (include item 2: value 20 + value at capacity 2 with items 0,1)
  
dp[4][5] = max(dp[3][5], 15 + dp[3][3]) = max(32, 37) = 37
  (include item 3: value 15 + value at capacity 3 with items 0,1,2)

Answer: 37

Backtracking:
At dp[4][5]=37: 37 ≠ dp[3][5]=32 → item 3 included
  Move to dp[3][3]=22
At dp[3][3]=22: 22 = dp[2][3]=22 → item 2 NOT included
  Move to dp[2][3]=22
At dp[2][3]=22: 22 ≠ dp[1][3]=12 → item 1 included
  Move to dp[1][2]=12
At dp[1][2]=12: 12 ≠ dp[0][2]=0 → item 0 included
  Move to dp[0][0]=0
  
Selected items: {0, 1, 3} with values {12, 10, 15} = 37
Weights: {2, 1, 2} = 5 ✓
```

### Example 2: Equal Subset Sum Partition

**Problem:** nums=[1,5,11,5], can partition into equal sums?

**Solution:**
```python
nums = [1, 5, 11, 5]
total_sum = 22
target = 11  # Need subset summing to 11

Converting to subset sum problem:
Can we find subset summing to 11?

DP array (subset sum):
Initially: dp = [T, F, F, F, F, F, F, F, F, F, F, F]
           (only dp[0] = True, empty subset)

After num=1:
  dp[1] = dp[1] OR dp[0] = F OR T = T
  dp = [T, T, F, F, F, F, F, F, F, F, F, F]

After num=5:
  dp[6] = dp[6] OR dp[1] = F OR T = T
  dp[5] = dp[5] OR dp[0] = F OR T = T
  dp = [T, T, F, F, F, T, T, F, F, F, F, F]

After num=11:
  dp[11] = dp[11] OR dp[0] = F OR T = T ✓ Found!
  (but continue to see full picture)
  dp[6] remains T (already true)
  dp = [T, T, F, F, F, T, T, F, F, F, F, T]

After num=5:
  dp[11] = dp[11] OR dp[6] = T OR T = T
  dp[10] = dp[10] OR dp[5] = F OR T = T
  dp[6] = dp[6] OR dp[1] = T OR T = T
  dp = [T, T, F, F, F, T, T, F, F, F, T, T]

Result: dp[11] = True
Partitions: {1, 5, 5} = 11 and {11} = 11
```

### Example 3: Count Subset Sum

**Problem:** nums=[1,1,2,3], target=4, count subsets summing to 4

**Solution:**
```python
nums = [1, 1, 2, 3]
target = 4

DP array (count ways):
Initially: dp = [1, 0, 0, 0, 0]
           (1 way to make 0: empty subset)

After num=1:
  dp[1] += dp[0] = 0 + 1 = 1
  dp = [1, 1, 0, 0, 0]
  (one way to make 1: {1})

After num=1 (second occurrence):
  dp[2] += dp[1] = 0 + 1 = 1
  dp[1] += dp[0] = 1 + 1 = 2
  dp = [1, 2, 1, 0, 0]
  (two ways to make 1: {1} and {1})
  (one way to make 2: {1,1})

After num=2:
  dp[4] += dp[2] = 0 + 1 = 1
  dp[3] += dp[1] = 0 + 2 = 2
  dp[2] += dp[0] = 1 + 1 = 2
  dp = [1, 2, 2, 2, 1]
  (one way to make 4: {1,1,2})

After num=3:
  dp[4] += dp[1] = 1 + 2 = 3 ✓
  dp = [1, 2, 2, 2, 3]

Result: 3 ways to make 4
  {1, 3} (using first 1)
  {1, 3} (using second 1)
  {1, 1, 2}
```

### Example 4: Target Sum with +/- Assignment

**Problem:** nums=[1,1,1,1,1], target=3, how many ways?

**Solution:**
```python
nums = [1, 1, 1, 1, 1]
target = 3

Convert to subset sum:
sum(nums) = 5
P - N = 3 (where P is positive subset, N is negative)
P + N = 5
=> 2P = 8 => P = 4

Problem: count subsets summing to 4

DP: count ways to make sum 4 from [1,1,1,1,1]

After processing all five 1's:
dp[0] = 1  (empty)
dp[1] = 5  (any single 1)
dp[2] = 10 (C(5,2) = choose 2 from 5)
dp[3] = 10 (C(5,3) = choose 3 from 5)
dp[4] = 5  (C(5,4) = choose 4 from 5)

Result: 5 ways to assign signs
  +1+1+1+1-1 = 3
  +1+1+1-1+1 = 3
  +1+1-1+1+1 = 3
  +1-1+1+1+1 = 3
  -1+1+1+1+1 = 3
```

## Edge Cases

### 1. Empty Items List
**Scenario:** No items to choose from

**Challenge:** No value can be achieved

**Solution:** Return 0

```python
def knapsack_01(weights, values, capacity):
    if not weights:
        return 0  # No items, no value
```

### 2. Zero Capacity
**Scenario:** Knapsack capacity is 0

**Challenge:** Can't take any items

**Solution:** Return 0

```python
if capacity <= 0:
    return 0
```

### 3. All Items Too Heavy
**Scenario:** Every item's weight > capacity

**Challenge:** Can't include any item

**Solution:** Algorithm handles correctly, returns 0

```python
# Example: weights=[10,20,30], capacity=5
# All items excluded, dp[n][5] = 0
```

### 4. Single Item
**Scenario:** Only one item available

**Challenge:** Binary decision: take it or not

**Solution:** Take it if it fits

```python
# weights=[3], values=[10], capacity=5
# Result: 10 (take the item)

# weights=[10], values=[100], capacity=5
# Result: 0 (can't take it)
```

### 5. Equal Partition with Odd Sum
**Scenario:** Array sum is odd

**Challenge:** Can't split odd number equally

**Solution:** Return False immediately

```python
def equal_partition(nums):
    if sum(nums) % 2 != 0:
        return False  # Impossible to partition
```

### 6. Subset Sum with Negative Numbers
**Scenario:** Array contains negative numbers

**Challenge:** DP table index can't be negative

**Solution:** Adjust by adding offset, or use different approach

```python
# If negatives present, offset all values
# Or use different DP formulation with signed indices
```

### 7. Target Sum Exceeds Total
**Scenario:** Target > sum of all positive elements

**Challenge:** Impossible to reach

**Solution:** Return 0

```python
def target_sum(nums, target):
    if abs(target) > sum(nums):
        return 0  # Impossible
```

### 8. All Items Same Weight/Value
**Scenario:** All items identical

**Challenge:** Choosing any K items gives same result

**Solution:** Algorithm works, just picks first K that fit

```python
# weights=[2,2,2,2], values=[5,5,5,5], capacity=6
# Can take 3 items, value = 15
# Doesn't matter which 3
```

## Common Pitfalls

### ❌ Pitfall 1: Using Greedy for 0/1 Knapsack

**What happens:**
```python
# WRONG: Greedy by value density
def knapsack_greedy_wrong(weights, values, capacity):
    items = sorted(zip(values, weights), key=lambda x: x[0]/x[1], reverse=True)
    total = 0
    
    for val, wt in items:
        if capacity >= wt:
            total += val
            capacity -= wt
    
    return total

# Example: weights=[1,2,3], values=[6,10,12], capacity=5
# Greedy: picks item 0 (ratio 6.0), then can't fit others
# Result: 6
# Optimal: pick items 1,2 → value 22 ❌
```

**Why it's wrong:** 0/1 knapsack doesn't have greedy choice property!

**Correct approach:** Use dynamic programming.

### ❌ Pitfall 2: Off-by-One in Array Indexing

**What happens:**
```python
# WRONG: Confusion between i (DP row) and array index
for i in range(1, n + 1):
    for w in range(capacity + 1):
        # WRONG: using i instead of i-1 for array access
        if weights[i] > w:  # IndexError when i = n!
            dp[i][w] = dp[i-1][w]
```

**Why it's wrong:** DP table has n+1 rows (0 to n), but array has indices 0 to n-1.

**Correct approach:**
```python
# CORRECT: i-1 for array access
if weights[i-1] > w:  # Item i is at index i-1
    dp[i][w] = dp[i-1][w]
```

### ❌ Pitfall 3: Processing Forward in Space-Optimized Version

**What happens:**
```python
# WRONG: Processing forward overwrites values we need
dp = [0] * (capacity + 1)

for i in range(n):
    for w in range(weights[i], capacity + 1):  # Forward!
        dp[w] = max(dp[w], values[i] + dp[w - weights[i]])
        # dp[w - weights[i]] was already updated in this iteration!
```

**Why it's wrong:** We might use an item multiple times (unbounded knapsack).

**Correct approach:**
```python
# CORRECT: Process backwards
for w in range(capacity, weights[i] - 1, -1):  # Backward!
    dp[w] = max(dp[w], values[i] + dp[w - weights[i]])
```

### ❌ Pitfall 4: Not Checking Weight Constraint

**What happens:**
```python
# WRONG: Always trying both options
dp[i][w] = max(dp[i-1][w], 
              values[i-1] + dp[i-1][w - weights[i-1]])
# Crashes when w < weights[i-1] (negative index!)
```

**Why it's wrong:** Can't include item if it doesn't fit.

**Correct approach:**
```python
# CORRECT: Check weight first
if weights[i-1] > w:
    dp[i][w] = dp[i-1][w]  # Can't include
else:
    dp[i][w] = max(dp[i-1][w], 
                   values[i-1] + dp[i-1][w - weights[i-1]])
```

### ❌ Pitfall 5: Forgetting Base Cases

**What happens:**
```python
# WRONG: Not initializing base cases
dp = [[None] * (capacity + 1) for _ in range(n + 1)]
# Later: dp[i][w] = max(..., values[i] + dp[i-1][...])
# Crashes! dp[0][...] is None
```

**Why it's wrong:** Recurrence assumes base cases are set.

**Correct approach:**
```python
# CORRECT: Initialize base cases
dp = [[0] * (capacity + 1) for _ in range(n + 1)]
# Row 0 and column 0 are already 0
```

### ❌ Pitfall 6: Equal Partition - Not Checking Sum Parity

**What happens:**
```python
# WRONG: Trying to partition odd sum
def equal_partition_wrong(nums):
    target = sum(nums) // 2  # Integer division!
    # If sum is 7, target = 3
    # Even if we find subset summing to 3, 
    # remaining is 4 ≠ 3 ❌
```

**Why it's wrong:** Odd sum can't be split equally.

**Correct approach:**
```python
# CORRECT: Check parity first
total = sum(nums)
if total % 2 != 0:
    return False
target = total // 2
```

### ❌ Pitfall 7: Subset Sum - Using Wrong Boolean Logic

**What happens:**
```python
# WRONG: Overwriting instead of OR
for num in nums:
    for s in range(target, num - 1, -1):
        dp[s] = dp[s - num]  # Lost previous value!
```

**Why it's wrong:** We might have already found a way to make sum s.

**Correct approach:**
```python
# CORRECT: OR with previous value
dp[s] = dp[s] or dp[s - num]
```

## Variations and Extensions

### Variation 1: Unbounded Knapsack

**Description:** Can take unlimited copies of each item.

**When to use:** When items are reusable (e.g., coin change).

**Key difference:** Process forward (can reuse items).

**Implementation:**
```python
def unbounded_knapsack(weights: List[int], values: List[int], capacity: int) -> int:
    """
    Unbounded knapsack: unlimited copies of each item.
    
    Time: O(n × W)
    Space: O(W)
    """
    dp = [0] * (capacity + 1)
    
    for w in range(1, capacity + 1):
        for i in range(len(weights)):
            if weights[i] <= w:
                # Can use item i multiple times
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]
```

### Variation 2: Minimum Subset Sum Difference

**Description:** Partition array to minimize difference between subset sums.

**When to use:** Load balancing, fairness problems.

**Implementation:**
```python
def min_subset_sum_difference(nums: List[int]) -> int:
    """
    Find minimum difference between two subset sums.
    
    Approach: Find all possible subset sums up to total/2,
    then minimize |2*s - total| where s is a possible sum.
    
    Time: O(n × sum)
    Space: O(sum)
    """
    total = sum(nums)
    target = total // 2
    
    # Find all possible sums up to target
    dp = [False] * (target + 1)
    dp[0] = True
    
    for num in nums:
        for s in range(target, num - 1, -1):
            dp[s] = dp[s] or dp[s - num]
    
    # Find largest s where dp[s] = True
    for s in range(target, -1, -1):
        if dp[s]:
            # One subset has sum s, other has sum (total - s)
            return abs(total - 2 * s)
    
    return total  # Shouldn't reach here
```

### Variation 3: Count of Subset Sum (Multiple Solutions)

**Description:** Count all subsets with given sum.

**When to use:** Combinatorial counting problems.

**Implementation:**
```python
def count_subset_sum(nums: List[int], target: int) -> int:
    """
    Count number of subsets summing to target.
    
    Time: O(n × target)
    Space: O(target)
    """
    dp = [0] * (target + 1)
    dp[0] = 1  # Empty subset
    
    for num in nums:
        for s in range(target, num - 1, -1):
            dp[s] += dp[s - num]  # Add ways, don't overwrite
    
    return dp[target]
```

### Variation 4: Bounded Knapsack

**Description:** Each item has limited quantity (can take 0 to k_i copies).

**When to use:** Inventory with limited stock.

**Implementation:**
```python
def bounded_knapsack(weights: List[int], values: List[int], 
                     counts: List[int], capacity: int) -> int:
    """
    Bounded knapsack: each item has limited quantity.
    
    weights[i], values[i], counts[i] for each item type i
    
    Time: O(n × W × max(counts))
    Space: O(W)
    """
    dp = [0] * (capacity + 1)
    
    for i in range(len(weights)):
        # Process item i with counts[i] available
        for _ in range(counts[i]):
            for w in range(capacity, weights[i] - 1, -1):
                dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    
    return dp[capacity]
```

### Variation 5: Multi-dimensional Knapsack

**Description:** Multiple constraints (e.g., weight AND volume).

**When to use:** Resource allocation with multiple constraints.

**Implementation:**
```python
def knapsack_2d(weights: List[int], volumes: List[int], 
                values: List[int], max_weight: int, max_volume: int) -> int:
    """
    Knapsack with two constraints: weight AND volume.
    
    Time: O(n × W × V)
    Space: O(W × V)
    """
    dp = [[0] * (max_volume + 1) for _ in range(max_weight + 1)]
    
    for i in range(len(weights)):
        for w in range(max_weight, weights[i] - 1, -1):
            for v in range(max_volume, volumes[i] - 1, -1):
                dp[w][v] = max(dp[w][v], 
                              dp[w - weights[i]][v - volumes[i]] + values[i])
    
    return dp[max_weight][max_volume]
```

## Practice Problems

### Beginner

1. **Partition Equal Subset Sum** - Can partition into two equal sums?
   - LeetCode #416 (Partition Equal Subset Sum)
   - Direct application of 0/1 knapsack

2. **Subset Sum Problem** - Does subset with given sum exist?
   - Classic variant, foundation for many problems

3. **Target Sum** - Assign +/- to reach target
   - LeetCode #494 (Target Sum)
   - Convert to subset sum

### Intermediate

1. **0/1 Knapsack (Classic)** - Basic implementation
   - Understand all variations (recursive, memo, tabulation, space-opt)

2. **Coin Change** - Minimum coins to make amount
   - LeetCode #322 (Coin Change)
   - Unbounded knapsack variant

3. **Ones and Zeroes** - Max strings with limited 0s and 1s
   - LeetCode #474 (Ones and Zeroes)
   - 2D knapsack (two constraints)

4. **Last Stone Weight II** - Minimize difference in stone weights
   - LeetCode #1049 (Last Stone Weight II)
   - Minimum subset sum difference

5. **Partition to K Equal Sum Subsets** - Can partition into K equal parts?
   - LeetCode #698 (Partition to K Equal Sum Subsets)
   - Extension of equal partition

### Advanced

1. **Profitable Schemes** - Count schemes meeting profit and group size
   - LeetCode #879 (Profitable Schemes)
   - 2D knapsack with counting

2. **Number of Dice Rolls With Target Sum** - Count ways to roll target
   - LeetCode #1155
   - Bounded knapsack variant

3. **Combination Sum IV** - Count combinations (order matters)
   - LeetCode #377 (Combination Sum IV)
   - Unbounded knapsack with permutations

4. **Shopping Offers** - Optimize shopping with bundle discounts
   - LeetCode #638 (Shopping Offers)
   - Multi-item knapsack

5. **Paint House III** - Color houses with cost and neighborhood constraints
   - LeetCode #1473 (Paint House III)
   - Complex state DP related to knapsack

6. **Maximum Profit in Job Scheduling** - Schedule non-overlapping jobs
   - LeetCode #1235
   - Weighted interval scheduling (DP)

## Real-World Applications

### Industry Use Cases

1. **Resource Allocation in Cloud Computing**
   - **How it's used:** Allocate VMs to servers with CPU/memory constraints
   - **Why it's effective:** Maximize utilization while respecting limits
   - **Scale:** AWS, Azure use variants for bin packing

2. **Portfolio Selection in Finance**
   - **How it's used:** Select stocks to maximize returns within budget
   - **Why it's effective:** Each stock is 0/1 (buy or don't buy)
   - **Constraints:** Budget limit, risk limits, sector diversification

3. **Cutting Stock Problem in Manufacturing**
   - **How it's used:** Cut raw materials to minimize waste
   - **Why it's effective:** Each cut pattern is selected or not
   - **Applications:** Paper mills, steel cutting, fabric cutting

4. **Project Selection**
   - **How it's used:** Select projects to maximize ROI within budget
   - **Why it's effective:** Each project is discrete (do or don't do)
   - **Scale:** Government budgeting, R&D planning

5. **Cargo Loading**
   - **How it's used:** Load cargo containers/trucks to maximize value
   - **Why it's effective:** Each package included or excluded
   - **Constraints:** Weight limit, volume limit

6. **Menu Planning / Diet Optimization**
   - **How it's used:** Select foods to maximize nutrition within calorie budget
   - **Why it's effective:** Binary food choices with constraints
   - **Example:** Military meal planning, athlete nutrition

### Popular Implementations

- **OR-Tools (Google):** Knapsack solver for optimization
- **CPLEX, Gurobi:** Commercial solvers use advanced DP techniques
- **Dynamic Programming in Databases:** Query optimization uses DP
- **Compilers:** Register allocation is a variant of knapsack
- **Network Design:** Selecting connections to optimize cost/bandwidth
- **Cryptocurrencies:** Transaction selection in block creation (knapsack variant)

### Practical Scenarios

- **Backup Selection:** Which files to backup given storage limit?
- **Feature Selection in ML:** Select features maximizing model quality
- **Advertising:** Select ads to show given time/space constraints
- **Course Scheduling:** Select courses maximizing credits within time budget
- **Investment:** Select startups to invest in within capital limit
- **Task Assignment:** Assign tasks to workers maximizing completion
- **Bandwidth Allocation:** Distribute bandwidth to maximize throughput
- **Cache Management:** Select data to cache for max hit rate

## Related Topics

### Prerequisites to Review

- **Recursion** - Foundation for understanding DP
- **Memoization** - Top-down DP technique
- **Arrays and 2D Arrays** - DP table representation
- **Time Complexity** - Understanding pseudo-polynomial time

### Next Steps

- **Fibonacci Numbers (DP)** - Simpler DP pattern to build intuition
- **Longest Common Subsequence** - Another classic DP
- **Coin Change Problems** - Unbounded knapsack variants
- **Palindromic Subsequence** - String DP
- **Backtracking** - When DP isn't applicable

### Similar Concepts

- **Subset Sum Problem** - Special case of knapsack (weight = value)
- **Bin Packing** - Dual of knapsack (minimize bins vs maximize value)
- **Multi-Choice Knapsack** - Must pick exactly one from each group
- **Quadratic Knapsack** - Values depend on item interactions
- **Two-Dimensional Knapsack** - Multiple constraint dimensions

### Further Reading

- **"Introduction to Algorithms" (CLRS)** - Chapter 15 on Dynamic Programming
  - Rigorous treatment of DP including knapsack variants
  
- **"Dynamic Programming for Coding Interviews"** - Meenakshi & Kamal Rawat
  - Practical DP patterns with 50+ problems
  
- **LeetCode Dynamic Programming Study Plan:**
  - https://leetcode.com/study-plan/dynamic-programming/
  - Structured progression through DP problems
  
- **GeeksforGeeks 0/1 Knapsack:**
  - https://www.geeksforgeeks.org/0-1-knapsack-problem-dp-10/
  - Multiple approaches with visualizations
  
- **"The Art of Computer Programming Vol. 3" by Knuth**
  - Historical algorithms for optimization problems
  
- **Research Papers:**
  - "A PTAS for the Multiple Knapsack Problem" - Chekuri & Khanna
  - Advanced approximation algorithms
