# Greedy Algorithms Pattern

**Difficulty:** Intermediate
**Prerequisites:** Arrays, Sorting, Basic algorithm analysis, Problem-solving strategies
**Estimated Reading Time:** 25 minutes

## Introduction

A greedy algorithm is a problem-solving strategy that makes the locally optimal choice at each step with the hope of finding a global optimum. The key insight is that by always choosing what appears best right now, you can arrive at an overall optimal solution - but this only works for certain types of problems where local optimality leads to global optimality.

**Why it matters:** Greedy algorithms are among the most elegant and efficient algorithmic strategies when applicable. They often provide O(n log n) or O(n) solutions to problems that might otherwise require exponential time with brute force. They're used everywhere from task scheduling to network routing to compression algorithms.

**Real-world analogy:** Imagine you're hiking to a mountain peak in dense fog. A greedy approach would be: "At each step, move in the direction that goes most steeply upward." This works perfectly if there's only one peak (convex landscape), but fails if there are local peaks that aren't the highest (non-convex). Similarly, greedy algorithms work when the problem structure guarantees that local optimal choices lead to global optimality.

## Core Concepts

### Key Principles

1. **Greedy choice property:** A global optimum can be arrived at by selecting local optimums. Making the locally optimal choice at each step leads to a globally optimal solution.

2. **Optimal substructure:** An optimal solution to the problem contains optimal solutions to subproblems. After making a greedy choice, the remaining problem has the same structure.

3. **Irrevocability:** Once a greedy choice is made, it's never reconsidered. We commit to decisions without backtracking.

4. **Proof of correctness:** Greedy algorithms require proof that the greedy strategy actually works. Not all problems have greedy solutions!

### Essential Terms

- **Greedy choice:** The locally optimal decision made at each step
- **Feasible solution:** A solution that satisfies all constraints
- **Optimal solution:** The best feasible solution according to the objective function
- **Greedy stays ahead:** Proof technique showing greedy is at least as good as any other solution
- **Exchange argument:** Proof technique where we show we can exchange non-greedy choices with greedy ones without worsening the solution
- **Matroid:** Mathematical structure that guarantees greedy algorithms work

### Visual Overview

```
Problem: Maximize value by selecting items with constraints

Greedy Approach Decision Tree:

Start
  │
  ├─ Choice 1: Pick best current option ✓
  │   │
  │   └─ Choice 2: Pick best from remaining ✓
  │       │
  │       └─ Choice 3: Pick best from remaining ✓
  │           │
  │           └─ Solution Found!

vs.

Optimal Exhaustive Search (for comparison):

Start
  ├─ Try option A
  │   ├─ Try option B
  │   │   └─ Try option C → Solution 1
  │   └─ Try option C
  │       └─ Try option B → Solution 2
  ├─ Try option B
  │   ├─ Try option A → Solution 3
  │   └─ Try option C → Solution 4
  └─ Try option C → ...many more solutions

Greedy: O(n log n) - one path
Exhaustive: O(2^n) - all paths
```

**Key Insight:** Greedy works when the problem has the property that an optimal solution can be constructed by making greedy choices. This is NOT true for all problems!

### When Greedy Works vs. Doesn't Work

**Works for:**
- Minimum spanning tree (Kruskal's, Prim's)
- Shortest path (Dijkstra's - for non-negative weights)
- Activity selection / interval scheduling
- Huffman coding
- Fractional knapsack

**Doesn't work for:**
- 0/1 knapsack (need dynamic programming)
- Longest path problem
- Shortest path with negative weights
- Traveling salesman problem

## How It Works

### General Greedy Algorithm Pattern

1. **Cast the problem as one where we make a choice**
   - Identify what decision needs to be made at each step
   
2. **Define what "greedy" means for this problem**
   - Determine the selection criteria (max value, min cost, earliest deadline, etc.)
   
3. **Prove that greedy choice is safe**
   - Show that there's always an optimal solution that includes the greedy choice
   
4. **Demonstrate optimal substructure**
   - Show that after making greedy choice, remaining problem has same structure
   
5. **Implement the greedy strategy**
   - Often involves sorting by the greedy criterion
   - Iterate and make greedy choices
   - Build up solution incrementally

### Visual Walkthrough: Activity Selection Problem

**Problem:** Given N activities with start/end times, select maximum number of non-overlapping activities.

Activities: `[(1,3), (2,5), (4,7), (1,8), (5,9), (8,10), (9,11), (11,14), (13,16)]`

```
Timeline visualization:
Time:  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16
       ┌──A──┐
       └─────B────┐
             ┌────C────┐
       └──────────D──────────┐
                   ┌─────E────┐
                         ┌──F──┐
                            ┌──G───┐
                                  ┌─────H─────┐
                                        └──────I──────┐

Greedy Strategy: Always pick activity that finishes earliest

Step 1: Sort by end time
[(1,3), (2,5), (4,7), (1,8), (5,9), (8,10), (9,11), (11,14), (13,16)]
   A      B      C      D      E      F       G       H        I

Step 2: Pick first (earliest ending)
Selected: [A (1,3)]
Current end time: 3

Step 3: Pick next that starts after 3
Skip B (starts at 2 < 3)
Pick C (starts at 4 ≥ 3)
Selected: [A, C]
Current end time: 7

Step 4: Pick next that starts after 7
Skip D, E (start before 7)
Pick F (starts at 8 ≥ 7)
Selected: [A, C, F]
Current end time: 10

Step 5: Pick next that starts after 10
Skip G (starts at 9 < 10)
Pick H (starts at 11 ≥ 10)
Selected: [A, C, F, H]
Current end time: 14

Step 6: Pick next that starts after 14
No more activities
Final: [A, C, F, H] - 4 activities

Why this is optimal:
- By always picking earliest ending, we leave maximum time for future activities
- Any other choice would end later, leaving less room
```

**State Changes Table:**

| Step | Consider | Start | End | Current End | Compatible? | Action | Selected |
|------|----------|-------|-----|-------------|-------------|--------|----------|
| 1 | A | 1 | 3 | 0 | Yes | Pick | [A] |
| 2 | B | 2 | 5 | 3 | No (2<3) | Skip | [A] |
| 3 | C | 4 | 7 | 3 | Yes (4≥3) | Pick | [A,C] |
| 4 | D | 1 | 8 | 7 | No (1<7) | Skip | [A,C] |
| 5 | E | 5 | 9 | 7 | No (5<7) | Skip | [A,C] |
| 6 | F | 8 | 10 | 7 | Yes (8≥7) | Pick | [A,C,F] |
| 7 | G | 9 | 11 | 10 | No (9<10) | Skip | [A,C,F] |
| 8 | H | 11 | 14 | 10 | Yes (11≥10) | Pick | [A,C,F,H] |
| 9 | I | 13 | 16 | 14 | No (13<14) | Skip | [A,C,F,H] |

## Implementation

### Python Implementation

```python
from typing import List, Tuple
import heapq

def activity_selection(activities: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Select maximum number of non-overlapping activities.
    Classic greedy algorithm - always pick earliest finishing activity.
    
    Args:
        activities: List of (start_time, end_time) tuples
        
    Returns:
        List of selected non-overlapping activities
        
    Time Complexity: O(n log n) - dominated by sorting
    Space Complexity: O(n) for result
    
    Example:
        >>> activity_selection([(1,3), (2,5), (4,7)])
        [(1, 3), (4, 7)]
    """
    if not activities:
        return []
    
    # Step 1: Sort by end time (greedy criterion)
    sorted_activities = sorted(activities, key=lambda x: x[1])
    
    # Step 2: Always pick first (earliest ending)
    selected = [sorted_activities[0]]
    current_end = sorted_activities[0][1]
    
    # Step 3: Pick next compatible activity
    for start, end in sorted_activities[1:]:
        if start >= current_end:  # Non-overlapping
            selected.append((start, end))
            current_end = end
    
    return selected


def fractional_knapsack(weights: List[int], values: List[int], capacity: int) -> float:
    """
    Fractional knapsack: can take fractions of items.
    Greedy works here! (Unlike 0/1 knapsack)
    
    Args:
        weights: Weight of each item
        values: Value of each item  
        capacity: Maximum weight capacity
        
    Returns:
        Maximum value achievable
        
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    Example:
        >>> fractional_knapsack([10, 20, 30], [60, 100, 120], 50)
        240.0  # Take all of item 3, all of item 2, none of item 1
    """
    if not weights or capacity <= 0:
        return 0.0
    
    # Calculate value per unit weight
    items = []
    for i in range(len(weights)):
        value_per_weight = values[i] / weights[i]
        items.append((value_per_weight, weights[i], values[i]))
    
    # Sort by value per weight (descending)
    items.sort(reverse=True, key=lambda x: x[0])
    
    total_value = 0.0
    remaining_capacity = capacity
    
    for value_per_weight, weight, value in items:
        if remaining_capacity >= weight:
            # Take entire item
            total_value += value
            remaining_capacity -= weight
        else:
            # Take fraction of item
            fraction = remaining_capacity / weight
            total_value += value * fraction
            break  # Knapsack is full
    
    return total_value


def minimum_platforms(arrivals: List[int], departures: List[int]) -> int:
    """
    Find minimum railway platforms needed for train schedule.
    Greedy approach: track simultaneous trains at each time point.
    
    Args:
        arrivals: Arrival times of trains
        departures: Departure times of trains
        
    Returns:
        Minimum platforms needed
        
    Time Complexity: O(n log n)
    Space Complexity: O(1)
    
    Example:
        >>> minimum_platforms([900, 940, 950], [910, 1200, 1120])
        2  # At 950, trains 1 and 2 are both present
    """
    if not arrivals:
        return 0
    
    # Sort both arrays
    arrivals = sorted(arrivals)
    departures = sorted(departures)
    
    platforms_needed = 0
    max_platforms = 0
    
    i = j = 0
    n = len(arrivals)
    
    while i < n and j < n:
        if arrivals[i] <= departures[j]:
            # Train arriving, need one more platform
            platforms_needed += 1
            max_platforms = max(max_platforms, platforms_needed)
            i += 1
        else:
            # Train departing, free up a platform
            platforms_needed -= 1
            j += 1
    
    return max_platforms


def jump_game(nums: List[int]) -> bool:
    """
    Determine if you can reach last index. Each element is max jump length.
    Greedy: track furthest reachable position.
    
    Args:
        nums: Array where nums[i] is max jump from position i
        
    Returns:
        True if can reach last index
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Example:
        >>> jump_game([2,3,1,1,4])
        True  # Jump 1 step to index 1, then 3 steps to end
        >>> jump_game([3,2,1,0,4])
        False  # Stuck at index 3 (value 0)
    """
    if not nums or len(nums) == 1:
        return True
    
    max_reach = 0
    
    for i in range(len(nums)):
        # If current position is beyond what we can reach, fail
        if i > max_reach:
            return False
        
        # Update furthest we can reach
        max_reach = max(max_reach, i + nums[i])
        
        # Early termination: if we can reach the end
        if max_reach >= len(nums) - 1:
            return True
    
    return max_reach >= len(nums) - 1


def minimum_add_parentheses_valid(s: str) -> int:
    """
    Minimum additions to make parentheses string valid.
    Greedy: track unmatched opening and closing parentheses.
    
    Args:
        s: String of parentheses
        
    Returns:
        Minimum additions needed
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Example:
        >>> minimum_add_parentheses_valid("())")
        1  # Add one '(' at start
        >>> minimum_add_parentheses_valid("(((")
        3  # Add three ')' at end
    """
    open_needed = 0   # ')' needed to close unmatched '('
    close_needed = 0  # '(' needed for unmatched ')'
    
    for char in s:
        if char == '(':
            open_needed += 1
        elif char == ')':
            if open_needed > 0:
                open_needed -= 1  # Match with previous '('
            else:
                close_needed += 1  # No '(' to match, need to add one
    
    return open_needed + close_needed


def remove_duplicate_letters(s: str) -> str:
    """
    Remove duplicate letters so result is smallest in lexicographical order.
    Greedy with stack: remove characters if we can place smaller one later.
    
    Args:
        s: Input string
        
    Returns:
        Smallest lexicographical string with unique characters
        
    Time Complexity: O(n)
    Space Complexity: O(1) - at most 26 letters
    
    Example:
        >>> remove_duplicate_letters("bcabc")
        "abc"  # Remove 'b' and 'c', keep first occurrence
        >>> remove_duplicate_letters("cbacdcbc")
        "acdb"
    """
    # Count occurrences of each character
    last_occurrence = {char: i for i, char in enumerate(s)}
    
    stack = []
    in_stack = set()
    
    for i, char in enumerate(s):
        if char in in_stack:
            continue  # Already in result, skip
        
        # Remove larger characters from stack if they appear later
        while stack and stack[-1] > char and last_occurrence[stack[-1]] > i:
            removed = stack.pop()
            in_stack.remove(removed)
        
        stack.append(char)
        in_stack.add(char)
    
    return ''.join(stack)


def maximum_length_pair_chain(pairs: List[List[int]]) -> int:
    """
    Find longest chain of pairs where pair[i][1] < pair[j][0].
    Greedy: same as activity selection - sort by end value.
    
    Args:
        pairs: List of [start, end] pairs
        
    Returns:
        Maximum chain length
        
    Time Complexity: O(n log n)
    Space Complexity: O(1)
    
    Example:
        >>> maximum_length_pair_chain([[1,2], [2,3], [3,4]])
        2  # Chain: [1,2] -> [3,4]
    """
    if not pairs:
        return 0
    
    # Sort by end value (greedy criterion)
    pairs.sort(key=lambda x: x[1])
    
    count = 1
    current_end = pairs[0][1]
    
    for i in range(1, len(pairs)):
        if pairs[i][0] > current_end:  # Can chain
            count += 1
            current_end = pairs[i][1]
    
    return count


def valid_palindrome_ii(s: str) -> bool:
    """
    Check if string can be palindrome after deleting at most one character.
    Greedy: use two pointers, when mismatch try deleting from either side.
    
    Args:
        s: Input string
        
    Returns:
        True if can be palindrome with ≤1 deletion
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Example:
        >>> valid_palindrome_ii("aba")
        True  # Already palindrome
        >>> valid_palindrome_ii("abca")
        True  # Delete 'c' -> "aba"
        >>> valid_palindrome_ii("abc")
        False
    """
    def is_palindrome_range(s: str, left: int, right: int) -> bool:
        """Check if s[left:right+1] is palindrome."""
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True
    
    left, right = 0, len(s) - 1
    
    while left < right:
        if s[left] != s[right]:
            # Try deleting from left or right
            return (is_palindrome_range(s, left + 1, right) or 
                    is_palindrome_range(s, left, right - 1))
        left += 1
        right -= 1
    
    return True


def largest_palindromic_number(num: str) -> str:
    """
    Construct largest palindromic number from digits (no leading zeros).
    Greedy: place largest digits at edges, working toward center.
    
    Args:
        num: String of digits
        
    Returns:
        Largest palindromic number as string
        
    Time Complexity: O(n)
    Space Complexity: O(1) - at most 10 digits (0-9)
    
    Example:
        >>> largest_palindromic_number("444947137")
        "7449447"
    """
    from collections import Counter
    
    freq = Counter(num)
    
    # Build first half (largest to smallest)
    first_half = []
    middle = ""
    
    for digit in "9876543210":
        count = freq[digit]
        
        if count == 0:
            continue
        
        # For first half, use pairs
        pairs = count // 2
        
        # Avoid leading zeros
        if digit == '0' and not first_half:
            middle = '0' if count % 2 == 1 else middle
            continue
        
        first_half.extend([digit] * pairs)
        
        # Track largest odd count for middle
        if count % 2 == 1 and (not middle or digit > middle):
            middle = digit
    
    if not first_half and not middle:
        return "0"
    
    # Build result
    first = ''.join(first_half)
    second = first[::-1]
    
    return first + middle + second


# Usage Examples
if __name__ == "__main__":
    # Example 1: Activity selection
    activities = [(1, 3), (2, 5), (4, 7), (1, 8), (5, 9)]
    selected = activity_selection(activities)
    print(f"Selected activities: {selected}")
    # Output: [(1, 3), (4, 7), (8, 10)]
    
    # Example 2: Fractional knapsack
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    max_value = fractional_knapsack(weights, values, capacity)
    print(f"Maximum value: {max_value}")
    # Output: 240.0
    
    # Example 3: Minimum platforms
    arrivals = [900, 940, 950, 1100, 1500, 1800]
    departures = [910, 1200, 1120, 1130, 1900, 2000]
    platforms = minimum_platforms(arrivals, departures)
    print(f"Minimum platforms needed: {platforms}")
    # Output: 3
    
    # Example 4: Jump game
    can_jump = jump_game([2, 3, 1, 1, 4])
    print(f"Can reach end: {can_jump}")
    # Output: True
    
    # Example 5: Remove duplicate letters
    result = remove_duplicate_letters("bcabc")
    print(f"Smallest unique string: {result}")
    # Output: "abc"
```

### Code Explanation

**Key Design Decisions:**

1. **Why sort in activity selection?**
   - Sorting by end time ensures we always pick the activity that finishes earliest
   - This leaves maximum room for subsequent activities
   - The greedy choice (earliest ending) is provably optimal

2. **Why track value per weight in fractional knapsack?**
   - We want maximum value per unit capacity
   - Taking items with highest value density first is optimal
   - Can prove this by exchange argument: swapping any non-greedy choice makes it worse

3. **Why two pointers in minimum platforms?**
   - We're essentially counting overlapping intervals
   - When a train arrives, we need a platform (+1)
   - When a train departs, we free a platform (-1)
   - Track maximum simultaneous trains

4. **Why track max_reach in jump game?**
   - We don't need to find the actual path, just if end is reachable
   - At each position, we can reach any position up to i + nums[i]
   - If we ever can't reach position i, we fail

5. **Why use stack in remove duplicate letters?**
   - Stack maintains the result in progress
   - We can pop characters if we know they appear later
   - Last occurrence tracking ensures we don't remove last instance
   - Greedy: always try to place smaller character earlier

## Complexity Analysis

### Time Complexity

**Most greedy algorithms:**
- **Sorting phase:** O(n log n) - often the dominant factor
- **Greedy selection:** O(n) - single pass through sorted data
- **Overall:** **O(n log n)**

**Some greedy algorithms** (no sorting needed):
- **Linear scan:** O(n) - examples: jump game, valid parentheses
- **Overall:** **O(n)**

**Why sorting dominates?**
1. Many greedy algorithms require data in a specific order
2. Sorting takes O(n log n)
3. After sorting, greedy selection is typically O(n)
4. Therefore: O(n log n + n) = O(n log n)

### Space Complexity

- **Sorting in-place:** O(1) auxiliary space (or O(log n) for recursion stack)
- **Result storage:** O(n) in worst case
- **Overall:** Typically **O(1) to O(n)**

### Comparison with Alternatives

| Problem Type | Greedy | Dynamic Programming | Backtracking | When to Use |
|-------------|---------|---------------------|--------------|-------------|
| **Activity Selection** | O(n log n) ✓ | O(n²) | O(2ⁿ) | Greedy optimal |
| **Fractional Knapsack** | O(n log n) ✓ | N/A | N/A | Greedy works |
| **0/1 Knapsack** | ❌ Wrong | O(nW) ✓ | O(2ⁿ) | Need DP |
| **Min Platforms** | O(n log n) ✓ | O(n²) | O(2ⁿ) | Greedy optimal |
| **Coin Change** | ❌ Sometimes | O(nW) ✓ | O(2ⁿ) | DP needed |
| **Graph Shortest Path** | O(E log V) ✓ | O(VE) | Exponential | Dijkstra (greedy) |

**When Greedy Wins:**
- Problem has greedy choice property and optimal substructure
- O(n log n) or O(n) vs exponential alternatives
- Simple to implement and understand
- Provably correct for the specific problem

**When Greedy Fails:**
- 0/1 knapsack: {weights: [1,2,3], values: [6,10,12], capacity: 5}
  - Greedy by value density: take item 1 (6/1=6) → total value = 6
  - Optimal: take items 2+3 → total value = 22
- Coin change with coins {1, 3, 4}, target 6:
  - Greedy: 4+1+1 = 3 coins
  - Optimal: 3+3 = 2 coins

## Examples

### Example 1: Activity Selection - Maximum Activities

**Problem:** Select maximum activities from: `[(1,4), (3,5), (0,6), (5,7), (3,9), (5,9), (6,10), (8,11), (8,12), (2,14), (12,16)]`

**Solution:**
```python
activities = [(1,4), (3,5), (0,6), (5,7), (3,9), (5,9), (6,10), (8,11), (8,12), (2,14), (12,16)]

# Step 1: Sort by end time
sorted_acts = [(1,4), (3,5), (0,6), (5,7), (3,9), (5,9), (6,10), (8,11), (8,12), (2,14), (12,16)]

# Step 2: Select greedily
# Pick (1,4), end=4
# Skip (3,5), starts at 3 < 4
# Skip (0,6), starts at 0 < 4
# Pick (5,7), starts at 5 ≥ 4, end=7
# Skip (3,9), (5,9), starts before 7
# Skip (6,10), starts at 6 < 7
# Pick (8,11), starts at 8 ≥ 7, end=11
# Skip (8,12), starts before 11
# Skip (2,14), starts before 11
# Pick (12,16), starts at 12 ≥ 11

# Result: [(1,4), (5,7), (8,11), (12,16)] - 4 activities
```

**Visualization:**
```
Time:  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16
       ┌─────────┐                                          (0,6)
          ┌───┐                                             (1,4) ✓
                └──┐                                        (3,5)
                └──────┐                                    (3,9)
                      ┌──┐                                  (5,7) ✓
                      └──────┐                              (5,9)
                         ┌────┐                             (6,10)
                               ┌────┐                       (8,11) ✓
                               └──────┐                     (8,12)
          └──────────────────────────────┐                 (2,14)
                                        └──────┐            (12,16) ✓
```

### Example 2: Fractional Knapsack

**Problem:** Weights=[10,20,30], Values=[60,100,120], Capacity=50

**Solution:**
```python
# Calculate value per weight:
# Item 0: 60/10 = 6.0
# Item 1: 100/20 = 5.0
# Item 2: 120/30 = 4.0

# Sort by value density (descending): [Item0, Item1, Item2]

# Greedy selection:
# Take all of Item 0: weight=10, value=60, remaining=40
# Take all of Item 1: weight=20, value=100, remaining=20
# Take 20/30 of Item 2: weight=20, value=80, remaining=0

# Total value: 60 + 100 + 80 = 240

# Breakdown:
# Item 0: 100% × 60 = 60
# Item 1: 100% × 100 = 100
# Item 2: 66.7% × 120 = 80
# Total: 240
```

**Why greedy works here:**
```
Any other combination gives less value:

Option A (greedy): 100% × Item0 + 100% × Item1 + 66.7% × Item2
                  = 60 + 100 + 80 = 240

Option B: 100% × Item0 + 100% × Item2 + 0% × Item1
         = 60 + 120 + 0 = 180 (worse!)

Option C: 100% × Item1 + 100% × Item2 + 0% × Item0
         = 100 + 120 + 0 = 220 (worse!)
         
Greedy (highest density first) is provably optimal for fractional knapsack.
```

### Example 3: Minimum Platforms Needed

**Problem:** Arrivals=[900,940,950,1100,1500,1800], Departures=[910,1200,1120,1130,1900,2000]

**Solution:**
```python
# Already sorted
arrivals =   [900, 940, 950, 1100, 1500, 1800]
departures = [910, 1200, 1120, 1130, 1900, 2000]

# Simulation:
i=0, j=0, platforms=0, max=0
├─ arr[0]=900 ≤ dep[0]=910 → +1 platform, max=1, i=1
├─ arr[1]=940 > dep[0]=910 → -1 platform, platforms=0, j=1
├─ arr[1]=940 ≤ dep[1]=1200 → +1 platform, max=1, i=2
├─ arr[2]=950 ≤ dep[1]=1200 → +1 platform, max=2, i=3
├─ arr[3]=1100 ≤ dep[1]=1200 → +1 platform, max=3, i=4
├─ arr[4]=1500 > dep[1]=1200 → -1 platform, platforms=2, j=2
├─ arr[4]=1500 > dep[2]=1120 → -1 platform, platforms=1, j=3
├─ arr[4]=1500 > dep[3]=1130 → -1 platform, platforms=0, j=4
├─ arr[4]=1500 ≤ dep[4]=1900 → +1 platform, platforms=1, i=5
├─ arr[5]=1800 ≤ dep[4]=1900 → +1 platform, platforms=2, i=6
└─ Done

Maximum platforms needed: 3 (at time 1100)
```

**Timeline:**
```
Time:  900  940  950  1100 1120 1130 1200 1500 1800 1900 2000
       [────────────────]                                     Train 1
            [────────────────────────────────]                Train 2
                 [────────────────────────]                   Train 3
                      [──────────────────────]                Train 4
                                        [────────────────]    Train 5
                                             [──────────────] Train 6

At 1100: Trains 1,2,3,4 all present → 4 platforms? 
Wait, let me recheck...

Actually at time 1100:
- Train 1 (900-910): departed
- Train 2 (940-1200): present ✓
- Train 3 (950-1120): present ✓
- Train 4 (1100-1130): arriving ✓
→ 3 platforms needed
```

### Example 4: Jump Game - Can Reach End?

**Problem:** nums = [2,3,1,1,4], can we reach last index?

**Solution:**
```python
nums = [2, 3, 1, 1, 4]
Index: 0  1  2  3  4

Greedy: track furthest reachable position

i=0: nums[0]=2, max_reach = max(0, 0+2) = 2
     Can reach indices 0,1,2

i=1: nums[1]=3, max_reach = max(2, 1+3) = 4
     Can reach indices 0,1,2,3,4 ✓ (reached end!)
     
Early termination: max_reach ≥ 4, return True

Visualization:
Index: [0] [1] [2] [3] [4]
Value:  2   3   1   1   4
        └─┬─┘
          Can reach 0+2=2
            └───┬───┘
                Can reach 1+3=4 (the end!)
```

**Counterexample (can't reach):**
```python
nums = [3, 2, 1, 0, 4]
Index: 0  1  2  3  4

i=0: max_reach = 0+3 = 3
i=1: max_reach = max(3, 1+2) = 3
i=2: max_reach = max(3, 2+1) = 3
i=3: max_reach = max(3, 3+0) = 3 (stuck!)
i=4: i=4 > max_reach=3 → return False

Index: [0] [1] [2] [3] [4]
Value:  3   2   1   0   4
        └───────┬───┘
                max jump to index 3
                     └X (value 0, can't jump further)
                        └ Index 4 unreachable!
```

## Edge Cases

### 1. Empty Input
**Scenario:** Empty array or list

**Challenge:** No data to process

**Solution:** Return appropriate default value

```python
def activity_selection(activities):
    if not activities:
        return []  # No activities to select
    
def fractional_knapsack(weights, values, capacity):
    if not weights:
        return 0.0  # No value can be achieved
```

### 2. Single Element
**Scenario:** Only one item/activity

**Challenge:** No choice to make

**Solution:** Return that single element (if valid)

```python
# Activity selection with 1 activity
activities = [(1, 3)]
# Result: [(1, 3)] - select the only activity

# Jump game with single element
nums = [0]
# Result: True - already at end
```

### 3. All Elements Equal
**Scenario:** All items have same value/weight ratio

**Challenge:** Greedy choice doesn't distinguish

**Solution:** Any order works (stable sort preserves original order)

```python
# Fractional knapsack: all items have value/weight = 2
items = [(10, 20), (5, 10), (20, 40)]
# All ratios = 2, so any selection order gives same result
```

### 4. Capacity Zero
**Scenario:** Knapsack capacity is 0

**Challenge:** Can't take any items

**Solution:** Return 0

```python
def fractional_knapsack(weights, values, capacity):
    if capacity <= 0:
        return 0.0
```

### 5. Overlapping Intervals All Day
**Scenario:** All activities overlap completely

**Challenge:** Can only select one

**Solution:** Greedy correctly selects shortest duration

```python
# All activities overlap: [(1,10), (1,10), (1,10)]
# Greedy selects first one (or any, all have same end time)
# Result: 1 activity selected (correct!)
```

### 6. No Feasible Solution
**Scenario:** Jump game where position has value 0 and isn't the end

**Challenge:** Stuck, can't proceed

**Solution:** Return False

```python
nums = [0, 1, 2]
# At index 0, value=0, can't jump anywhere
# Result: False (unless array length is 1)
```

### 7. Negative Values
**Scenario:** Activities with negative start times, or negative weights

**Challenge:** Ensure comparisons work correctly

**Solution:** Algorithm handles naturally if comparison logic is sound

```python
# Activities can have negative times (e.g., time before epoch)
activities = [(-10, -5), (-3, 2), (1, 4)]
# Sort by end time works: -5 < 2 < 4
# Result: all three can be selected if non-overlapping
```

### 8. Very Large Numbers
**Scenario:** Weights/values near integer limits

**Challenge:** Potential overflow in calculations

**Solution:** Use floating point or arbitrary precision

```python
# Python handles large integers automatically
weights = [10**9, 2*10**9]
values = [10**15, 2*10**15]
# value/weight calculations work correctly
```

## Common Pitfalls

### ❌ Pitfall 1: Using Greedy When It Doesn't Work

**What happens:**
```python
# WRONG: Greedy for 0/1 Knapsack
def knapsack_greedy_wrong(weights, values, capacity):
    # Sort by value density
    items = sorted(zip(values, weights), key=lambda x: x[0]/x[1], reverse=True)
    
    total_value = 0
    for value, weight in items:
        if capacity >= weight:
            total_value += value
            capacity -= weight
    
    return total_value

# Example where it fails:
weights = [1, 2, 3]
values = [6, 10, 12]
capacity = 5

# Greedy picks: Item 0 (6/1=6), then can't fit item 2
# Total: 6
# Optimal: Items 1+2 (10+12=22) ❌
```

**Why it's wrong:** 0/1 knapsack doesn't have greedy choice property. Taking highest value density first doesn't guarantee global optimum.

**Correct approach:** Use dynamic programming for 0/1 knapsack.

### ❌ Pitfall 2: Wrong Greedy Criterion

**What happens:**
```python
# WRONG: Sort by start time instead of end time
def activity_selection_wrong(activities):
    # Sorting by start time
    sorted_acts = sorted(activities, key=lambda x: x[0])
    
    selected = [sorted_acts[0]]
    current_end = sorted_acts[0][1]
    
    for start, end in sorted_acts[1:]:
        if start >= current_end:
            selected.append((start, end))
            current_end = end
    
    return selected

# Example: [(1,10), (2,3), (4,5)]
# Sorted by start: [(1,10), (2,3), (4,5)]
# Wrong greedy: picks (1,10), can't pick others
# Result: 1 activity
# Optimal: (2,3), (4,5) → 2 activities ❌
```

**Why it's wrong:** Sorting by start time doesn't minimize overlap. An activity starting early might block many later activities.

**Correct approach:** Sort by end time - finishing early leaves maximum room.

### ❌ Pitfall 3: Not Proving Greedy Works

**What happens:**
```python
# WRONG: Assuming greedy works without proof
def coin_change_greedy_wrong(coins, amount):
    """
    This ONLY works for canonical coin systems (like US coins).
    Fails for arbitrary coin sets!
    """
    coins.sort(reverse=True)
    count = 0
    
    for coin in coins:
        while amount >= coin:
            amount -= coin
            count += 1
    
    return count if amount == 0 else -1

# Example where it fails:
coins = [1, 3, 4]
amount = 6

# Greedy: 4 + 1 + 1 = 3 coins
# Optimal: 3 + 3 = 2 coins ❌
```

**Why it's wrong:** Greedy coin change only works for specific coin systems (canonical). For arbitrary coins, greedy can be suboptimal.

**Correct approach:** Use dynamic programming for general coin change.

### ❌ Pitfall 4: Modifying Input Data

**What happens:**
```python
# WRONG: Modifying input array
def fractional_knapsack_wrong(weights, values, capacity):
    # Sorting modifies original arrays!
    weights.sort()
    values.sort()
    # Now weights and values are no longer aligned!
```

**Why it's wrong:** Sorting separately destroys the correspondence between weights and values.

**Correct approach:**
```python
# CORRECT: Keep pairs together
items = list(zip(weights, values))
items.sort(key=lambda x: x[1]/x[0], reverse=True)
# Or sort indices
```

### ❌ Pitfall 5: Off-by-One Errors in Comparisons

**What happens:**
```python
# WRONG: Using > instead of >=
def activity_selection_wrong(activities):
    sorted_acts = sorted(activities, key=lambda x: x[1])
    selected = [sorted_acts[0]]
    current_end = sorted_acts[0][1]
    
    for start, end in sorted_acts[1:]:
        # WRONG: should be >=
        if start > current_end:  
            selected.append((start, end))
            current_end = end
    
    return selected

# Example: [(1,2), (2,3)]
# Activity 2 starts exactly when activity 1 ends
# With >: rejected (wrong! activities don't overlap)
# With >=: accepted (correct!)
```

**Why it's wrong:** If one activity ends at time T and another starts at time T, they don't overlap. Should use `>=`.

**Correct approach:** Use `start >= current_end` for non-overlapping.

### ❌ Pitfall 6: Not Handling Ties Correctly

**What happens:**
```python
# WRONG: Unstable when values are equal
def activity_selection_unstable(activities):
    # If multiple activities end at same time, 
    # which do we pick?
    sorted_acts = sorted(activities, key=lambda x: x[1])
    
    # If activities [(1,5), (2,5)] both end at 5,
    # which is better? Need secondary criterion!
```

**Why it's wrong:** When greedy criterion ties, need tiebreaker rule.

**Correct approach:**
```python
# CORRECT: Use multiple sort keys
sorted_acts = sorted(activities, key=lambda x: (x[1], x[0]))
# Sort by end time, then by start time
```

### ❌ Pitfall 7: Forgetting Edge Cases

**What happens:**
```python
# WRONG: Doesn't handle empty input
def jump_game_wrong(nums):
    max_reach = 0
    
    for i in range(len(nums)):  # Crashes if nums is empty!
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])
    
    return max_reach >= len(nums) - 1
```

**Why it's wrong:** Empty array causes issues; single element needs special handling.

**Correct approach:**
```python
def jump_game_correct(nums):
    if not nums or len(nums) == 1:
        return True  # Handle edge cases first
    # ... rest of algorithm
```

## Variations and Extensions

### Variation 1: Activity Selection with Weights/Values

**Description:** Select activities to maximize total value (not just count).

**When to use:** When activities have different importance/profit.

**Key differences:**
- Not just counting activities
- Weighted interval scheduling problem
- Greedy doesn't always work - need DP!

**Implementation:**
```python
def weighted_activity_selection_dp(activities: List[Tuple[int, int, int]]) -> int:
    """
    Select non-overlapping activities to maximize value.
    activities: [(start, end, value)]
    
    Greedy DOESN'T work here! Need DP.
    
    Time: O(n log n + n²)
    Space: O(n)
    """
    if not activities:
        return 0
    
    # Sort by end time
    activities.sort(key=lambda x: x[1])
    n = len(activities)
    
    # dp[i] = max value using activities 0..i
    dp = [0] * n
    dp[0] = activities[0][2]
    
    for i in range(1, n):
        # Option 1: Include current activity
        include_value = activities[i][2]
        
        # Find last non-overlapping activity
        for j in range(i - 1, -1, -1):
            if activities[j][1] <= activities[i][0]:
                include_value += dp[j]
                break
        
        # Option 2: Exclude current activity
        exclude_value = dp[i - 1]
        
        dp[i] = max(include_value, exclude_value)
    
    return dp[n - 1]
```

### Variation 2: Gas Station Circuit

**Description:** Can you complete circuit if gas[i] is gas at station i and cost[i] is cost to reach next station?

**When to use:** Circuit/cycle problems with local gains/costs.

**Implementation:**
```python
def can_complete_circuit(gas: List[int], cost: List[int]) -> int:
    """
    Find starting gas station to complete circuit.
    
    Greedy insight: If total gas >= total cost, solution exists.
    Start from position where we first run out.
    
    Time: O(n)
    Space: O(1)
    
    Example:
        gas = [1,2,3,4,5], cost = [3,4,5,1,2]
        Output: 3 (start from station 3)
    """
    total_gas = sum(gas)
    total_cost = sum(cost)
    
    if total_gas < total_cost:
        return -1  # Impossible
    
    current_gas = 0
    start = 0
    
    for i in range(len(gas)):
        current_gas += gas[i] - cost[i]
        
        if current_gas < 0:
            # Can't reach next station from current start
            # Try starting from next station
            start = i + 1
            current_gas = 0
    
    return start
```

### Variation 3: Minimum Number of Arrows to Burst Balloons

**Description:** Balloons on wall at various positions, find minimum arrows to burst all.

**When to use:** Interval covering problems.

**Implementation:**
```python
def find_min_arrow_shots(points: List[List[int]]) -> int:
    """
    Find minimum arrows to burst all balloons.
    Each balloon is interval [start, end].
    Arrow at position x bursts all balloons containing x.
    
    Greedy: Sort by end, shoot arrow at end of first balloon.
    
    Time: O(n log n)
    Space: O(1)
    
    Example:
        points = [[10,16],[2,8],[1,6],[7,12]]
        Output: 2
    """
    if not points:
        return 0
    
    # Sort by end position
    points.sort(key=lambda x: x[1])
    
    arrows = 1
    arrow_pos = points[0][1]
    
    for start, end in points[1:]:
        if start > arrow_pos:
            # Need new arrow
            arrows += 1
            arrow_pos = end
    
    return arrows
```

### Variation 4: Task Scheduler with Cooldown

**Description:** Schedule tasks with cooldown period to minimize total time.

**When to use:** Scheduling with constraints.

**Implementation:**
```python
def least_interval(tasks: List[str], n: int) -> int:
    """
    Schedule tasks with cooldown of n intervals between same task.
    
    Greedy: Always schedule most frequent available task.
    
    Time: O(N) where N is number of tasks
    Space: O(1) - at most 26 different tasks
    
    Example:
        tasks = ["A","A","A","B","B","B"], n = 2
        Output: 8 (A -> B -> idle -> A -> B -> idle -> A -> B)
    """
    from collections import Counter
    import heapq
    
    freq = Counter(tasks)
    max_heap = [-count for count in freq.values()]
    heapq.heapify(max_heap)
    
    time = 0
    
    while max_heap:
        temp = []
        
        # Try to schedule n+1 tasks (1 cycle)
        for _ in range(n + 1):
            if max_heap:
                temp.append(heapq.heappop(max_heap))
            time += 1
            
            if not max_heap and not temp:
                break
        
        # Put back tasks that still have instances
        for count in temp:
            if count + 1 < 0:  # Still has remaining tasks
                heapq.heappush(max_heap, count + 1)
        
        # Adjust time if we finished early in this cycle
        if not max_heap:
            time -= (n + 1 - len(temp))
    
    return time
```

### Variation 5: Reorganize String (No Adjacent Duplicates)

**Description:** Rearrange string so no two adjacent characters are the same.

**When to use:** Arrangement/permutation problems with constraints.

**Implementation:**
```python
def reorganize_string(s: str) -> str:
    """
    Rearrange string so no adjacent characters are same.
    
    Greedy: Always place most frequent character (that wasn't just placed).
    
    Time: O(n log k) where k is number of unique characters
    Space: O(k)
    
    Example:
        "aab" -> "aba"
        "aaab" -> "" (impossible)
    """
    from collections import Counter
    import heapq
    
    freq = Counter(s)
    
    # Check if possible
    max_freq = max(freq.values())
    if max_freq > (len(s) + 1) // 2:
        return ""  # Impossible
    
    # Max heap of (frequency, character)
    max_heap = [(-count, char) for char, count in freq.items()]
    heapq.heapify(max_heap)
    
    result = []
    prev_freq, prev_char = 0, ''
    
    while max_heap:
        freq, char = heapq.heappop(max_heap)
        result.append(char)
        
        # Put back previous character if it still has count
        if prev_freq < 0:
            heapq.heappush(max_heap, (prev_freq, prev_char))
        
        # Update previous
        prev_freq = freq + 1  # Decrease count (freq is negative)
        prev_char = char
    
    return ''.join(result)
```

## Practice Problems

### Beginner

1. **Assign Cookies** - Greedy assignment to maximize happiness
   - LeetCode #455 (Assign Cookies)
   - Simple greedy: sort both arrays, assign smallest cookie to smallest greed

2. **Lemonade Change** - Can you provide change for all customers?
   - LeetCode #860 (Lemonade Change)
   - Greedy tracking of bill denominations

3. **Maximum Subarray** - Find contiguous subarray with largest sum (Kadane's)
   - LeetCode #53 (Maximum Subarray)
   - Greedy decision: extend or restart subarray

4. **Best Time to Buy and Sell Stock** - Maximize profit with one transaction
   - LeetCode #121 (Best Time to Buy and Sell Stock)
   - Greedy: track minimum price seen so far

### Intermediate

1. **Jump Game** - Can you reach the last index?
   - LeetCode #55 (Jump Game)
   - Greedy: track furthest reachable position

2. **Jump Game II** - Minimum jumps to reach end
   - LeetCode #45 (Jump Game II)
   - Greedy: jump to position with maximum reach

3. **Partition Labels** - Partition string into maximum parts
   - LeetCode #763 (Partition Labels)
   - Greedy: track last occurrence, extend partition

4. **Remove K Digits** - Remove K digits to form smallest number
   - LeetCode #402 (Remove K Digits)
   - Greedy with stack: remove larger digits when possible

5. **Gas Station** - Find starting station to complete circuit
   - LeetCode #134 (Gas Station)
   - Greedy: start from position where tank first goes negative

6. **Task Scheduler** - Schedule tasks with cooldown
   - LeetCode #621 (Task Scheduler)
   - Greedy: always schedule most frequent available task

7. **Non-overlapping Intervals** - Remove minimum to make non-overlapping
   - LeetCode #435 (Non-overlapping Intervals)
   - Activity selection variant

### Advanced

1. **Merge Triplets to Form Target** - Can you form target triplet?
   - LeetCode #1899 (Merge Triplets to Form Target Triplet)
   - Greedy: include triplets that don't exceed target

2. **Maximum Number of Events That Can Be Attended** - Attend max events
   - LeetCode #1353 (Maximum Number of Events That Can Be Attended)
   - Greedy with heap: attend earliest ending available event

3. **Minimum Number of Arrows to Burst Balloons** - Shoot minimum arrows
   - LeetCode #452 (Minimum Number of Arrows to Burst Balloons)
   - Interval covering greedy

4. **Queue Reconstruction by Height** - Reconstruct queue
   - LeetCode #406 (Queue Reconstruction by Height)
   - Greedy: process tallest first, insert by k value

5. **Remove Duplicate Letters** - Smallest lexicographical string
   - LeetCode #316 (Remove Duplicate Letters)
   - Greedy with stack and last occurrence tracking

6. **Create Maximum Number** - Create largest number from two arrays
   - LeetCode #321 (Create Maximum Number)
   - Complex greedy: merge two monotonic stacks

7. **Candy** - Distribute candy with neighbor constraints
   - LeetCode #135 (Candy)
   - Greedy: two passes (left-to-right, right-to-left)

## Real-World Applications

### Industry Use Cases

1. **Operating System Process Scheduling**
   - **How it's used:** CPU schedulers use greedy algorithms (Shortest Job First, Earliest Deadline First)
   - **Why it's effective:** Minimizes average waiting time or maximizes throughput
   - **Algorithms:** SJF, EDF, Rate-Monotonic Scheduling

2. **Network Routing**
   - **How it's used:** Dijkstra's algorithm (greedy) finds shortest paths in networks
   - **Why it's effective:** O(E log V) for finding shortest paths from source
   - **Applications:** Internet routing protocols (OSPF), GPS navigation

3. **Data Compression**
   - **How it's used:** Huffman coding (greedy) builds optimal prefix-free codes
   - **Why it's effective:** Minimizes expected code length based on character frequency
   - **Usage:** ZIP files, JPEG compression, network protocols

4. **Job Scheduling in Manufacturing**
   - **How it's used:** Schedule jobs on machines to minimize makespan or tardiness
   - **Why it's effective:** Greedy heuristics like SPT (Shortest Processing Time) give good approximations
   - **Scale:** Production lines, cloud computing resource allocation

5. **Financial Trading**
   - **How it's used:** Best time to buy/sell stocks - greedy algorithm finds optimal transaction points
   - **Why it's effective:** Single pass through prices, O(n) time
   - **Applications:** Trading algorithms, portfolio optimization

6. **Video Streaming**
   - **How it's used:** Adaptive bitrate streaming selects quality greedily based on current bandwidth
   - **Why it's effective:** Immediate decisions maximize current quality
   - **Example:** Netflix, YouTube quality selection

### Popular Implementations

- **Linux CFS Scheduler:** Greedy scheduling for fairness
- **Dijkstra in Network Protocols:** OSPF routing uses Dijkstra's greedy shortest path
- **Huffman Coding:** Used in ZIP, GZIP, PNG compression
- **Prim's/Kruskal's MST:** Network design, circuit design
- **Fractional Knapsack:** Resource allocation in cloud computing
- **Interval Scheduling:** Conference room booking systems, airport gate assignment
- **Task Scheduling:** Jenkins CI/CD, Kubernetes pod scheduling

### Practical Scenarios

- **Meeting Room Allocation:** Assign rooms to maximize meetings scheduled
- **Advertisement Placement:** Select ads to maximize revenue (greedy by CPC/CPM)
- **Cache Replacement:** LRU (Least Recently Used) is greedy eviction policy
- **Load Balancing:** Assign requests to servers greedily (least loaded, round-robin)
- **Delivery Route Optimization:** Greedy insertion heuristics for TSP approximation
- **Bandwidth Allocation:** Allocate network bandwidth to flows greedily
- **Exam Scheduling:** Schedule exams to minimize conflicts (graph coloring greedy)
- **Coin Change Machines:** Dispense change using greedy (works for canonical systems)

## Related Topics

### Prerequisites to Review

- **Sorting Algorithms** - Most greedy algorithms start with sorting
- **Arrays and Lists** - Basic data structures for greedy problems
- **Priority Queues/Heaps** - Used in advanced greedy algorithms (Dijkstra, Huffman)
- **Big-O Analysis** - Understanding complexity of greedy choices

### Next Steps

- **Dynamic Programming** - Alternative when greedy doesn't work
- **Graph Algorithms** - Dijkstra, Prim's, Kruskal's are greedy
- **Interval Scheduling Theory** - Deeper study of activity selection variants
- **Approximation Algorithms** - When greedy gives approximate solutions
- **Matroid Theory** - Mathematical framework for when greedy works

### Similar Concepts

- **Divide and Conquer** - Both make irreversible decisions, but D&C splits problem
- **Branch and Bound** - Explores search space more carefully than greedy
- **Heuristics** - Greedy is often used as heuristic for hard problems
- **Local Search** - Makes local improvements, related to greedy
- **Primal-Dual Algorithms** - Combine greedy with optimization theory

### Further Reading

- **"Introduction to Algorithms" (CLRS)** - Chapter 16 on Greedy Algorithms
  - Rigorous proofs of greedy correctness
  - Activity selection, Huffman codes, Matroid theory
  
- **"Algorithm Design" by Kleinberg & Tardos** - Chapter 4
  - Interval scheduling, exchange arguments
  - Excellent explanations of when greedy works
  
- **"The Algorithm Design Manual" by Skiena** - Greedy chapter
  - Practical greedy heuristics
  - War stories from real applications
  
- **LeetCode Greedy Tag:**
  - https://leetcode.com/tag/greedy/
  - 200+ greedy problems with discussions
  
- **GeeksforGeeks Greedy Algorithms:**
  - https://www.geeksforgeeks.org/greedy-algorithms/
  - Comprehensive tutorials with visualizations
  
- **Academic Papers:**
  - "Greedy Algorithms" by Allan Borodin
  - Research on approximation ratios for greedy heuristics
