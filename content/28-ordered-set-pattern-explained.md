# Ordered Set Pattern

**Difficulty:** Intermediate to Advanced
**Prerequisites:** Binary Search Trees, Set operations, Sorting
**Estimated Reading Time:** 35 minutes

## Introduction

An Ordered Set (also called Sorted Set or TreeSet) is a data structure that maintains elements in sorted order while supporting efficient insertion, deletion, and lookup operations. Unlike regular sets that only track membership, ordered sets maintain elements in a specific order and support range queries, finding predecessors/successors, and order statistics.

**Why it matters:** Ordered sets are crucial for problems requiring both set membership and ordering. They appear in interval scheduling, range queries, finding closest elements, and maintaining sliding window statistics. In Python, we use `sortedcontainers.SortedList` or implement with binary search. Languages like C++ and Java have built-in implementations (std::set, TreeSet). Understanding ordered sets bridges the gap between arrays (ordered but slow insertions) and hash sets (fast operations but unordered).

**Real-world analogy:** Think of an ordered set like a self-organizing bookshelf that always keeps books sorted by title. When you add a new book, it automatically slides into the correct alphabetical position. You can quickly answer questions like "What's the book right before 'Moby Dick'?" or "How many books are between 'Alice' and 'Zelda'?" Unlike a regular array where you'd have to shift everything, this bookshelf efficiently maintains order. Unlike a hash table where books are scattered randomly, this bookshelf lets you find the nearest book or walk through titles in order.

## Core Concepts

### Key Principles

1. **Sorted Order Maintenance:** Elements are automatically kept in sorted order based on a comparator function.

2. **Efficient Operations:** Insertion, deletion, and search in O(log n) time using self-balancing trees (typically Red-Black trees or AVL trees).

3. **Order Statistics:** Can efficiently answer queries like "find kth smallest" or "count elements in range".

4. **Range Queries:** Support finding elements within a given range efficiently.

5. **No Duplicates (usually):** Standard sets don't allow duplicates, though multisets do.

### Essential Terms

- **Ordered Set / Sorted Set / TreeSet:** Set maintaining elements in sorted order
- **Self-Balancing Tree:** Underlying structure (Red-Black tree, AVL tree)
- **Lower Bound:** Smallest element ≥ target
- **Upper Bound:** Smallest element > target
- **Floor:** Largest element ≤ target
- **Ceiling:** Smallest element ≥ target
- **Rank:** Position/index of element in sorted order
- **Range Query:** Finding elements within [low, high]

### Visual Overview

```
Ordered Set containing: [3, 7, 12, 15, 20, 25]

Underlying Red-Black Tree (conceptual):
         15
        /  \
      7     20
     / \   /  \
    3  12 17  25

Operations:
add(10):     [3, 7, 10, 12, 15, 20, 25]
remove(7):   [3, 10, 12, 15, 20, 25]
floor(14):   12 (largest ≤ 14)
ceiling(14): 15 (smallest ≥ 14)
rank(15):    3 (0-indexed: 3rd position)
range(10, 20): [10, 12, 15, 20]
```

## How It Works

### Basic Operations

**Step 1: Insertion**
- Insert element into tree maintaining BST property
- Rebalance tree (rotations) to maintain O(log n) height
- Element automatically placed in sorted position

**Step 2: Deletion**
- Find element in tree
- Remove node and reconnect children
- Rebalance tree if necessary

**Step 3: Search/Membership**
- Binary search through tree structure
- O(log n) time complexity

**Step 4: Range Queries**
- Use tree structure to efficiently skip irrelevant subtrees
- Only visit nodes in range

### Detailed Walkthrough Example

**Problem:** Calendar scheduling - check for conflicts
**Ordered Set stores:** [start_time, end_time] intervals sorted by start time

```
Calendar (ordered by start time):
Events: [(9:00, 10:00), (11:00, 12:00), (14:00, 15:00)]

Query: Can we book (10:30, 11:30)?

Step 1: Find potential conflicts
  Use ceiling(10:30) to find first event starting ≥ 10:30
  ceiling(10:30) = (11:00, 12:00)
  
  Use floor(10:30) to find last event starting ≤ 10:30
  floor(10:30) = (9:00, 10:00)

Step 2: Check conflicts
  Previous event: (9:00, 10:00)
    Ends at 10:00, new starts at 10:30 ✓ No overlap
  
  Next event: (11:00, 12:00)
    Starts at 11:00, new ends at 11:30 ✗ Overlap!
    
  Result: Cannot book (conflict with 11:00-12:00 event)

Visualization:
9:00    10:00 10:30    11:00 11:30 12:00    14:00    15:00
|----------|   |---------|...|-----|         |---------|
  Event 1      New Event     Event 2           Event 3
                   ↑          ↑
                   └──────────┘ Overlap!
```

## Implementation

### Python Implementation - Using SortedContainers

```python
from sortedcontainers import SortedList, SortedDict, SortedSet
from typing import List, Tuple, Optional
import bisect

class OrderedSetExample:
    """
    Examples using ordered sets (SortedList from sortedcontainers).
    
    Python doesn't have built-in ordered set, so we use sortedcontainers.
    Install: pip install sortedcontainers
    
    Alternative: Implement manually with binary search on list (slower)
    """
    
    def __init__(self):
        self.sorted_set = SortedList()
    
    def add(self, val: int) -> bool:
        """
        Add value to set.
        
        Time: O(log n)
        """
        if val in self.sorted_set:
            return False
        self.sorted_set.add(val)
        return True
    
    def remove(self, val: int) -> bool:
        """
        Remove value from set.
        
        Time: O(log n)
        """
        try:
            self.sorted_set.remove(val)
            return True
        except ValueError:
            return False
    
    def contains(self, val: int) -> bool:
        """
        Check if value exists.
        
        Time: O(log n)
        """
        return val in self.sorted_set
    
    def floor(self, val: int) -> Optional[int]:
        """
        Find largest element <= val.
        
        Time: O(log n)
        """
        idx = self.sorted_set.bisect_right(val)
        if idx == 0:
            return None  # No element <= val
        return self.sorted_set[idx - 1]
    
    def ceiling(self, val: int) -> Optional[int]:
        """
        Find smallest element >= val.
        
        Time: O(log n)
        """
        idx = self.sorted_set.bisect_left(val)
        if idx == len(self.sorted_set):
            return None  # No element >= val
        return self.sorted_set[idx]
    
    def lower(self, val: int) -> Optional[int]:
        """
        Find largest element < val.
        
        Time: O(log n)
        """
        idx = self.sorted_set.bisect_left(val)
        if idx == 0:
            return None
        return self.sorted_set[idx - 1]
    
    def higher(self, val: int) -> Optional[int]:
        """
        Find smallest element > val.
        
        Time: O(log n)
        """
        idx = self.sorted_set.bisect_right(val)
        if idx == len(self.sorted_set):
            return None
        return self.sorted_set[idx]
    
    def range_query(self, low: int, high: int) -> List[int]:
        """
        Find all elements in [low, high].
        
        Time: O(log n + k) where k is result size
        """
        left_idx = self.sorted_set.bisect_left(low)
        right_idx = self.sorted_set.bisect_right(high)
        return list(self.sorted_set[left_idx:right_idx])
    
    def kth_smallest(self, k: int) -> Optional[int]:
        """
        Find kth smallest element (0-indexed).
        
        Time: O(1)
        """
        if k < 0 or k >= len(self.sorted_set):
            return None
        return self.sorted_set[k]
    
    def rank(self, val: int) -> int:
        """
        Find number of elements < val.
        
        Time: O(log n)
        """
        return self.sorted_set.bisect_left(val)


def merge_similar_items(items1: List[List[int]], items2: List[List[int]]) -> List[List[int]]:
    """
    Merge two item lists, summing values for same keys.
    
    LeetCode #2363: Merge Similar Items
    
    Args:
        items1: List of [value, weight] pairs
        items2: List of [value, weight] pairs
        
    Returns:
        Merged list sorted by value
        
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    Example:
        >>> merge_similar_items([[1,1],[4,5],[3,8]], [[3,1],[1,5]])
        [[1,6],[3,9],[4,5]]
    """
    # Use SortedDict to maintain sorted order by key
    merged = SortedDict()
    
    for value, weight in items1:
        merged[value] = merged.get(value, 0) + weight
    
    for value, weight in items2:
        merged[value] = merged.get(value, 0) + weight
    
    return [[k, v] for k, v in merged.items()]


def find_132_pattern(nums: List[int]) -> bool:
    """
    Find if there exists i < j < k such that nums[i] < nums[k] < nums[j].
    
    LeetCode #456: 132 Pattern
    
    Args:
        nums: Array of integers
        
    Returns:
        True if 132 pattern exists
        
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    Example:
        >>> find_132_pattern([3,1,4,2])
        True  # 1 < 2 < 4
    """
    sorted_set = SortedList()
    min_so_far = float('inf')
    
    for num in nums:
        # For current num (playing role of k), 
        # find elements in sorted_set (potential j values) > num
        # We need: min_so_far < num < j
        
        # Find smallest element > num in sorted_set
        idx = sorted_set.bisect_right(num)
        
        # Check if there's any element in sorted_set that satisfies pattern
        if idx < len(sorted_set) and min_so_far < num:
            return True
        
        sorted_set.add(num)
        min_so_far = min(min_so_far, num)
    
    return False


class MyCalendar:
    """
    Calendar booking system preventing double bookings.
    
    LeetCode #729: My Calendar I
    
    Time Complexity: O(log n) per booking
    Space Complexity: O(n)
    
    Example:
        >>> cal = MyCalendar()
        >>> cal.book(10, 20)
        True
        >>> cal.book(15, 25)
        False  # Conflicts with [10, 20)
        >>> cal.book(20, 30)
        True
    """
    
    def __init__(self):
        self.events = SortedList()  # Store (start, end) tuples
    
    def book(self, start: int, end: int) -> bool:
        """
        Try to book event [start, end).
        
        Returns True if booking successful, False if conflict.
        """
        # Find position where this event would be inserted
        idx = self.events.bisect_left((start, end))
        
        # Check conflict with previous event
        if idx > 0:
            prev_start, prev_end = self.events[idx - 1]
            if prev_end > start:
                return False  # Previous event extends into this one
        
        # Check conflict with next event
        if idx < len(self.events):
            next_start, next_end = self.events[idx]
            if end > next_start:
                return False  # This event extends into next one
        
        # No conflicts, book the event
        self.events.add((start, end))
        return True


def longest_continuous_subarray(nums: List[int], limit: int) -> int:
    """
    Find longest subarray where |max - min| <= limit.
    
    LeetCode #1438: Longest Continuous Subarray With Absolute Diff <= Limit
    
    Uses ordered set to maintain min/max in sliding window.
    
    Args:
        nums: Array of integers
        limit: Maximum allowed difference
        
    Returns:
        Length of longest valid subarray
        
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    Example:
        >>> longest_continuous_subarray([8,2,4,7], 4)
        2  # [2,4] or [4,7]
    """
    sorted_window = SortedList()
    left = 0
    max_length = 0
    
    for right in range(len(nums)):
        sorted_window.add(nums[right])
        
        # While window invalid, shrink from left
        while sorted_window[-1] - sorted_window[0] > limit:
            sorted_window.remove(nums[left])
            left += 1
        
        max_length = max(max_length, right - left + 1)
    
    return max_length


def count_smaller_after_self(nums: List[int]) -> List[int]:
    """
    Count numbers smaller than each element to its right.
    
    LeetCode #315: Count of Smaller Numbers After Self
    
    Args:
        nums: Array of integers
        
    Returns:
        Array where result[i] = count of smaller elements after nums[i]
        
    Time Complexity: O(n log n)
    Space Complexity: O(n)
    
    Example:
        >>> count_smaller_after_self([5,2,6,1])
        [2,1,1,0]  # 5 has [2,1], 2 has [1], 6 has [1], 1 has []
    """
    result = []
    sorted_seen = SortedList()
    
    # Process from right to left
    for num in reversed(nums):
        # Count elements in sorted_seen that are < num
        count = sorted_seen.bisect_left(num)
        result.append(count)
        sorted_seen.add(num)
    
    return result[::-1]


def max_sum_subarray_with_constraint(nums: List[int], k: int) -> int:
    """
    Find max sum of subarray with length <= k where elements satisfy constraint.
    
    Uses ordered set to efficiently find valid subarrays.
    
    Args:
        nums: Array of integers
        k: Maximum subarray length
        
    Returns:
        Maximum sum
        
    Time Complexity: O(n log k)
    Space Complexity: O(k)
    """
    sorted_window = SortedList()
    max_sum = float('-inf')
    current_sum = 0
    left = 0
    
    for right in range(len(nums)):
        sorted_window.add(nums[right])
        current_sum += nums[right]
        
        # Maintain window size <= k
        while right - left + 1 > k:
            sorted_window.remove(nums[left])
            current_sum -= nums[left]
            left += 1
        
        # Check if current window is valid
        # Example constraint: max - min <= threshold
        if sorted_window[-1] - sorted_window[0] <= 100:
            max_sum = max(max_sum, current_sum)
    
    return max_sum


class OrderedMultiset:
    """
    Ordered multiset allowing duplicates.
    
    Useful when we need to maintain sorted order with repeated elements.
    """
    
    def __init__(self):
        self.elements = SortedList()
    
    def add(self, val: int) -> None:
        """Add element (duplicates allowed)."""
        self.elements.add(val)
    
    def remove(self, val: int) -> bool:
        """Remove one instance of element."""
        try:
            self.elements.remove(val)
            return True
        except ValueError:
            return False
    
    def count(self, val: int) -> int:
        """Count occurrences of value."""
        return self.elements.count(val)
    
    def median(self) -> float:
        """Find median of all elements."""
        n = len(self.elements)
        if n == 0:
            return 0
        if n % 2 == 1:
            return self.elements[n // 2]
        return (self.elements[n // 2 - 1] + self.elements[n // 2]) / 2.0


# Example usage and testing
if __name__ == "__main__":
    print("=== Basic Ordered Set Operations ===")
    oset = OrderedSetExample()
    
    oset.add(5)
    oset.add(2)
    oset.add(8)
    oset.add(3)
    print(f"Set: {oset.sorted_set}")
    
    print(f"Floor of 4: {oset.floor(4)}")  # 3
    print(f"Ceiling of 4: {oset.ceiling(4)}")  # 5
    print(f"Lower than 5: {oset.lower(5)}")  # 3
    print(f"Higher than 5: {oset.higher(5)}")  # 8
    print(f"Range [2, 6]: {oset.range_query(2, 6)}")  # [2, 3, 5]
    print(f"2nd smallest: {oset.kth_smallest(1)}")  # 3
    print(f"Rank of 5: {oset.rank(5)}")  # 2
    print()
    
    print("=== Merge Similar Items ===")
    items1 = [[1, 1], [4, 5], [3, 8]]
    items2 = [[3, 1], [1, 5]]
    print(f"Items1: {items1}")
    print(f"Items2: {items2}")
    print(f"Merged: {merge_similar_items(items1, items2)}")
    print()
    
    print("=== 132 Pattern ===")
    nums = [3, 1, 4, 2]
    print(f"Array: {nums}")
    print(f"Has 132 pattern: {find_132_pattern(nums)}")
    print()
    
    print("=== My Calendar ===")
    cal = MyCalendar()
    print(f"Book [10, 20): {cal.book(10, 20)}")
    print(f"Book [15, 25): {cal.book(15, 25)}")
    print(f"Book [20, 30): {cal.book(20, 30)}")
    print()
    
    print("=== Longest Continuous Subarray ===")
    nums = [8, 2, 4, 7]
    limit = 4
    print(f"Array: {nums}, Limit: {limit}")
    print(f"Length: {longest_continuous_subarray(nums, limit)}")
    print()
    
    print("=== Count Smaller After Self ===")
    nums = [5, 2, 6, 1]
    print(f"Array: {nums}")
    print(f"Counts: {count_smaller_after_self(nums)}")
```

### Code Explanation

**OrderedSetExample:**
- Wraps SortedList to provide ordered set operations
- `floor`/`ceiling`: Use bisect methods to find bounds
- `range_query`: Efficient slicing after finding boundaries
- All operations leverage binary search for O(log n) time

**Merge Similar Items:**
- SortedDict maintains keys in sorted order
- Automatically handles merging and sorting
- Simple iteration over sorted keys for output

**132 Pattern:**
- Maintains sorted set of candidates
- Uses bisect to find potential pattern matches
- Tracks minimum value seen so far

**MyCalendar:**
- Stores events as sorted tuples
- Uses bisect to find insertion position
- Checks adjacent events for conflicts

**Longest Continuous Subarray:**
- Sliding window with ordered set
- Efficiently tracks min/max in window
- Shrinks window when constraint violated

**Count Smaller After Self:**
- Processes array right to left
- Uses bisect_left to count smaller elements
- Maintains sorted list of seen elements

## Complexity Analysis

### Time Complexity

**Basic Operations (using self-balancing tree):**
- **Insert:** O(log n) - traverse tree height
- **Delete:** O(log n) - find and rebalance
- **Search:** O(log n) - binary search
- **Floor/Ceiling:** O(log n) - binary search
- **Range Query:** O(log n + k) where k is result size
- **Kth Element:** O(log n) for tree, O(1) for array-based

**Why O(log n)?** Self-balancing trees (Red-Black, AVL) maintain height proportional to log n.

### Space Complexity

**Storage:**
- **Space:** O(n) for n elements
- **Tree overhead:** Additional pointers (2-3× element size)

**SortedList (Python):**
- Uses chunked array approach
- O(n) space with better cache locality than trees

### Comparison with Alternatives

| Data Structure | Insert | Delete | Search | Min/Max | Range Query | Order |
|----------------|--------|--------|--------|---------|-------------|-------|
| **Ordered Set** | O(log n) | O(log n) | O(log n) | O(1) | O(log n + k) | ✓ Sorted |
| **Hash Set** | O(1) | O(1) | O(1) | O(n) | O(n) | ✗ Unordered |
| **Sorted Array** | O(n) | O(n) | O(log n) | O(1) | O(log n + k) | ✓ Sorted |
| **Heap** | O(log n) | O(log n) | O(n) | O(1) | N/A | ✓ Partial |
| **BST (unbalanced)** | O(n) | O(n) | O(n) | O(n) | O(n) | ✓ Sorted |

**When to use Ordered Set:**
- Need both fast operations AND sorted order
- Frequent range queries
- Finding predecessors/successors
- Order statistics (kth element, rank)
- Can't afford O(n) insertion of sorted array

## Examples

### Example 1: Floor and Ceiling

**Operations:**
```
set = OrderedSet([3, 7, 12, 15, 20])
floor(10)?
ceiling(10)?
floor(7)?
ceiling(15)?
```

**Trace:**

```
Set: [3, 7, 12, 15, 20]

floor(10):
  Binary search for largest element ≤ 10
  Check: 3 ≤ 10 ✓, 7 ≤ 10 ✓, 12 ≤ 10 ✗
  Answer: 7 ✓

ceiling(10):
  Binary search for smallest element ≥ 10
  Check: 3 < 10, 7 < 10, 12 ≥ 10 ✓
  Answer: 12 ✓

floor(7):
  Exact match exists
  Answer: 7 ✓

ceiling(15):
  Exact match exists
  Answer: 15 ✓
```

### Example 2: Calendar Bookings

**Bookings:**
```
book(10, 20)
book(15, 25)
book(20, 30)
book(5, 10)
```

**Trace:**

```
Initial: events = []

book(10, 20):
  Check previous: none
  Check next: none
  Book successful
  events = [(10, 20)]
  Return True ✓

book(15, 25):
  Find position: between (10, 20) and end
  Check previous: (10, 20)
    prev_end = 20 > 15 = start ✗ Conflict!
  Return False ✗

book(20, 30):
  Find position: after (10, 20)
  Check previous: (10, 20)
    prev_end = 20 ≤ 20 = start ✓ No conflict
  Check next: none
  Book successful
  events = [(10, 20), (20, 30)]
  Return True ✓

book(5, 10):
  Find position: before (10, 20)
  Check previous: none
  Check next: (10, 20)
    end = 10 ≤ 10 = next_start ✓ No conflict
  Book successful
  events = [(5, 10), (10, 20), (20, 30)]
  Return True ✓

Final calendar:
5      10      20      30
|------|-------|-------|
 Event1 Event2  Event3
```

### Example 3: Longest Continuous Subarray

**Input:** nums = [10,1,2,4,7,2], limit = 5
**Output:** 4

**Trace:**

```
Sliding window with sorted set to track min/max

left=0, right=0: window=[10]
  sorted=[10], max-min=0 ≤ 5 ✓
  length=1

left=0, right=1: window=[10,1]
  sorted=[1,10], max-min=9 > 5 ✗
  Shrink: remove 10, left=1
  sorted=[1], max-min=0 ≤ 5 ✓
  length=1

left=1, right=2: window=[1,2]
  sorted=[1,2], max-min=1 ≤ 5 ✓
  length=2

left=1, right=3: window=[1,2,4]
  sorted=[1,2,4], max-min=3 ≤ 5 ✓
  length=3

left=1, right=4: window=[1,2,4,7]
  sorted=[1,2,4,7], max-min=6 > 5 ✗
  Shrink: remove 1, left=2
  sorted=[2,4,7], max-min=5 ≤ 5 ✓
  length=3

left=2, right=5: window=[2,4,7,2]
  sorted=[2,2,4,7], max-min=5 ≤ 5 ✓
  length=4 ✓

Maximum length: 4
Subarray: [2,4,7,2]
```

### Example 4: Count Smaller After Self

**Input:** [5,2,6,1]
**Output:** [2,1,1,0]

**Trace:**

```
Process right to left, maintain sorted set

i=3, num=1:
  sorted_seen = []
  count < 1: bisect_left(1) = 0
  result = [0]
  sorted_seen = [1]

i=2, num=6:
  sorted_seen = [1]
  count < 6: bisect_left(6) = 1
  result = [0, 1]
  sorted_seen = [1, 6]

i=1, num=2:
  sorted_seen = [1, 6]
  count < 2: bisect_left(2) = 1
  result = [0, 1, 1]
  sorted_seen = [1, 2, 6]

i=0, num=5:
  sorted_seen = [1, 2, 6]
  count < 5: bisect_left(5) = 2
  result = [0, 1, 1, 2]
  sorted_seen = [1, 2, 5, 6]

Reverse result: [2, 1, 1, 0] ✓

Explanation:
  5: has [2, 1] smaller on right (count=2)
  2: has [1] smaller on right (count=1)
  6: has [1] smaller on right (count=1)
  1: has [] smaller on right (count=0)
```

## Edge Cases

### 1. Empty Set
**Scenario:** Operations on empty ordered set
**Challenge:** No elements to query
**Solution:** Return None or appropriate default
**Code example:**
```python
oset = OrderedSetExample()
assert oset.floor(5) is None
assert oset.ceiling(5) is None
```

### 2. Single Element
**Scenario:** Set with one element
**Challenge:** Element is both min and max
**Solution:** Handle boundary cases
**Code example:**
```python
oset = OrderedSetExample()
oset.add(5)
assert oset.floor(10) == 5
assert oset.ceiling(3) == 5
```

### 3. All Elements Equal
**Scenario:** Multiset with repeated values
**Challenge:** Need to track duplicates
**Solution:** Use multiset variant
**Code example:**
```python
mset = OrderedMultiset()
mset.add(5)
mset.add(5)
mset.add(5)
assert mset.count(5) == 3
```

### 4. Query Outside Range
**Scenario:** floor/ceiling of values outside set range
**Challenge:** No valid answer
**Solution:** Return None
**Code example:**
```python
oset = OrderedSetExample()
oset.add(5)
oset.add(10)
assert oset.floor(3) is None  # All elements > 3
assert oset.ceiling(15) is None  # All elements < 15
```

### 5. Empty Range Query
**Scenario:** Range with no elements
**Challenge:** No matches
**Solution:** Return empty list
**Code example:**
```python
oset = OrderedSetExample()
oset.add(1)
oset.add(10)
assert oset.range_query(3, 7) == []
```

### 6. Duplicate Insertions
**Scenario:** Adding same element multiple times
**Challenge:** Sets don't allow duplicates
**Solution:** Return False or use multiset
**Code example:**
```python
oset = OrderedSetExample()
assert oset.add(5) == True
assert oset.add(5) == False  # Already exists
```

### 7. Large Numbers
**Scenario:** Very large or very small numbers
**Challenge:** Numerical overflow (in some languages)
**Solution:** Python handles arbitrary precision
**Code example:**
```python
oset = OrderedSetExample()
oset.add(10**100)
oset.add(-10**100)
# Works fine in Python
```

## Common Pitfalls

### ❌ Pitfall 1: Using Regular Set When Order Needed
**What happens:** Can't find floor/ceiling efficiently
**Why it's wrong:** Hash sets are unordered
**Correct approach:**
```python
# WRONG:
regular_set = set([3, 7, 12])
# Can't efficiently find floor(10)!

# CORRECT:
from sortedcontainers import SortedList
sorted_set = SortedList([3, 7, 12])
floor_10 = sorted_set[sorted_set.bisect_right(10) - 1]
```

### ❌ Pitfall 2: Confusing Floor/Ceiling with Lower/Higher
**What happens:** Wrong element returned
**Why it's wrong:** Different definitions
**Correct approach:**
```python
# floor(x): largest element <= x (can equal x)
# ceiling(x): smallest element >= x (can equal x)
# lower(x): largest element < x (must be less)
# higher(x): smallest element > x (must be greater)

# If set = [3, 5, 7]:
# floor(5) = 5, ceiling(5) = 5
# lower(5) = 3, higher(5) = 7
```

### ❌ Pitfall 3: Not Checking for None Returns
**What happens:** NoneType error on operations
**Why it's wrong:** Floor/ceiling can return None
**Correct approach:**
```python
# WRONG:
floor_val = oset.floor(x)
result = floor_val + 1  # Error if floor_val is None!

# CORRECT:
floor_val = oset.floor(x)
if floor_val is not None:
    result = floor_val + 1
```

### ❌ Pitfall 4: Forgetting Set Doesn't Allow Duplicates
**What happens:** Duplicate values disappear
**Why it's wrong:** Sets have unique elements
**Correct approach:**
```python
# WRONG for counting frequencies:
from sortedcontainers import SortedSet
counts = SortedSet([1, 2, 2, 3])  # Becomes {1, 2, 3}!

# CORRECT:
from sortedcontainers import SortedList
counts = SortedList([1, 2, 2, 3])  # Keeps [1, 2, 2, 3]
# Or use Counter + sort when needed
```

### ❌ Pitfall 5: Using Sorted Array for Frequent Insertions
**What happens:** O(n) per insertion
**Why it's wrong:** Inefficient for dynamic data
**Correct approach:**
```python
# WRONG for many insertions:
sorted_arr = []
for val in values:
    bisect.insort(sorted_arr, val)  # O(n) per insertion

# CORRECT:
from sortedcontainers import SortedList
sorted_list = SortedList()
for val in values:
    sorted_list.add(val)  # O(log n) per insertion
```

### ❌ Pitfall 6: Not Importing SortedContainers
**What happens:** Python doesn't have built-in ordered set
**Why it's wrong:** Need external library or manual implementation
**Correct approach:**
```python
# WRONG (doesn't exist):
from collections import OrderedSet  # Not a thing!

# CORRECT:
from sortedcontainers import SortedList, SortedSet
# Or: pip install sortedcontainers
```

### ❌ Pitfall 7: Using bisect on Unsorted List
**What happens:** Incorrect results
**Why it's wrong:** Bisect assumes sorted data
**Correct approach:**
```python
# WRONG:
arr = [5, 2, 8, 1]
idx = bisect.bisect_left(arr, 3)  # Wrong! Array not sorted

# CORRECT:
arr = [1, 2, 5, 8]  # Sorted first
idx = bisect.bisect_left(arr, 3)  # Correct
```

## Variations and Extensions

### Variation 1: Multiset (with duplicates)
**Description:** Ordered collection allowing duplicates
**When to use:** Need frequency counts with ordering
**Implementation:** Use SortedList instead of SortedSet

### Variation 2: Interval Tree
**Description:** Store intervals, query for overlaps
**When to use:** Calendar scheduling, range conflicts
**Key differences:** More complex, O(log n + k) queries

### Variation 3: Order Statistics Tree
**Description:** Tree supporting rank and select
**When to use:** Need frequent rank queries
**Implementation:** Augmented red-black tree

### Variation 4: Sliding Window Median
**Description:** Maintain median in sliding window
**When to use:** Stream processing
**Implementation:** Two ordered sets (min/max heaps) or single SortedList

## Practice Problems

### Beginner
1. **Merge Similar Items** - Merge and sort items
   - LeetCode #2363

2. **Contains Duplicate III** - Find near-duplicates
   - LeetCode #220

3. **Intersection of Multiple Arrays** - Find common elements
   - LeetCode #2248

### Intermediate
1. **My Calendar I** - Book events without conflicts
   - LeetCode #729

2. **Longest Continuous Subarray** - Max/min difference constraint
   - LeetCode #1438

3. **132 Pattern** - Find specific ordering pattern
   - LeetCode #456

4. **Count of Smaller Numbers After Self** - Inversion counting
   - LeetCode #315

5. **Median of Data Stream** - Running median
   - LeetCode #295

### Advanced
1. **My Calendar III** - Count maximum overlaps
   - LeetCode #732

2. **Falling Squares** - Height tracking
   - LeetCode #699

3. **Count of Range Sum** - Range sum queries
   - LeetCode #327

4. **Reverse Pairs** - Count specific inversions
   - LeetCode #493

## Real-World Applications

### Industry Use Cases

1. **Event Scheduling:** Calendar applications use ordered sets to detect conflicts
2. **Time Series Analysis:** Maintaining sorted timestamps for efficient queries
3. **Range Queries:** Database indexing for range searches
4. **Leaderboards:** Gaming systems with rank queries
5. **Stock Trading:** Order books with price-time priority

### Popular Implementations

- **C++ std::set** - Red-Black tree
- **Java TreeSet** - Red-Black tree  
- **Python SortedContainers** - Sorted list with chunking
- **Redis Sorted Sets** - Skip list based

### Practical Scenarios

- **Meeting room scheduling**
- **Finding closest timestamps**
- **Percentile calculations**
- **Moving window statistics**
- **IP address range matching**

## Related Topics

### Prerequisites to Review
- **Binary Search** - Foundation for operations
- **Binary Search Trees** - Underlying structure
- **Self-Balancing Trees** - Implementation details

### Next Steps
- **Interval Trees** - Advanced interval queries
- **Segment Trees** - Range update/query
- **Fenwick Trees** - Prefix sum queries

### Similar Concepts
- **Priority Queues** - Partial ordering
- **Skip Lists** - Alternative implementation
- **B-Trees** - Database indexing

### Further Reading
- "Introduction to Algorithms" (CLRS) - Red-Black Trees
- SortedContainers documentation
- C++ STL documentation for std::set
- [Sorted Containers](http://www.grantjenks.com/docs/sortedcontainers/)
