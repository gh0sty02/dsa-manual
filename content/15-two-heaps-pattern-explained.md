# Two Heaps Pattern

**Difficulty:** Intermediate to Advanced
**Prerequisites:** Heaps (Priority Queues), Arrays, Basic statistics (median concept)
**Estimated Reading Time:** 50 minutes

## Introduction

The Two Heaps Pattern is a technique that uses two heaps (a max-heap and a min-heap) working together to efficiently solve problems involving finding medians, maintaining balanced partitions, or tracking both smallest and largest elements in a dynamic dataset. This pattern elegantly divides data into two parts to enable O(log n) insertions and O(1) median access.

**Why it matters:** Finding the median in a constantly changing dataset is crucial in many applications: real-time analytics, streaming data processing, financial trading systems, and monitoring systems. Without the two heaps pattern, you'd need to sort the entire dataset repeatedly (O(n log n) each time), but with two heaps, you can maintain the median with just O(log n) insertion time and O(1) retrieval time.

**Real-world analogy:** Imagine a bookstore that wants to always know the median price of books in stock. They organize books on two shelves: the left shelf holds the cheaper half (organized with most expensive at the top), and the right shelf holds the more expensive half (organized with cheapest at the top). The median is always at the top of one of these shelves! When a new book arrives, they just need to place it on the correct shelf and maybe swap one book between shelves to keep them balanced. This is exactly how two heaps work!

## Core Concepts

### Key Principles

1. **Dual Heap Structure:** Use two heaps to partition data:
   - **Max-Heap (left side):** Stores the smaller half of numbers, with largest at top
   - **Min-Heap (right side):** Stores the larger half of numbers, with smallest at top

2. **Balance Invariant:** Keep heaps balanced in size:
   - Sizes differ by at most 1: `|max_heap.size - min_heap.size| ≤ 1`
   - This ensures median is always accessible at heap tops

3. **Ordering Invariant:** Maintain proper ordering:
   - All elements in max-heap ≤ all elements in min-heap
   - `max_heap.top() ≤ min_heap.top()`

4. **Median Location:**
   - If heaps are equal size: median = average of both tops
   - If one heap is larger: median = top of larger heap

### Essential Terms

- **Heap:** A tree-based data structure that maintains parent-child ordering
- **Max-Heap:** Parent nodes are larger than children (largest element at root)
- **Min-Heap:** Parent nodes are smaller than children (smallest element at root)
- **Median:** Middle value in a sorted dataset (50th percentile)
- **Heapify:** Process of converting array to heap structure (O(n))
- **Rebalancing:** Moving elements between heaps to maintain balance
- **Priority Queue:** Abstract data type typically implemented as a heap

### Visual Overview

```
Two Heaps Maintaining Median:

Initial state (empty):
    Max-Heap          Min-Heap
   (smaller half)   (larger half)
       [ ]              [ ]

After inserting 5:
    Max-Heap          Min-Heap
       [5]              [ ]
    Median = 5

After inserting 15:
    Max-Heap          Min-Heap
       [5]              [15]
    Median = (5+15)/2 = 10

After inserting 1:
    Max-Heap          Min-Heap
       [5]              [15]
       [1]
    
    Rebalance needed! Max-heap has 2, min-heap has 1
    
    After rebalance:
    Max-Heap          Min-Heap
       [1]              [5]
                        [15]
    Median = 5 (top of larger heap)

After inserting 3:
    Max-Heap          Min-Heap
       [3]              [5]
       [1]              [15]
    Median = (3+5)/2 = 4

Visual representation of heap structure:

Max-Heap [3,1]:        Min-Heap [5,15]:
       3                      5
      /                      /
     1                      15

Property maintained:
- All values in max-heap ≤ all values in min-heap
- max(max-heap) = 3 ≤ 5 = min(min-heap) ✓
```

## How It Works

### Finding Median in Data Stream - Step by Step

**Problem:** Design a data structure that supports adding numbers and finding the median.

**Algorithm:**

1. **Insert number:**
   - If max-heap is empty OR number ≤ max-heap.top(): add to max-heap
   - Otherwise: add to min-heap
   - Rebalance if size difference > 1

2. **Rebalance:**
   - If max-heap.size > min-heap.size + 1: move max-heap.top() to min-heap
   - If min-heap.size > max-heap.size + 1: move min-heap.top() to max-heap

3. **Find median:**
   - If max-heap.size > min-heap.size: return max-heap.top()
   - If min-heap.size > max-heap.size: return min-heap.top()
   - If equal sizes: return (max-heap.top() + min-heap.top()) / 2

**Detailed Walkthrough:**

```
Stream: [5, 15, 1, 3, 8, 7, 9, 10]

Step 1: Add 5
  Max-heap empty, add to max-heap
  Max-Heap: [5]
  Min-Heap: []
  Sizes: 1, 0 → No rebalance needed
  Median: 5

Step 2: Add 15
  15 > 5 (max-heap top), add to min-heap
  Max-Heap: [5]
  Min-Heap: [15]
  Sizes: 1, 1 → Balanced
  Median: (5 + 15) / 2 = 10.0

Step 3: Add 1
  1 ≤ 5 (max-heap top), add to max-heap
  Max-Heap: [5, 1]
  Min-Heap: [15]
  Sizes: 2, 1 → max-heap larger by 1, OK
  Median: 5 (top of larger max-heap)

Step 4: Add 3
  3 ≤ 5 (max-heap top), add to max-heap
  Max-Heap: [5, 3, 1]
  Min-Heap: [15]
  Sizes: 3, 1 → Difference > 1, REBALANCE!
  
  Move 5 from max-heap to min-heap:
  Max-Heap: [3, 1]
  Min-Heap: [5, 15]
  Sizes: 2, 2 → Balanced
  Median: (3 + 5) / 2 = 4.0

Step 5: Add 8
  8 > 3 (max-heap top), add to min-heap
  Max-Heap: [3, 1]
  Min-Heap: [5, 8, 15]
  Sizes: 2, 3 → Difference > 1, REBALANCE!
  
  Move 5 from min-heap to max-heap:
  Max-Heap: [5, 3, 1]
  Min-Heap: [8, 15]
  Sizes: 3, 2 → max-heap larger by 1, OK
  Median: 5

Step 6: Add 7
  7 > 5 (max-heap top), add to min-heap
  Max-Heap: [5, 3, 1]
  Min-Heap: [7, 8, 15]
  Sizes: 3, 3 → Balanced
  Median: (5 + 7) / 2 = 6.0

Step 7: Add 9
  9 > 5, add to min-heap
  Max-Heap: [5, 3, 1]
  Min-Heap: [7, 8, 9, 15]
  Sizes: 3, 4 → Difference > 1, REBALANCE!
  
  Move 7 from min-heap to max-heap:
  Max-Heap: [7, 5, 3, 1]
  Min-Heap: [8, 9, 15]
  Sizes: 4, 3 → max-heap larger by 1, OK
  Median: 7

Step 8: Add 10
  10 > 7, add to min-heap
  Max-Heap: [7, 5, 3, 1]
  Min-Heap: [8, 9, 10, 15]
  Sizes: 4, 4 → Balanced
  Median: (7 + 8) / 2 = 7.5

Final state:
  Max-Heap (smaller half): [7, 5, 3, 1]
  Min-Heap (larger half): [8, 9, 10, 15]
  Sorted order: [1, 3, 5, 7 | 8, 9, 10, 15]
  Median: (7 + 8) / 2 = 7.5 ✓
```

## Implementation

### Median Finder Implementation

```python
import heapq
from typing import Optional

class MedianFinder:
    """
    Find median from a data stream.
    
    Uses two heaps:
    - max_heap: stores smaller half (implemented as negated min-heap)
    - min_heap: stores larger half
    
    Attributes:
        max_heap: Smaller half of numbers (largest at top)
        min_heap: Larger half of numbers (smallest at top)
    """
    
    def __init__(self):
        """
        Initialize data structure.
        
        Note: Python's heapq is a min-heap, so we negate values
        to simulate a max-heap for the smaller half.
        """
        self.max_heap = []  # Smaller half (negated for max-heap behavior)
        self.min_heap = []  # Larger half
    
    def addNum(self, num: int) -> None:
        """
        Add a number to the data structure.
        
        Args:
            num: Number to add
            
        Time Complexity: O(log n) - heap insertion and potential rebalance
        Space Complexity: O(1) - only adds one element
        """
        # Step 1: Add to appropriate heap
        if not self.max_heap or num <= -self.max_heap[0]:
            # Add to max-heap (smaller half)
            # Negate value because heapq is min-heap
            heapq.heappush(self.max_heap, -num)
        else:
            # Add to min-heap (larger half)
            heapq.heappush(self.min_heap, num)
        
        # Step 2: Rebalance heaps if size difference > 1
        if len(self.max_heap) > len(self.min_heap) + 1:
            # Max-heap too large, move top element to min-heap
            value = -heapq.heappop(self.max_heap)  # Negate back to positive
            heapq.heappush(self.min_heap, value)
        elif len(self.min_heap) > len(self.max_heap) + 1:
            # Min-heap too large, move top element to max-heap
            value = heapq.heappop(self.min_heap)
            heapq.heappush(self.max_heap, -value)  # Negate for max-heap
    
    def findMedian(self) -> float:
        """
        Find the median of all numbers added so far.
        
        Returns:
            The median value
            
        Time Complexity: O(1) - just accessing heap tops
        Space Complexity: O(1)
        """
        # If max-heap has more elements, median is its top
        if len(self.max_heap) > len(self.min_heap):
            return float(-self.max_heap[0])  # Negate back to positive
        
        # If min-heap has more elements, median is its top
        if len(self.min_heap) > len(self.max_heap):
            return float(self.min_heap[0])
        
        # If equal sizes, median is average of both tops
        return (-self.max_heap[0] + self.min_heap[0]) / 2.0


# Usage example
mf = MedianFinder()
mf.addNum(1)
print(mf.findMedian())  # Output: 1.0

mf.addNum(2)
print(mf.findMedian())  # Output: 1.5

mf.addNum(3)
print(mf.findMedian())  # Output: 2.0
```

### Sliding Window Median Implementation

```python
import heapq
from typing import List
from collections import defaultdict

class SlidingWindowMedian:
    """
    Find median in a sliding window using two heaps with lazy deletion.
    
    Challenge: Elements leave the window, but heap doesn't support
    arbitrary deletion. Solution: Track removed elements and ignore
    them when they appear at heap tops.
    """
    
    def __init__(self):
        """Initialize heaps and removal tracking."""
        self.max_heap = []  # Smaller half
        self.min_heap = []  # Larger half
        self.removed = defaultdict(int)  # Count of removed elements
        self.max_heap_size = 0  # Actual size (excluding removed)
        self.min_heap_size = 0
    
    def _clean_heap_top(self, heap: List[int], is_max_heap: bool) -> None:
        """
        Remove invalidated elements from heap top.
        
        Args:
            heap: The heap to clean
            is_max_heap: True if this is the max-heap (values are negated)
        """
        while heap:
            value = heap[0]
            actual_value = -value if is_max_heap else value
            
            if self.removed[actual_value] > 0:
                # This element was removed, pop it
                heapq.heappop(heap)
                self.removed[actual_value] -= 1
                if self.removed[actual_value] == 0:
                    del self.removed[actual_value]
            else:
                break  # Top is valid
    
    def add(self, num: int) -> None:
        """Add number to heaps."""
        if not self.max_heap or num <= -self.max_heap[0]:
            heapq.heappush(self.max_heap, -num)
            self.max_heap_size += 1
        else:
            heapq.heappush(self.min_heap, num)
            self.min_heap_size += 1
        
        self._rebalance()
    
    def remove(self, num: int) -> None:
        """
        Mark number as removed (lazy deletion).
        
        Args:
            num: Number to remove
        """
        self.removed[num] += 1
        
        # Update size counters
        if num <= -self.max_heap[0]:
            self.max_heap_size -= 1
        else:
            self.min_heap_size -= 1
        
        self._rebalance()
    
    def _rebalance(self) -> None:
        """Rebalance heaps to maintain size invariant."""
        # Clean tops before checking sizes
        self._clean_heap_top(self.max_heap, True)
        self._clean_heap_top(self.min_heap, False)
        
        # Rebalance if needed
        if self.max_heap_size > self.min_heap_size + 1:
            value = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, value)
            self.max_heap_size -= 1
            self.min_heap_size += 1
            self._clean_heap_top(self.max_heap, True)
        elif self.min_heap_size > self.max_heap_size + 1:
            value = heapq.heappop(self.min_heap)
            heapq.heappush(self.max_heap, -value)
            self.min_heap_size -= 1
            self.max_heap_size += 1
            self._clean_heap_top(self.min_heap, False)
    
    def find_median(self) -> float:
        """Get current median."""
        self._clean_heap_top(self.max_heap, True)
        self._clean_heap_top(self.min_heap, False)
        
        if self.max_heap_size > self.min_heap_size:
            return float(-self.max_heap[0])
        if self.min_heap_size > self.max_heap_size:
            return float(self.min_heap[0])
        return (-self.max_heap[0] + self.min_heap[0]) / 2.0


def medianSlidingWindow(nums: List[int], k: int) -> List[float]:
    """
    Find median of each sliding window of size k.
    
    Args:
        nums: Array of numbers
        k: Window size
        
    Returns:
        List of medians for each window position
        
    Time Complexity: O(n * log k) where n is length of nums
        - Each add/remove is O(log k)
        - n-k+1 windows, each needs add and remove
    Space Complexity: O(k) for heaps
    """
    if not nums or k == 0:
        return []
    
    finder = SlidingWindowMedian()
    result = []
    
    # Initialize first window
    for i in range(k):
        finder.add(nums[i])
    result.append(finder.find_median())
    
    # Slide window
    for i in range(k, len(nums)):
        finder.remove(nums[i - k])  # Remove leftmost element
        finder.add(nums[i])          # Add new element
        result.append(finder.find_median())
    
    return result


# Example usage
nums = [1, 3, -1, -3, 5, 3, 6, 7]
k = 3
print(medianSlidingWindow(nums, k))  
# Output: [1.0, -1.0, -1.0, 3.0, 5.0, 6.0]
```

### Maximize Capital (IPO Problem)

```python
def findMaximizedCapital(k: int, w: int, profits: List[int], 
                         capital: List[int]) -> int:
    """
    Select up to k projects to maximize capital.
    
    Strategy:
    - Use min-heap for projects we can't afford yet (by capital required)
    - Use max-heap for projects we can afford (by profit)
    - At each step, pick the most profitable affordable project
    
    Args:
        k: Maximum number of projects
        w: Initial capital
        profits: Profit from each project
        capital: Capital required for each project
        
    Returns:
        Maximum capital achievable
        
    Time Complexity: O(n log n) where n is number of projects
        - Initial sorting/heap creation: O(n log n)
        - k iterations of heap operations: O(k log n)
        - Total: O(n log n + k log n) = O(n log n)
    Space Complexity: O(n) for heaps
    """
    # Create min-heap of (capital_required, profit) for all projects
    # This keeps projects sorted by capital requirement
    min_capital_heap = [(capital[i], profits[i]) 
                        for i in range(len(profits))]
    heapq.heapify(min_capital_heap)
    
    # Max-heap for affordable projects (by profit)
    max_profit_heap = []
    
    # Current capital
    current_capital = w
    
    # Select up to k projects
    for _ in range(k):
        # Move all affordable projects to max-profit heap
        while min_capital_heap and min_capital_heap[0][0] <= current_capital:
            cap, prof = heapq.heappop(min_capital_heap)
            heapq.heappush(max_profit_heap, -prof)  # Negate for max-heap
        
        # If no affordable projects, we're done
        if not max_profit_heap:
            break
        
        # Take the most profitable affordable project
        profit = -heapq.heappop(max_profit_heap)  # Negate back
        current_capital += profit
    
    return current_capital


# Example usage
k = 2
w = 0
profits = [1, 2, 3]
capital = [0, 1, 1]
print(findMaximizedCapital(k, w, profits, capital))  # Output: 4
# Explanation: 
# Start with w=0
# Do project 0 (capital=0, profit=1): w becomes 1
# Do project 2 (capital=1, profit=3): w becomes 4
```

### Code Explanation

**Key Design Decisions:**

1. **Negating for Max-Heap:** Python's `heapq` only provides min-heap. To simulate max-heap, we negate values when inserting and negate back when extracting.

2. **Lazy Deletion (Sliding Window):** Instead of removing elements from middle of heap (O(n)), we mark them as removed and clean them when they reach the top (amortized O(log n)).

3. **Size Tracking:** Maintain separate size counters to account for logically removed elements that are still physically in the heap.

4. **Rebalancing Strategy:** Always maintain the invariant that heap sizes differ by at most 1. This guarantees O(1) median access.

5. **Capital vs Profit Heaps:** In IPO problem, we need two different orderings (by capital for affordability, by profit for selection), so we use two heaps with different priorities.

## Complexity Analysis

### Time Complexity

**MedianFinder Operations:**
- **addNum:** O(log n)
  - Heap insertion: O(log n)
  - Potential rebalance (move one element): O(log n)
  - Total: O(log n)

- **findMedian:** O(1)
  - Just access heap tops (constant time)

**For n operations:**
- Total: O(n log n)

**Sliding Window Median:**
- **Per window:** O(log k) for add/remove operations
- **Total:** O(n log k) where n = array length, k = window size

**Maximize Capital:**
- **Initial heap creation:** O(n log n)
- **k iterations:** Each iteration processes projects once: O(n log n)
- **Total:** O(n log n)

### Space Complexity

**MedianFinder:** O(n)
- Store all n elements across two heaps

**Sliding Window Median:** O(k)
- Heaps contain at most k elements from current window
- Removed map contains at most k elements

**Maximize Capital:** O(n)
- Two heaps store all n projects

### Comparison with Alternatives

| Approach | Add Time | Find Median Time | Space | Notes |
|----------|----------|------------------|-------|-------|
| Two Heaps | O(log n) | O(1) | O(n) | Optimal for streaming data |
| Sorted Array | O(n) | O(1) | O(n) | Too slow for insertions |
| BST (balanced) | O(log n) | O(log n) | O(n) | Complex to implement |
| Unsorted Array | O(1) | O(n log n) | O(n) | Fast add, slow median |
| Quick Select | O(1) | O(n) | O(n) | Fast on average, no sorting |

**Why Two Heaps is Best:**
- O(log n) insertion is much faster than O(n) for sorted array
- O(1) median access is instant vs O(n) for unsorted array
- Simpler than balanced BST
- Perfect for streaming/online algorithms where medians are frequently queried

## Examples

### Example 1: Basic Median Finding

**Problem:** Find median after each insertion.

**Input:** Stream = [5, 15, 1, 3]

**Step-by-step:**

```
Insert 5:
  Max-heap: [5], Min-heap: []
  Median: 5

Insert 15:
  Max-heap: [5], Min-heap: [15]
  Median: (5+15)/2 = 10.0

Insert 1:
  1 ≤ 5, add to max-heap
  Max-heap: [5,1], Min-heap: [15]
  Rebalance: move 5 to min-heap
  Max-heap: [1], Min-heap: [5,15]
  Median: 5

Insert 3:
  3 > 1, add to min-heap
  Max-heap: [1], Min-heap: [3,5,15]
  Rebalance: move 3 to max-heap
  Max-heap: [3,1], Min-heap: [5,15]
  Median: (3+5)/2 = 4.0

Results: [5, 10.0, 5, 4.0]
```

### Example 2: Sliding Window Median

**Problem:** Find median of each window of size 3.

**Input:** nums = [1, 3, -1, -3, 5, 3, 6, 7], k = 3

**Step-by-step:**

```
Window [1, 3, -1]:
  Sorted: [-1, 1, 3]
  Median: 1

Window [3, -1, -3]:
  Sorted: [-3, -1, 3]
  Median: -1

Window [-1, -3, 5]:
  Sorted: [-3, -1, 5]
  Median: -1

Window [-3, 5, 3]:
  Sorted: [-3, 3, 5]
  Median: 3

Window [5, 3, 6]:
  Sorted: [3, 5, 6]
  Median: 5

Window [3, 6, 7]:
  Sorted: [3, 6, 7]
  Median: 6

Result: [1, -1, -1, 3, 5, 6]
```

### Example 3: IPO (Maximize Capital)

**Problem:** Choose up to k=2 projects to maximize capital.

**Input:**
```
k = 2
w = 0 (initial capital)
profits = [1, 2, 3]
capital = [0, 1, 1]
```

**Solution:**

```
Initial state:
  Capital: 0
  Available projects by capital:
    Project 0: capital=0, profit=1
    Project 1: capital=1, profit=2
    Project 2: capital=1, profit=3

Step 1: Find affordable projects (capital ≤ 0)
  Affordable: [Project 0]
  Choose max profit: Project 0 (profit=1)
  Capital: 0 + 1 = 1

Step 2: Find affordable projects (capital ≤ 1)
  Affordable: [Project 1, Project 2]
  Choose max profit: Project 2 (profit=3)
  Capital: 1 + 3 = 4

Final capital: 4
```

### Example 4: Finding Median with Negative Numbers

**Problem:** Handle negative numbers correctly.

**Input:** Stream = [-1, -2, -3, -4, -5]

**Trace:**

```
Insert -1:
  Max-heap: [-1], Min-heap: []
  Median: -1

Insert -2:
  -2 ≤ -1, add to max-heap
  Max-heap: [-1, -2], Min-heap: []
  Rebalance: move -1 to min-heap
  Max-heap: [-2], Min-heap: [-1]
  Median: (-2 + -1)/2 = -1.5

Insert -3:
  -3 ≤ -2, add to max-heap
  Max-heap: [-2, -3], Min-heap: [-1]
  Median: -2

Insert -4:
  -4 ≤ -2, add to max-heap
  Max-heap: [-2, -3, -4], Min-heap: [-1]
  Rebalance: move -2 to min-heap
  Max-heap: [-3, -4], Min-heap: [-2, -1]
  Median: (-3 + -2)/2 = -2.5

Insert -5:
  -5 ≤ -3, add to max-heap
  Max-heap: [-3, -4, -5], Min-heap: [-2, -1]
  Median: -3

Results: [-1, -1.5, -2, -2.5, -3]
```

## Edge Cases

### 1. Empty Data Stream

**Scenario:** Finding median with no elements.

**Challenge:** Cannot compute median of empty set.

**Solution:**

```python
def findMedian(self) -> Optional[float]:
    """Return None if no elements."""
    if not self.max_heap and not self.min_heap:
        return None
    
    # Regular median logic...
```

### 2. Single Element

**Scenario:** Only one number added.

**Challenge:** Median is just that number.

**Solution:**
```python
# After adding 5:
# Max-heap: [5], Min-heap: []
# Median: 5 ✓

# Implementation handles this naturally
if len(self.max_heap) > len(self.min_heap):
    return float(-self.max_heap[0])  # Returns 5
```

### 3. All Same Numbers

**Scenario:** Stream of identical numbers.

**Challenge:** Median should always be that number.

**Solution:**
```python
# Stream: [5, 5, 5, 5, 5]
# Max-heap: [5, 5, 5], Min-heap: [5, 5]
# Median: (5 + 5) / 2 = 5.0 ✓

# Works correctly because all elements are equal
```

### 4. Strictly Increasing or Decreasing Sequence

**Scenario:** Numbers always increase/decrease.

**Challenge:** One heap might get all elements.

**Solution:**

```python
# Increasing: [1, 2, 3, 4, 5]
# All go to min-heap initially, but rebalancing maintains invariant

# After insertions and rebalancing:
# Max-heap: [1, 2], Min-heap: [3, 4, 5]
# Properly balanced ✓

# The rebalancing logic prevents one heap from growing unbounded
```

### 5. Duplicate Numbers in Sliding Window

**Scenario:** Window contains duplicate values.

**Challenge:** Removing duplicates must handle multiplicity.

**Solution:**

```python
# Use counter for removed elements
self.removed = defaultdict(int)

def remove(self, num):
    self.removed[num] += 1  # Handles duplicates
    # When cleaning: self.removed[num] -= 1
```

### 6. Window Size Larger Than Array

**Scenario:** k > len(nums) in sliding window.

**Challenge:** Invalid input.

**Solution:**

```python
def medianSlidingWindow(nums: List[int], k: int) -> List[float]:
    """Handle edge cases."""
    if not nums or k <= 0 or k > len(nums):
        return []
    
    # Regular logic...
```

### 7. Negative and Positive Numbers Mixed

**Scenario:** Stream contains both negative and positive numbers.

**Challenge:** Heap comparisons must work correctly.

**Solution:**

```python
# Python's heapq handles negative numbers correctly
# Max-heap simulation with negation works for any integers

# Stream: [-5, 10, -3, 7]
# Max-heap (negated): [5, 3], actual values: [-5, -3]
# Min-heap: [7, 10]
# Median: (-3 + 7) / 2 = 2.0 ✓
```

## Common Pitfalls

### ❌ Pitfall 1: Forgetting to Negate for Max-Heap

**What happens:** Max-heap behaves like min-heap, wrong median.

```python
# WRONG - Not negating for max-heap
def addNum(self, num):
    heapq.heappush(self.max_heap, num)  # This is still a min-heap!
```

**Why it's wrong:** Python's heapq is always a min-heap. Without negation, both heaps are min-heaps.

**Correct approach:**

```python
# CORRECT - Negate for max-heap behavior
def addNum(self, num):
    if not self.max_heap or num <= -self.max_heap[0]:
        heapq.heappush(self.max_heap, -num)  # Negate!
    
    # When retrieving:
    value = -self.max_heap[0]  # Negate back
```

### ❌ Pitfall 2: Not Rebalancing After Insert

**What happens:** Heaps become unbalanced, median is wrong.

```python
# WRONG - No rebalancing
def addNum(self, num):
    heapq.heappush(self.max_heap, -num)
    # Missing rebalance logic!
```

**Correct approach:**

```python
# CORRECT - Always rebalance after insert
def addNum(self, num):
    # Add to appropriate heap
    if not self.max_heap or num <= -self.max_heap[0]:
        heapq.heappush(self.max_heap, -num)
    else:
        heapq.heappush(self.min_heap, num)
    
    # Rebalance if size difference > 1
    if len(self.max_heap) > len(self.min_heap) + 1:
        val = -heapq.heappop(self.max_heap)
        heapq.heappush(self.min_heap, val)
    elif len(self.min_heap) > len(self.max_heap) + 1:
        val = heapq.heappop(self.min_heap)
        heapq.heappush(self.max_heap, -val)
```

### ❌ Pitfall 3: Wrong Median Calculation for Even Count

**What happens:** Returning wrong value when counts are even.

```python
# WRONG - Not handling even count correctly
def findMedian(self):
    if len(self.max_heap) > len(self.min_heap):
        return -self.max_heap[0]
    return self.min_heap[0]  # Wrong! Should average both
```

**Correct approach:**

```python
# CORRECT - Average both tops when equal sizes
def findMedian(self):
    if len(self.max_heap) > len(self.min_heap):
        return float(-self.max_heap[0])
    if len(self.min_heap) > len(self.max_heap):
        return float(self.min_heap[0])
    # Equal sizes - average both tops
    return (-self.max_heap[0] + self.min_heap[0]) / 2.0
```

### ❌ Pitfall 4: Not Cleaning Removed Elements in Sliding Window

**What happens:** Heap contains invalid elements, wrong median.

```python
# WRONG - Not removing invalidated elements from tops
def find_median(self):
    # Removed elements still at tops!
    return (-self.max_heap[0] + self.min_heap[0]) / 2.0
```

**Correct approach:**

```python
# CORRECT - Clean tops before accessing
def find_median(self):
    self._clean_heap_top(self.max_heap, True)
    self._clean_heap_top(self.min_heap, False)
    
    # Now tops are valid
    if self.max_heap_size > self.min_heap_size:
        return float(-self.max_heap[0])
    # ... rest of logic
```

### ❌ Pitfall 5: Using Physical Heap Size Instead of Logical Size

**What happens:** Rebalancing based on wrong sizes.

```python
# WRONG - Using len() which includes removed elements
def _rebalance(self):
    if len(self.max_heap) > len(self.min_heap) + 1:  # Wrong!
        # ...
```

**Correct approach:**

```python
# CORRECT - Track logical sizes
def _rebalance(self):
    if self.max_heap_size > self.min_heap_size + 1:
        # Use logical sizes that exclude removed elements
```

## Variations and Extensions

### Variation 1: K-th Largest Element in Stream

**Description:** Instead of median, find k-th largest element.

**When to use:** Leaderboards, top-k ranking systems.

**Implementation:**

```python
class KthLargest:
    """
    Find k-th largest element in a stream.
    
    Strategy: Maintain min-heap of size k containing k largest elements.
    The root is the k-th largest.
    """
    
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.heap = nums
        heapq.heapify(self.heap)
        
        # Keep only k largest elements
        while len(self.heap) > k:
            heapq.heappop(self.heap)
    
    def add(self, val: int) -> int:
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]  # K-th largest
```

### Variation 2: Multiple Percentiles (Not Just Median)

**Description:** Find any percentile (25th, 75th, 90th, etc.).

**When to use:** Statistical analysis, performance monitoring.

**Key difference:** Adjust heap balance ratio instead of 50-50 split.

**Implementation:**

```python
class PercentileFinder:
    """Find arbitrary percentile using two heaps."""
    
    def __init__(self, percentile: float):
        """
        Args:
            percentile: Value between 0 and 100
        """
        self.percentile = percentile / 100.0
        self.max_heap = []
        self.min_heap = []
    
    def add(self, num: int) -> None:
        # Add to heaps
        if not self.max_heap or num <= -self.max_heap[0]:
            heapq.heappush(self.max_heap, -num)
        else:
            heapq.heappush(self.min_heap, num)
        
        # Rebalance to maintain percentile ratio
        total = len(self.max_heap) + len(self.min_heap)
        target_max_size = int(total * self.percentile)
        
        while len(self.max_heap) > target_max_size:
            val = -heapq.heappop(self.max_heap)
            heapq.heappush(self.min_heap, val)
        
        while len(self.max_heap) < target_max_size:
            val = heapq.heappop(self.min_heap)
            heapq.heappush(self.max_heap, -val)
    
    def find_percentile(self) -> float:
        if len(self.max_heap) > len(self.min_heap):
            return float(-self.max_heap[0])
        return float(self.min_heap[0])
```

### Variation 3: Median of Ranges

**Description:** Find median of elements within certain value range.

**When to use:** Filtering outliers, robust statistics.

**Key difference:** Only consider elements within [low, high] range.

### Variation 4: Weighted Median

**Description:** Elements have weights, find weighted median.

**When to use:** Weighted voting systems, importance-based statistics.

**Key difference:** Track cumulative weights instead of counts.

### Variation 5: Two Heaps for Meeting Rooms

**Description:** Track available rooms using two heaps.

**When to use:** Resource allocation, scheduling problems.

**Implementation:**

```python
def minMeetingRooms(intervals: List[List[int]]) -> int:
    """
    Find minimum meeting rooms needed.
    
    Uses min-heap to track when rooms become free.
    """
    if not intervals:
        return 0
    
    intervals.sort(key=lambda x: x[0])  # Sort by start time
    heap = []  # Track end times of ongoing meetings
    
    for start, end in intervals:
        # If earliest ending meeting has ended, reuse room
        if heap and heap[0] <= start:
            heapq.heappop(heap)
        
        # Assign room (add end time)
        heapq.heappush(heap, end)
    
    return len(heap)  # Number of rooms needed
```

## Practice Problems

### Beginner

1. **Find Median from Data Stream** - Classic two heaps problem
   - LeetCode #295

2. **Kth Largest Element in a Stream** - Variant using single heap
   - LeetCode #703

3. **Last Stone Weight** - Max-heap simulation
   - LeetCode #1046

4. **Meeting Rooms II** - Min-heap for scheduling
   - LeetCode #253 (Premium)

### Intermediate

1. **Sliding Window Median** - Two heaps with lazy deletion
   - LeetCode #480

2. **Find Median from Data Stream** - Two heaps implementation
   - LeetCode #295

3. **IPO (Maximize Capital)** - Two heaps with different priorities
   - LeetCode #502

4. **Top K Frequent Elements** - Heap-based frequency counting
   - LeetCode #347

5. **Task Scheduler** - Max-heap for task prioritization
   - LeetCode #621

### Advanced

1. **Sliding Window Maximum** - Deque-based solution (related concept)
   - LeetCode #239

2. **Find K Pairs with Smallest Sums** - Multiple heaps
   - LeetCode #373

3. **Minimum Cost to Hire K Workers** - Heap-based optimization
   - LeetCode #857

4. **Next Interval** - Two heaps for interval problems
   - LeetCode #436

5. **Median of Two Sorted Arrays** - Binary search variant
   - LeetCode #4

## Real-World Applications

### Industry Use Cases

1. **Real-Time Analytics:** Streaming platforms like Kafka and Spark use heap-based structures to compute running statistics (median, percentiles) on live data streams without storing all historical data.

2. **Financial Trading:** High-frequency trading systems maintain medians and percentiles of stock prices in real-time to detect anomalies and trigger automated trades within milliseconds.

3. **Performance Monitoring:** Application Performance Monitoring (APM) tools track response time percentiles (p50, p95, p99) using two-heap structures to identify performance degradation without storing every request.

4. **Network Traffic Analysis:** Intrusion detection systems monitor packet size medians and percentiles to detect DDoS attacks and abnormal traffic patterns in real-time.

5. **Database Query Optimization:** Database systems maintain statistics on column values using histogram approximations built with heap structures for query plan optimization.

### Popular Implementations

- **Prometheus (Monitoring):** Uses quantile estimation algorithms based on heap structures for computing percentiles
  - Powers monitoring for Kubernetes, Docker, and cloud platforms

- **Apache Flink:** Stream processing framework implements approximate median/percentile computation
  - Used by Uber, Netflix, Alibaba for real-time analytics

- **PostgreSQL:** Maintains statistics on table columns using sampling and heap-based structures
  - Powers query optimization for millions of databases

- **Time-Series Databases (InfluxDB, TimescaleDB):** Compute rolling statistics efficiently
  - Used for IoT sensor data, financial tick data

### Practical Scenarios

- **Video Streaming Quality:** Netflix tracks median buffering times to adjust video quality
- **Ad Bidding Systems:** Real-time bidding platforms compute median bid prices for auction optimization
- **Healthcare Monitoring:** ICU systems track median vital signs (heart rate, blood pressure) to detect patient deterioration
- **Capacity Planning:** Cloud providers monitor resource usage percentiles to predict scaling needs
- **A/B Testing:** Experiment platforms compare median conversion rates between variants

## Related Topics

### Prerequisites to Review

- **Heaps (Priority Queues)** - Understanding heap operations is essential
- **Binary Trees** - Heaps are complete binary trees
- **Arrays** - Heap implementation using arrays
- **Sorting Algorithms** - Understanding why heaps help avoid full sorting
- **Statistics** - Understanding median, percentiles, and distributions

### Next Steps

- **Multiway Merge (K-way Merge)** - Using heaps to merge sorted lists
- **Interval Problems** - Using heaps for scheduling and overlap detection
- **Greedy Algorithms** - Heaps enable many greedy strategies
- **Top K Elements** - Finding k largest/smallest efficiently
- **Segment Trees** - For more complex range queries

### Similar Concepts

- **Balanced Binary Search Trees** - Alternative for order statistics
- **Skip Lists** - Probabilistic alternative to balanced trees
- **Order Statistics Trees** - Augmented BST for finding k-th element
- **Finger Trees** - Functional data structure for sequences with fast access

### Further Reading

- "Introduction to Algorithms" (CLRS) - Chapter 6: Heapsort
- "Algorithm Design Manual" by Skiena - Heap applications and priority queues
- "Competitive Programming 4" - Advanced heap techniques
- [Streaming Algorithms](https://en.wikipedia.org/wiki/Streaming_algorithm) - Computing statistics on data streams
- [Two Heaps Pattern on LeetCode](https://leetcode.com/tag/heap-priority-queue/) - Practice problems
- [Quantile Estimation](https://www.cs.umd.edu/~samir/498/Greenwald.pdf) - Research paper on approximate quantiles
