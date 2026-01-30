# Top 'K' Elements Pattern

**Difficulty:** Intermediate
**Prerequisites:** Arrays, Heaps (Priority Queues), Basic sorting concepts, Hash Maps
**Estimated Reading Time:** 25 minutes

## Introduction

The Top 'K' Elements pattern is a powerful algorithmic technique for efficiently finding the K largest, smallest, or most frequent elements in a dataset. Instead of sorting the entire dataset (which would take O(n log n) time), this pattern uses heaps to maintain only K elements at any time, achieving better time complexity in many scenarios.

**Why it matters:** This pattern appears everywhere in real-world applications - from finding trending topics on social media, to identifying top-selling products, to recommender systems suggesting the most relevant items. Companies like Netflix, Amazon, and Google use variations of this pattern billions of times per day.

**Real-world analogy:** Imagine you're a teacher grading 1,000 exams and want to identify the top 10 students. You don't need to sort all 1,000 exams - you can keep a rolling list of the current top 10. As you review each exam, if it's better than the worst score in your top 10, you replace that worst score. This way, you only maintain 10 scores in memory instead of sorting all 1,000. That's exactly how the Top 'K' Elements pattern works!

## Core Concepts

### Key Principles

1. **Heap-based selection:** Use a min-heap (for K largest) or max-heap (for K smallest) to maintain exactly K elements. The heap's top element is always the "boundary" element - the smallest in our K largest, or largest in our K smallest.

2. **Constant space for K:** By maintaining only K elements in the heap, we use O(K) space regardless of how large the input is, making this pattern memory-efficient.

3. **Early rejection:** Elements that don't qualify for the top K are immediately discarded, avoiding unnecessary comparisons and memory usage.

4. **Frequency-based variations:** For problems involving frequency (most common elements), combine hash maps with heaps to track counts and select top K frequent items.

### Essential Terms

- **Heap (Priority Queue):** A tree-based data structure that maintains elements in a specific order (min or max). Python's `heapq` module implements a min-heap by default.
- **Min-Heap:** A heap where the smallest element is always at the root. Used for finding K largest elements.
- **Max-Heap:** A heap where the largest element is always at the root. Used for finding K smallest elements.
- **Heap Size K:** Maintaining exactly K elements in the heap ensures optimal memory usage.
- **Boundary Element:** The heap's root - it represents the threshold for whether a new element should enter the heap.

### Visual Overview

```
Finding Top 3 Largest from [3, 1, 5, 12, 2, 11, 9]

Step 1: Build min-heap of size 3 with first 3 elements
         1
        / \
       3   5

Step 2: Process 12 (12 > 1, so replace 1)
         3
        / \
       5  12

Step 3: Process 2 (2 < 3, skip)
         3
        / \
       5  12

Step 4: Process 11 (11 > 3, so replace 3)
         5
        / \
       11  12

Step 5: Process 9 (9 > 5, so replace 5)
         9
        / \
       11  12

Result: [9, 11, 12] - the 3 largest elements
```

**Key Insight:** We use a MIN-heap for finding MAX elements because:
- The smallest element in our heap is the "weakest" of our top K
- Any new element must be larger than this weakest element to qualify
- This allows us to quickly identify and replace the weakest element

## How It Works

### Algorithm Steps for Finding K Largest Elements

1. **Initialize a min-heap** with the first K elements from the array
2. **Iterate through remaining elements** (from index K to end)
3. **For each element:**
   - Compare it with the heap's minimum (root element)
   - If current element > heap minimum:
     - Remove the minimum from heap
     - Insert the current element
   - If current element ≤ heap minimum:
     - Skip it (not in top K)
4. **Return the heap** - it contains the K largest elements

### Visual Walkthrough: Finding Top 3 Numbers

Let's trace through finding the top 3 largest numbers in `[3, 1, 5, 12, 2, 11, 9]`:

```
Initial Array: [3, 1, 5, 12, 2, 11, 9]
K = 3

Step 1: Build heap with first 3 elements [3, 1, 5]
Heap: [1, 3, 5]  (min-heap representation)
Visual:    1
          / \
         3   5

Step 2: Process element 12
Compare: 12 > 1 (heap min)? YES
Action: Remove 1, add 12
Heap: [3, 5, 12]
Visual:    3
          / \
         5  12

Step 3: Process element 2
Compare: 2 > 3 (heap min)? NO
Action: Skip
Heap: [3, 5, 12] (unchanged)

Step 4: Process element 11
Compare: 11 > 3 (heap min)? YES
Action: Remove 3, add 11
Heap: [5, 11, 12]
Visual:    5
          / \
         11  12

Step 5: Process element 9
Compare: 9 > 5 (heap min)? YES
Action: Remove 5, add 9
Heap: [9, 11, 12]
Visual:    9
          / \
         11  12

Final Result: [9, 11, 12]
```

**State Changes Summary:**
```
Element | Heap Before | Compare | Action      | Heap After
--------|-------------|---------|-------------|------------
3       | []          | -       | Add         | [3]
1       | [3]         | -       | Add         | [1,3]
5       | [1,3]       | -       | Add         | [1,3,5]
12      | [1,3,5]     | 12 > 1  | Replace 1   | [3,5,12]
2       | [3,5,12]    | 2 < 3   | Skip        | [3,5,12]
11      | [3,5,12]    | 11 > 3  | Replace 3   | [5,11,12]
9       | [5,11,12]   | 9 > 5   | Replace 5   | [9,11,12]
```

## Implementation

### Python Implementation

```python
import heapq
from typing import List, Tuple

def find_k_largest_numbers(nums: List[int], k: int) -> List[int]:
    """
    Find the K largest numbers in an array using a min-heap.
    
    Args:
        nums: List of integers to search through
        k: Number of largest elements to find
        
    Returns:
        List of K largest numbers (not necessarily sorted)
        
    Time Complexity: O(n log k) where n is length of nums
    Space Complexity: O(k) for the heap
    
    Examples:
        >>> find_k_largest_numbers([3, 1, 5, 12, 2, 11], 3)
        [5, 12, 11]
        
        >>> find_k_largest_numbers([5, 12, 11, -1, 12], 2)
        [12, 12]
    """
    if k <= 0 or not nums:
        return []
    
    # Handle edge case where k >= len(nums)
    if k >= len(nums):
        return nums
    
    # Step 1: Create a min-heap with first k elements
    # heapq in Python is a min-heap by default
    min_heap = nums[:k]
    heapq.heapify(min_heap)  # O(k) operation
    
    # Step 2: Process remaining elements
    for i in range(k, len(nums)):
        # If current element is larger than smallest in heap
        if nums[i] > min_heap[0]:  # O(1) to peek at root
            heapq.heapreplace(min_heap, nums[i])  # O(log k)
            # heapreplace is more efficient than heappop + heappush
    
    return min_heap


def find_k_smallest_numbers(nums: List[int], k: int) -> List[int]:
    """
    Find the K smallest numbers using a max-heap.
    
    Args:
        nums: List of integers to search through
        k: Number of smallest elements to find
        
    Returns:
        List of K smallest numbers
        
    Time Complexity: O(n log k)
    Space Complexity: O(k)
    
    Note: Python's heapq only supports min-heap, so we negate
    values to simulate a max-heap.
    """
    if k <= 0 or not nums:
        return []
    
    if k >= len(nums):
        return nums
    
    # Create max-heap by negating values (min-heap of negatives)
    max_heap = [-x for x in nums[:k]]
    heapq.heapify(max_heap)
    
    for i in range(k, len(nums)):
        # If current element is smaller than largest in heap
        # (comparing negatives: -nums[i] > max_heap[0])
        if nums[i] < -max_heap[0]:
            heapq.heapreplace(max_heap, -nums[i])
    
    # Convert back to positive values
    return [-x for x in max_heap]


def find_kth_largest(nums: List[int], k: int) -> int:
    """
    Find the Kth largest element (not the K largest elements).
    
    Args:
        nums: List of integers
        k: Position of the element (1-indexed, 1 = largest)
        
    Returns:
        The Kth largest element
        
    Time Complexity: O(n log k)
    Space Complexity: O(k)
    
    Example:
        >>> find_kth_largest([3, 2, 1, 5, 6, 4], 2)
        5  # Second largest element
    """
    # Use min-heap of size k
    min_heap = nums[:k]
    heapq.heapify(min_heap)
    
    for i in range(k, len(nums)):
        if nums[i] > min_heap[0]:
            heapq.heapreplace(min_heap, nums[i])
    
    # The root of the heap is the Kth largest
    return min_heap[0]


def top_k_frequent(nums: List[int], k: int) -> List[int]:
    """
    Find K most frequent elements using hash map + heap.
    
    Args:
        nums: List of integers
        k: Number of most frequent elements to find
        
    Returns:
        List of K most frequent elements
        
    Time Complexity: O(n log k) where n is length of nums
    Space Complexity: O(n) for the frequency map
    
    Example:
        >>> top_k_frequent([1,1,1,2,2,3], 2)
        [1, 2]
    """
    if k <= 0 or not nums:
        return []
    
    # Step 1: Build frequency map - O(n)
    from collections import Counter
    freq_map = Counter(nums)
    
    # Step 2: Use min-heap to find k most frequent
    # Heap stores tuples: (frequency, number)
    min_heap = []
    
    for num, freq in freq_map.items():
        if len(min_heap) < k:
            heapq.heappush(min_heap, (freq, num))  # O(log k)
        elif freq > min_heap[0][0]:
            heapq.heapreplace(min_heap, (freq, num))  # O(log k)
    
    # Extract just the numbers from (frequency, number) tuples
    return [num for freq, num in min_heap]


def k_closest_points(points: List[List[int]], k: int) -> List[List[int]]:
    """
    Find K closest points to origin (0, 0).
    
    Args:
        points: List of [x, y] coordinates
        k: Number of closest points to find
        
    Returns:
        K points closest to origin
        
    Time Complexity: O(n log k)
    Space Complexity: O(k)
    
    Example:
        >>> k_closest_points([[1,3], [-2,2], [5,8], [0,1]], 2)
        [[0,1], [-2,2]]
    """
    # Use max-heap to store k closest points
    # Store tuples: (-distance, point) - negate for max-heap behavior
    max_heap = []
    
    for point in points:
        x, y = point
        # Calculate squared distance (no need for sqrt for comparison)
        dist = x * x + y * y
        
        if len(max_heap) < k:
            heapq.heappush(max_heap, (-dist, point))
        elif dist < -max_heap[0][0]:
            heapq.heapreplace(max_heap, (-dist, point))
    
    return [point for dist, point in max_heap]


# Usage Examples
if __name__ == "__main__":
    # Example 1: Find K largest numbers
    nums = [3, 1, 5, 12, 2, 11, 9]
    k = 3
    result = find_k_largest_numbers(nums, k)
    print(f"Top {k} largest from {nums}: {sorted(result, reverse=True)}")
    # Output: Top 3 largest from [3, 1, 5, 12, 2, 11, 9]: [12, 11, 9]
    
    # Example 2: Find Kth largest element
    kth = find_kth_largest([3, 2, 1, 5, 6, 4], 2)
    print(f"2nd largest element: {kth}")
    # Output: 2nd largest element: 5
    
    # Example 3: Top K frequent elements
    frequent = top_k_frequent([1, 1, 1, 2, 2, 3, 4, 4, 4, 4], 2)
    print(f"Top 2 frequent: {frequent}")
    # Output: Top 2 frequent: [1, 4] or [4, 1]
    
    # Example 4: K closest points to origin
    points = [[1, 3], [-2, 2], [5, 8], [0, 1]]
    closest = k_closest_points(points, 2)
    print(f"2 closest points: {closest}")
    # Output: 2 closest points: [[0, 1], [-2, 2]]
```

### Code Explanation

**Key Design Decisions:**

1. **Why `heapreplace` instead of `heappop` + `heappush`?**
   - `heapreplace` is a single operation that's more efficient
   - It removes the root and adds the new element in one step
   - Reduces the number of heap rebalancing operations

2. **Why check `nums[i] > min_heap[0]` before replacing?**
   - Avoids unnecessary heap operations when element doesn't qualify
   - The comparison is O(1), but heap operations are O(log k)
   - Significant performance gain on large datasets

3. **Why use tuples for frequency/distance problems?**
   - Python's heapq compares tuples element-wise
   - `(frequency, number)` naturally orders by frequency first
   - If frequencies tie, it compares numbers as tiebreaker

4. **Why negate values for max-heap simulation?**
   - Python only has min-heap (heapq)
   - Negating values: `-5 < -3` but originally `5 > 3`
   - This flips the ordering to create max-heap behavior

## Complexity Analysis

### Time Complexity

- **Building initial heap:** O(k) using heapify
- **Processing remaining (n-k) elements:** Each heap operation is O(log k)
- **Overall:** O(k + (n-k) log k) = **O(n log k)**

**Why O(n log k)?**
1. We process n elements total
2. For each element, we might do a heap operation (log k)
3. Therefore: n × log k operations

**Best Case:** O(n) - When all elements after the first k are smaller than the kth largest (no heap operations needed)

**Average Case:** O(n log k) - Some elements require heap operations

**Worst Case:** O(n log k) - All elements require heap operations (e.g., sorted array)

### Space Complexity

- **Heap storage:** O(k) - we maintain exactly k elements
- **Additional space:** O(1) - only a few variables
- **Overall:** **O(k)**

For frequency-based problems:
- **Frequency map:** O(n) - in worst case, all elements are unique
- **Heap:** O(k)
- **Overall:** **O(n)** - dominated by the frequency map

### Comparison with Alternatives

| Approach | Time Complexity | Space Complexity | When to Use |
|----------|----------------|------------------|-------------|
| **Top K Pattern (Heap)** | O(n log k) | O(k) | When k << n; memory constrained; streaming data |
| **Full Sort** | O(n log n) | O(1) or O(n) | When k ≈ n; need all elements sorted |
| **QuickSelect** | O(n) average, O(n²) worst | O(1) | When k is moderate; one-time query; can modify array |
| **Counting Sort** | O(n + r) | O(r) | When range r is small; integers only |

**When Top K Pattern Wins:**
- k is much smaller than n (e.g., top 10 from 1 million items)
- Streaming data where you can't store everything
- Multiple queries with different k values
- Memory is limited

**Example:** Finding top 10 from 1 million items:
- Heap: 1M × log(10) ≈ 3.3M operations, 10 elements stored
- Full sort: 1M × log(1M) ≈ 20M operations, 1M elements stored
- **Heap is ~6× faster and uses 100,000× less memory!**

## Examples

### Example 1: Find Top 3 Largest Numbers

**Problem:** Given `[3, 1, 5, 12, 2, 11]`, find the 3 largest numbers.

**Solution:**
```python
nums = [3, 1, 5, 12, 2, 11]
k = 3

# Step-by-step trace
min_heap = [3, 1, 5]  # First k elements
heapq.heapify(min_heap)  # Result: [1, 3, 5]

# Process 12: 12 > 1 → replace
heapq.heapreplace(min_heap, 12)  # [3, 5, 12]

# Process 2: 2 < 3 → skip

# Process 11: 11 > 3 → replace
heapq.heapreplace(min_heap, 11)  # [5, 11, 12]

# Result: [5, 11, 12]
```

**Visualization:**
```
Initial: [3, 1, 5, 12, 2, 11]
         ↓  ↓  ↓
Heap:   [1, 3, 5]

Process 12 (12 > 1):
         ❌ ↓  ↓
Heap:   [3, 5, 12]

Process 2 (2 < 3): Skip

Process 11 (11 > 3):
         ❌ ↓  ↓
Heap:   [5, 11, 12] ✓
```

### Example 2: Kth Largest Element

**Problem:** Find the 4th largest number in `[3, 2, 1, 5, 6, 4]`.

**Solution:**
```python
nums = [3, 2, 1, 5, 6, 4]
k = 4

# Build heap with first 4 elements
min_heap = [3, 2, 1, 5]
heapq.heapify(min_heap)  # [1, 2, 3, 5]

# Process 6: 6 > 1 → replace
heapq.heapreplace(min_heap, 6)  # [2, 3, 5, 6]

# Process 4: 4 > 2 → replace
heapq.heapreplace(min_heap, 4)  # [3, 4, 5, 6]

# The minimum (root) is the Kth largest
result = min_heap[0]  # 3
```

**Why the root is Kth largest:**
```
After processing all elements:
Heap contains: [3, 4, 5, 6] - the 4 largest elements

The minimum of these 4 largest = 4th largest overall
        3 ← This is the 4th largest
       / \
      4   5
     /
    6
```

### Example 3: Top K Frequent Elements

**Problem:** Find 2 most frequent numbers in `[1, 1, 1, 2, 2, 3]`.

**Solution:**
```python
nums = [1, 1, 1, 2, 2, 3]
k = 2

# Step 1: Count frequencies
freq_map = {1: 3, 2: 2, 3: 1}

# Step 2: Build heap with (frequency, number) tuples
min_heap = []

# Process (3, 1): heap = [(3, 1)]
# Process (2, 2): heap = [(2, 2), (3, 1)]
# Process (1, 3): 1 < 2, skip

# Result: [(2, 2), (3, 1)]
# Extract numbers: [2, 1]
```

**Visualization:**
```
Frequency Map:
  Number │ Count
  ───────┼───────
    1    │   3
    2    │   2
    3    │   1

Building heap (size k=2):
  Add (3,1):     (3,1)
  Add (2,2):    (2,2)
                /
              (3,1)
  Check (1,3): 1 < 2, skip

Final: Extract [1, 2] from [(2,2), (3,1)]
```

### Example 4: K Closest Points to Origin

**Problem:** Find 2 closest points to origin from `[[1,3], [-2,2], [5,8], [0,1]]`.

**Solution:**
```python
points = [[1,3], [-2,2], [5,8], [0,1]]
k = 2

# Calculate distances (squared):
# [1,3]: 1² + 3² = 10
# [-2,2]: 4 + 4 = 8
# [5,8]: 25 + 64 = 89
# [0,1]: 0 + 1 = 1

# Build max-heap (negate distances)
max_heap = [(-10, [1,3]), (-8, [-2,2])]

# Process [5,8]: distance 89
# 89 > 10 (current max), skip

# Process [0,1]: distance 1
# 1 < 10, replace (-10, [1,3])
max_heap = [(-8, [-2,2]), (-1, [0,1])]

# Result: [[0,1], [-2,2]]
```

**Distance Visualization:**
```
       y
       │
   3   │   • [1,3]
       │ ↗ dist=√10
   2   │ • [-2,2]
       │↗ dist=√8
   1   │• [0,1]
       │ dist=1
   ────┼─────────── x
   -2  0  1     5

Distances (squared):
[0,1]: 1     ← Closest
[-2,2]: 8    ← Second closest ✓
[1,3]: 10
[5,8]: 89
```

## Edge Cases

### 1. K Equals Array Length
**Scenario:** `k = len(nums)` - user wants all elements

**Challenge:** The algorithm would build a heap of the entire array, which is inefficient.

**Solution:** Return the original array immediately without heap operations.

```python
def find_k_largest_numbers(nums: List[int], k: int) -> List[int]:
    if k >= len(nums):
        return nums  # No need for heap operations
    # ... rest of algorithm
```

**Why this matters:** Building a heap of n elements takes O(n) time, but we don't need any ordering, so we can return in O(1) time.

### 2. K is Zero or Negative
**Scenario:** `k = 0` or `k < 0` - invalid input

**Challenge:** Heap of size 0 doesn't make sense; negative size is meaningless.

**Solution:** Return empty list.

```python
def find_k_largest_numbers(nums: List[int], k: int) -> List[int]:
    if k <= 0 or not nums:
        return []
    # ... rest of algorithm
```

### 3. Empty Input Array
**Scenario:** `nums = []` - no elements to search

**Challenge:** Can't build a heap from nothing.

**Solution:** Return empty list immediately.

```python
def find_k_largest_numbers(nums: List[int], k: int) -> List[int]:
    if not nums:
        return []
    # ... rest of algorithm
```

### 4. Array with Duplicates
**Scenario:** `nums = [5, 5, 5, 5]`, `k = 2`

**Challenge:** Multiple elements with same value might need to be in top K.

**Solution:** Algorithm handles this naturally - duplicates are treated as separate elements.

```python
# Example: [5, 5, 3, 3], k=2
# Result could be [5, 5] or [5, 3] depending on processing order
# Both are correct! Heap doesn't guarantee order within same values.

min_heap = [5, 5]  # Valid result
# or
min_heap = [5, 3]  # Also valid if 3's are processed differently
```

**Important:** If you need a specific tiebreaker (e.g., prefer earlier indices), modify the heap to store tuples: `(value, index)`.

### 5. Single Element Array
**Scenario:** `nums = [42]`, `k = 1`

**Challenge:** Array has only one element.

**Solution:** Return that single element (or the array itself).

```python
# If k >= len(nums), return nums
# nums = [42], k = 1 → returns [42]
# nums = [42], k = 3 → returns [42] (can't find 3 elements)
```

### 6. All Elements Identical
**Scenario:** `nums = [7, 7, 7, 7, 7]`, `k = 3`

**Challenge:** Every element is the "largest" element.

**Solution:** Algorithm works correctly - returns k copies of the value.

```python
nums = [7, 7, 7, 7, 7]
k = 3
result = find_k_largest_numbers(nums, k)  # [7, 7, 7]
```

### 7. Negative Numbers
**Scenario:** `nums = [-5, -2, -10, -1]`, `k = 2`

**Challenge:** Ensure comparison logic works with negative values.

**Solution:** Comparison operators work the same for negative numbers.

```python
# Finding 2 largest from [-5, -2, -10, -1]
# Largest: -1, -2
# Algorithm handles naturally: -1 > -2 > -5 > -10
```

### 8. Very Large K
**Scenario:** `k = 1000000` but `nums` has only 100 elements

**Challenge:** Can't find more elements than exist.

**Solution:** Return all elements when k > n.

```python
if k >= len(nums):
    return nums
```

## Common Pitfalls

### ❌ Pitfall 1: Using Wrong Heap Type

**What happens:**
```python
# WRONG: Using max-heap to find K largest
# This finds K smallest instead!
max_heap = [-x for x in nums[:k]]
heapq.heapify(max_heap)

for i in range(k, len(nums)):
    if nums[i] > -max_heap[0]:  # Wrong comparison
        heapq.heapreplace(max_heap, -nums[i])
```

**Why it's wrong:** A max-heap stores the largest elements at the top. To find the K largest overall, you need to reject elements smaller than the "weakest" of your top K - but in a max-heap, the top is the strongest!

**Correct approach:**
```python
# CORRECT: Use min-heap to find K largest
min_heap = nums[:k]
heapq.heapify(min_heap)

for i in range(k, len(nums)):
    if nums[i] > min_heap[0]:  # Compare with smallest of top K
        heapq.heapreplace(min_heap, nums[i])
```

**Memory trick:** "Use MIN heap for MAX elements" - the minimum in your heap is the boundary for what qualifies as "large enough."

### ❌ Pitfall 2: Sorting the Entire Array

**What happens:**
```python
# WRONG: Sorting defeats the purpose
def find_k_largest(nums, k):
    nums.sort(reverse=True)  # O(n log n)
    return nums[:k]
```

**Why it's wrong:** This is O(n log n) time, which is slower than O(n log k) when k << n. For k=10 and n=1,000,000, you're doing 100× more work than necessary!

**Correct approach:** Use the heap-based pattern for O(n log k) time.

**When sorting is acceptable:** If k ≈ n (e.g., finding top 90% of elements), then O(n log n) ≈ O(n log k), so sorting might be simpler.

### ❌ Pitfall 3: Not Handling Edge Cases

**What happens:**
```python
# WRONG: Crashes on empty array or k=0
def find_k_largest(nums, k):
    min_heap = nums[:k]  # IndexError if nums is empty!
    heapq.heapify(min_heap)
    # ...
```

**Why it's wrong:** Slicing an empty array with `nums[:k]` returns `[]`, and heapify on `[]` is OK, but you haven't handled the logical case.

**Correct approach:**
```python
def find_k_largest(nums, k):
    # Guard clauses at the start
    if k <= 0 or not nums:
        return []
    if k >= len(nums):
        return nums
    # Now proceed with algorithm
```

### ❌ Pitfall 4: Forgetting to Negate for Max-Heap in Python

**What happens:**
```python
# WRONG: Trying to use max-heap without negating
max_heap = nums[:k]  # This is still a min-heap!
heapq.heapify(max_heap)

for i in range(k, len(nums)):
    if nums[i] < max_heap[0]:  # Wrong: this finds K largest, not smallest
        heapq.heapreplace(max_heap, nums[i])
```

**Why it's wrong:** Python's heapq is ALWAYS a min-heap. You can't make it a max-heap by just changing the comparison.

**Correct approach:**
```python
# CORRECT: Negate values to simulate max-heap
max_heap = [-x for x in nums[:k]]
heapq.heapify(max_heap)

for i in range(k, len(nums)):
    if nums[i] < -max_heap[0]:  # Compare with largest of top K
        heapq.heapreplace(max_heap, -nums[i])

# Convert back to positive
return [-x for x in max_heap]
```

### ❌ Pitfall 5: Using heappop + heappush Instead of heapreplace

**What happens:**
```python
# LESS EFFICIENT: Two operations
if nums[i] > min_heap[0]:
    heapq.heappop(min_heap)      # O(log k)
    heapq.heappush(min_heap, nums[i])  # O(log k)
    # Total: 2 × O(log k) rebalancing operations
```

**Why it's suboptimal:** Two separate heap operations mean the heap is rebalanced twice.

**Correct approach:**
```python
# MORE EFFICIENT: Single operation
if nums[i] > min_heap[0]:
    heapq.heapreplace(min_heap, nums[i])  # O(log k)
    # Single rebalancing operation
```

**Performance impact:** On large datasets with many replacements, this can be 20-30% faster.

### ❌ Pitfall 6: Not Checking Before Replacing

**What happens:**
```python
# INEFFICIENT: Always replacing, even when not needed
for i in range(k, len(nums)):
    heapq.heapreplace(min_heap, nums[i])  # Replaces even if nums[i] is smaller!
```

**Why it's wrong:** If nums[i] < min_heap[0], replacing it just to remove it again wastes operations.

**Correct approach:**
```python
# EFFICIENT: Check first
for i in range(k, len(nums)):
    if nums[i] > min_heap[0]:  # O(1) comparison
        heapq.heapreplace(min_heap, nums[i])  # O(log k) only when needed
```

**Impact:** On arrays where most elements don't qualify (e.g., finding top 10 from sorted array), this saves enormous time.

## Variations and Extensions

### Variation 1: Top K Frequent Elements with Bucket Sort

**Description:** Alternative O(n) approach using bucket sort when K is close to the number of unique elements.

**When to use:** When K ≈ number of unique elements; when guaranteed integer frequencies.

**Key differences:**
- Time: O(n) instead of O(n log k)
- Space: O(n) for buckets
- Only works for frequency-based problems

**Implementation:**
```python
def top_k_frequent_bucket(nums: List[int], k: int) -> List[int]:
    """
    Find top K frequent using bucket sort - O(n) time.
    """
    from collections import Counter
    
    freq_map = Counter(nums)
    # Create buckets indexed by frequency
    buckets = [[] for _ in range(len(nums) + 1)]
    
    for num, freq in freq_map.items():
        buckets[freq].append(num)
    
    result = []
    # Iterate from highest frequency to lowest
    for freq in range(len(buckets) - 1, 0, -1):
        for num in buckets[freq]:
            result.append(num)
            if len(result) == k:
                return result
    
    return result
```

### Variation 2: Kth Largest in a Stream

**Description:** Maintain Kth largest element in a stream of numbers where elements arrive one at a time.

**When to use:** Real-time systems; infinite streams; when you can't store all elements.

**Key differences:**
- Fixed heap of size K maintained continuously
- O(log k) per insertion
- Can't batch process

**Implementation:**
```python
class KthLargest:
    """
    Maintain Kth largest element in a stream.
    
    Example:
        kth_largest = KthLargest(3, [4, 5, 8, 2])
        kth_largest.add(3)  # returns 4
        kth_largest.add(5)  # returns 5
        kth_largest.add(10) # returns 5
    """
    
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.min_heap = nums
        heapq.heapify(self.min_heap)
        
        # Keep only k largest elements
        while len(self.min_heap) > k:
            heapq.heappop(self.min_heap)
    
    def add(self, val: int) -> int:
        """
        Add new value and return current Kth largest.
        Time: O(log k)
        """
        if len(self.min_heap) < self.k:
            heapq.heappush(self.min_heap, val)
        elif val > self.min_heap[0]:
            heapq.heapreplace(self.min_heap, val)
        
        return self.min_heap[0]
```

### Variation 3: QuickSelect for Kth Element

**Description:** Use partition-based selection (like QuickSort) to find Kth element in average O(n) time.

**When to use:**
- One-time query (not streaming)
- Can modify the array
- Want average O(n) instead of O(n log k)

**Key differences:**
- Average O(n), worst O(n²)
- Finds exact Kth element, not K largest
- Modifies input array

**Implementation:**
```python
import random

def quickselect_kth_largest(nums: List[int], k: int) -> int:
    """
    Find Kth largest using QuickSelect - average O(n).
    
    Note: Modifies the input array!
    """
    # Convert to 0-indexed position from right
    # Kth largest is at index (len-k)
    return quickselect(nums, 0, len(nums) - 1, len(nums) - k)

def quickselect(nums: List[int], left: int, right: int, k: int) -> int:
    """
    Helper function for quickselect.
    """
    if left == right:
        return nums[left]
    
    # Random pivot for better average case
    pivot_index = random.randint(left, right)
    pivot_index = partition(nums, left, right, pivot_index)
    
    if k == pivot_index:
        return nums[k]
    elif k < pivot_index:
        return quickselect(nums, left, pivot_index - 1, k)
    else:
        return quickselect(nums, pivot_index + 1, right, k)

def partition(nums: List[int], left: int, right: int, pivot_index: int) -> int:
    """
    Partition array around pivot.
    """
    pivot_value = nums[pivot_index]
    # Move pivot to end
    nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
    store_index = left
    
    for i in range(left, right):
        if nums[i] < pivot_value:
            nums[i], nums[store_index] = nums[store_index], nums[i]
            store_index += 1
    
    # Move pivot to final position
    nums[right], nums[store_index] = nums[store_index], nums[right]
    return store_index
```

### Variation 4: Frequency Sorting with Custom Comparator

**Description:** Sort elements by frequency, with tiebreaker rules.

**When to use:** Need custom ordering beyond just frequency; need stable sort.

**Implementation:**
```python
def frequency_sort_custom(nums: List[int]) -> List[int]:
    """
    Sort by frequency (descending), then by value (ascending).
    
    Example:
        [1,1,2,2,2,3] → [2,2,2,1,1,3]
        Frequency: 2 appears 3 times, 1 appears 2 times, 3 appears 1 time
    """
    from collections import Counter
    
    freq_map = Counter(nums)
    
    # Sort using custom key: (-frequency, value)
    # Negative frequency for descending order
    return sorted(nums, key=lambda x: (-freq_map[x], x))
```

### Variation 5: Sliding Window Top K

**Description:** Find top K elements in each sliding window of size W.

**When to use:** Time-series analysis; moving averages; real-time analytics.

**Key differences:**
- Maintain heap while adding/removing elements
- Need to track element counts for removal

**Implementation:**
```python
from collections import Counter

def sliding_window_top_k(nums: List[int], k: int, window_size: int) -> List[List[int]]:
    """
    Find top K elements in each sliding window.
    
    Example:
        nums = [1,3,1,2,0,5], k=2, window_size=3
        Windows: [1,3,1], [3,1,2], [1,2,0], [2,0,5]
        Top 2: [3,1], [3,2], [2,1], [5,2]
    """
    if window_size > len(nums):
        return []
    
    result = []
    freq = Counter()
    
    for i in range(len(nums)):
        # Add new element
        freq[nums[i]] += 1
        
        # Remove old element from window
        if i >= window_size:
            freq[nums[i - window_size]] -= 1
            if freq[nums[i - window_size]] == 0:
                del freq[nums[i - window_size]]
        
        # Once we have a full window
        if i >= window_size - 1:
            # Find top k in current window
            top_k = heapq.nlargest(k, freq.items(), key=lambda x: x[1])
            result.append([num for num, count in top_k])
    
    return result
```

## Practice Problems

### Beginner

1. **Top K Numbers** - Find the K largest or smallest numbers in an unsorted array
   - LeetCode #215 (Kth Largest Element in an Array)
   - Practice with different K values and array sizes

2. **Kth Smallest Element** - Find the Kth smallest element using a max-heap
   - Start with understanding why we use max-heap for smallest
   - Try both heap and quickselect approaches

3. **Closest Points** - Find K closest points to origin (0,0)
   - LeetCode #973 (K Closest Points to Origin)
   - Practice calculating Euclidean distance

4. **Top K Frequent Words** - Find K most frequent words in a list
   - LeetCode #692 (Top K Frequent Words)
   - Combine hash map with heap

### Intermediate

1. **K Pairs with Smallest Sums** - Find K pairs with smallest sums from two arrays
   - LeetCode #373
   - Use heap to track pairs efficiently

2. **Sort Characters by Frequency** - Sort string characters by frequency
   - LeetCode #451 (Sort Characters By Frequency)
   - Similar to top K frequent, but return sorted string

3. **Reorganize String** - Rearrange string so no two same characters are adjacent
   - LeetCode #767 (Reorganize String)
   - Use max-heap to always pick most frequent available character

4. **Kth Largest Element in Stream** - Design class to maintain Kth largest in stream
   - LeetCode #703 (Kth Largest Element in a Stream)
   - Practice with streaming data pattern

5. **Find K Closest Elements** - Find K closest integers to target X in sorted array
   - LeetCode #658
   - Consider both heap and two-pointer approaches

6. **Connect Ropes** - Connect N ropes with minimum cost (always connect two shortest)
   - Practice greedy + heap pattern
   - Similar to Huffman coding

7. **Ugly Number II** - Find Kth number whose prime factors are only 2, 3, or 5
   - LeetCode #264
   - Use min-heap to generate sequence

### Advanced

1. **Rearrange String K Distance Apart** - Rearrange so same characters are K distance apart
   - LeetCode #358 (Rearrange String k Distance Apart)
   - Combine frequency heap with cooldown tracking

2. **Task Scheduler** - Schedule tasks with cooldown period to minimize total time
   - LeetCode #621 (Task Scheduler)
   - Use max-heap + queue for cooldown

3. **Smallest Range Covering K Lists** - Find smallest range containing at least one number from each K lists
   - LeetCode #632 (Smallest Range Covering Elements from K Lists)
   - Combine min-heap with range tracking

4. **Maximum Frequency Stack** - Design stack that pops most frequent element
   - LeetCode #895 (Maximum Frequency Stack)
   - Use frequency heap with timestamp tiebreaker

5. **Kth Smallest in Multiplication Table** - Find Kth smallest in M×N multiplication table
   - LeetCode #668
   - Use binary search + counting, not heap!

6. **Find Median from Data Stream** - Maintain median as numbers arrive
   - LeetCode #295 (Find Median from Data Stream)
   - Use two heaps: max-heap for smaller half, min-heap for larger half

7. **Sliding Window Median** - Find median in each sliding window
   - LeetCode #480 (Sliding Window Median)
   - Extend two-heap approach with element removal

## Real-World Applications

### Industry Use Cases

1. **Social Media Trending Topics**
   - **How it's used:** Twitter/X uses Top K to identify trending hashtags and topics in real-time across millions of tweets
   - **Why it's effective:** O(n log k) allows processing millions of tweets efficiently, keeping only top trending topics in memory
   - **Scale:** Processing 500M tweets/day to find top 100 trends → heap is 5000× more efficient than sorting

2. **E-Commerce Product Recommendations**
   - **How it's used:** Amazon finds "Top K most purchased together" or "Top K rated products" for recommendations
   - **Why it's effective:** Can update recommendations in real-time as purchases happen without recomputing entire dataset
   - **Implementation:** Streaming heap updates as each purchase occurs

3. **Search Engine Autocomplete**
   - **How it's used:** Google suggests top K most common completions for search queries
   - **Why it's effective:** Maintains top suggestions per prefix using frequency heaps
   - **Scale:** Billions of searches → only top 10 suggestions per prefix stored

4. **System Monitoring & Alerting**
   - **How it's used:** Find top K error messages, slowest queries, or highest resource consumers
   - **Why it's effective:** Real-time identification of issues without storing all metrics
   - **Example:** Finding top 10 slowest API endpoints from millions of requests

5. **Financial Trading**
   - **How it's used:** Identify top K movers (stocks with largest price changes)
   - **Why it's effective:** Maintains leaderboard in O(log k) per price update
   - **Real-time:** Must process thousands of price updates per second

### Popular Implementations

- **Python's heapq.nlargest() and nsmallest():** Built-in functions implementing Top K pattern
- **Java PriorityQueue:** Standard library heap used for Top K problems
- **Apache Kafka Streams:** Uses approximate Top K for streaming analytics
- **Redis Sorted Sets:** Efficiently maintains top-scored items with ZREVRANGE
- **Elasticsearch Aggregations:** Top hits aggregation uses heap-based selection
- **NumPy's argpartition():** Partial sorting for Top K in numerical arrays

### Practical Scenarios

- **Content Moderation:** Finding top K most reported posts/comments for review
- **Log Analysis:** Identifying top K IP addresses by request volume (DDoS detection)
- **Gaming Leaderboards:** Maintaining top K players by score
- **Video Streaming:** Selecting top K quality levels based on network conditions
- **Email Spam Detection:** Finding top K spam indicators (words, patterns)
- **Network Traffic:** Identifying top K bandwidth consumers
- **Inventory Management:** Tracking top K selling products for restocking
- **A/B Testing:** Finding top K performing variants
- **Music Streaming:** "Top K songs this week" in each genre
- **Ad Serving:** Selecting top K highest-bid ads for each auction

## Related Topics

### Prerequisites to Review

- **Heaps (Priority Queues)** - Essential foundation; understand heap operations and properties
- **Hash Maps** - Required for frequency-based Top K problems
- **Arrays and Array Manipulation** - Basic data structure for input
- **Time/Space Complexity Analysis** - To understand when to use this pattern

### Next Steps

- **K-way Merge Pattern** - Natural extension using heaps to merge K sorted lists
- **Sliding Window Pattern** - Combine with Top K for window-based problems
- **Two Heaps Pattern** - Find median using two heaps (max-heap + min-heap)
- **QuickSelect Algorithm** - Alternative O(n) average time for finding Kth element
- **Trie + Heap** - For problems like "Top K Frequent Words" with prefix matching

### Similar Concepts

- **Partial Sorting** - Top K is a form of partial sorting (only sort K elements)
- **Selection Algorithms** - Family of algorithms for finding Kth element
- **Order Statistics** - Mathematical concept of finding Kth order statistic
- **Streaming Algorithms** - Top K in streams connects to approximate counting algorithms
- **Reservoir Sampling** - Random sampling of K elements from stream

### Further Reading

- **Introduction to Algorithms (CLRS)** - Chapter 6 (Heapsort) and Chapter 9 (Medians and Order Statistics)
  - Deep dive into heap data structure and selection algorithms
  
- **LeetCode Explore Cards** - "Heap" card covers Top K patterns extensively
  - https://leetcode.com/explore/learn/card/heap/
  
- **GeeksforGeeks - K largest/smallest elements**
  - https://www.geeksforgeeks.org/k-largestor-smallest-elements-in-an-array/
  - Multiple approaches compared with analysis
  
- **Python heapq Documentation**
  - https://docs.python.org/3/library/heapq.html
  - Official documentation with examples and complexity analysis
  
- **"Algorithms" by Robert Sedgewick** - Section on Priority Queues
  - Comprehensive coverage of heap implementations and applications
  
- **System Design Interview Resources**
  - "Designing Data-Intensive Applications" by Martin Kleppmann
  - Real-world uses of Top K in distributed systems
