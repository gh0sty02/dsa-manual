# K-way Merge Pattern

**Difficulty:** Intermediate to Advanced
**Prerequisites:** Heaps (Priority Queues), Linked Lists, Arrays, Merge Sort concept
**Estimated Reading Time:** 20 minutes

## Introduction

The K-way Merge pattern is an efficient technique for merging K sorted arrays, lists, or sequences into a single sorted output. Instead of merging them one-by-one (which would be inefficient), this pattern uses a min-heap to simultaneously track the smallest unprocessed element from each of the K inputs, ensuring optimal performance.

**Why it matters:** This pattern is fundamental to external sorting (sorting data larger than memory), distributed systems (merging results from multiple servers), and database query optimization. It's also the backbone of merge sort and appears in real-time data processing pipelines where sorted streams need to be combined.

**Real-world analogy:** Imagine you're a librarian organizing books from K different sorted shelves onto one master shelf. You could take all books from shelf 1, then shelf 2, etc., and re-sort everything - but that's wasteful! Instead, you look at the first book on each shelf, pick the one that comes first alphabetically, place it on the master shelf, and repeat. You're always choosing the "next smallest" from among K candidates. That's K-way merge!

## Core Concepts

### Key Principles

1. **Multiple sorted inputs:** Each of the K inputs is already sorted internally. We leverage this property to avoid re-sorting.

2. **Min-heap for selection:** A min-heap of size K tracks the current smallest element from each input. The heap root is always the global minimum among unprocessed elements.

3. **Lazy evaluation:** We only process elements as needed, one at a time, rather than loading all K inputs into memory.

4. **Pointer advancement:** After extracting an element from input i, we advance that input's pointer to its next element and insert it into the heap.

### Essential Terms

- **K-way merge:** Merging K separate sorted sequences into one sorted sequence
- **Min-heap:** Priority queue that maintains the smallest element at the root
- **Heap entry:** Typically a tuple `(value, list_index, element_index)` to track which input an element came from
- **Input exhaustion:** When an input list has no more elements to contribute
- **Stable merge:** Maintains relative order of equal elements from the same source

### Visual Overview

```
K=3 Sorted Lists:
List 0: [1, 4, 7]
List 1: [2, 5, 8]
List 2: [3, 6, 9]

Min-Heap Process:

Step 1: Initialize heap with first element from each list
Heap: [(1,0,0), (2,1,0), (3,2,0)]
       1
      / \
     2   3

Step 2: Extract min (1 from list 0), add next from list 0
Heap: [(2,1,0), (3,2,0), (4,0,1)]
Output: [1]

Step 3: Extract min (2 from list 1), add next from list 1
Heap: [(3,2,0), (4,0,1), (5,1,1)]
Output: [1, 2]

Continue until all elements processed...
Final Output: [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

**Key Insight:** By maintaining only K elements in the heap (one from each list), we achieve O(log K) insertion/extraction time rather than O(log N) where N is the total number of elements.

## How It Works

### Algorithm Steps for K-way Merge

1. **Initialize min-heap** with the first element from each of the K inputs
   - Each heap entry: `(value, source_index, element_index)`
   
2. **While heap is not empty:**
   - Extract minimum element from heap (this is the next smallest overall)
   - Add it to the result
   - If the source of this element has more elements:
     - Insert the next element from that source into heap
   
3. **Return merged result**

### Visual Walkthrough: Merging 3 Lists

Let's trace through merging:
- List 0: `[1, 4, 5]`
- List 1: `[1, 3, 4]`
- List 2: `[2, 6]`

```
═══════════════════════════════════════════════════════════════
INITIALIZATION
═══════════════════════════════════════════════════════════════
List 0: [1, 4, 5]    pointer: 0 → 1
List 1: [1, 3, 4]    pointer: 0 → 1
List 2: [2, 6]       pointer: 0 → 2

Min-Heap: [(1,0,0), (1,1,0), (2,2,0)]
Structure:     1,0,0
              /     \
          1,1,0     2,2,0
          
Result: []

═══════════════════════════════════════════════════════════════
STEP 1: Extract (1, list 0, index 0)
═══════════════════════════════════════════════════════════════
List 0: [1, 4, 5]    pointer: 1 → 4
List 1: [1, 3, 4]    pointer: 0 → 1
List 2: [2, 6]       pointer: 0 → 2

Action: Remove (1,0,0), add (4,0,1)
Min-Heap: [(1,1,0), (2,2,0), (4,0,1)]
Structure:     1,1,0
              /     \
          2,2,0     4,0,1

Result: [1]

═══════════════════════════════════════════════════════════════
STEP 2: Extract (1, list 1, index 0)
═══════════════════════════════════════════════════════════════
List 0: [1, 4, 5]    pointer: 1 → 4
List 1: [1, 3, 4]    pointer: 1 → 3
List 2: [2, 6]       pointer: 0 → 2

Action: Remove (1,1,0), add (3,1,1)
Min-Heap: [(2,2,0), (3,1,1), (4,0,1)]
Structure:     2,2,0
              /     \
          3,1,1     4,0,1

Result: [1, 1]

═══════════════════════════════════════════════════════════════
STEP 3: Extract (2, list 2, index 0)
═══════════════════════════════════════════════════════════════
List 0: [1, 4, 5]    pointer: 1 → 4
List 1: [1, 3, 4]    pointer: 1 → 3
List 2: [2, 6]       pointer: 1 → 6

Action: Remove (2,2,0), add (6,2,1)
Min-Heap: [(3,1,1), (4,0,1), (6,2,1)]
Structure:     3,1,1
              /     \
          4,0,1     6,2,1

Result: [1, 1, 2]

═══════════════════════════════════════════════════════════════
STEP 4: Extract (3, list 1, index 1)
═══════════════════════════════════════════════════════════════
List 0: [1, 4, 5]    pointer: 1 → 4
List 1: [1, 3, 4]    pointer: 2 → 4
List 2: [2, 6]       pointer: 1 → 6

Action: Remove (3,1,1), add (4,1,2)
Min-Heap: [(4,0,1), (4,1,2), (6,2,1)]
Structure:     4,0,1
              /     \
          4,1,2     6,2,1

Result: [1, 1, 2, 3]

═══════════════════════════════════════════════════════════════
Continue this process...
═══════════════════════════════════════════════════════════════
Final Result: [1, 1, 2, 3, 4, 4, 5, 6]
```

**State Transition Table:**

| Step | Extract | From List | Heap Before | Heap After | Result |
|------|---------|-----------|-------------|------------|--------|
| Init | - | - | empty | [1₀,1₁,2₂] | [] |
| 1 | 1 | List 0 | [1₀,1₁,2₂] | [1₁,2₂,4₀] | [1] |
| 2 | 1 | List 1 | [1₁,2₂,4₀] | [2₂,3₁,4₀] | [1,1] |
| 3 | 2 | List 2 | [2₂,3₁,4₀] | [3₁,4₀,6₂] | [1,1,2] |
| 4 | 3 | List 1 | [3₁,4₀,6₂] | [4₀,4₁,6₂] | [1,1,2,3] |
| 5 | 4 | List 0 | [4₀,4₁,6₂] | [4₁,5₀,6₂] | [1,1,2,3,4] |
| 6 | 4 | List 1 | [4₁,5₀,6₂] | [5₀,6₂] | [1,1,2,3,4,4] |
| 7 | 5 | List 0 | [5₀,6₂] | [6₂] | [1,1,2,3,4,4,5] |
| 8 | 6 | List 2 | [6₂] | [] | [1,1,2,3,4,4,5,6] |

## Implementation

### Python Implementation

```python
import heapq
from typing import List, Optional

class ListNode:
    """Definition for singly-linked list node."""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_k_sorted_arrays(arrays: List[List[int]]) -> List[int]:
    """
    Merge K sorted arrays into one sorted array using min-heap.
    
    Args:
        arrays: List of K sorted arrays
        
    Returns:
        Single merged sorted array
        
    Time Complexity: O(N log K) where N is total elements, K is number of arrays
    Space Complexity: O(K) for the heap + O(N) for result
    
    Examples:
        >>> merge_k_sorted_arrays([[1,4,5], [1,3,4], [2,6]])
        [1, 1, 2, 3, 4, 4, 5, 6]
        
        >>> merge_k_sorted_arrays([[1], [2], [3]])
        [1, 2, 3]
    """
    if not arrays:
        return []
    
    result = []
    min_heap = []
    
    # Step 1: Initialize heap with first element from each array
    # Heap entry: (value, array_index, element_index)
    for i in range(len(arrays)):
        if arrays[i]:  # Check if array is not empty
            heapq.heappush(min_heap, (arrays[i][0], i, 0))
    
    # Step 2: Extract min and add next element from same array
    while min_heap:
        value, array_idx, element_idx = heapq.heappop(min_heap)
        result.append(value)
        
        # If there are more elements in the same array, add next one
        if element_idx + 1 < len(arrays[array_idx]):
            next_value = arrays[array_idx][element_idx + 1]
            heapq.heappush(min_heap, (next_value, array_idx, element_idx + 1))
    
    return result


def merge_k_sorted_lists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Merge K sorted linked lists into one sorted linked list.
    
    Args:
        lists: List of K sorted linked list heads
        
    Returns:
        Head of merged sorted linked list
        
    Time Complexity: O(N log K)
    Space Complexity: O(K) for heap
    
    Note: This is LeetCode #23 - Merge k Sorted Lists
    
    Example:
        Input: [[1,4,5], [1,3,4], [2,6]]
        Output: 1→1→2→3→4→4→5→6
    """
    if not lists:
        return None
    
    # Min-heap to store (node.val, unique_id, node)
    # unique_id prevents comparison issues when values are equal
    min_heap = []
    
    # Initialize heap with head of each list
    for i, node in enumerate(lists):
        if node:
            # Use index as tiebreaker to avoid comparing nodes
            heapq.heappush(min_heap, (node.val, i, node))
    
    # Dummy head for result list
    dummy = ListNode(0)
    current = dummy
    
    # Counter for unique IDs (needed when adding new nodes)
    unique_id = len(lists)
    
    while min_heap:
        val, _, node = heapq.heappop(min_heap)
        
        # Add to result list
        current.next = node
        current = current.next
        
        # If there's a next node in this list, add it to heap
        if node.next:
            heapq.heappush(min_heap, (node.next.val, unique_id, node.next))
            unique_id += 1
    
    return dummy.next


def kth_smallest_in_m_sorted_lists(lists: List[List[int]], k: int) -> int:
    """
    Find Kth smallest element across M sorted lists.
    
    Args:
        lists: M sorted lists
        k: Position of element to find (1-indexed)
        
    Returns:
        Kth smallest element
        
    Time Complexity: O(K log M) - only process K elements
    Space Complexity: O(M) for heap
    
    Example:
        >>> kth_smallest_in_m_sorted_lists([[2,6,8], [3,6,10], [5,8,11]], 5)
        6  # Elements in order: 2,3,5,6,6... 5th is 6
    """
    if not lists or k <= 0:
        return -1
    
    min_heap = []
    
    # Initialize with first element from each list
    for i in range(len(lists)):
        if lists[i]:
            heapq.heappush(min_heap, (lists[i][0], i, 0))
    
    # Extract K-1 elements (to position at Kth)
    count = 0
    result = -1
    
    while min_heap and count < k:
        value, list_idx, element_idx = heapq.heappop(min_heap)
        result = value
        count += 1
        
        # Add next element from same list
        if element_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][element_idx + 1]
            heapq.heappush(min_heap, (next_val, list_idx, element_idx + 1))
    
    return result if count == k else -1


def kth_smallest_in_sorted_matrix(matrix: List[List[int]], k: int) -> int:
    """
    Find Kth smallest element in row and column sorted matrix.
    
    Args:
        matrix: n x n matrix sorted in rows and columns
        k: Position of element to find
        
    Returns:
        Kth smallest element
        
    Time Complexity: O(K log n) where n is matrix dimension
    Space Complexity: O(n) for heap
    
    Example:
        matrix = [
            [1,  5,  9],
            [10, 11, 13],
            [12, 13, 15]
        ]
        k = 8
        Output: 13  (elements: 1,5,9,10,11,12,13,13,15)
    """
    if not matrix or not matrix[0]:
        return -1
    
    n = len(matrix)
    min_heap = []
    
    # Initialize heap with first element of each row
    for i in range(min(n, k)):  # Only need first k rows
        heapq.heappush(min_heap, (matrix[i][0], i, 0))
    
    count = 0
    result = -1
    
    while min_heap and count < k:
        value, row, col = heapq.heappop(min_heap)
        result = value
        count += 1
        
        # Add next element in same row
        if col + 1 < n:
            heapq.heappush(min_heap, (matrix[row][col + 1], row, col + 1))
    
    return result


def smallest_range_covering_k_lists(lists: List[List[int]]) -> List[int]:
    """
    Find smallest range that includes at least one number from each list.
    
    Args:
        lists: K sorted lists
        
    Returns:
        [start, end] of smallest range
        
    Time Complexity: O(N log K) where N is total elements
    Space Complexity: O(K)
    
    Example:
        lists = [[4,10,15,24,26], [0,9,12,20], [5,18,22,30]]
        Output: [20,24]
        Explanation: Range [20,24] contains 24 from list 1, 20 from list 2, 22 from list 3
    """
    min_heap = []
    current_max = float('-inf')
    
    # Initialize heap with first element from each list
    for i in range(len(lists)):
        if lists[i]:
            heapq.heappush(min_heap, (lists[i][0], i, 0))
            current_max = max(current_max, lists[i][0])
    
    # Track smallest range
    range_start, range_end = 0, float('inf')
    
    while len(min_heap) == len(lists):  # All lists must be represented
        current_min, list_idx, element_idx = heapq.heappop(min_heap)
        
        # Update smallest range if current range is smaller
        if current_max - current_min < range_end - range_start:
            range_start = current_min
            range_end = current_max
        
        # Move to next element in the list that had minimum
        if element_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][element_idx + 1]
            heapq.heappush(min_heap, (next_val, list_idx, element_idx + 1))
            current_max = max(current_max, next_val)
        else:
            # One list exhausted, can't maintain coverage
            break
    
    return [range_start, range_end]


# Usage Examples
if __name__ == "__main__":
    # Example 1: Merge K sorted arrays
    arrays = [[1, 4, 5], [1, 3, 4], [2, 6]]
    merged = merge_k_sorted_arrays(arrays)
    print(f"Merged arrays: {merged}")
    # Output: [1, 1, 2, 3, 4, 4, 5, 6]
    
    # Example 2: Kth smallest in M sorted lists
    lists = [[2, 6, 8], [3, 6, 10], [5, 8, 11]]
    kth = kth_smallest_in_m_sorted_lists(lists, 5)
    print(f"5th smallest: {kth}")
    # Output: 6
    
    # Example 3: Kth smallest in sorted matrix
    matrix = [
        [1, 5, 9],
        [10, 11, 13],
        [12, 13, 15]
    ]
    kth_matrix = kth_smallest_in_sorted_matrix(matrix, 8)
    print(f"8th smallest in matrix: {kth_matrix}")
    # Output: 13
    
    # Example 4: Smallest range covering K lists
    range_lists = [[4, 10, 15, 24, 26], [0, 9, 12, 20], [5, 18, 22, 30]]
    range_result = smallest_range_covering_k_lists(range_lists)
    print(f"Smallest range: {range_result}")
    # Output: [20, 24]
```

### Code Explanation

**Key Design Decisions:**

1. **Why store tuples (value, index, position) in heap?**
   - `value`: Needed for min-heap ordering
   - `index`: Tracks which input list this element came from
   - `position`: Tracks position in that list, so we know what to add next
   - This allows us to "refill" from the correct source

2. **Why use dummy node for linked list merge?**
   - Simplifies edge cases (empty result, single element)
   - Avoids special handling for the first node
   - Standard linked list pattern for building results

3. **Why unique_id in linked list version?**
   - Python's heapq compares entire tuples if first elements are equal
   - ListNode objects aren't comparable by default
   - Using unique_id as second element avoids comparing nodes
   - Alternative: implement `__lt__` method on ListNode

4. **Smallest range algorithm insight:**
   - Current range is `[min_in_heap, max_seen_so_far]`
   - We advance the minimum to try to shrink the range
   - We stop when one list is exhausted (can't maintain coverage)
   - This greedy approach guarantees finding the optimal range

5. **Matrix as K-way merge:**
   - Each row is a sorted list
   - We merge rows just like merging K arrays
   - Optimization: Only initialize first k rows (don't need more for Kth element)

## Complexity Analysis

### Time Complexity

**For merging K sorted arrays/lists:**
- **Initialization:** O(K) to add first element from each list to heap
- **Main loop:** Process N total elements, each heap operation is O(log K)
- **Overall:** O(K + N log K) = **O(N log K)**

**Why N log K?**
1. We process N elements total (from all K lists combined)
2. Each element requires:
   - One heappop: O(log K)
   - One heappush: O(log K)
3. Therefore: N × (log K + log K) = N × 2 log K = O(N log K)

**For Kth smallest:**
- **Only process K elements** instead of N
- **Overall:** **O(K log M)** where M is number of lists
- Significant speedup when K << N

**For smallest range:**
- **Worst case:** Process all N elements
- **Each step:** heappop O(log K) + heappush O(log K)
- **Overall:** **O(N log K)**

### Space Complexity

- **Heap storage:** O(K) - stores at most K elements (one from each list)
- **Result array:** O(N) - stores all merged elements
- **Overall:** **O(K) for heap, O(N) for output**
- **Auxiliary:** O(1) - only a few pointers/variables

**For linked list merge:**
- **Heap:** O(K)
- **Result:** O(1) - we're reusing existing nodes, not creating new ones
- **Overall:** **O(K)**

### Comparison with Alternatives

| Approach | Time Complexity | Space Complexity | When to Use |
|----------|----------------|------------------|-------------|
| **K-way Merge (Heap)** | O(N log K) | O(K) | Standard choice; balanced performance |
| **Merge pairs iteratively** | O(N K) or O(N log K)* | O(N) | When K is very small (K ≤ 4) |
| **Merge all then sort** | O(N log N) | O(N) | Never use! Ignores sorted property |
| **Tournament tree** | O(N log K) | O(K) | Similar to heap; more complex |
| **External merge sort** | O(N log K) | O(B) blocks | For data larger than memory |

*If merging in balanced tree fashion (merge 1+2, 3+4, then results), it's O(N log K)

**When K-way Merge Wins:**
- Multiple sorted inputs to combine
- K is moderate to large (K > 4)
- Want optimal time complexity O(N log K)
- Limited memory (only O(K) needed)

**Example:** Merging 1000 sorted lists of 1000 elements each:
- K-way merge: 1M × log(1000) ≈ 10M operations
- Naive (merge pairs): 1M × 1000 = 1B operations
- **K-way merge is 100× faster!**

## Examples

### Example 1: Merge 3 Sorted Arrays

**Problem:** Merge `[[1,4,7], [2,5,8], [3,6,9]]`

**Solution:**
```python
arrays = [[1,4,7], [2,5,8], [3,6,9]]

# Initialize heap with first element from each
min_heap = [(1,0,0), (2,1,0), (3,2,0)]
result = []

# Step-by-step execution:
# Pop (1,0,0), push (4,0,1) → result = [1]
# Pop (2,1,0), push (5,1,1) → result = [1,2]
# Pop (3,2,0), push (6,2,1) → result = [1,2,3]
# Pop (4,0,1), push (7,0,2) → result = [1,2,3,4]
# Pop (5,1,1), push (8,1,2) → result = [1,2,3,4,5]
# Pop (6,2,1), push (9,2,2) → result = [1,2,3,4,5,6]
# Pop (7,0,2), no more → result = [1,2,3,4,5,6,7]
# Pop (8,1,2), no more → result = [1,2,3,4,5,6,7,8]
# Pop (9,2,2), no more → result = [1,2,3,4,5,6,7,8,9]

# Final: [1,2,3,4,5,6,7,8,9]
```

**Visualization:**
```
Arrays:  [1,4,7]  [2,5,8]  [3,6,9]
          ↓        ↓        ↓
Heap:    [1]      [2]      [3]

Extract 1, add 4:
Arrays:  [1,4,7]  [2,5,8]  [3,6,9]
            ↓      ↓        ↓
Heap:      [2]    [3]      [4]

Result builds: [1] → [1,2] → [1,2,3] → ...
```

### Example 2: Kth Smallest Across M Lists

**Problem:** Find 5th smallest in `[[2,6,8], [3,6,10], [5,8,11]]`

**Solution:**
```python
lists = [[2,6,8], [3,6,10], [5,8,11]]
k = 5

# All elements in order: 2, 3, 5, 6, 6, 8, 8, 10, 11
# 5th element is 6

# Trace:
# Init heap: [(2,0,0), (3,1,0), (5,2,0)]
# 
# Count=1: Pop 2, add 6 from list 0 → heap: [(3,1,0), (5,2,0), (6,0,1)]
# Count=2: Pop 3, add 6 from list 1 → heap: [(5,2,0), (6,0,1), (6,1,1)]
# Count=3: Pop 5, add 8 from list 2 → heap: [(6,0,1), (6,1,1), (8,2,1)]
# Count=4: Pop 6, add 8 from list 0 → heap: [(6,1,1), (8,0,2), (8,2,1)]
# Count=5: Pop 6 ← This is our answer!
#
# Result: 6
```

**Key insight:** We don't need to merge all elements, just extract K of them!

### Example 3: Kth Smallest in Sorted Matrix

**Problem:** Find 8th smallest in matrix:
```
[1,  5,  9]
[10, 11, 13]
[12, 13, 15]
```

**Solution:**
```python
matrix = [[1,5,9], [10,11,13], [12,13,15]]
k = 8

# Elements in order: 1, 5, 9, 10, 11, 12, 13, 13, 15
# 8th is 13

# Trace (treating each row as a sorted list):
# Init: [(1,0,0), (10,1,0), (12,2,0)]
# 
# Extract 1, add 5: [(5,0,1), (10,1,0), (12,2,0)]
# Extract 5, add 9: [(9,0,2), (10,1,0), (12,2,0)]
# Extract 9: [(10,1,0), (12,2,0)]  (row 0 exhausted)
# Extract 10, add 11: [(11,1,1), (12,2,0)]
# Extract 11, add 13: [(12,2,0), (13,1,2)]
# Extract 12, add 13: [(13,1,2), (13,2,1)]
# Extract 13: [(13,2,1)]
# Extract 13: []
#
# 8th element extracted: 13
```

**Matrix visualization:**
```
     col0  col1  col2
row0: [1]   5     9
row1: 10    11    13
row2: 12    13    15
      ↓
Start with first element of each row

After extracting 1, move right in row 0:
     col0  col1  col2
row0: 1    [5]    9
row1: 10    11    13
row2: 12    13    15
```

### Example 4: Smallest Range Covering K Lists

**Problem:** Find smallest range in `[[4,10,15,24,26], [0,9,12,20], [5,18,22,30]]`

**Solution:**
```python
lists = [[4,10,15,24,26], [0,9,12,20], [5,18,22,30]]

# Goal: Find [start, end] that includes ≥1 number from each list

# Trace:
# Init: heap=[(0,1,0), (4,0,0), (5,2,0)], max=5, range=[0,5]
#       (covers: 0 from list1, 4 from list0, 5 from list2)
#
# Pop 0, add 9: heap=[(4,0,0), (5,2,0), (9,1,1)], max=9, range=[4,9]
#               (covers: 4 from list0, 5 from list2, 9 from list1) ✓ Better!
#
# Pop 4, add 10: heap=[(5,2,0), (9,1,1), (10,0,1)], max=10, range=[5,10]
#                (covers: 5 from list2, 9 from list1, 10 from list0) ✓ Better!
#
# Pop 5, add 18: heap=[(9,1,1), (10,0,1), (18,2,1)], max=18, range=[9,18]
#                (covers: 9 from list1, 10 from list0, 18 from list2)
#
# Pop 9, add 12: heap=[(10,0,1), (12,1,2), (18,2,1)], max=18, range=[10,18]
#
# Pop 10, add 15: heap=[(12,1,2), (15,0,2), (18,2,1)], max=18, range=[12,18]
#
# Pop 12, add 20: heap=[(15,0,2), (18,2,1), (20,1,3)], max=20, range=[15,20]
#
# Pop 15, add 24: heap=[(18,2,1), (20,1,3), (24,0,3)], max=24, range=[18,24]
#
# Pop 18, add 22: heap=[(20,1,3), (22,2,2), (24,0,3)], max=24, range=[20,24]
#                 (covers: 20 from list1, 22 from list2, 24 from list0) ✓ Better!
#
# Pop 20, no more in list1 → STOP (can't maintain coverage)
#
# Best range: [20, 24]
```

**Why [20,24] is optimal:**
```
List 0: 4  10  15  [24] 26
List 1: 0   9  12  [20]
List 2: 5  18  [22] 30
              └─┬─┘
          Range [20,24] has length 4

Any other range including all lists is larger:
[18,24] = length 6
[4,30] = length 26
etc.
```

## Edge Cases

### 1. Empty Input Lists
**Scenario:** Some or all input lists are empty

**Challenge:** Can't initialize heap with empty lists; heap might become empty mid-process.

**Solution:** Filter out empty lists before processing.

```python
def merge_k_sorted_arrays(arrays: List[List[int]]) -> List[int]:
    # Filter empty arrays
    arrays = [arr for arr in arrays if arr]
    
    if not arrays:
        return []
    
    # Now proceed with algorithm
```

### 2. Single List Input
**Scenario:** K=1, only one sorted list

**Challenge:** Heap operations are wasteful for single list.

**Solution:** Return the list directly.

```python
def merge_k_sorted_arrays(arrays: List[List[int]]) -> List[int]:
    if len(arrays) == 1:
        return arrays[0][:]  # Return copy
    
    # Otherwise use heap
```

### 3. Lists of Vastly Different Lengths
**Scenario:** One list has 1M elements, others have 10 each

**Challenge:** After short lists are exhausted, we're essentially copying one long list.

**Solution:** Algorithm handles naturally, but consider switching to direct copy when K=1 lists remain.

```python
# When heap size drops to 1:
if len(min_heap) == 1:
    # Just append remaining elements from last list
    _, list_idx, element_idx = min_heap[0]
    result.extend(arrays[list_idx][element_idx:])
    break
```

### 4. Duplicate Values Across Lists
**Scenario:** Same value appears in multiple lists

**Challenge:** Need stable merge (maintain relative order from same source).

**Solution:** Use (value, list_index) tuples; Python's heapq is stable for equal values.

```python
# Heap naturally handles:
# (5, 0, ...) comes before (5, 1, ...)
# So list 0's 5 is processed before list 1's 5
```

### 5. All Elements Equal
**Scenario:** All K lists contain only the same value

**Challenge:** Every heap extraction yields the same value.

**Solution:** Algorithm works correctly, just extracts N identical values.

```python
# Example: [[5,5,5], [5,5], [5,5,5,5]]
# Result: [5,5,5,5,5,5,5,5,5]
# Perfectly valid!
```

### 6. Negative Numbers
**Scenario:** Lists contain negative values

**Challenge:** Ensure comparison logic works.

**Solution:** Min-heap comparison works identically for negatives.

```python
# Example: [[-10,-5,-1], [-8,-3,0]]
# Min-heap correctly orders: -10 < -8 < -5 < -3 < -1 < 0
```

### 7. Kth Element Beyond Total Count
**Scenario:** `k > N` (asking for 100th element when only 50 exist)

**Challenge:** Can't find element that doesn't exist.

**Solution:** Return -1 or None to indicate invalid K.

```python
def kth_smallest(lists, k):
    total_elements = sum(len(lst) for lst in lists)
    if k > total_elements:
        return -1  # or raise ValueError
    
    # Proceed with algorithm
```

### 8. Matrix Not Square
**Scenario:** Matrix is M × N where M ≠ N

**Challenge:** Algorithm assumes we can iterate rows and columns.

**Solution:** Works fine; adjust column bound check.

```python
def kth_smallest_matrix(matrix, k):
    rows, cols = len(matrix), len(matrix[0])
    
    # When adding next in row, check column bound
    if col + 1 < cols:  # Use cols, not rows
        heapq.heappush(heap, ...)
```

## Common Pitfalls

### ❌ Pitfall 1: Forgetting Unique ID for Linked List Heap

**What happens:**
```python
# WRONG: This crashes when values are equal!
min_heap = []
for node in lists:
    if node:
        heapq.heappush(min_heap, (node.val, node))

# Error: '<' not supported between ListNode instances
```

**Why it's wrong:** When two nodes have equal values, Python tries to compare the second tuple element (the nodes themselves). ListNode doesn't implement `__lt__`.

**Correct approach:**
```python
# CORRECT: Use unique ID as tiebreaker
min_heap = []
for i, node in enumerate(lists):
    if node:
        heapq.heappush(min_heap, (node.val, i, node))
        #                                  ↑ prevents comparing nodes
```

### ❌ Pitfall 2: Not Checking if List Has More Elements

**What happens:**
```python
# WRONG: IndexError when list is exhausted
while min_heap:
    value, idx, pos = heapq.heappop(min_heap)
    result.append(value)
    
    # Crashes if pos+1 is out of bounds!
    heapq.heappush(min_heap, (arrays[idx][pos + 1], idx, pos + 1))
```

**Why it's wrong:** When a list is exhausted, trying to access next element causes IndexError.

**Correct approach:**
```python
# CORRECT: Check bounds before adding
while min_heap:
    value, idx, pos = heapq.heappop(min_heap)
    result.append(value)
    
    if pos + 1 < len(arrays[idx]):  # Check if more elements exist
        heapq.heappush(min_heap, (arrays[idx][pos + 1], idx, pos + 1))
```

### ❌ Pitfall 3: Initializing Heap with All Elements

**What happens:**
```python
# WRONG: Defeats the purpose of K-way merge!
min_heap = []
for array in arrays:
    for value in array:  # Adding ALL elements!
        heapq.heappush(min_heap, value)

result = []
while min_heap:
    result.append(heapq.heappop(min_heap))
```

**Why it's wrong:** 
- Time complexity becomes O(N log N) instead of O(N log K)
- Space complexity becomes O(N) instead of O(K)
- You're just doing heapsort, not K-way merge!

**Correct approach:** Only maintain K elements in heap (one from each list).

### ❌ Pitfall 4: Not Handling Empty Input Lists

**What happens:**
```python
# WRONG: Crashes on empty lists
for i in range(len(arrays)):
    # Assumes arrays[i] has at least one element!
    heapq.heappush(min_heap, (arrays[i][0], i, 0))
```

**Why it's wrong:** If `arrays[i]` is empty, `arrays[i][0]` raises IndexError.

**Correct approach:**
```python
# CORRECT: Check if array is non-empty
for i in range(len(arrays)):
    if arrays[i]:  # Only add if non-empty
        heapq.heappush(min_heap, (arrays[i][0], i, 0))
```

### ❌ Pitfall 5: Losing Track of Which List an Element Came From

**What happens:**
```python
# WRONG: Can't refill from correct list!
min_heap = []
for array in arrays:
    if array:
        heapq.heappush(min_heap, array[0])  # Just the value!

while min_heap:
    value = heapq.heappop(min_heap)
    result.append(value)
    # How do we know which array to refill from? 🤔
```

**Why it's wrong:** Without tracking source, we don't know which list to get the next element from.

**Correct approach:** Store (value, list_index, position) tuples.

### ❌ Pitfall 6: Using Wrong Data Structure

**What happens:**
```python
# WRONG: Using sorted list instead of heap
import bisect

sorted_elements = []
# Maintaining sorted list with bisect.insort
# Each insertion: O(n) to shift elements
# Total: O(n²) instead of O(n log k)
```

**Why it's wrong:** Sorted list has O(N) insertion time; heap has O(log K). For large N, this is catastrophically slow.

**Correct approach:** Use heap (priority queue) with O(log K) operations.

### ❌ Pitfall 7: Modifying Original Lists

**What happens:**
```python
# WRONG: Mutating input
def merge_k_sorted_arrays(arrays):
    for array in arrays:
        array.pop(0)  # Modifying input!
```

**Why it's wrong:**
- Violates principle of not modifying inputs
- `pop(0)` on list is O(n) operation
- Caller's data is destroyed

**Correct approach:** Use indices to track position, don't modify original lists.

## Variations and Extensions

### Variation 1: Divide and Conquer Merge

**Description:** Merge K lists by repeatedly merging pairs (like merge sort).

**When to use:** When K is very small (K ≤ 4); when heap overhead is significant.

**Key differences:**
- Time: O(N log K) same as heap, but with different constants
- Simpler implementation for small K
- Better cache locality (sequential merges)

**Implementation:**
```python
def merge_two_lists(l1: List[int], l2: List[int]) -> List[int]:
    """Merge two sorted lists."""
    result = []
    i, j = 0, 0
    
    while i < len(l1) and j < len(l2):
        if l1[i] <= l2[j]:
            result.append(l1[i])
            i += 1
        else:
            result.append(l2[j])
            j += 1
    
    result.extend(l1[i:])
    result.extend(l2[j:])
    return result

def merge_k_lists_divide_conquer(lists: List[List[int]]) -> List[int]:
    """
    Merge K lists using divide and conquer.
    
    Time: O(N log K)
    Space: O(N) for intermediate results
    """
    if not lists:
        return []
    if len(lists) == 1:
        return lists[0]
    
    while len(lists) > 1:
        merged = []
        
        # Merge pairs
        for i in range(0, len(lists), 2):
            if i + 1 < len(lists):
                merged.append(merge_two_lists(lists[i], lists[i + 1]))
            else:
                merged.append(lists[i])
        
        lists = merged
    
    return lists[0]
```

### Variation 2: External K-way Merge (Disk-based)

**Description:** Merge sorted files larger than RAM using disk I/O.

**When to use:** Big data processing; sorting data that doesn't fit in memory.

**Key differences:**
- Reads/writes to disk in blocks
- Minimizes I/O operations
- Used in external merge sort

**Implementation concept:**
```python
def external_k_way_merge(file_paths: List[str], output_path: str, buffer_size: int):
    """
    Merge K sorted files using limited memory.
    
    Args:
        file_paths: Paths to K sorted input files
        output_path: Where to write merged output
        buffer_size: Size of input buffer per file (in bytes)
    """
    # Open all input files
    input_files = [open(path, 'r') for path in file_paths]
    output_file = open(output_path, 'w')
    
    # Min-heap: (value, file_index)
    min_heap = []
    
    # Read first value from each file
    for i, f in enumerate(input_files):
        line = f.readline()
        if line:
            heapq.heappush(min_heap, (int(line.strip()), i))
    
    # Merge using heap
    while min_heap:
        value, file_idx = heapq.heappop(min_heap)
        output_file.write(f"{value}\n")
        
        # Read next from same file
        line = input_files[file_idx].readline()
        if line:
            heapq.heappush(min_heap, (int(line.strip()), file_idx))
    
    # Cleanup
    for f in input_files:
        f.close()
    output_file.close()
```

### Variation 3: K Pairs with Largest Sum

**Description:** Find K pairs (one from each of two arrays) with largest sums.

**When to use:** Combination problems; optimization problems.

**Implementation:**
```python
def k_pairs_largest_sums(nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
    """
    Find K pairs with largest sums.
    
    Example:
        nums1 = [1,7,11], nums2 = [2,4,6], k = 3
        Output: [[11,6], [7,6], [11,4]]
        
    Time: O(K log K)
    """
    if not nums1 or not nums2:
        return []
    
    # Max-heap: (-sum, i, j)
    max_heap = []
    
    # Start with largest from each array
    heapq.heappush(max_heap, (-(nums1[-1] + nums2[-1]), len(nums1)-1, len(nums2)-1))
    visited = {(len(nums1)-1, len(nums2)-1)}
    
    result = []
    
    while max_heap and len(result) < k:
        neg_sum, i, j = heapq.heappop(max_heap)
        result.append([nums1[i], nums2[j]])
        
        # Add adjacent pairs
        if i - 1 >= 0 and (i-1, j) not in visited:
            heapq.heappush(max_heap, (-(nums1[i-1] + nums2[j]), i-1, j))
            visited.add((i-1, j))
        
        if j - 1 >= 0 and (i, j-1) not in visited:
            heapq.heappush(max_heap, (-(nums1[i] + nums2[j-1]), i, j-1))
            visited.add((i, j-1))
    
    return result
```

### Variation 4: Merge Intervals from K Lists

**Description:** Merge overlapping intervals from K sorted interval lists.

**When to use:** Calendar scheduling; resource allocation; time-series data.

**Implementation:**
```python
def merge_k_interval_lists(interval_lists: List[List[List[int]]]) -> List[List[int]]:
    """
    Merge K sorted interval lists.
    
    Example:
        [[1,3],[5,7]], [[2,4],[6,8]]
        → [[1,4],[5,8]]
    """
    # First, merge all intervals into one sorted list
    merged_list = merge_k_sorted_arrays(interval_lists)
    
    # Then merge overlapping intervals
    result = []
    for interval in merged_list:
        if not result or result[-1][1] < interval[0]:
            result.append(interval)
        else:
            result[-1][1] = max(result[-1][1], interval[1])
    
    return result
```

## Practice Problems

### Beginner

1. **Merge Two Sorted Lists** - Start with merging just 2 lists
   - LeetCode #21 (Merge Two Sorted Lists)
   - Foundation for K-way merge

2. **Merge Sorted Array** - Merge two sorted arrays in-place
   - LeetCode #88 (Merge Sorted Array)
   - Practice two-pointer technique

3. **Find Kth Smallest Element in Two Sorted Arrays** - K=2 case
   - Simpler version of K-way merge
   - Build intuition for the pattern

### Intermediate

1. **Merge k Sorted Lists** - Classic K-way merge problem
   - LeetCode #23 (Merge k Sorted Lists)
   - THE problem to master this pattern

2. **Kth Smallest Element in a Sorted Matrix** - Each row is a sorted list
   - LeetCode #378 (Kth Smallest Element in a Sorted Matrix)
   - Combine K-way merge with early termination

3. **Find K Pairs with Smallest Sums** - Generate pairs from two arrays
   - LeetCode #373 (Find K Pairs with Smallest Sums)
   - Variation with pair generation

4. **Ugly Number II** - Generate sequence using K-way merge concept
   - LeetCode #264 (Ugly Number II)
   - Merge three sequences (multiples of 2, 3, 5)

5. **Smallest Range Covering Elements from K Lists** - Find minimal range
   - LeetCode #632 (Smallest Range Covering Elements from K Lists)
   - Advanced K-way merge application

### Advanced

1. **Median of Two Sorted Arrays** - Find median in O(log(m+n))
   - LeetCode #4 (Median of Two Sorted Arrays)
   - Binary search + merge concept

2. **Merge k Sorted Interval Lists** - Merge intervals from K lists
   - Combine K-way merge with interval merging
   - Calendar/scheduling applications

3. **Find Median from Data Stream with K Streams** - Multiple data streams
   - Extension of LeetCode #295
   - Real-time K-way merge

4. **Super Ugly Number** - Generalize ugly number to N primes
   - LeetCode #313 (Super Ugly Number)
   - K-way merge with K prime factors

5. **Merge Large Files** - External merge sort simulation
   - Practice disk I/O patterns
   - Real-world big data problem

## Real-World Applications

### Industry Use Cases

1. **Database Query Processing**
   - **How it's used:** Merging results from multiple sorted indexes or shards
   - **Why it's effective:** Each database shard returns sorted results; K-way merge combines them efficiently
   - **Scale:** Distributed databases like Cassandra use this for range queries across nodes

2. **Log Aggregation & Analysis**
   - **How it's used:** Merging timestamp-sorted logs from multiple servers
   - **Why it's effective:** Each server's logs are chronologically sorted; merge for global timeline
   - **Tools:** Splunk, ELK Stack use K-way merge for distributed log search

3. **External Sorting**
   - **How it's used:** Sorting files larger than available RAM
   - **Why it's effective:** Split file into sorted chunks, merge using limited memory
   - **Applications:** MapReduce shuffle phase, sorting terabyte-scale datasets

4. **Search Engine Result Merging**
   - **How it's used:** Google merges results from multiple index servers
   - **Why it's effective:** Each server returns top results sorted by relevance; merge for final ranking
   - **Scale:** Thousands of index servers contributing to each search

5. **Time-Series Data Processing**
   - **How it's used:** Merging sensor data streams from multiple IoT devices
   - **Why it's effective:** Each device produces time-sorted data; merge for global view
   - **Applications:** Industrial monitoring, financial tick data

6. **Video Streaming CDN**
   - **How it's used:** Merging sorted quality level manifests from multiple CDN nodes
   - **Why it's effective:** Select optimal video chunks from K servers
   - **Example:** Netflix, YouTube adaptive bitrate streaming

### Popular Implementations

- **Unix `sort` command:** Uses external K-way merge for files larger than memory
- **Apache Hadoop/Spark:** Shuffle phase uses K-way merge for distributed sorting
- **SQLite:** Merge sort for ORDER BY queries uses K-way merge
- **PostgreSQL:** Multi-way merge for index scans and parallel query execution
- **Lucene/Elasticsearch:** Merging inverted index segments uses K-way merge
- **RocksDB:** LSM tree compaction merges sorted runs using K-way merge
- **Python `heapq.merge()`:** Built-in K-way merge iterator for sorted iterables

### Practical Scenarios

- **Multi-tenant SaaS:** Merge per-tenant sorted data for global leaderboard
- **Genomics:** Merge sorted DNA sequence alignments from parallel processing
- **Financial Trading:** Merge order books from multiple exchanges
- **Distributed Caching:** Merge sorted keys from multiple cache nodes for range queries
- **Version Control:** Merge commits from multiple branches (Git merge-base)
- **Data Warehousing:** Merge sorted fact tables from different time partitions
- **Real-time Analytics:** Merge pre-aggregated results from multiple time windows
- **Machine Learning:** Merge sorted feature vectors from distributed feature engineering

## Related Topics

### Prerequisites to Review

- **Heaps (Priority Queues)** - Absolute must; understand heapify, push, pop operations
- **Merge Sort** - K-way merge is a generalization of 2-way merge
- **Linked Lists** - Many problems involve merging sorted linked lists
- **Two Pointers** - Foundation for merging two sorted arrays

### Next Steps

- **Top K Elements Pattern** - Uses heaps similarly for selection problems
- **Two Heaps Pattern** - Extend to using min-heap + max-heap together
- **External Sorting** - Apply K-way merge to disk-based sorting
- **Parallel Algorithms** - K-way merge in multi-threaded/distributed context
- **Streaming Algorithms** - Merge infinite sorted streams

### Similar Concepts

- **Tournament Tree** - Alternative to heap for K-way selection
- **Binary Merge** - Special case where K=2
- **N-way Merge Join** - Database join algorithm using K-way merge
- **Multi-way Quicksort** - Partitioning K ways instead of 2
- **LSM Trees** - Use K-way merge for compaction in databases

### Further Reading

- **"Introduction to Algorithms" (CLRS)** - Chapter 6 (Heapsort) and external sorting
  - Deep dive into heap-based algorithms
  
- **"Database System Concepts" by Silberschatz** - Chapter on query processing
  - How databases use K-way merge for sorting and joins
  
- **LeetCode Problems:**
  - #23 Merge k Sorted Lists (Must solve!)
  - #378 Kth Smallest Element in Sorted Matrix
  - #632 Smallest Range Covering Elements from K Lists
  
- **System Design:**
  - "Designing Data-Intensive Applications" by Martin Kleppmann
  - Chapter on sorted string tables (SSTables) and LSM trees
  
- **External Sorting:**
  - "The Art of Computer Programming Vol. 3" by Knuth
  - Comprehensive coverage of multi-way merge sorting
  
- **Distributed Systems:**
  - Google's MapReduce paper - describes shuffle/merge phase
  - Apache Spark documentation on sorting and shuffling
