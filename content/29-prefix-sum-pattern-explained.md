# Prefix Sum Pattern

**Difficulty:** Beginner to Intermediate
**Prerequisites:** Arrays, Basic arithmetic, Hash maps
**Estimated Reading Time:** 35 minutes

## Introduction

The Prefix Sum pattern is a technique for preprocessing an array to enable constant-time range sum queries. A prefix sum array stores the cumulative sum of elements from the start of the array up to each index, allowing us to calculate the sum of any subarray in O(1) time after O(n) preprocessing.

**Why it matters:** Prefix sums are fundamental in competitive programming and technical interviews. They appear in problems involving subarray sums, range queries, 2D matrix operations, and optimization problems. The pattern transforms O(n) range queries into O(1) operations, making it essential for solving problems with multiple queries efficiently. Companies like Google, Amazon, and Facebook frequently test understanding of prefix sums in interviews.

**Real-world analogy:** Imagine you're reading a book and want to know how many pages are in any chapter range. Instead of counting pages every time, you create a page counter at the end of each chapter showing total pages up to that point. Chapter 1 ends at page 50, Chapter 2 at page 120, Chapter 3 at page 200, etc. Now, to find how many pages are in Chapters 2-3, you simply calculate: pages at end of Chapter 3 (200) minus pages at end of Chapter 1 (50) = 150 pages. This is exactly how prefix sums work!

## Core Concepts

### Key Principles

1. **Cumulative Storage:** Store running totals rather than individual values, enabling range queries.

2. **Range Calculation:** Sum of range [i, j] = prefix[j] - prefix[i-1] (handle boundaries carefully).

3. **Constant Time Queries:** After O(n) preprocessing, any range sum query takes O(1) time.

4. **Space-Time Tradeoff:** Use O(n) extra space to achieve O(1) query time.

5. **Extension to 2D:** Prefix sums work on matrices too, enabling rectangle sum queries in O(1).

### Essential Terms

- **Prefix Sum Array:** Array where element at index i stores sum of all elements from 0 to i
- **Range Sum:** Sum of elements in a subarray [i, j]
- **Cumulative Sum:** Running total up to a position
- **Subarray:** Contiguous portion of array
- **Hash Map Optimization:** Using hash maps to find subarrays with specific sums
- **2D Prefix Sum:** Extension to matrices for rectangle queries

### Visual Overview

```
Original array: [3, 1, 4, 1, 5, 9, 2]
Index:           0  1  2  3  4  5  6

Prefix sum:     [3, 4, 8, 9, 14, 23, 25]
Calculation:
prefix[0] = 3
prefix[1] = 3 + 1 = 4
prefix[2] = 3 + 1 + 4 = 8
prefix[3] = 3 + 1 + 4 + 1 = 9
...

Range sum query: sum[2, 5] (elements 4+1+5+9)
Formula: prefix[5] - prefix[1] = 23 - 4 = 19 ✓
Verification: 4 + 1 + 5 + 9 = 19 ✓

Visual:
[3, 1, | 4, 1, 5, 9, | 2]
       ^              ^
    prefix[1]=4   prefix[5]=23
    
Range sum = 23 - 4 = 19
```

## How It Works

### Basic Algorithm

**Step 1: Build Prefix Sum Array**
- Initialize array of same length
- First element equals first element of original
- Each subsequent element = previous prefix + current element

**Step 2: Query Range Sum**
- For range [i, j]: return prefix[j] - prefix[i-1]
- Handle boundary: if i=0, return prefix[j]

**Step 3: Optimization with Hash Map**
- For finding subarrays with target sum
- Store prefix sums in hash map with indices
- Check if (current_prefix - target) exists in map

### Detailed Walkthrough Example

**Problem:** Count subarrays with sum equal to k
**Input:** nums = [1, 2, 3], k = 3
**Output:** 2 (subarrays [1,2] and [3])

```
Build prefix sums and use hash map:

Initial state:
nums = [1, 2, 3]
k = 3
prefix_count = {0: 1}  # sum 0 appears once (before array)
count = 0
current_sum = 0

Step 1: Process nums[0] = 1
  current_sum = 0 + 1 = 1
  target = current_sum - k = 1 - 3 = -2
  Check if -2 in prefix_count? No
  count = 0
  Add current_sum to map: prefix_count = {0: 1, 1: 1}

Step 2: Process nums[1] = 2
  current_sum = 1 + 2 = 3
  target = current_sum - k = 3 - 3 = 0
  Check if 0 in prefix_count? Yes! count += prefix_count[0] = 1
  count = 1
  Add current_sum to map: prefix_count = {0: 1, 1: 1, 3: 1}
  
  Found subarray: [1, 2] with sum 3 ✓

Step 3: Process nums[2] = 3
  current_sum = 3 + 3 = 6
  target = current_sum - k = 6 - 3 = 3
  Check if 3 in prefix_count? Yes! count += prefix_count[3] = 1
  count = 2
  Add current_sum to map: prefix_count = {0: 1, 1: 1, 3: 1, 6: 1}
  
  Found subarray: [3] with sum 3 ✓

Final answer: 2 subarrays ✓

Explanation:
When current_sum - k appears in map, it means there's a 
subarray ending at current position with sum k.

Example: At index 2, current_sum = 6, k = 3
Looking for prefix_sum = 3 means:
  sum[0 to 2] - sum[0 to i] = k
  6 - 3 = 3 ✓
  This represents subarray from i+1 to 2, which is [3]
```

## Implementation

### Python Implementation - Comprehensive Prefix Sum Solutions

```python
from typing import List, Dict
from collections import defaultdict

class PrefixSum:
    """
    Basic prefix sum implementation for range queries.
    
    Time Complexity:
        - Build: O(n)
        - Query: O(1)
    Space Complexity: O(n)
    """
    
    def __init__(self, nums: List[int]):
        """
        Build prefix sum array.
        
        Args:
            nums: Input array
        """
        self.prefix = [0] * (len(nums) + 1)  # Extra element for easier boundary handling
        for i in range(len(nums)):
            self.prefix[i + 1] = self.prefix[i] + nums[i]
    
    def range_sum(self, left: int, right: int) -> int:
        """
        Get sum of elements in range [left, right] inclusive.
        
        Args:
            left: Left boundary (0-indexed)
            right: Right boundary (0-indexed)
            
        Returns:
            Sum of elements in range
            
        Time: O(1)
        
        Example:
            >>> ps = PrefixSum([1, 2, 3, 4, 5])
            >>> ps.range_sum(1, 3)
            9  # 2 + 3 + 4
        """
        return self.prefix[right + 1] - self.prefix[left]


def find_middle_index(nums: List[int]) -> int:
    """
    Find index where left sum equals right sum.
    
    LeetCode #1991: Find the Middle Index in Array
    
    Args:
        nums: Array of integers
        
    Returns:
        Leftmost middle index, or -1 if none exists
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Example:
        >>> find_middle_index([2,3,-1,8,4])
        3  # left sum = 2+3+(-1) = 4, right sum = 4
    """
    total_sum = sum(nums)
    left_sum = 0
    
    for i in range(len(nums)):
        # Right sum = total - left - current
        right_sum = total_sum - left_sum - nums[i]
        
        if left_sum == right_sum:
            return i
        
        left_sum += nums[i]
    
    return -1


def left_right_difference(nums: List[int]) -> List[int]:
    """
    For each index, calculate absolute difference between left and right sums.
    
    LeetCode #2574: Left and Right Sum Differences
    
    Args:
        nums: Array of integers
        
    Returns:
        Array of absolute differences
        
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Example:
        >>> left_right_difference([10,4,8,3])
        [15,1,11,22]
    """
    n = len(nums)
    left_sum = [0] * n
    right_sum = [0] * n
    
    # Build left sum array
    for i in range(1, n):
        left_sum[i] = left_sum[i - 1] + nums[i - 1]
    
    # Build right sum array
    for i in range(n - 2, -1, -1):
        right_sum[i] = right_sum[i + 1] + nums[i + 1]
    
    # Calculate differences
    return [abs(left_sum[i] - right_sum[i]) for i in range(n)]


def max_size_subarray_sum_equals_k(nums: List[int], k: int) -> int:
    """
    Find length of longest subarray with sum equal to k.
    
    LeetCode #325: Maximum Size Subarray Sum Equals k
    
    Uses hash map to store first occurrence of each prefix sum.
    
    Args:
        nums: Array of integers
        k: Target sum
        
    Returns:
        Length of longest subarray, or 0 if none exists
        
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Example:
        >>> max_size_subarray_sum_equals_k([1,-1,5,-2,3], 3)
        4  # [1,-1,5,-2]
    """
    prefix_sum = {0: -1}  # Map prefix sum -> earliest index
    current_sum = 0
    max_length = 0
    
    for i in range(len(nums)):
        current_sum += nums[i]
        
        # Check if there's a prefix sum such that current - prefix = k
        target = current_sum - k
        if target in prefix_sum:
            length = i - prefix_sum[target]
            max_length = max(max_length, length)
        
        # Only store first occurrence for maximum length
        if current_sum not in prefix_sum:
            prefix_sum[current_sum] = i
    
    return max_length


def num_subarrays_with_sum(nums: List[int], goal: int) -> int:
    """
    Count subarrays with sum equal to goal (binary array).
    
    LeetCode #930: Binary Subarrays With Sum
    
    Args:
        nums: Binary array (only 0s and 1s)
        goal: Target sum
        
    Returns:
        Count of subarrays with sum equal to goal
        
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Example:
        >>> num_subarrays_with_sum([1,0,1,0,1], 2)
        4  # [1,0,1], [1,0,1,0], [0,1,0,1], [1,0,1]
    """
    prefix_count = defaultdict(int)
    prefix_count[0] = 1  # Empty prefix
    
    current_sum = 0
    count = 0
    
    for num in nums:
        current_sum += num
        
        # Find how many prefix sums = current_sum - goal
        target = current_sum - goal
        count += prefix_count[target]
        
        prefix_count[current_sum] += 1
    
    return count


def subarray_sum_divisible_by_k(nums: List[int], k: int) -> int:
    """
    Count subarrays with sum divisible by k.
    
    LeetCode #974: Subarray Sums Divisible by K
    
    Key insight: If prefix_sum[i] % k == prefix_sum[j] % k,
    then sum(nums[i+1:j+1]) is divisible by k.
    
    Args:
        nums: Array of integers
        k: Divisor
        
    Returns:
        Count of subarrays with sum divisible by k
        
    Time Complexity: O(n)
    Space Complexity: O(k)
    
    Example:
        >>> subarray_sum_divisible_by_k([4,5,0,-2,-3,1], 5)
        7
    """
    # Map remainder -> count
    remainder_count = defaultdict(int)
    remainder_count[0] = 1  # Empty prefix has remainder 0
    
    current_sum = 0
    count = 0
    
    for num in nums:
        current_sum += num
        remainder = current_sum % k
        
        # In Python, -3 % 5 = 2 (always positive)
        # But to be safe: remainder = (remainder % k + k) % k
        
        # If this remainder seen before, all those positions
        # form valid subarrays with current position
        count += remainder_count[remainder]
        
        remainder_count[remainder] += 1
    
    return count


def sum_of_absolute_differences(nums: List[int]) -> List[int]:
    """
    For each element, sum of absolute differences with all other elements.
    
    LeetCode #1685: Sum of Absolute Differences in a Sorted Array
    
    For sorted array, can use prefix sums cleverly:
    For position i:
      - Elements to left: all are smaller
      - Elements to right: all are larger
    
    Args:
        nums: Sorted array
        
    Returns:
        Array of absolute difference sums
        
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Example:
        >>> sum_of_absolute_differences([2,3,5])
        [4,3,5]  # |2-3|+|2-5|=4, |3-2|+|3-5|=3, |5-2|+|5-3|=5
    """
    n = len(nums)
    
    # Build prefix sum
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]
    
    result = []
    
    for i in range(n):
        # Sum of differences with elements to the left
        # Each element to left contributes: nums[i] - nums[j]
        # Sum = i * nums[i] - sum(nums[0:i])
        left_sum = i * nums[i] - prefix[i]
        
        # Sum of differences with elements to the right
        # Each element to right contributes: nums[j] - nums[i]
        # Sum = sum(nums[i+1:n]) - (n - i - 1) * nums[i]
        right_sum = (prefix[n] - prefix[i + 1]) - (n - i - 1) * nums[i]
        
        result.append(left_sum + right_sum)
    
    return result


def subarray_sum_equals_k(nums: List[int], k: int) -> int:
    """
    Count subarrays with sum equal to k.
    
    LeetCode #560: Subarray Sum Equals K
    
    Classic prefix sum + hash map problem.
    
    Args:
        nums: Array of integers
        k: Target sum
        
    Returns:
        Count of subarrays
        
    Time Complexity: O(n)
    Space Complexity: O(n)
    
    Example:
        >>> subarray_sum_equals_k([1,1,1], 2)
        2  # [1,1] appears twice
    """
    prefix_count = defaultdict(int)
    prefix_count[0] = 1
    
    current_sum = 0
    count = 0
    
    for num in nums:
        current_sum += num
        
        # If current_sum - k exists, those positions form valid subarrays
        target = current_sum - k
        count += prefix_count[target]
        
        prefix_count[current_sum] += 1
    
    return count


class NumMatrix:
    """
    2D prefix sum for rectangle sum queries.
    
    LeetCode #304: Range Sum Query 2D - Immutable
    
    Time Complexity:
        - Build: O(m * n)
        - Query: O(1)
    Space Complexity: O(m * n)
    """
    
    def __init__(self, matrix: List[List[int]]):
        """
        Build 2D prefix sum.
        
        prefix[i][j] = sum of all elements in rectangle (0,0) to (i-1,j-1)
        """
        if not matrix or not matrix[0]:
            self.prefix = [[]]
            return
        
        m, n = len(matrix), len(matrix[0])
        self.prefix = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Inclusion-exclusion principle
                self.prefix[i][j] = (
                    matrix[i - 1][j - 1] +
                    self.prefix[i - 1][j] +
                    self.prefix[i][j - 1] -
                    self.prefix[i - 1][j - 1]
                )
    
    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        """
        Sum of rectangle from (row1, col1) to (row2, col2) inclusive.
        
        Uses inclusion-exclusion:
        result = total - top - left + top_left_overlap
        
        Example:
            >>> matrix = [[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]]
            >>> nm = NumMatrix(matrix)
            >>> nm.sumRegion(2, 1, 4, 3)
            8
        """
        r1, c1, r2, c2 = row1 + 1, col1 + 1, row2 + 1, col2 + 1
        
        return (
            self.prefix[r2][c2] -
            self.prefix[r1 - 1][c2] -
            self.prefix[r2][c1 - 1] +
            self.prefix[r1 - 1][c1 - 1]
        )


def ways_to_split_array(nums: List[int]) -> int:
    """
    Count valid ways to split array where left sum >= right sum.
    
    LeetCode #2270: Number of Ways to Split Array
    
    Args:
        nums: Array of integers
        
    Returns:
        Count of valid split positions
        
    Time Complexity: O(n)
    Space Complexity: O(1)
    
    Example:
        >>> ways_to_split_array([10,4,-8,7])
        2  # Split at index 0 or 1
    """
    total = sum(nums)
    left_sum = 0
    count = 0
    
    # Can split at positions 0 to n-2 (need at least 1 element on right)
    for i in range(len(nums) - 1):
        left_sum += nums[i]
        right_sum = total - left_sum
        
        if left_sum >= right_sum:
            count += 1
    
    return count


# Example usage and testing
if __name__ == "__main__":
    print("=== Basic Prefix Sum ===")
    ps = PrefixSum([1, 2, 3, 4, 5])
    print(f"Array: [1, 2, 3, 4, 5]")
    print(f"Range sum [1, 3]: {ps.range_sum(1, 3)}")  # 2+3+4 = 9
    print(f"Range sum [0, 4]: {ps.range_sum(0, 4)}")  # 1+2+3+4+5 = 15
    print()
    
    print("=== Find Middle Index ===")
    nums = [2, 3, -1, 8, 4]
    print(f"Array: {nums}")
    print(f"Middle index: {find_middle_index(nums)}")  # 3
    print()
    
    print("=== Left Right Difference ===")
    nums = [10, 4, 8, 3]
    print(f"Array: {nums}")
    print(f"Differences: {left_right_difference(nums)}")
    print()
    
    print("=== Max Size Subarray Sum = k ===")
    nums = [1, -1, 5, -2, 3]
    k = 3
    print(f"Array: {nums}, k={k}")
    print(f"Max length: {max_size_subarray_sum_equals_k(nums, k)}")
    print()
    
    print("=== Binary Subarrays With Sum ===")
    nums = [1, 0, 1, 0, 1]
    goal = 2
    print(f"Array: {nums}, goal={goal}")
    print(f"Count: {num_subarrays_with_sum(nums, goal)}")
    print()
    
    print("=== Subarray Sums Divisible by K ===")
    nums = [4, 5, 0, -2, -3, 1]
    k = 5
    print(f"Array: {nums}, k={k}")
    print(f"Count: {subarray_sum_divisible_by_k(nums, k)}")
    print()
    
    print("=== Sum of Absolute Differences ===")
    nums = [2, 3, 5]
    print(f"Array: {nums}")
    print(f"Results: {sum_of_absolute_differences(nums)}")
    print()
    
    print("=== Subarray Sum Equals K ===")
    nums = [1, 1, 1]
    k = 2
    print(f"Array: {nums}, k={k}")
    print(f"Count: {subarray_sum_equals_k(nums, k)}")
    print()
    
    print("=== 2D Matrix Range Sum ===")
    matrix = [
        [3, 0, 1, 4, 2],
        [5, 6, 3, 2, 1],
        [1, 2, 0, 1, 5],
        [4, 1, 0, 1, 7],
        [1, 0, 3, 0, 5]
    ]
    nm = NumMatrix(matrix)
    print(f"Sum of region (2,1) to (4,3): {nm.sumRegion(2, 1, 4, 3)}")
```

### Code Explanation

**Basic PrefixSum Class:**
- Stores cumulative sums with extra element at start for easier boundary handling
- `range_sum()` uses simple subtraction for O(1) queries

**Find Middle Index:**
- Maintains left sum while iterating
- Calculates right sum as total - left - current
- Returns first index where they're equal

**Max Size Subarray Sum:**
- Hash map stores first occurrence of each prefix sum
- For maximum length, keep earliest occurrence only
- Check if `current_sum - k` exists in map

**Subarray Sum Divisible by K:**
- Uses modulo arithmetic on prefix sums
- If two prefix sums have same remainder, subarray between them is divisible by k
- Counts occurrences of each remainder

**Sum of Absolute Differences:**
- For sorted array, uses formula-based approach
- Left sum: `i * nums[i] - prefix[i]`
- Right sum: `(total - prefix[i+1]) - (n-i-1) * nums[i]`

**2D Prefix Sum (NumMatrix):**
- Uses inclusion-exclusion principle
- Build: add current + top + left - top-left overlap
- Query: total - top - left + top-left overlap

## Complexity Analysis

### Time Complexity

**Building Prefix Sum:**
- **Time:** O(n) for 1D, O(m*n) for 2D
- **Why?** Single pass through array/matrix

**Range Query:**
- **Time:** O(1)
- **Why?** Simple subtraction of two values

**With Hash Map (finding subarrays):**
- **Time:** O(n)
- **Why?** Single pass, O(1) hash map operations

**Without Prefix Sum (naive range queries):**
- **Time:** O(n) per query
- **Why?** Must sum elements in range each time

### Space Complexity

**Prefix Sum Array:**
- **Space:** O(n) for 1D, O(m*n) for 2D
- **Extra storage equals input size**

**With Hash Map:**
- **Space:** O(n)
- **Why?** Store at most n different prefix sums

**Space-Optimized Version:**
- **Space:** O(1) if only need running sum
- **But:** Can't answer arbitrary range queries

### Comparison with Alternatives

| Approach | Build Time | Query Time | Space | Multiple Queries? |
|----------|------------|------------|-------|-------------------|
| **Prefix Sum** | O(n) | O(1) | O(n) | Excellent |
| **Naive Sum** | O(1) | O(n) | O(1) | Poor for many queries |
| **Segment Tree** | O(n) | O(log n) | O(n) | Good, supports updates |
| **Fenwick Tree** | O(n log n) | O(log n) | O(n) | Good, supports updates |

**When to use Prefix Sum:**
- Multiple range sum queries
- Array doesn't change (immutable)
- Need O(1) query time
- Finding subarrays with specific properties

**When NOT to use:**
- Array frequently updated (use segment/Fenwick tree)
- Need other range operations (min, max, gcd)
- Only one or two queries (naive is fine)

## Examples

### Example 1: Basic Range Sum

**Array:** [1, 2, 3, 4, 5]
**Query:** Sum of range [1, 3]

**Trace:**

```
Build prefix sum:
Original: [1, 2, 3, 4, 5]
Prefix:   [0, 1, 3, 6, 10, 15]
          ↑  ↑  ↑  ↑  ↑   ↑
      empty  1  1+2 ...  sum(all)

Query sum[1, 3]:
  Elements: [2, 3, 4]
  
  Method 1 (naive): 2 + 3 + 4 = 9
  
  Method 2 (prefix sum):
    prefix[3+1] - prefix[1]
    = prefix[4] - prefix[1]
    = 10 - 1
    = 9 ✓

Why it works:
  prefix[4] = sum[0..3] = 1+2+3+4 = 10
  prefix[1] = sum[0..0] = 1
  Difference = sum[1..3] = 2+3+4 = 9
```

### Example 2: Subarray Sum Equals K

**Input:** nums = [1, 2, 3], k = 3
**Output:** 2

**Detailed Trace:**

```
Goal: Count subarrays with sum = 3

Initialize:
  prefix_count = {0: 1}
  current_sum = 0
  count = 0

Index 0, num=1:
  current_sum = 0 + 1 = 1
  target = current_sum - k = 1 - 3 = -2
  Is -2 in map? No
  count = 0
  Update map: {0: 1, 1: 1}

Index 1, num=2:
  current_sum = 1 + 2 = 3
  target = current_sum - k = 3 - 3 = 0
  Is 0 in map? Yes! (appears 1 time)
  count = 0 + 1 = 1
  Update map: {0: 1, 1: 1, 3: 1}
  
  Found subarray [1,2] with sum 3 ✓

Index 2, num=3:
  current_sum = 3 + 3 = 6
  target = current_sum - k = 6 - 3 = 3
  Is 3 in map? Yes! (appears 1 time)
  count = 1 + 1 = 2
  Update map: {0: 1, 1: 1, 3: 1, 6: 1}
  
  Found subarray [3] with sum 3 ✓

Final: count = 2

Subarrays found:
1. [1, 2]: sum = 3
2. [3]: sum = 3
```

### Example 3: Subarrays Divisible by K

**Input:** nums = [4, 5, 0, -2, -3, 1], k = 5
**Output:** 7

**Trace:**

```
Track remainders of prefix sums mod k:

remainder_count = {0: 1}
current_sum = 0
count = 0

Index 0, num=4:
  current_sum = 4
  remainder = 4 % 5 = 4
  count += remainder_count[4] = 0
  remainder_count = {0: 1, 4: 1}

Index 1, num=5:
  current_sum = 9
  remainder = 9 % 5 = 4
  count += remainder_count[4] = 1 (found [4,5]!)
  count = 1
  remainder_count = {0: 1, 4: 2}

Index 2, num=0:
  current_sum = 9
  remainder = 9 % 5 = 4
  count += remainder_count[4] = 2 (found [4,5,0] and [5,0]!)
  count = 3
  remainder_count = {0: 1, 4: 3}

Index 3, num=-2:
  current_sum = 7
  remainder = 7 % 5 = 2
  count += remainder_count[2] = 0
  remainder_count = {0: 1, 4: 3, 2: 1}

Index 4, num=-3:
  current_sum = 4
  remainder = 4 % 5 = 4
  count += remainder_count[4] = 3 (found 3 more!)
  count = 6
  remainder_count = {0: 1, 4: 4, 2: 1}

Index 5, num=1:
  current_sum = 5
  remainder = 5 % 5 = 0
  count += remainder_count[0] = 1 (found [4,5,0,-2,-3,1]!)
  count = 7
  remainder_count = {0: 2, 4: 4, 2: 1}

Final count: 7 ✓

All subarrays divisible by 5:
[4, 5], [5], [0], [5, 0], [4, 5, 0], 
[-2, -3], [4, 5, 0, -2, -3, 1]
```

### Example 4: 2D Range Sum

**Matrix:**
```
[3, 0, 1, 4, 2]
[5, 6, 3, 2, 1]
[1, 2, 0, 1, 5]
[4, 1, 0, 1, 7]
[1, 0, 3, 0, 5]
```

**Query:** Sum of region (2,1) to (4,3)

**Trace:**

```
Target region (highlighted with *):
[3, 0, 1, 4, 2]
[5, 6, 3, 2, 1]
[1, 2*, 0*, 1*, 5]
[4, 1*, 0*, 1*, 7]
[1, 0*, 3*, 0*, 5]

Elements in region:
Row 2: 2, 0, 1
Row 3: 1, 0, 1
Row 4: 0, 3, 0
Sum = 2+0+1+1+0+1+0+3+0 = 8

Using 2D prefix sum:
prefix[i][j] = sum of rectangle (0,0) to (i-1,j-1)

Query for (2,1) to (4,3):
  r1=3, c1=2, r2=5, c2=4 (adjusted to 1-indexed)
  
  result = prefix[5][4] 
         - prefix[2][4]    (subtract top rectangle)
         - prefix[5][1]    (subtract left rectangle)
         + prefix[2][1]    (add back overlap)
  
  = 58 - 28 - 24 + 2
  = 8 ✓

Visual explanation:
  ┌─────────┐
  │ Top(28) │
  ├─┬───────┤
  │L│Target │
  │e│  (?)  │
  │f│       │
  │t│       │
  │(│       │
  │2│       │
  │4│       │
  │)│       │
  └─┴───────┘
    └─58────┘

Total - Top - Left + Overlap = Target
58 - 28 - 24 + 2 = 8
```

## Edge Cases

### 1. Empty Array
**Scenario:** Input array is empty
**Challenge:** No elements to sum
**Solution:** Return 0 or empty result
**Code example:**
```python
if not nums:
    return 0
```

### 2. Single Element
**Scenario:** Array with one element
**Challenge:** Only one possible range
**Solution:** prefix = [0, nums[0]]
**Code example:**
```python
nums = [5]
prefix = [0, 5]
range_sum(0, 0) = 5
```

### 3. All Zeros
**Scenario:** Array of all zeros
**Challenge:** All prefix sums are 0
**Solution:** Many subarrays have sum 0
**Code example:**
```python
nums = [0, 0, 0]
# Count subarrays with sum 0
# All subarrays: 6 subarrays total
```

### 4. Negative Numbers
**Scenario:** Array contains negative values
**Challenge:** Prefix sums can decrease
**Solution:** Works same way, use signed arithmetic
**Code example:**
```python
nums = [1, -2, 3]
prefix = [0, 1, -1, 2]
```

### 5. Target Sum Not Achievable
**Scenario:** No subarray sums to target
**Challenge:** Return 0 or empty
**Solution:** Hash map won't find matches
**Code example:**
```python
nums = [1, 2, 3]
k = 10
# No subarray sums to 10
result = 0
```

### 6. Entire Array as Answer
**Scenario:** Whole array is the answer
**Challenge:** Need to check full range
**Solution:** prefix_count[0] = 1 handles this
**Code example:**
```python
nums = [1, 2, 3]
k = 6
# Entire array sums to 6
```

### 7. Multiple Subarrays Same Sum
**Scenario:** Many subarrays with target sum
**Challenge:** Count all occurrences
**Solution:** Hash map tracks counts
**Code example:**
```python
nums = [1, 1, 1]
k = 2
# Two subarrays: [1,1] at indices (0,1) and (1,2)
```

## Common Pitfalls

### ❌ Pitfall 1: Off-by-One in Range Query
**What happens:** Wrong elements summed
**Why it's wrong:** Incorrect index adjustment
**Correct approach:**
```python
# WRONG:
def range_sum(left, right):
    return prefix[right] - prefix[left]  # Misses left element!

# CORRECT:
def range_sum(left, right):
    return prefix[right + 1] - prefix[left]  # Includes both ends
```

### ❌ Pitfall 2: Not Initializing Hash Map with 0
**What happens:** Miss subarrays starting at index 0
**Why it's wrong:** No prefix to subtract for full array
**Correct approach:**
```python
# WRONG:
prefix_count = {}  # Missing base case!

# CORRECT:
prefix_count = {0: 1}  # Empty prefix has sum 0
```

### ❌ Pitfall 3: Building Prefix Array Too Small
**What happens:** Index out of bounds
**Why it's wrong:** Need n+1 elements for n-element array
**Correct approach:**
```python
# WRONG:
prefix = [0] * len(nums)  # Too small!

# CORRECT:
prefix = [0] * (len(nums) + 1)  # Extra element for boundary
```

### ❌ Pitfall 4: Forgetting to Handle Negative Remainders
**What happens:** Wrong count for divisibility problems
**Why it's wrong:** Some languages have negative remainders
**Correct approach:**
```python
# WRONG (can be negative in some languages):
remainder = current_sum % k

# CORRECT (ensure positive):
remainder = ((current_sum % k) + k) % k
# Or in Python (already handles it correctly):
remainder = current_sum % k  # Always >= 0 in Python
```

### ❌ Pitfall 5: Updating Hash Map Before Checking
**What happens:** Count includes current element
**Why it's wrong:** Should check then update
**Correct approach:**
```python
# WRONG:
prefix_count[current_sum] += 1  # Update first
count += prefix_count[current_sum - k]  # Then check

# CORRECT:
count += prefix_count[current_sum - k]  # Check first
prefix_count[current_sum] += 1  # Then update
```

### ❌ Pitfall 6: Wrong 2D Prefix Sum Formula
**What happens:** Incorrect rectangle sums
**Why it's wrong:** Inclusion-exclusion principle violated
**Correct approach:**
```python
# WRONG:
prefix[i][j] = matrix[i][j] + prefix[i-1][j] + prefix[i][j-1]

# CORRECT:
prefix[i][j] = (matrix[i][j] + 
                prefix[i-1][j] + 
                prefix[i][j-1] - 
                prefix[i-1][j-1])  # Subtract overlap!
```

### ❌ Pitfall 7: Storing All Prefix Sums When Only Count Needed
**What happens:** Unnecessary space usage
**Why it's wrong:** Only need frequency counts
**Correct approach:**
```python
# WRONG (wastes space):
all_prefix_sums = []  # Stores every value
for num in nums:
    current_sum += num
    all_prefix_sums.append(current_sum)

# CORRECT:
prefix_count = defaultdict(int)  # Only stores counts
for num in nums:
    current_sum += num
    prefix_count[current_sum] += 1
```

## Variations and Extensions

### Variation 1: Running Maximum/Minimum
**Description:** Track running max/min instead of sum
**When to use:** Need range max/min queries
**Implementation:** Use segment tree or sparse table instead

### Variation 2: XOR Prefix
**Description:** Store XOR instead of sum
**When to use:** Finding subarrays with XOR equal to k
**Implementation:**
```python
def count_xor_subarrays(nums, k):
    xor_count = {0: 1}
    current_xor = 0
    count = 0
    
    for num in nums:
        current_xor ^= num
        target = current_xor ^ k
        count += xor_count.get(target, 0)
        xor_count[current_xor] = xor_count.get(current_xor, 0) + 1
    
    return count
```

### Variation 3: Product Prefix
**Description:** Store running product
**When to use:** Subarray product problems
**Challenge:** Handle zeros separately
**Implementation:**
```python
# Count subarrays with product < k
def num_subarray_product_less_than_k(nums, k):
    if k <= 1:
        return 0
    
    product = 1
    count = 0
    left = 0
    
    for right in range(len(nums)):
        product *= nums[right]
        
        while product >= k:
            product //= nums[left]
            left += 1
        
        count += right - left + 1
    
    return count
```

### Variation 4: Difference Array
**Description:** Store differences between consecutive elements
**When to use:** Range update operations
**Implementation:**
```python
class DifferenceArray:
    def __init__(self, nums):
        self.diff = [0] * len(nums)
        self.diff[0] = nums[0]
        for i in range(1, len(nums)):
            self.diff[i] = nums[i] - nums[i-1]
    
    def range_add(self, left, right, val):
        """Add val to all elements in [left, right]."""
        self.diff[left] += val
        if right + 1 < len(self.diff):
            self.diff[right + 1] -= val
    
    def get_array(self):
        """Reconstruct array from differences."""
        result = [0] * len(self.diff)
        result[0] = self.diff[0]
        for i in range(1, len(self.diff)):
            result[i] = result[i-1] + self.diff[i]
        return result
```

## Practice Problems

### Beginner
1. **Find the Middle Index in Array** - Basic prefix sum
   - LeetCode #1991

2. **Left and Right Sum Differences** - Build both prefix arrays
   - LeetCode #2574

3. **Running Sum of 1d Array** - Build prefix sum array
   - LeetCode #1480

4. **Find Pivot Index** - Same as middle index
   - LeetCode #724

### Intermediate
1. **Subarray Sum Equals K** - Classic hash map + prefix sum
   - LeetCode #560

2. **Maximum Size Subarray Sum Equals k** - Store earliest index
   - LeetCode #325

3. **Binary Subarrays With Sum** - Count with target sum
   - LeetCode #930

4. **Subarray Sums Divisible by K** - Modulo arithmetic
   - LeetCode #974

5. **Sum of Absolute Differences in Sorted Array** - Formula-based
   - LeetCode #1685

6. **Continuous Subarray Sum** - Multiple of k
   - LeetCode #523

7. **Range Sum Query - Immutable** - Basic prefix sum class
   - LeetCode #303

8. **Range Sum Query 2D - Immutable** - 2D prefix sum
   - LeetCode #304

### Advanced
1. **Count of Range Sum** - Complex counting
   - LeetCode #327

2. **Maximum Sum of 3 Non-Overlapping Subarrays** - Dynamic programming + prefix
   - LeetCode #689

3. **Minimum Operations to Reduce X to Zero** - Two pointer + prefix
   - LeetCode #1658

4. **Make Sum Divisible by P** - Remove smallest subarray
   - LeetCode #1590

## Real-World Applications

### Industry Use Cases

1. **Database Query Optimization:** Materialized views use prefix sums for fast aggregate queries
2. **Financial Analysis:** Running totals for profit/loss, cumulative returns
3. **Image Processing:** Summed area tables for rapid rectangle sum queries
4. **Analytics Dashboards:** Real-time cumulative metrics
5. **Scientific Computing:** Numerical integration, cumulative distribution functions

### Popular Implementations

- **NumPy cumsum()** - Python scientific computing
- **Pandas cumsum()** - Data analysis library
- **SQL Window Functions** - OVER clause with cumulative sums
- **Summed Area Tables** - Computer vision and image processing

### Practical Scenarios

- **Stock price analysis** - Moving averages, cumulative returns
- **E-commerce** - Running sales totals, revenue dashboards
- **Gaming** - Experience points accumulation, score tracking
- **Resource monitoring** - Cumulative CPU/memory usage
- **Weather data** - Accumulated rainfall, temperature anomalies

## Related Topics

### Prerequisites to Review
- **Arrays** - Foundation for prefix sums
- **Hash Maps** - For finding subarrays with properties
- **Basic Arithmetic** - Cumulative operations

### Next Steps
- **Segment Trees** - Range queries with updates
- **Fenwick Trees (BIT)** - Space-efficient range queries
- **Sparse Tables** - Range minimum/maximum queries
- **Difference Arrays** - Range update operations

### Similar Concepts
- **Running Statistics** - Online algorithms for mean, variance
- **Cumulative Distribution Function** - Probability theory
- **Integration** - Mathematical cumulative sum analog
- **Sliding Window** - Related technique for subarrays

### Further Reading
- "Introduction to Algorithms" (CLRS) - Dynamic Programming chapter
- "Competitive Programming" by Halim - Range Query section
- [Prefix Sum - CP-Algorithms](https://cp-algorithms.com/data_structures/segment_tree.html)
- [LeetCode Prefix Sum Tag](https://leetcode.com/tag/prefix-sum/)
- "The Algorithm Design Manual" by Skiena - Data structures
