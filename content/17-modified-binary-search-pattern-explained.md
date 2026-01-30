# Modified Binary Search Pattern

**Difficulty:** Intermediate to Advanced
**Prerequisites:** Binary Search, Arrays, Sorting concepts, Two Pointers
**Estimated Reading Time:** 55 minutes

## Introduction

The Modified Binary Search Pattern extends the classic binary search algorithm to solve a wider range of problems beyond simple element lookup. While traditional binary search finds an exact match in a sorted array, modified binary search can find ranges, closest elements, rotated array elements, bitonic peaks, and much more—all while maintaining the O(log n) time complexity that makes binary search so powerful.

**Why it matters:** Binary search's logarithmic time complexity makes it one of the most efficient search algorithms. For an array of 1 billion elements, binary search needs only about 30 comparisons maximum! Modified binary search extends this efficiency to complex problems: finding insertion positions, searching in rotated arrays, finding peaks in bitonic arrays, and identifying ranges. Mastering these variations unlocks solutions to problems that would otherwise require O(n) linear scans.

**Real-world analogy:** Imagine you're looking for a house to buy in a neighborhood where houses are numbered sequentially, but some numbers are missing. Traditional binary search would only help if you knew the exact house number. Modified binary search is like a smarter realtor who can find: "the smallest house number ≥ 500" (ceiling), "the largest house number ≤ 500" (floor), or "the first house in a rotated street where numbers wrapped around." This flexibility while maintaining speed is the power of modified binary search!

## Core Concepts

### Key Principles

1. **Monotonicity Requirement:** Binary search works on sorted or partially sorted data. The key insight is finding a property that divides the search space into two parts: one where the property holds, one where it doesn't.

2. **Search Space Reduction:** Each comparison eliminates half the search space, leading to O(log n) time complexity. The challenge is determining which half to eliminate based on the specific problem.

3. **Boundary Conditions:** Modified binary search often finds boundaries or thresholds rather than exact matches. Understanding when to use `<`, `<=`, `>`, `>=` is crucial.

4. **Template Pattern:** Most variations follow the same structure:
   - Initialize `left` and `right` pointers
   - While `left <= right` (or `left < right` for some variations)
   - Calculate `mid = left + (right - left) // 2`
   - Adjust pointers based on comparison
   - Return appropriate value

### Essential Terms

- **Binary Search:** Search algorithm that repeatedly divides search interval in half
- **Sorted Array:** Array where elements are in non-decreasing (or non-increasing) order
- **Ceiling:** Smallest element ≥ target
- **Floor:** Largest element ≤ target
- **Rotated Array:** Sorted array rotated at some pivot point
- **Bitonic Array:** Array that first increases, then decreases (or vice versa)
- **Search Space:** Current range being searched
- **Pivot:** Point where array rotation or direction change occurs
- **Order-agnostic:** Works for both ascending and descending order
- **Infinite Array:** Array of unknown size (can't access length directly)

### Visual Overview

```
Standard Binary Search:
Array: [1, 3, 5, 7, 9, 11, 13, 15], Target: 7

Step 1: left=0, right=7, mid=3
        [1, 3, 5, |7|, 9, 11, 13, 15]
                   ↑
        arr[mid]=7 == target, FOUND!

Ceiling (smallest element ≥ target):
Array: [1, 3, 5, 7, 9, 11, 13, 15], Target: 6

Step 1: left=0, right=7, mid=3
        [1, 3, 5, |7|, 9, 11, 13, 15]
        6 < 7, search left half
        
Step 2: left=0, right=2, mid=1
        [1, |3|, 5]
        6 > 3, search right half
        
Step 3: left=2, right=2, mid=2
        [|5|]
        6 > 5, move left=3
        
Result: arr[3]=7 (ceiling of 6)

Rotated Array:
Array: [7, 9, 11, 13, 1, 3, 5], Target: 3

        [7, 9, 11, 13, |1, 3, 5]
         ←sorted→      ←sorted→
         
Rotation point between 13 and 1
One half is always sorted!

Bitonic Array (find peak):
Array: [1, 3, 8, 12, 10, 7, 4, 2]

        [1, 3, 8, |12|, 10, 7, 4, 2]
         ↗  ↗  ↗   ↑   ↘  ↘  ↘  ↘
                  peak
        
Left of peak: increasing
Right of peak: decreasing
```

## How It Works

### Order-Agnostic Binary Search - Step by Step

**Problem:** Search in an array that could be sorted ascending or descending.

**Algorithm:**

1. Determine sort order by comparing first and last elements
2. Perform binary search with appropriate comparison operators
3. Return index if found, -1 otherwise

**Detailed Walkthrough:**

```
Example 1: Ascending array
Array: [1, 3, 5, 7, 9, 11], Target: 7

Step 1: Determine order
  arr[0]=1 < arr[5]=11 → Ascending order
  
Step 2: Binary search (ascending)
  left=0, right=5, mid=2
  [1, 3, |5|, 7, 9, 11]
  arr[mid]=5 < 7 → search right half
  
  left=3, right=5, mid=4
  [7, |9|, 11]
  arr[mid]=9 > 7 → search left half
  
  left=3, right=3, mid=3
  [|7|]
  arr[mid]=7 == 7 → FOUND at index 3!

Example 2: Descending array
Array: [11, 9, 7, 5, 3, 1], Target: 7

Step 1: Determine order
  arr[0]=11 > arr[5]=1 → Descending order
  
Step 2: Binary search (descending)
  left=0, right=5, mid=2
  [11, 9, |7|, 5, 3, 1]
  arr[mid]=7 == 7 → FOUND at index 2!
```

### Finding Ceiling - Step by Step

**Problem:** Find the smallest element ≥ target.

**Algorithm:**

1. Perform binary search
2. When `arr[mid] >= target`, this could be ceiling, but check left half for smaller ceiling
3. When `arr[mid] < target`, search right half
4. Track the best ceiling found so far

**Detailed Walkthrough:**

```
Array: [2, 4, 6, 8, 10, 12], Target: 7

Step 1: left=0, right=5, mid=2
        [2, 4, |6|, 8, 10, 12]
        arr[mid]=6 < 7
        6 is not ceiling (too small)
        Search right: left=3
        
Step 2: left=3, right=5, mid=4
        [8, |10|, 12]
        arr[mid]=10 >= 7
        10 could be ceiling, but check left for smaller
        ceiling_candidate = 10
        Search left: right=3
        
Step 3: left=3, right=3, mid=3
        [|8|]
        arr[mid]=8 >= 7
        8 is better ceiling than 10
        ceiling_candidate = 8
        Search left: right=2
        
Step 4: left=3, right=2
        left > right, exit loop
        
Result: ceiling_candidate = 8 (smallest element ≥ 7)

Edge Case - Target larger than all elements:
Array: [2, 4, 6, 8], Target: 10

After search: ceiling_candidate remains unset
Result: -1 (no ceiling exists)
```

### Searching in Rotated Sorted Array - Step by Step

**Problem:** Find target in sorted array rotated at unknown pivot.

**Key Insight:** At least one half is always sorted!

**Algorithm:**

1. Find mid point
2. Determine which half is sorted
3. Check if target is in sorted half's range
4. If yes, search sorted half; otherwise search other half

**Detailed Walkthrough:**

```
Array: [7, 9, 11, 13, 1, 3, 5], Target: 3

Step 1: left=0, right=6, mid=3
        [7, 9, 11, |13|, 1, 3, 5]
        
        Check which half is sorted:
        arr[left]=7 <= arr[mid]=13 → Left half sorted
        
        Is target in sorted half [7,13]?
        3 is NOT in [7, 13]
        Search right half: left=4

Step 2: left=4, right=6, mid=5
        [1, |3|, 5]
        
        Check which half is sorted:
        arr[left]=1 <= arr[mid]=3 → Left half sorted
        
        Is target in sorted half [1,3]?
        3 IS in [1, 3]
        
        Actually arr[mid]=3 == target
        FOUND at index 5!

Another Example:
Array: [4, 5, 6, 7, 0, 1, 2], Target: 0

Step 1: left=0, right=6, mid=3
        [4, 5, 6, |7|, 0, 1, 2]
        
        arr[left]=4 <= arr[mid]=7 → Left half sorted
        Target 0 NOT in [4, 7]
        Search right: left=4

Step 2: left=4, right=6, mid=5
        [0, |1|, 2]
        
        arr[left]=0 <= arr[mid]=1 → Left half sorted
        Target 0 in [0, 1]
        Search left: right=4

Step 3: left=4, right=4, mid=4
        [|0|]
        arr[mid]=0 == target
        FOUND at index 4!
```

### Finding Bitonic Peak - Step by Step

**Problem:** Find maximum element in bitonic array (increases then decreases).

**Algorithm:**

1. Calculate mid
2. If `arr[mid] > arr[mid+1]`, peak is in left half (including mid)
3. Otherwise, peak is in right half
4. Continue until left == right

**Detailed Walkthrough:**

```
Array: [1, 3, 8, 12, 10, 7, 4, 2]

Step 1: left=0, right=7, mid=3
        [1, 3, 8, |12|, 10, 7, 4, 2]
        
        Compare arr[mid]=12 with arr[mid+1]=10
        12 > 10 → We're in decreasing part or at peak
        Peak is at mid or to the left
        right=mid=3

Step 2: left=0, right=3, mid=1
        [1, |3|, 8, 12]
        
        Compare arr[mid]=3 with arr[mid+1]=8
        3 < 8 → We're in increasing part
        Peak is to the right
        left=mid+1=2

Step 3: left=2, right=3, mid=2
        [|8|, 12]
        
        Compare arr[mid]=8 with arr[mid+1]=12
        8 < 12 → Still increasing
        Peak is to the right
        left=mid+1=3

Step 4: left=3, right=3
        left == right → Found peak!
        Peak is at index 3, value=12

Verification:
[1, 3, 8, |12|, 10, 7, 4, 2]
         ↗  ↗  ↗  ↑  ↘  ↘  ↘  ↘
                  peak
```

## Implementation

### Order-Agnostic Binary Search

```python
from typing import List

def order_agnostic_binary_search(arr: List[int], target: int) -> int:
    """
    Search for target in array sorted in unknown order (ascending or descending).
    
    Args:
        arr: Sorted array (ascending or descending)
        target: Element to find
        
    Returns:
        Index of target if found, -1 otherwise
        
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    if not arr:
        return -1
    
    left, right = 0, len(arr) - 1
    
    # Determine sort order
    is_ascending = arr[left] < arr[right]
    
    while left <= right:
        mid = left + (right - left) // 2  # Avoid overflow
        
        if arr[mid] == target:
            return mid
        
        if is_ascending:
            # Ascending order logic
            if arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        else:
            # Descending order logic
            if arr[mid] > target:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1  # Not found


# Example usage
arr_asc = [1, 3, 5, 7, 9, 11]
arr_desc = [11, 9, 7, 5, 3, 1]
print(order_agnostic_binary_search(arr_asc, 7))   # Output: 3
print(order_agnostic_binary_search(arr_desc, 7))  # Output: 2
```

### Ceiling of a Number

```python
def find_ceiling(arr: List[int], target: int) -> int:
    """
    Find the smallest element greater than or equal to target.
    
    Args:
        arr: Sorted array in ascending order
        target: Value to find ceiling for
        
    Returns:
        Index of ceiling element, -1 if no ceiling exists
        
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    if not arr:
        return -1
    
    # If target is greater than largest element, no ceiling exists
    if target > arr[-1]:
        return -1
    
    left, right = 0, len(arr) - 1
    ceiling_index = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid  # Exact match is the ceiling
        
        if arr[mid] > target:
            # arr[mid] could be ceiling, but search left for smaller ceiling
            ceiling_index = mid
            right = mid - 1
        else:
            # arr[mid] < target, search right
            left = mid + 1
    
    return ceiling_index


# Floor of a number (largest element <= target)
def find_floor(arr: List[int], target: int) -> int:
    """
    Find the largest element less than or equal to target.
    
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    if not arr or target < arr[0]:
        return -1
    
    left, right = 0, len(arr) - 1
    floor_index = -1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        
        if arr[mid] < target:
            # arr[mid] could be floor, but search right for larger floor
            floor_index = mid
            left = mid + 1
        else:
            # arr[mid] > target, search left
            right = mid - 1
    
    return floor_index


# Example usage
arr = [2, 4, 6, 8, 10, 12, 14]
print(find_ceiling(arr, 5))   # Output: 2 (element 6)
print(find_ceiling(arr, 8))   # Output: 3 (element 8)
print(find_floor(arr, 5))     # Output: 1 (element 4)
print(find_floor(arr, 8))     # Output: 3 (element 8)
```

### Next Letter (Circular Array)

```python
def next_greatest_letter(letters: List[str], target: str) -> str:
    """
    Find smallest letter in array that is greater than target.
    Array is circular (wraps around).
    
    Args:
        letters: Sorted array of lowercase letters
        target: Target letter
        
    Returns:
        Next greatest letter (wraps to first if target >= last)
        
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    n = len(letters)
    left, right = 0, n - 1
    
    # If target >= last letter, answer is first letter (circular)
    if target >= letters[-1]:
        return letters[0]
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if letters[mid] > target:
            # This could be answer, search left for potentially smaller
            right = mid - 1
        else:
            # letters[mid] <= target, search right
            left = mid + 1
    
    # left points to the next greatest letter
    return letters[left]


# Example usage
letters = ['a', 'c', 'f', 'h']
print(next_greatest_letter(letters, 'f'))  # Output: 'h'
print(next_greatest_letter(letters, 'b'))  # Output: 'c'
print(next_greatest_letter(letters, 'h'))  # Output: 'a' (wraps)
```

### Number Range (First and Last Position)

```python
def search_range(nums: List[int], target: int) -> List[int]:
    """
    Find starting and ending position of target in sorted array.
    
    Args:
        nums: Sorted array with possible duplicates
        target: Value to find range for
        
    Returns:
        [start_index, end_index] or [-1, -1] if not found
        
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    def find_boundary(is_finding_left: bool) -> int:
        """
        Find left or right boundary of target.
        
        Args:
            is_finding_left: True for leftmost, False for rightmost
        """
        left, right = 0, len(nums) - 1
        boundary = -1
        
        while left <= right:
            mid = left + (right - left) // 2
            
            if nums[mid] == target:
                boundary = mid
                if is_finding_left:
                    # Found target, search left for first occurrence
                    right = mid - 1
                else:
                    # Found target, search right for last occurrence
                    left = mid + 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return boundary
    
    left_bound = find_boundary(True)
    if left_bound == -1:
        return [-1, -1]  # Target not found
    
    right_bound = find_boundary(False)
    return [left_bound, right_bound]


# Example usage
nums = [5, 7, 7, 8, 8, 8, 10]
print(search_range(nums, 8))   # Output: [3, 5]
print(search_range(nums, 7))   # Output: [1, 2]
print(search_range(nums, 6))   # Output: [-1, -1]
```

### Search in Rotated Sorted Array

```python
def search_rotated(nums: List[int], target: int) -> int:
    """
    Search for target in rotated sorted array.
    
    Args:
        nums: Sorted array rotated at some pivot
        target: Element to find
        
    Returns:
        Index of target if found, -1 otherwise
        
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    if not nums:
        return -1
    
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # Determine which half is sorted
        if nums[left] <= nums[mid]:
            # Left half is sorted
            if nums[left] <= target < nums[mid]:
                # Target is in sorted left half
                right = mid - 1
            else:
                # Target is in right half
                left = mid + 1
        else:
            # Right half is sorted
            if nums[mid] < target <= nums[right]:
                # Target is in sorted right half
                left = mid + 1
            else:
                # Target is in left half
                right = mid - 1
    
    return -1


# Example usage
nums = [4, 5, 6, 7, 0, 1, 2]
print(search_rotated(nums, 0))  # Output: 4
print(search_rotated(nums, 3))  # Output: -1
```

### Find Bitonic Array Maximum

```python
def find_peak_element(nums: List[int]) -> int:
    """
    Find peak element in bitonic array (increases then decreases).
    
    Args:
        nums: Bitonic array
        
    Returns:
        Index of peak element
        
    Time Complexity: O(log n)
    Space Complexity: O(1)
    """
    if not nums:
        return -1
    
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[mid + 1]:
            # We're in decreasing part or at peak
            # Peak is at mid or to the left
            right = mid
        else:
            # We're in increasing part
            # Peak is to the right of mid
            left = mid + 1
    
    # left == right, pointing to peak
    return left


# Example usage
nums = [1, 3, 8, 12, 10, 7, 4, 2]
peak_idx = find_peak_element(nums)
print(f"Peak at index {peak_idx}, value: {nums[peak_idx]}")  
# Output: Peak at index 3, value: 12
```

### Search in Infinite Sorted Array

```python
def search_infinite_array(reader, target: int) -> int:
    """
    Search in sorted array of unknown size.
    
    Strategy: Find bounds first using exponential search,
    then use binary search.
    
    Args:
        reader: Array reader interface with get(index) method
        target: Element to find
        
    Returns:
        Index of target if found, -1 otherwise
        
    Time Complexity: O(log n) where n is index of target
    Space Complexity: O(1)
    """
    # Find bounds using exponential search
    left, right = 0, 1
    
    # Double right pointer until we find a range containing target
    while reader.get(right) < target:
        left = right
        right *= 2  # Exponential increase
    
    # Binary search in the found range
    while left <= right:
        mid = left + (right - left) // 2
        value = reader.get(mid)
        
        if value == target:
            return mid
        elif value < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


# Simulated infinite array reader
class ArrayReader:
    def __init__(self, arr):
        self.arr = arr
    
    def get(self, index):
        if index >= len(self.arr):
            return float('inf')  # Simulate infinite array
        return self.arr[index]


# Example usage
arr = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
reader = ArrayReader(arr)
print(search_infinite_array(reader, 11))  # Output: 5
```

### Code Explanation

**Key Design Decisions:**

1. **Avoiding Overflow:** Use `mid = left + (right - left) // 2` instead of `mid = (left + right) // 2` to avoid integer overflow in languages with fixed-size integers.

2. **Boundary Updates:**
   - For finding exact match: `left = mid + 1` or `right = mid - 1`
   - For finding boundary: Keep potential answer and continue searching

3. **Loop Condition:**
   - `while left <= right`: Use when you need to check if left equals right
   - `while left < right`: Use when convergence itself gives the answer

4. **Handling Edge Cases:**
   - Empty array: Return -1 immediately
   - Target out of range: Check before search
   - Single element: Handle naturally with loop condition

5. **Sorted Half Detection (Rotated Array):**
   - Compare `arr[left]` with `arr[mid]` to determine which half is sorted
   - Then check if target is in the sorted half's range

## Complexity Analysis

### Time Complexity

**All Binary Search Variations: O(log n)**

**Why O(log n)?**
- Each iteration eliminates half the search space
- After k iterations, search space = n / 2^k
- When search space = 1: n / 2^k = 1 → 2^k = n → k = log₂(n)
- Therefore, maximum iterations = O(log n)

**Specific Cases:**
- **Standard Binary Search:** O(log n)
- **Ceiling/Floor:** O(log n) - same as standard
- **Number Range:** O(log n) - two binary searches
- **Rotated Array:** O(log n) - one pass through array
- **Bitonic Peak:** O(log n) - converges to peak
- **Infinite Array:** O(log n) where n is position of target
  - Exponential search to find bounds: O(log n)
  - Binary search in bounds: O(log n)
  - Total: O(log n)

**Comparison to Linear Search:**
```
Array size:         100        1,000      1,000,000    1,000,000,000
Linear Search:      100        1,000      1,000,000    1,000,000,000
Binary Search:      7          10         20           30
```

### Space Complexity

**Iterative Binary Search: O(1)**
- Only use constant extra space for pointers (left, right, mid)
- No recursion stack

**Recursive Binary Search: O(log n)**
- Recursion depth is O(log n)
- Each recursive call adds to call stack
- Can be optimized to O(1) using iteration

### Comparison with Alternatives

| Approach | Time | Space | When to Use |
|----------|------|-------|-------------|
| Binary Search | O(log n) | O(1) | Sorted array, exact match |
| Linear Search | O(n) | O(1) | Unsorted, small arrays |
| Hash Table | O(1) avg | O(n) | Multiple lookups, no order |
| Interpolation Search | O(log log n) avg, O(n) worst | O(1) | Uniformly distributed data |
| Exponential Search | O(log n) | O(1) | Unbounded/infinite arrays |

## Examples

### Example 1: Find Ceiling

**Problem:** Find smallest element ≥ 10 in array.

**Input:** arr = [2, 4, 6, 8, 12, 14, 16], target = 10

**Step-by-step:**

```
Initial: left=0, right=6, ceiling=-1

Step 1: mid=3, arr[3]=8
        8 < 10
        Search right: left=4

Step 2: left=4, right=6, mid=5
        arr[5]=14
        14 >= 10 → Potential ceiling
        ceiling=5
        Search left for smaller: right=4

Step 3: left=4, right=4, mid=4
        arr[4]=12
        12 >= 10 → Better ceiling
        ceiling=4
        Search left: right=3

Step 4: left=4, right=3
        left > right, exit

Result: ceiling=4, arr[4]=12
```

### Example 2: Search in Rotated Array

**Problem:** Find 1 in rotated array.

**Input:** nums = [6, 7, 8, 9, 1, 2, 3, 4, 5], target = 1

**Step-by-step:**

```
Step 1: left=0, right=8, mid=4
        nums = [6, 7, 8, 9, |1|, 2, 3, 4, 5]
        
        Which half is sorted?
        nums[left]=6 > nums[mid]=1
        → Right half is sorted [1,2,3,4,5]
        
        Is target in sorted half?
        1 <= 1 <= 5 → Yes!
        But wait, nums[mid]=1 == target
        FOUND at index 4!

Alternative path if mid wasn't exact match:
nums = [6, 7, 8, 9, 2, 3, 4, 5], target = 3

Step 1: left=0, right=7, mid=3
        nums[left]=6 <= nums[mid]=9
        → Left half sorted [6,7,8,9]
        
        Is 3 in [6,9]? No
        Search right: left=4

Step 2: left=4, right=7, mid=5
        nums[left]=2 <= nums[mid]=3
        → Left half sorted [2,3]
        
        Is 3 in [2,3]? Yes
        Actually nums[mid]=3 == target
        FOUND at index 5!
```

### Example 3: Find Bitonic Peak

**Problem:** Find maximum in [1, 4, 7, 10, 8, 5, 2].

**Step-by-step:**

```
Step 1: left=0, right=6, mid=3
        [1, 4, 7, |10|, 8, 5, 2]
        
        nums[mid]=10 > nums[mid+1]=8
        → Decreasing or at peak
        right=mid=3

Step 2: left=0, right=3, mid=1
        [1, |4|, 7, 10]
        
        nums[mid]=4 < nums[mid+1]=7
        → Still increasing
        left=mid+1=2

Step 3: left=2, right=3, mid=2
        [|7|, 10]
        
        nums[mid]=7 < nums[mid+1]=10
        → Still increasing
        left=mid+1=3

Step 4: left=3, right=3
        Found peak at index 3, value=10
```

### Example 4: Number Range

**Problem:** Find range of 8 in [5, 7, 7, 8, 8, 8, 10].

**Finding Left Boundary:**

```
Step 1: left=0, right=6, mid=3
        [5, 7, 7, |8|, 8, 8, 10]
        
        nums[mid]=8 == target
        Found target, but search left for first occurrence
        boundary=3, right=2

Step 2: left=0, right=2, mid=1
        [5, |7|, 7]
        
        nums[mid]=7 < 8
        Search right: left=2

Step 3: left=2, right=2, mid=2
        [|7|]
        
        nums[mid]=7 < 8
        Search right: left=3

Step 4: left=3, right=2
        Exit, left_boundary=3
```

**Finding Right Boundary:**

```
Step 1: left=0, right=6, mid=3
        nums[mid]=8 == target
        Found target, search right for last occurrence
        boundary=3, left=4

Step 2: left=4, right=6, mid=5
        [8, |8|, 10]
        
        nums[mid]=8 == target
        boundary=5, left=6

Step 3: left=6, right=6, mid=6
        [|10|]
        
        nums[mid]=10 > 8
        right=5

Step 4: left=6, right=5
        Exit, right_boundary=5

Result: [3, 5] (indices of first and last 8)
```

## Edge Cases

### 1. Empty Array

**Scenario:** Array is empty or None.

**Challenge:** Cannot perform binary search.

**Solution:**

```python
def binary_search(arr, target):
    if not arr:  # Handles None or empty list
        return -1
    
    # Regular binary search logic...
```

### 2. Single Element Array

**Scenario:** Array has only one element.

**Challenge:** Edge case for loop conditions.

**Solution:**
```python
# arr = [5], target = 5
# left=0, right=0, mid=0
# arr[mid]=5 == target → Found!

# Handled naturally by loop condition
```

### 3. Target Not in Array

**Scenario:** Target doesn't exist in array.

**Challenge:** Return appropriate value (-1, None, or special marker).

**Solution:**

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Not found
```

### 4. All Elements Same

**Scenario:** All elements in array are identical.

**Challenge:** Still need O(log n) complexity.

**Solution:**

```python
# arr = [7, 7, 7, 7, 7], target = 7
# Returns first occurrence (index 0)
# Binary search still works, O(log n)

# For finding range [0, 4]:
# Left boundary search: O(log n)
# Right boundary search: O(log n)
```

### 5. Ceiling/Floor When Target Out of Range

**Scenario:** Target smaller than all elements or larger than all.

**Challenge:** No ceiling/floor exists.

**Solution:**

```python
def find_ceiling(arr, target):
    if not arr or target > arr[-1]:
        return -1  # No ceiling exists
    
    # Regular search...

def find_floor(arr, target):
    if not arr or target < arr[0]:
        return -1  # No floor exists
    
    # Regular search...
```

### 6. Rotated Array with No Rotation

**Scenario:** Rotation point is at index 0 (array is actually sorted).

**Challenge:** Algorithm should still work.

**Solution:**

```python
# arr = [1, 2, 3, 4, 5], target = 3
# Left half is always sorted in this case
# Algorithm works correctly, finds target at index 2
```

### 7. Bitonic Array That's Only Increasing

**Scenario:** Array increases but never decreases.

**Challenge:** Peak is at the end.

**Solution:**

```python
# arr = [1, 3, 5, 7, 9, 11]
# Algorithm converges to last element
# Returns index 5 (peak = 11)
```

### 8. Duplicate Elements in Rotated Array

**Scenario:** Rotated array with duplicates.

**Challenge:** Can't determine which half is sorted.

**Solution:**

```python
def search_with_duplicates(nums, target):
    """
    Handle duplicates in rotated array.
    Worst case O(n) when all elements are same.
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # When duplicates prevent determining sorted half
        if nums[left] == nums[mid] == nums[right]:
            left += 1
            right -= 1
        elif nums[left] <= nums[mid]:
            # Left half sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            # Right half sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    
    return -1
```

## Common Pitfalls

### ❌ Pitfall 1: Integer Overflow in Mid Calculation

**What happens:** For large arrays, (left + right) can overflow.

```python
# WRONG - Potential overflow in some languages
mid = (left + right) // 2
```

**Why it's wrong:** In languages with fixed-size integers (Java, C++), adding large values can overflow.

**Correct approach:**

```python
# CORRECT - Avoids overflow
mid = left + (right - left) // 2
```

### ❌ Pitfall 2: Wrong Loop Condition

**What happens:** Infinite loop or missing the target.

```python
# WRONG for exact match search
def binary_search_wrong(arr, target):
    left, right = 0, len(arr) - 1
    
    while left < right:  # Should be left <= right
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

**Why it's wrong:** When `left == right`, we have one element left to check, but loop exits.

**Correct approach:**

```python
# CORRECT
while left <= right:  # Check equality too
    # ... search logic
```

### ❌ Pitfall 3: Not Updating Pointers Correctly

**What happens:** Infinite loop.

```python
# WRONG - Infinite loop possible
def binary_search_wrong(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid  # Should be mid + 1
        else:
            right = mid  # Should be mid - 1
    
    return -1
```

**Why it's wrong:** If `arr[mid] != target`, we've already checked mid, so we should exclude it. Using `left = mid` or `right = mid` can cause infinite loop.

**Correct approach:**

```python
# CORRECT
if arr[mid] < target:
    left = mid + 1  # Exclude mid
else:
    right = mid - 1  # Exclude mid
```

### ❌ Pitfall 4: Wrong Comparison for Rotated Array

**What happens:** Can't determine which half is sorted.

```python
# WRONG - Incorrect sorted half detection
def search_rotated_wrong(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        
        # WRONG: Comparing mid with right
        if nums[mid] < nums[right]:
            # This doesn't tell us if left half is sorted
            # ...
```

**Correct approach:**

```python
# CORRECT - Compare left with mid
if nums[left] <= nums[mid]:
    # Left half is sorted
    if nums[left] <= target < nums[mid]:
        right = mid - 1
    else:
        left = mid + 1
else:
    # Right half is sorted
    if nums[mid] < target <= nums[right]:
        left = mid + 1
    else:
        right = mid - 1
```

### ❌ Pitfall 5: Forgetting to Handle Duplicates in Range Search

**What happens:** Return first occurrence instead of range.

```python
# WRONG - Returns one occurrence, not range
def search_range_wrong(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return [mid, mid]  # Only returns one position!
        # ...
```

**Correct approach:**

```python
# CORRECT - Find boundaries separately
def search_range_correct(nums, target):
    def find_left():
        left, right = 0, len(nums) - 1
        result = -1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                result = mid
                right = mid - 1  # Continue searching left
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return result
    
    def find_right():
        left, right = 0, len(nums) - 1
        result = -1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                result = mid
                left = mid + 1  # Continue searching right
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return result
    
    left_bound = find_left()
    if left_bound == -1:
        return [-1, -1]
    return [left_bound, find_right()]
```

## Variations and Extensions

### Variation 1: Binary Search on Answer Space

**Description:** Binary search on the answer itself rather than array indices.

**When to use:** Optimization problems where you're finding minimum/maximum value satisfying a condition.

**Implementation:**

```python
def find_minimum_days(weights: List[int], capacity: int) -> int:
    """
    Find minimum capacity of ship to ship all packages in 'days' days.
    
    Binary search on the answer (capacity).
    """
    def can_ship(cap: int) -> bool:
        """Check if we can ship with given capacity."""
        days_needed = 1
        current_load = 0
        
        for weight in weights:
            if current_load + weight > cap:
                days_needed += 1
                current_load = weight
            else:
                current_load += weight
        
        return days_needed <= capacity
    
    # Binary search on capacity
    left = max(weights)  # Minimum possible capacity
    right = sum(weights)  # Maximum possible capacity
    
    while left < right:
        mid = left + (right - left) // 2
        if can_ship(mid):
            right = mid  # Try smaller capacity
        else:
            left = mid + 1  # Need larger capacity
    
    return left
```

### Variation 2: Search in 2D Matrix

**Description:** Binary search in row-wise and column-wise sorted matrix.

**When to use:** Searching in sorted 2D structures.

**Implementation:**

```python
def search_matrix(matrix: List[List[int]], target: int) -> bool:
    """
    Search in matrix where:
    - Each row is sorted left to right
    - First element of each row > last element of previous row
    
    Treat as flattened sorted array.
    """
    if not matrix or not matrix[0]:
        return False
    
    m, n = len(matrix), len(matrix[0])
    left, right = 0, m * n - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        # Convert 1D index to 2D coordinates
        row, col = mid // n, mid % n
        mid_val = matrix[row][col]
        
        if mid_val == target:
            return True
        elif mid_val < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False
```

### Variation 3: Find K-th Smallest in Sorted Matrix

**Description:** Binary search on value range combined with counting.

**When to use:** Finding k-th element in structured data.

**Implementation:**

```python
def kth_smallest(matrix: List[List[int]], k: int) -> int:
    """
    Find k-th smallest element in row and column sorted matrix.
    
    Binary search on the value range.
    """
    n = len(matrix)
    left, right = matrix[0][0], matrix[n-1][n-1]
    
    def count_less_equal(mid: int) -> int:
        """Count elements <= mid."""
        count = 0
        col = n - 1
        for row in range(n):
            while col >= 0 and matrix[row][col] > mid:
                col -= 1
            count += col + 1
        return count
    
    while left < right:
        mid = left + (right - left) // 2
        if count_less_equal(mid) < k:
            left = mid + 1
        else:
            right = mid
    
    return left
```

### Variation 4: Find Minimum in Rotated Sorted Array

**Description:** Find the pivot point (minimum element) in rotated array.

**When to use:** Preprocessing for rotated array problems.

**Implementation:**

```python
def find_min(nums: List[int]) -> int:
    """
    Find minimum element in rotated sorted array.
    
    Time Complexity: O(log n)
    """
    left, right = 0, len(nums) - 1
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] > nums[right]:
            # Minimum is in right half
            left = mid + 1
        else:
            # Minimum is in left half (including mid)
            right = mid
    
    return nums[left]
```

### Variation 5: Square Root (Integer)

**Description:** Find floor of square root using binary search.

**When to use:** Mathematical computation without built-in functions.

**Implementation:**

```python
def my_sqrt(x: int) -> int:
    """
    Compute floor of square root of x.
    
    Binary search in range [0, x].
    """
    if x < 2:
        return x
    
    left, right = 1, x // 2
    
    while left <= right:
        mid = left + (right - left) // 2
        square = mid * mid
        
        if square == x:
            return mid
        elif square < x:
            left = mid + 1
        else:
            right = mid - 1
    
    return right  # Floor of square root
```

## Practice Problems

### Beginner

1. **Binary Search** - Classic binary search implementation
   - LeetCode #704

2. **First Bad Version** - Binary search to find boundary
   - LeetCode #278

3. **Search Insert Position** - Find index to insert target
   - LeetCode #35

4. **Valid Perfect Square** - Check if number is perfect square
   - LeetCode #367

### Intermediate

1. **Find Ceiling of Number** - Smallest element >= target
   - Similar to LeetCode #744 (Next Greatest Letter)

2. **Find First and Last Position** - Find range of target
   - LeetCode #34

3. **Search in Rotated Sorted Array** - Classic rotated array search
   - LeetCode #33

4. **Find Peak Element** - Find any peak in array
   - LeetCode #162

5. **Find Minimum in Rotated Sorted Array** - Find rotation point
   - LeetCode #153

6. **Search in Rotated Sorted Array II** - With duplicates
   - LeetCode #81

7. **Single Element in Sorted Array** - Find unique element
   - LeetCode #540

### Advanced

1. **Median of Two Sorted Arrays** - Binary search on smaller array
   - LeetCode #4

2. **Find K-th Smallest in Sorted Matrix** - Value range binary search
   - LeetCode #378

3. **Split Array Largest Sum** - Binary search on answer
   - LeetCode #410

4. **Capacity To Ship Packages** - Binary search on capacity
   - LeetCode #1011

5. **Koko Eating Bananas** - Binary search on eating speed
   - LeetCode #875

6. **Minimum Number of Days to Make m Bouquets** - Binary search on days
   - LeetCode #1482

## Real-World Applications

### Industry Use Cases

1. **Database Indexing:** B-trees and B+ trees use binary search principles for fast database queries, enabling millions of lookups per second in systems like MySQL, PostgreSQL, and MongoDB.

2. **Version Control Systems:** Git uses binary search (git bisect) to find the commit that introduced a bug by efficiently narrowing down thousands of commits to the problematic one.

3. **Auto-complete Systems:** Search engines and IDEs use binary search on sorted dictionaries to provide instant auto-complete suggestions as you type.

4. **Resource Allocation:** Cloud platforms use binary search on answer space to find optimal resource allocation (minimum servers, maximum throughput, optimal capacity).

5. **Computer Graphics:** Ray tracing and collision detection use binary space partitioning (BSP) trees with binary search for fast spatial queries.

### Popular Implementations

- **std::binary_search (C++ STL):** Standard library implementation
  - Used in competitive programming and production systems

- **Arrays.binarySearch (Java):** Built-in binary search
  - Powers many Java applications and Android apps

- **bisect module (Python):** Binary search and insertion
  - Used in data analysis, scientific computing

- **Database B-tree indices:** Generalized binary search trees
  - Core of relational database performance

### Practical Scenarios

- **Network Routing:** Finding optimal routes in routing tables using binary search on IP ranges
- **Memory Management:** Allocating memory blocks using binary search on free block lists
- **Game Development:** Pathfinding with binary space partitioning for collision detection
- **E-commerce:** Price range filtering, product search by attributes
- **Real-time Systems:** Finding time slots in scheduling systems

## Related Topics

### Prerequisites to Review

- **Arrays and Lists** - Understanding indexed data structures
- **Sorting Algorithms** - Binary search requires sorted data
- **Recursion** - Recursive binary search implementations
- **Mathematical Induction** - Understanding why binary search works
- **Big-O Notation** - Understanding logarithmic complexity

### Next Steps

- **Binary Search Trees** - Tree structure enabling binary search
- **Balanced Trees** - AVL, Red-Black trees for dynamic sorted data
- **Segment Trees** - Range queries with binary search principles
- **Binary Indexed Trees (Fenwick)** - Efficient prefix sums with binary structure
- **Ternary Search** - Extension for unimodal functions

### Similar Concepts

- **Divide and Conquer** - Binary search is a D&C algorithm
- **Two Pointers** - Similar pointer manipulation patterns
- **Sliding Window** - Both reduce search space systematically
- **Exponential Search** - Combines exponential growth with binary search

### Further Reading

- "Introduction to Algorithms" (CLRS) - Chapter on searching algorithms
- "Algorithm Design Manual" by Skiena - Binary search variations
- "Programming Pearls" by Bentley - Classic binary search problems
- [Binary Search Visualization](https://visualgo.net/en/binarysearch) - Interactive demonstrations
- [LeetCode Binary Search](https://leetcode.com/tag/binary-search/) - Practice problems
- "Competitive Programming 4" - Advanced binary search techniques
