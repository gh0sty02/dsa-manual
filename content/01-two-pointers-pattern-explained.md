# Two Pointers Pattern

**Difficulty:** Beginner to Medium
**Prerequisites:** Arrays, Basic sorting concepts, Problem-solving fundamentals
**Estimated Reading Time:** 25 minutes

## Introduction

The Two Pointers pattern is a fundamental algorithmic technique that uses two pointers to iterate through a data structure, typically an array or linked list, to solve problems efficiently. Instead of using nested loops (which would be O(n²)), we use two pointers moving through the data in a coordinated manner to achieve O(n) time complexity.

**Why it matters:** This pattern transforms problems that would typically require O(n²) time complexity into O(n) solutions, making them significantly faster for large datasets. It's one of the most frequently asked patterns in technical interviews and appears in real-world applications like text processing, data deduplication, and stream processing.

**Real-world analogy:** Imagine you're organizing books on a shelf. Instead of comparing every book with every other book (which would take forever), you use two hands - one starting from the left and one from the right. You move them towards each other based on certain criteria (like alphabetical order). This coordinated movement of two "pointers" is exactly how the pattern works!

## Core Concepts

### Key Principles

1. **Pointer initialization:** Typically start with pointers at different positions (both at start, one at start and one at end, or at specific positions based on the problem).

2. **Movement strategy:** Pointers move based on certain conditions - they might move towards each other, in the same direction at different speeds, or in opposite directions.

3. **Termination condition:** The algorithm terminates when pointers meet, cross each other, or reach specific boundaries.

4. **Optimal time complexity:** By processing each element at most once, we achieve linear O(n) time complexity instead of quadratic O(n²).

### Essential Terms

- **Left/Start pointer:** Usually begins at the start of the array (index 0)
- **Right/End pointer:** Usually begins at the end of the array (index n-1)
- **Slow/Fast pointers:** Two pointers moving at different speeds (covered in Fast & Slow pattern)
- **Window:** The subarray between two pointers
- **In-place modification:** Changing the array without using extra space

### Visual Overview

```
Type 1: Opposite Direction (Common for sorted arrays)
[1, 2, 3, 4, 5, 6, 7, 8]
 ↑                    ↑
left               right
 
Pointers move towards each other

Type 2: Same Direction (Common for removal/deduplication)
[1, 1, 2, 2, 3, 3]
 ↑ ↑
 i j

Both pointers move forward, j faster than i

Type 3: Sliding Window
[1, 2, 3, 4, 5, 6, 7, 8]
 ↑     ↑
left  right

Window expands/contracts based on condition
```

## How It Works

### General Algorithm Steps

**Type 1: Opposite Direction (e.g., Pair Sum)**
1. Initialize left pointer at index 0
2. Initialize right pointer at index n-1
3. While left < right:
   - Calculate result based on elements at both pointers
   - If result matches target: return or record
   - If result is less than target: move left pointer right (left++)
   - If result is greater than target: move right pointer left (right--)
4. Return result or indicate not found

**Type 2: Same Direction (e.g., Remove Duplicates)**
1. Initialize slow pointer i at 0
2. Initialize fast pointer j at 1
3. While j < n:
   - If arr[i] != arr[j]: 
     - Increment i
     - Set arr[i] = arr[j]
   - Increment j
4. Return i + 1 as the new length

### Step-by-Step Example: Pair with Target Sum

Problem: Find two numbers in sorted array [1, 2, 3, 4, 6] that sum to 6

```
Step 1: Initialize pointers
[1, 2, 3, 4, 6]
 ↑           ↑
left=0    right=4

Step 2: Check sum
sum = 1 + 6 = 7
7 > 6, so move right pointer left

Step 3:
[1, 2, 3, 4, 6]
 ↑        ↑
left=0  right=3

sum = 1 + 4 = 5
5 < 6, so move left pointer right

Step 4:
[1, 2, 3, 4, 6]
    ↑     ↑
  left=1 right=3

sum = 2 + 4 = 6
Found! Return [1, 3]
```

## How to Identify This Pattern

Recognizing when to use Two Pointers is critical for solving problems efficiently. Here are the key indicators:

### Primary Indicators ✓

**Input is sorted or can be sorted**
- Array/list is sorted in ascending or descending order
- Problem allows you to sort the input without losing information
- Keywords: "sorted array", "sorted list"
- Example: "Given a sorted array of integers..."

**Looking for pairs, triplets, or subsets with specific criteria**
- Finding two/three/four numbers that sum to a target
- Finding elements that satisfy a mathematical relationship
- Comparing or combining elements from different positions
- Keywords: "pair", "triplet", "two sum", "three sum"
- Example: "Find two numbers that add up to target"

**Need to compare elements from opposite ends**
- Processing from both start and end of array
- Checking for palindrome properties
- Finding optimal pairs by comparing extremes
- Keywords: "palindrome", "container", "from both ends"
- Example: "Check if string is a palindrome"

**In-place modification required with O(1) space**
- Cannot use extra data structures
- Must work within existing array
- Removing/rearranging elements without extra space
- Keywords: "in-place", "O(1) space", "without extra space"
- Example: "Remove duplicates in-place"

**Partitioning or rearranging elements**
- Separating elements based on a condition
- Dutch National Flag problem variants
- Moving elements to specific positions
- Keywords: "partition", "rearrange", "move zeros"
- Example: "Move all zeros to the end"

### Common Problem Phrases 🔑

Watch for these exact phrases in problem statements:
- "Find pair/triplet/quadruplet that sums to..."
- "Two Sum" / "Three Sum" / "Four Sum"
- "Remove duplicates from sorted array"
- "Is this a palindrome?"
- "Container with most water"
- "Trapping rain water"
- "Sort colors" / "Dutch National Flag"
- "Partition array"
- "Squaring a sorted array"
- "Backspace string compare"
- "Minimum window sort"

### When NOT to Use Two Pointers ✗

**Array is unsorted and CANNOT be sorted**
- If sorting destroys critical information (like original indices)
- When order preservation is essential
- → Use Hash Map instead

**Need to find ALL contiguous subarrays**
- Looking for substrings or subarrays
- Window-based problems
- → Use Sliding Window instead

**Working with Linked Lists**
- Pointer manipulation on linked structures
- → Use Fast & Slow Pointers pattern

**Need frequency counting or lookups**
- Tracking occurrences of elements
- → Use Hash Map pattern

### Quick Decision Checklist ✅

Ask yourself these questions:

1. **Is the input sorted?** → Strong signal for Two Pointers
2. **Can I sort the input without losing information?** → Probably Two Pointers
3. **Am I finding pairs/triplets that satisfy a condition?** → Likely Two Pointers
4. **Do I need to process from both ends?** → Two Pointers
5. **Is O(1) extra space required?** → Consider Two Pointers
6. **Am I partitioning or rearranging?** → Two Pointers

If you answered YES to any of these, Two Pointers is likely the right choice!

### Visual Recognition 👁️

**Two Pointers Pattern Looks Like:**
```
Array: [1, 2, 3, 4, 5, 6, 7, 8]
        ↑                    ↑
       left                right
```

**Moving towards each other:**
- left moves →
- right moves ←
- Process until they meet

**Or both moving same direction:**
```
Array: [1, 2, 3, 4, 5, 6, 7, 8]
        ↑  ↑
       slow fast
```

### Example Pattern Matching 💡

**Problem: "Find two numbers in a sorted array that sum to target"**

Analysis:
- ✓ Array is sorted
- ✓ Looking for a pair
- ✓ Can use two pointers (start and end)
- ✓ O(1) space solution exists

**Verdict: USE TWO POINTERS** ✓

**Problem: "Find longest substring with k distinct characters"**

Analysis:
- ✗ Looking for substring (contiguous window)
- ✗ Not about pairs
- ✗ Need to track character counts

**Verdict: USE SLIDING WINDOW** (Not Two Pointers) ✗

**Problem: "Find two numbers that sum to target (unsorted array)"**

Analysis:
- ✗ Array is unsorted
- ? Could sort, but loses indices
- ✓ Need pairs, but indices matter

**Verdict: USE HASH MAP** (Not Two Pointers) ✗

### Pattern vs Problem Type 📊

| Problem Type | Two Pointers? | Alternative |
|--------------|---------------|-------------|
| Sorted array pair sum | ✅ YES | Hash Map (if unsorted) |
| Triplet sum | ✅ YES | - |
| Quadruplet sum | ✅ YES | - |
| Remove duplicates (sorted) | ✅ YES | - |
| Palindrome validation | ✅ YES | - |
| Container with most water | ✅ YES | - |
| Partition array | ✅ YES | - |
| Longest substring | ❌ NO | Sliding Window |
| Subarray sum | ❌ NO | Sliding Window/Prefix Sum |
| Unsorted pair sum | ❌ NO | Hash Map |
| Detect cycle | ❌ NO | Fast & Slow Pointers |

### Red Flags 🚩

These suggest TWO POINTERS might NOT be the right choice:
- Problem says "substring" or "subarray" → Likely Sliding Window
- Array is unsorted and order matters → Likely Hash Map
- Need to count frequencies → Likely Hash Map
- Linked list mentioned → Likely Fast & Slow Pointers
- Finding cycles → Likely Fast & Slow Pointers

### Green Flags 🟢

These are STRONG indicators for TWO POINTERS:
- Problem says "sorted array"
- "Find pair/triplet that sums to..."
- "Is palindrome"
- "Remove duplicates in-place"
- "Partition" or "Dutch National Flag"
- "Container" or "water" problems
- "Comparing from both ends"



## Implementation

### Problem 1: Pair with Target Sum

```python
from typing import List, Optional

def pair_with_target_sum(arr: List[int], target: int) -> Optional[List[int]]:
    """
    Find indices of two numbers that add up to target in a sorted array.
    
    Args:
        arr: Sorted array of integers
        target: Target sum to find
    
    Returns:
        List of two indices if found, None otherwise
    
    Time Complexity: O(n) - single pass through array
    Space Complexity: O(1) - only two pointers used
    """
    left, right = 0, len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        
        if current_sum == target:
            return [left, right]  # Found the pair
        elif current_sum < target:
            left += 1  # Need larger sum, move left pointer right
        else:
            right -= 1  # Need smaller sum, move right pointer left
    
    return None  # No pair found


# Alternative: Using hash map (when array is not sorted)
def pair_with_target_sum_hashmap(arr: List[int], target: int) -> Optional[List[int]]:
    """
    Find indices using hash map approach - works for unsorted arrays.
    
    Time Complexity: O(n)
    Space Complexity: O(n) - hash map storage
    """
    num_map = {}  # num -> index
    
    for i, num in enumerate(arr):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    
    return None


# Usage Example
if __name__ == "__main__":
    arr = [1, 2, 3, 4, 6]
    target = 6
    
    result = pair_with_target_sum(arr, target)
    print(f"Indices: {result}")  # Output: [1, 3] (2 + 4 = 6)
    
    # Unsorted array example
    arr2 = [4, 1, 6, 3, 2]
    result2 = pair_with_target_sum_hashmap(arr2, 6)
    print(f"Indices (unsorted): {result2}")  # Output: [1, 2] (1 + 6 = 7... wait)
```

### Problem 2: Remove Duplicates (Find Non-Duplicate Number Instances)

```python
def remove_duplicates(arr: List[int]) -> int:
    """
    Remove duplicates from sorted array in-place.
    
    Args:
        arr: Sorted array with possible duplicates
    
    Returns:
        Length of array after removing duplicates
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - in-place modification
    """
    if not arr:
        return 0
    
    # Pointer for position of next unique element
    next_unique = 1
    
    # Iterate through array starting from second element
    for i in range(1, len(arr)):
        # If current element is different from previous
        if arr[i] != arr[next_unique - 1]:
            arr[next_unique] = arr[i]
            next_unique += 1
    
    return next_unique


# Usage Example
arr = [2, 3, 3, 3, 6, 9, 9]
length = remove_duplicates(arr)
print(f"New length: {length}")  # Output: 4
print(f"Array: {arr[:length]}")  # Output: [2, 3, 6, 9]
```

### Problem 3: Squaring a Sorted Array

```python
def make_squares(arr: List[int]) -> List[int]:
    """
    Square all elements of sorted array and return in sorted order.
    Array may contain negative numbers.
    
    Args:
        arr: Sorted array (can have negative numbers)
    
    Returns:
        New array with squared values in sorted order
    
    Time Complexity: O(n)
    Space Complexity: O(n) - for result array
    """
    n = len(arr)
    squares = [0] * n
    left, right = 0, n - 1
    highest_square_idx = n - 1  # Fill from right to left
    
    while left <= right:
        left_square = arr[left] ** 2
        right_square = arr[right] ** 2
        
        # Place larger square at the end
        if left_square > right_square:
            squares[highest_square_idx] = left_square
            left += 1
        else:
            squares[highest_square_idx] = right_square
            right -= 1
        
        highest_square_idx -= 1
    
    return squares


# Usage Example
arr = [-2, -1, 0, 2, 3]
result = make_squares(arr)
print(f"Squared array: {result}")  # Output: [0, 1, 4, 4, 9]
```

### Problem 4: Triplet Sum to Zero

```python
def search_triplets(arr: List[int]) -> List[List[int]]:
    """
    Find all unique triplets that sum to zero.
    
    Args:
        arr: Array of integers
    
    Returns:
        List of triplets that sum to zero
    
    Time Complexity: O(n²) - O(n) for outer loop, O(n) for two pointers
    Space Complexity: O(n) - for sorting (depends on sort implementation)
    """
    arr.sort()  # Sort array first
    triplets = []
    
    for i in range(len(arr) - 2):
        # Skip duplicates for first number
        if i > 0 and arr[i] == arr[i - 1]:
            continue
        
        # Use two pointers for remaining array
        search_pair(arr, -arr[i], i + 1, triplets)
    
    return triplets


def search_pair(arr: List[int], target_sum: int, left: int, triplets: List[List[int]]):
    """Helper function to find pairs that sum to target."""
    right = len(arr) - 1
    
    while left < right:
        current_sum = arr[left] + arr[right]
        
        if current_sum == target_sum:
            triplets.append([-target_sum, arr[left], arr[right]])
            left += 1
            right -= 1
            
            # Skip duplicates
            while left < right and arr[left] == arr[left - 1]:
                left += 1
            while left < right and arr[right] == arr[right + 1]:
                right -= 1
                
        elif current_sum < target_sum:
            left += 1
        else:
            right -= 1


# Usage Example
arr = [-3, 0, 1, 2, -1, 1, -2]
result = search_triplets(arr)
print(f"Triplets: {result}")  # Output: [[-3, 1, 2], [-2, 0, 2], [-2, 1, 1], [-1, 0, 1]]
```

### Problem 5: Triplet Sum Close to Target

```python
def triplet_sum_close_to_target(arr: List[int], target: int) -> int:
    """
    Find triplet sum closest to target.
    
    Args:
        arr: Array of integers
        target: Target sum
    
    Returns:
        Sum of triplet closest to target
    
    Time Complexity: O(n²)
    Space Complexity: O(n) - for sorting
    """
    arr.sort()
    smallest_diff = float('inf')
    closest_sum = 0
    
    for i in range(len(arr) - 2):
        left = i + 1
        right = len(arr) - 1
        
        while left < right:
            current_sum = arr[i] + arr[left] + arr[right]
            diff = abs(target - current_sum)
            
            # Update closest sum if current is closer
            if diff < smallest_diff:
                smallest_diff = diff
                closest_sum = current_sum
            
            # Move pointers based on comparison
            if current_sum < target:
                left += 1
            elif current_sum > target:
                right -= 1
            else:
                return current_sum  # Exact match
    
    return closest_sum


# Usage Example
arr = [-1, 0, 2, 3]
target = 3
result = triplet_sum_close_to_target(arr, target)
print(f"Closest sum: {result}")  # Output: 2 (from triplet [-1, 0, 3])
```

### Problem 6: Triplets with Smaller Sum

```python
def triplet_with_smaller_sum(arr: List[int], target: int) -> int:
    """
    Count triplets with sum less than target.
    
    Args:
        arr: Array of integers
        target: Target sum
    
    Returns:
        Count of triplets with sum < target
    
    Time Complexity: O(n²)
    Space Complexity: O(n) - for sorting
    """
    arr.sort()
    count = 0
    
    for i in range(len(arr) - 2):
        left = i + 1
        right = len(arr) - 1
        
        while left < right:
            current_sum = arr[i] + arr[left] + arr[right]
            
            if current_sum < target:
                # All triplets between left and right will have sum < target
                count += right - left
                left += 1
            else:
                right -= 1
    
    return count


# Variation: Return all triplets
def triplet_with_smaller_sum_list(arr: List[int], target: int) -> List[List[int]]:
    """Return all triplets with sum less than target."""
    arr.sort()
    triplets = []
    
    for i in range(len(arr) - 2):
        left = i + 1
        right = len(arr) - 1
        
        while left < right:
            current_sum = arr[i] + arr[left] + arr[right]
            
            if current_sum < target:
                # Add all valid triplets with current left
                for j in range(left + 1, right + 1):
                    triplets.append([arr[i], arr[left], arr[j]])
                left += 1
            else:
                right -= 1
    
    return triplets


# Usage Example
arr = [-1, 0, 2, 3]
target = 3
count = triplet_with_smaller_sum(arr, target)
print(f"Count: {count}")  # Output: 2
```

### Problem 7: Dutch National Flag Problem

```python
def dutch_flag_sort(arr: List[int]) -> None:
    """
    Sort array containing 0s, 1s, and 2s in-place.
    
    Args:
        arr: Array with only 0, 1, 2 values
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(1) - in-place sorting
    """
    # Three pointers: low for 0s, mid for 1s, high for 2s
    low, mid, high = 0, 0, len(arr) - 1
    
    while mid <= high:
        if arr[mid] == 0:
            # Swap with low pointer
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        elif arr[mid] == 1:
            # Already in correct position
            mid += 1
        else:  # arr[mid] == 2
            # Swap with high pointer
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1
            # Don't increment mid as swapped element needs checking


# Usage Example
arr = [1, 0, 2, 1, 0]
dutch_flag_sort(arr)
print(f"Sorted: {arr}")  # Output: [0, 0, 1, 1, 2]
```

### Problem 8: Quadruple Sum to Target

```python
def search_quadruplets(arr: List[int], target: int) -> List[List[int]]:
    """
    Find all unique quadruplets that sum to target.
    
    Args:
        arr: Array of integers
        target: Target sum
    
    Returns:
        List of quadruplets that sum to target
    
    Time Complexity: O(n³) - O(n²) for two loops, O(n) for two pointers
    Space Complexity: O(n) - for sorting
    """
    arr.sort()
    quadruplets = []
    
    for i in range(len(arr) - 3):
        # Skip duplicates for first number
        if i > 0 and arr[i] == arr[i - 1]:
            continue
        
        for j in range(i + 1, len(arr) - 2):
            # Skip duplicates for second number
            if j > i + 1 and arr[j] == arr[j - 1]:
                continue
            
            # Two pointers for remaining elements
            search_pairs(arr, target, i, j, quadruplets)
    
    return quadruplets


def search_pairs(arr: List[int], target: int, first: int, second: int, 
                 quadruplets: List[List[int]]):
    """Helper to find pairs that complete quadruplet."""
    left = second + 1
    right = len(arr) - 1
    
    while left < right:
        quad_sum = arr[first] + arr[second] + arr[left] + arr[right]
        
        if quad_sum == target:
            quadruplets.append([arr[first], arr[second], arr[left], arr[right]])
            left += 1
            right -= 1
            
            # Skip duplicates
            while left < right and arr[left] == arr[left - 1]:
                left += 1
            while left < right and arr[right] == arr[right + 1]:
                right -= 1
                
        elif quad_sum < target:
            left += 1
        else:
            right -= 1


# Usage Example
arr = [4, 1, 2, -1, 1, -3]
target = 1
result = search_quadruplets(arr, target)
print(f"Quadruplets: {result}")  # Output: [[-3, -1, 1, 4], [-3, 1, 1, 2]]
```

### Problem 9: Comparing Strings with Backspaces

```python
def backspace_compare(str1: str, str2: str) -> bool:
    """
    Compare two strings containing backspaces (#).
    
    Args:
        str1: First string with possible backspaces
        str2: Second string with possible backspaces
    
    Returns:
        True if strings are equal after processing backspaces
    
    Time Complexity: O(n + m)
    Space Complexity: O(1) - only pointers used
    """
    # Start from end of both strings
    index1 = len(str1) - 1
    index2 = len(str2) - 1
    
    while index1 >= 0 or index2 >= 0:
        # Get next valid character from str1
        i1 = get_next_valid_char_index(str1, index1)
        # Get next valid character from str2
        i2 = get_next_valid_char_index(str2, index2)
        
        # If both reached end, strings are equal
        if i1 < 0 and i2 < 0:
            return True
        
        # If one reached end or characters don't match
        if i1 < 0 or i2 < 0 or str1[i1] != str2[i2]:
            return False
        
        index1 = i1 - 1
        index2 = i2 - 1
    
    return True


def get_next_valid_char_index(s: str, index: int) -> int:
    """Find next valid character index considering backspaces."""
    backspace_count = 0
    
    while index >= 0:
        if s[index] == '#':
            backspace_count += 1
        elif backspace_count > 0:
            backspace_count -= 1
        else:
            break
        index -= 1
    
    return index


# Usage Example
str1 = "xy#z"
str2 = "xzz#"
result = backspace_compare(str1, str2)
print(f"Equal: {result}")  # Output: True (both become "xz")
```

### Problem 10: Minimum Window Sort

```python
def shortest_window_sort(arr: List[int]) -> int:
    """
    Find minimum subarray that needs to be sorted for entire array to be sorted.
    
    Args:
        arr: Array of integers
    
    Returns:
        Length of minimum subarray to sort
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    low, high = 0, len(arr) - 1
    
    # Find first element out of order from beginning
    while low < len(arr) - 1 and arr[low] <= arr[low + 1]:
        low += 1
    
    # Array is already sorted
    if low == len(arr) - 1:
        return 0
    
    # Find first element out of order from end
    while high > 0 and arr[high] >= arr[high - 1]:
        high -= 1
    
    # Find min and max in the subarray
    subarray_max = max(arr[low:high + 1])
    subarray_min = min(arr[low:high + 1])
    
    # Extend subarray to include elements larger than subarray_min
    while low > 0 and arr[low - 1] > subarray_min:
        low -= 1
    
    # Extend subarray to include elements smaller than subarray_max
    while high < len(arr) - 1 and arr[high + 1] < subarray_max:
        high += 1
    
    return high - low + 1


# Usage Example
arr = [1, 2, 5, 3, 7, 10, 9, 12]
result = shortest_window_sort(arr)
print(f"Minimum window length: {result}")  # Output: 5 (need to sort [5,3,7,10,9])
```

## Complexity Analysis

### Time Complexity

**General Two Pointers:**
- **Best Case:** O(n) - Linear scan when solution found early
- **Average Case:** O(n) - Each element visited at most once
- **Worst Case:** O(n) - Complete traversal needed

**Triplet Problems (with sorting):**
- **Time:** O(n²) - O(n log n) for sorting + O(n) outer loop × O(n) inner two-pointer scan

**Quadruplet Problems:**
- **Time:** O(n³) - Two nested loops + two-pointer scan

**Why O(n)?** 
In the basic two-pointer approach, each pointer moves at most n times total. Even though we have two pointers, we're not doing n × n work because:
- Each element is visited at most twice (once by each pointer)
- Total operations = O(2n) = O(n)

### Space Complexity

- **Space:** O(1) for basic two-pointer operations
- **Space:** O(n) when sorting is required (depends on sorting algorithm)
- **Space:** O(k) where k is the number of results to return

### Comparison with Alternatives

| Approach | Time (Avg) | Space | When to Use |
|----------|------------|-------|-------------|
| Two Pointers | O(n) or O(n²) | O(1) | Sorted arrays, in-place operations |
| Hash Map | O(n) | O(n) | Unsorted arrays, need fast lookup |
| Brute Force | O(n²) or O(n³) | O(1) | Small inputs, simple implementation |
| Sorting + Binary Search | O(n log n) | O(n) | When multiple searches needed |

## Examples

### Example 1: Basic Pair Sum (Success Case)

**Problem:** Find pair summing to 9 in [2, 5, 9, 11]

```
Input: arr = [2, 5, 9, 11], target = 9
Expected Output: [0, 1] (2 + 5 = 7... wait, let me recalculate)

Wait, 2 + 5 = 7, not 9. Let me use target = 11:

Step 1: [2, 5, 9, 11]
        ↑          ↑
       left=0    right=3
Sum = 2 + 11 = 13 > 11, move right left

Step 2: [2, 5, 9, 11]
        ↑       ↑
       left=0 right=2
Sum = 2 + 9 = 11 ✓
Output: [0, 2]
```

### Example 2: Remove Duplicates

**Problem:** Remove duplicates from [2, 2, 2, 11]

```
Initial: [2, 2, 2, 11]
         ↑ ↑
         i j=1

Step 1: arr[0] == arr[1] (2 == 2), move j
        [2, 2, 2, 11]
         ↑    ↑
         i=0  j=2

Step 2: arr[0] == arr[2] (2 == 2), move j
        [2, 2, 2, 11]
         ↑       ↑
         i=0     j=3

Step 3: arr[0] != arr[3] (2 != 11)
        i++, arr[i] = arr[j]
        [2, 11, 2, 11]
            ↑      ↑
            i=1    j=3

Step 4: j reaches end
Output: length = 2, array = [2, 11, ...]
```

### Example 3: Squaring Sorted Array with Negatives

**Problem:** Square [-3, -1, 0, 1, 4] and keep sorted

```
Input: [-3, -1, 0, 1, 4]
Output: [0, 1, 1, 9, 16]

Process (filling from right to left):
Step 1: [-3, -1, 0, 1, 4]  →  [_, _, _, _, 16]
         ↑              ↑
        left          right
        9 < 16, place 16, right--

Step 2: [-3, -1, 0, 1, _]  →  [_, _, _, 9, 16]
         ↑           ↑
        left       right
        9 > 1, place 9, left++

Step 3: [_, -1, 0, 1, _]  →  [_, _, 1, 9, 16]
            ↑      ↑
          left   right
        1 == 1, place 1, right--

Step 4: [_, -1, 0, _, _]  →  [_, 1, 1, 9, 16]
            ↑   ↑
          left right
        1 > 0, place 1, left++

Step 5: [_, _, 0, _, _]  →  [0, 1, 1, 9, 16]
               ↑
          left==right
        place 0, done
```

### Example 4: Triplet Sum to Zero

**Problem:** Find triplets in [-3, 0, 1, 2, -1, 1, -2]

```
After sorting: [-3, -2, -1, 0, 1, 1, 2]

Iteration i=0 (num = -3):
Target pair sum = 3
[-3, -2, -1, 0, 1, 1, 2]
         ↑              ↑
        left=1       right=6

-2 + 2 = 0 < 3, left++
-1 + 2 = 1 < 3, left++
0 + 2 = 2 < 3, left++
1 + 2 = 3 ✓ → Found: [-3, 1, 2]

Continue for other values of i...
```

## Edge Cases

### 1. Empty or Single Element Array
**Scenario:** arr = [] or arr = [5]
**Challenge:** No pairs possible with less than 2 elements
**Solution:** Check length at the start
```python
if len(arr) < 2:
    return None  # or [] depending on problem
```

### 2. No Solution Exists
**Scenario:** Finding pair with sum 100 in [1, 2, 3, 4, 5]
**Challenge:** Pointers will cross without finding answer
**Solution:** Return None/empty when pointers meet without finding solution
```python
while left < right:
    # ... search logic
return None  # After loop completes without finding
```

### 3. All Elements Are Same
**Scenario:** Remove duplicates from [5, 5, 5, 5, 5]
**Challenge:** Result should be single element
**Solution:** Algorithm naturally handles this
```python
# Result: [5, _, _, _, _] with length 1
```

### 4. Array with Duplicates in Triplet/Quadruplet
**Scenario:** [-1, -1, 2, 2] finding triplets
**Challenge:** Avoid returning duplicate triplets
**Solution:** Skip duplicate values
```python
if i > 0 and arr[i] == arr[i-1]:
    continue  # Skip duplicate
```

### 5. Negative and Positive Numbers
**Scenario:** Squaring array with mixed signs
**Challenge:** Largest square could be at either end
**Solution:** Use two pointers from both ends
```python
# Compare absolute values or squares from both ends
```

### 6. Target at Boundaries
**Scenario:** Pair sum where answer is first and last element
**Challenge:** Must check initial pointer positions
**Solution:** Algorithm naturally checks this first
```python
# First comparison: arr[0] + arr[n-1]
```

## Common Pitfalls

### ❌ Pitfall 1: Not Sorting When Required
**What happens:** Two-pointer logic fails on unsorted arrays for problems requiring sorted order
**Why it's wrong:** 
```python
# Wrong: Using two pointers on unsorted array for pair sum
arr = [4, 1, 3, 2]
left, right = 0, 3
# arr[0] + arr[3] = 4 + 2 = 6, but we might miss 1 + 3 = 4
```
**Correct approach:**
```python
# Always sort first when problem requires it
arr.sort()  # Now: [1, 2, 3, 4]
left, right = 0, len(arr) - 1
# Systematic scan will find all pairs
```

### ❌ Pitfall 2: Forgetting to Skip Duplicates
**What happens:** Returns duplicate triplets/quadruplets
**Why it's wrong:**
```python
# Input: [-1, -1, 2]
# Without skipping: returns [-1, -1, 2] twice
for i in range(len(arr)):
    # Missing: if i > 0 and arr[i] == arr[i-1]: continue
```
**Correct approach:**
```python
for i in range(len(arr)):
    if i > 0 and arr[i] == arr[i-1]:
        continue  # Skip duplicates
    # ... rest of logic
```

### ❌ Pitfall 3: Moving Both Pointers When Only One Should Move
**What happens:** Skips potential solutions
**Why it's wrong:**
```python
if current_sum == target:
    # Wrong: moving both pointers unconditionally
    left += 1
    right -= 1
    # Might skip valid solutions in between
```
**Correct approach:**
```python
if current_sum == target:
    # Record result
    result.append([left, right])
    # Then move based on logic (could be both, could be one)
    left += 1
    right -= 1
    # Skip duplicates if needed
```

### ❌ Pitfall 4: Not Handling Pointer Boundary Conditions
**What happens:** Index out of bounds errors
**Why it's wrong:**
```python
while left < right:
    # ... logic
    left += 1  # Might exceed right
    right -= 1  # Might go below left
# Missing: checks that left and right are valid
```
**Correct approach:**
```python
while left < right:  # This condition prevents crossing
    # ... logic
    if some_condition:
        left += 1
    else:
        right -= 1
# Condition ensures pointers never cross invalidly
```

### ❌ Pitfall 5: Incorrect Initialization for In-Place Problems
**What happens:** Overwrites needed values
**Why it's wrong:**
```python
# Remove duplicates - wrong initialization
next_unique = 0  # Should be 1
for i in range(len(arr)):
    # Will overwrite arr[0] incorrectly
```
**Correct approach:**
```python
next_unique = 1  # First element always stays
for i in range(1, len(arr)):  # Start from second element
    if arr[i] != arr[next_unique - 1]:
        arr[next_unique] = arr[i]
        next_unique += 1
```

## Variations and Extensions

### Variation 1: Three Pointers (Dutch National Flag)
**Description:** Uses three pointers to partition array into three sections
**When to use:** Sorting arrays with limited distinct values (0, 1, 2)
**Key differences:** Third pointer manages a middle section
**Implementation:**
```python
# low: boundary for 0s
# mid: current element
# high: boundary for 2s
# Invariant: [0...low-1] are 0s, [low...mid-1] are 1s, [high+1...n] are 2s
```

### Variation 2: Two Pointers Same Direction Different Speed
**Description:** Both pointers move forward but at different speeds
**When to use:** In-place array modifications, removing elements
**Key differences:** No opposite-direction movement
**Example:**
```python
# i: slow pointer (write position)
# j: fast pointer (read position)
i, j = 0, 0
while j < len(arr):
    if condition:
        arr[i] = arr[j]
        i += 1
    j += 1
```

### Variation 3: Expanding Window
**Description:** Start pointers together, expand outward
**When to use:** Finding palindromes, expanding around center
**Implementation:**
```python
def expand_around_center(s, left, right):
    while left >= 0 and right < len(s) and s[left] == s[right]:
        left -= 1
        right += 1
    return right - left - 1  # Length of palindrome
```

### Variation 4: Multiple Passes with Different Pointer Strategies
**Description:** First pass identifies range, second pass processes
**When to use:** Complex in-place modifications
**Example:** Minimum Window Sort uses this approach

### Variation 5: Two Pointers with Hash Map
**Description:** Combine pointers with hash map for O(1) lookups
**When to use:** When you need fast existence checks while traversing
**Implementation:**
```python
# Track elements between pointers
window_map = {}
left = right = 0
while right < len(arr):
    window_map[arr[right]] = window_map.get(arr[right], 0) + 1
    # ... condition check
    right += 1
```

## Practice Problems

### Beginner
1. **Two Sum (LeetCode #1)** - Find two numbers that add up to target
   - Classic introduction to two pointers vs hash map

2. **Valid Palindrome (LeetCode #125)** - Check if string is palindrome
   - Two pointers from both ends

3. **Remove Element (LeetCode #27)** - Remove all instances of value
   - Two pointers same direction

4. **Merge Sorted Array (LeetCode #88)** - Merge two sorted arrays in-place
   - Two pointers from end

### Intermediate
1. **3Sum (LeetCode #15)** - Find all triplets that sum to zero
   - Classic triplet problem with duplicate handling

2. **Container With Most Water (LeetCode #11)** - Find max area between lines
   - Two pointers optimization problem

3. **Sort Colors (LeetCode #75)** - Dutch National Flag problem
   - Three pointers partitioning

4. **3Sum Closest (LeetCode #16)** - Find triplet sum closest to target
   - Variation of triplet sum

5. **4Sum (LeetCode #18)** - Find all quadruplets summing to target
   - Extension to quadruplets

6. **Partition Labels (LeetCode #763)** - Partition string into segments
   - Two pointers with character tracking

7. **Backspace String Compare (LeetCode #844)** - Compare strings with backspaces
   - Two pointers from end with state

### Advanced
1. **Trapping Rain Water (LeetCode #42)** - Calculate trapped water
   - Two pointers with height tracking

2. **Minimum Window Substring (LeetCode #76)** - Find minimum window containing all characters
   - Sliding window variant

3. **Longest Mountain in Array (LeetCode #845)** - Find longest mountain subarray
   - Multiple pointer passes

4. **Subarrays with K Different Integers (LeetCode #992)** - Count subarrays
   - Advanced sliding window

## Real-World Applications

### Industry Use Cases

1. **Data Deduplication:** Removing duplicates in large sorted datasets (log processing, database cleanup)
   - Used in data warehouses to clean and merge duplicate records efficiently

2. **Two-Factor Authentication:** Comparing user input with time-sensitive codes
   - Backspace comparison logic used in password/PIN verification

3. **Network Packet Processing:** Finding matching packet pairs in streams
   - Used in TCP/IP to match request-response pairs

4. **Financial Trading:** Finding pairs of stocks that meet correlation criteria
   - High-frequency trading systems use two-pointer techniques for real-time pair analysis

### Popular Implementations

- **Python's `sorted()` with `key` and two pointers:** Used internally for efficient comparison operations
- **Git diff algorithm:** Uses two-pointer technique to find longest common subsequences
- **Database JOIN operations:** Two-pointer merge join for sorted tables
- **Compression algorithms:** LZ77 uses two-pointer sliding window approach

### Practical Scenarios

- **Autocomplete systems:** Finding word pairs that form valid completions
- **Image processing:** Comparing pixel arrays for similarity detection
- **Genomic sequencing:** Finding matching DNA sequence pairs
- **E-commerce:** Finding product combinations that meet price constraints (e.g., "Frequently bought together")
- **Meeting room scheduling:** Finding overlapping time slots using interval merging
- **Social networks:** Finding mutual friends (intersection of sorted friend lists)

## Related Topics

### Prerequisites to Review
- **Array manipulation** - Essential for understanding pointer movement
- **Sorting algorithms** - Many two-pointer problems require sorted input
- **Time/Space complexity** - Understanding why two pointers is O(n) vs O(n²)

### Next Steps
- **Sliding Window Pattern** - Natural extension for subarray problems
- **Fast & Slow Pointers** - Applied to linked lists and cycle detection
- **Binary Search** - Another divide-and-conquer approach for sorted arrays
- **Hash Maps** - Alternative approach for many two-pointer problems

### Similar Concepts
- **Merge Sort** - Uses two pointers to merge sorted arrays
- **Quick Sort partitioning** - Uses two pointers to partition around pivot
- **String matching algorithms** - KMP uses two pointers for pattern matching

### Further Reading
- [LeetCode Two Pointers Study Guide](https://leetcode.com/tag/two-pointers/)
- [GeeksforGeeks Two Pointer Technique](https://www.geeksforgeeks.org/two-pointers-technique/)
- Introduction to Algorithms (CLRS) - Chapter on sorting and searching
- Algorithm Design Manual by Skiena - Section on pointer-based techniques
- [Educative.io: Grokking Coding Interview Patterns](https://www.educative.io/courses/grokking-coding-interview)
