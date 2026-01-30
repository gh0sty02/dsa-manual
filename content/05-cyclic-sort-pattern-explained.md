# Cyclic Sort Pattern

**Difficulty:** Easy to Medium
**Prerequisites:** Arrays, Basic sorting concepts
**Estimated Reading Time:** 22 minutes

## Introduction

Cyclic Sort is a brilliant and elegant pattern for sorting arrays containing numbers in a given range (typically 1 to n or 0 to n). Unlike traditional comparison-based sorting algorithms that take O(n log n) time, Cyclic Sort achieves O(n) time complexity by utilizing the fact that we know exactly where each number should be placed.

**Why it matters:** This pattern is a game-changer for interview problems involving finding missing numbers, duplicates, or corrupt data in arrays with numbers in a specific range. It's a favorite in technical interviews because it tests your ability to think beyond standard algorithms. Companies value candidates who can recognize when a specialized algorithm outperforms generic solutions.

**Real-world analogy:** Imagine you're organizing a deck of numbered cards (1 through 52) that got shuffled. Instead of comparing cards and swapping them like typical sorting, you look at each card and directly place it in its correct position. Card #7 goes to position 7, Card #23 goes to position 23. This direct placement is exactly what Cyclic Sort does - it uses the value itself to determine where it belongs!

## Core Concepts

### Key Principles

1. **Direct placement:** Each number tells us its correct index. Number `i` belongs at index `i-1` (for 1-indexed) or index `i` (for 0-indexed).

2. **Swap until correct:** Keep swapping the current number to its correct position until the current position has the right number.

3. **Each element moved once:** Each number is swapped at most once to its final position, giving O(n) time.

4. **In-place sorting:** No extra space needed beyond the input array.

5. **Range requirement:** Only works when numbers are in range [1, n] or [0, n] or similar consecutive ranges.

### Essential Terms

- **Correct index:** The position where a number should be placed (value - 1 for 1-indexed)
- **Cyclic swap:** Swapping elements until each reaches its correct position
- **Missing number:** A number in the range that doesn't appear in the array
- **Duplicate:** A number that appears more than once
- **Corrupt pair:** A pair where one number is wrong

### Visual Overview

```
Example: Sort [3, 1, 5, 4, 2]

Target: Each number i should be at index i-1
        Number 1 → index 0
        Number 2 → index 1
        Number 3 → index 2
        etc.

Initial: [3, 1, 5, 4, 2]
         i=0

Step 1: nums[0]=3, should be at index 2
        Swap 3 and nums[2]=5
        [5, 1, 3, 4, 2]

Step 2: nums[0]=5, should be at index 4
        Swap 5 and nums[4]=2
        [2, 1, 3, 4, 5]

Step 3: nums[0]=2, should be at index 1
        Swap 2 and nums[1]=1
        [1, 2, 3, 4, 5]

Step 4: nums[0]=1, correct position! Move i++

Final: [1, 2, 3, 4, 5] ✓

Total swaps: 3 (each number moved at most once)
```

## How It Works

### Cyclic Sort Algorithm

1. **Initialize pointer i = 0**
2. **While i < n:**
   - Calculate correct index for current number
   - If number is not at correct position:
     - Swap it to correct position
     - Don't increment i (check new number at position i)
   - Else:
     - Number is correct, move to next (i++)
3. **Array is now sorted**

### Why It's O(n)

Each number is placed in its correct position in one swap. Even though we have a while loop inside another while loop, the inner while can execute at most n times total across all iterations because each swap puts at least one number in its final position.

### Step-by-Step Example: Finding Missing Number

Problem: Find missing number in [4, 0, 3, 1] (range 0 to n)

```
Array: [4, 0, 3, 1]
Range: 0 to 4 (n=4, so range is 0 to n)

Goal: Place each number at its own index
      0 → index 0
      1 → index 1
      2 → index 2
      3 → index 3
      4 → index 4 (but we only have 4 positions!)

Step 1: i=0, nums[0]=4
Correct index for 4 = 4
But we only have indices 0-3!
Skip numbers >= n
i++

Step 2: i=1, nums[1]=0
Correct index for 0 = 0
nums[0] ≠ 0, so swap
[0, 4, 3, 1]
Don't increment i

Step 3: i=1, nums[1]=4
Skip (4 >= n), i++

Step 4: i=2, nums[2]=3
Correct index for 3 = 3
nums[3] ≠ 3, so swap
[0, 4, 1, 3]
Don't increment i

Step 5: i=2, nums[2]=1
Correct index for 1 = 1
nums[1] ≠ 1, so swap
[0, 1, 4, 3]
Don't increment i

Step 6: i=2, nums[2]=4
Skip, i++

Step 7: i=3, nums[3]=3
Correct! i++

Final: [0, 1, 4, 3]

Now find first index where nums[i] ≠ i:
Index 2: nums[2]=4, but should be 2
Missing number = 2
```

## How to Identify This Pattern

Cyclic Sort is a specialized pattern with very specific use cases. Here's how to recognize it:

### Primary Indicators ✓

**Array contains numbers in range [1, n] or [0, n]**
- Numbers are consecutive or in specific bounded range
- Array size is n, contains numbers from 1 to n (possibly with missing/duplicates)
- Keywords: "array of size n", "numbers 1 to n", "range [1,n]"
- Example: "Given array of size 5 with numbers from 1 to 5..."

**Looking for missing numbers**
- Find which numbers are absent from range
- Identify gaps in sequence
- Keywords: "missing number", "find missing", "absent"
- Example: "Find the missing number from 1 to n"

**Looking for duplicate numbers**
- Identify numbers appearing more than once
- Find repeated values
- Keywords: "duplicate", "repeated", "appears twice"
- Example: "Find all duplicate numbers"

**"Corrupt" or "wrong" numbers mentioned**
- Numbers that shouldn't be there
- Values out of place
- Keywords: "corrupt", "wrong", "mismatch", "error"
- Example: "Find the corrupt pair"

**Need O(n) time AND O(1) space**
- Linear time requirement
- Constant space constraint
- Cannot use hash set
- Keywords: "O(n) time", "O(1) space", "constant space"
- Example: "Solve in linear time without extra space"

**Value-as-index relationship possible**
- Each number can tell you its correct position
- Number i belongs at index i-1 (or i for 0-indexed)
- This is the KEY insight for Cyclic Sort!

### Common Problem Phrases 🔑

Watch for these exact phrases:
- "Find missing number from 1 to n"
- "Find all missing numbers"
- "Find the duplicate number"
- "Find all duplicates in the array"
- "Set mismatch" / "Find corrupt pair"
- "First missing positive"
- "Find k missing positive numbers"
- "Numbers from 1 to n"
- "Cyclic sort"
- "Array contains numbers in range..."

### The Golden Rule 🏆

**If each number can tell you EXACTLY where it belongs (its index), use Cyclic Sort!**

**For range [1, n]:** Number k → Index k-1
**For range [0, n-1]:** Number k → Index k

This is the fundamental insight!

### When NOT to Use Cyclic Sort ✗

**Range is too large**
- Numbers from 1 to 1,000,000 in array of size 100
- Range much larger than array size
- → Use Hash Set

**Numbers not in consecutive range**
- Random values
- No pattern to range
- → Use other sorting or Hash Set

**Cannot modify original array**
- Need to preserve original
- Read-only array
- → Use Hash Set (but needs O(n) space)

**Numbers can be negative**
- Negative values don't map to indices
- → Special handling or different approach

**Need stable sort**
- Original order matters
- → Use merge sort or other stable sort

### Quick Decision Checklist ✅

Ask yourself:

1. **Numbers in range [1,n] or [0,n]?** → Strong Cyclic Sort signal
2. **Array size matches the range?** → Cyclic Sort
3. **Looking for missing/duplicate?** → Cyclic Sort
4. **Can use value as index (value-1)?** → Cyclic Sort
5. **Need O(n) time AND O(1) space?** → Cyclic Sort
6. **Problem says "1 to n" or "0 to n"?** → Cyclic Sort

If YES to questions 1 AND (2 OR 3), it's definitely Cyclic Sort!

### Algorithm Signature 📝

**Cyclic Sort Template:**
```python
i = 0
while i < len(nums):
    correct_index = nums[i] - 1  # For range [1,n]
    # correct_index = nums[i]    # For range [0,n-1]
    
    if nums[i] != nums[correct_index]:
        # Swap to correct position
        nums[i], nums[correct_index] = nums[correct_index], nums[i]
        # Don't increment i - check new number at i
    else:
        i += 1  # Number correct, move forward
```

**Key Points:**
- Don't increment i after swap
- Swap until correct number at position i
- Each number placed at most once
- O(n) time even with nested loop!

### Visual Recognition 👁️

**If you think: "Number 3 should be at position 3", that's Cyclic Sort!**

```
Array: [3, 1, 5, 4, 2]
Goal:  [1, 2, 3, 4, 5]

Number 3 should be at index 2 (3-1)
Number 1 should be at index 0 (1-1)
Number 5 should be at index 4 (5-1)
...

Each number knows where it belongs!
```

**Process:**
```
[3, 1, 5, 4, 2]
 ↓ swap 3 to index 2
[5, 1, 3, 4, 2]
 ↓ swap 5 to index 4
[2, 1, 3, 4, 5]
 ↓ swap 2 to index 1
[1, 2, 3, 4, 5] ✓
```

### Example Pattern Matching 💡

**Problem: "Find missing number in [0,n]"**

Analysis:
- ✓ Range [0, n] explicitly given
- ✓ Array size n+1 with one missing
- ✓ Can place each number at its index
- ✓ O(n) time, O(1) space needed

**Verdict: USE CYCLIC SORT** ✓

**Problem: "Find all missing numbers in array with numbers 1 to n"**

Analysis:
- ✓ Range [1, n] with size n
- ✓ Looking for missing numbers
- ✓ Each number i → index i-1

**Verdict: USE CYCLIC SORT** ✓

**Problem: "Find duplicate in array of numbers 1 to 1,000,000"**

Analysis:
- ✗ Range much larger than array
- ✗ Cannot use value as index directly
- ? Can use Floyd's if array size is small

**Verdict: USE HASH SET or FLOYD'S** (Not Cyclic Sort) ✗

**Problem: "Find two numbers that sum to target"**

Analysis:
- ✗ Not about consecutive range
- ✗ Not about missing/duplicate

**Verdict: USE TWO POINTERS or HASH MAP** (Not Cyclic Sort) ✗

### Pattern vs Problem Type 📊

| Problem Type | Cyclic Sort? | Alternative |
|--------------|--------------|-------------|
| Missing number [1,n] | ✅ YES | XOR, Sum formula |
| Find all missing [1,n] | ✅ YES | Hash Set |
| Duplicate in [1,n] | ✅ YES | Hash Set, Floyd's |
| All duplicates [1,n] | ✅ YES | Hash Set |
| First missing positive | ✅ YES | Hash Set |
| Corrupt pair [1,n] | ✅ YES | Hash Set |
| Missing in large range | ❌ NO | Hash Set |
| Two sum | ❌ NO | Two Pointers/Hash Map |
| Sort random numbers | ❌ NO | Comparison sort |

### Range Variations 🔢

| Range | Correct Index Formula | Example |
|-------|----------------------|---------|
| [1, n] | `index = value - 1` | 3 → index 2 |
| [0, n-1] | `index = value` | 3 → index 3 |
| [k, k+n-1] | `index = value - k` | 5 → index 2 (if k=3) |

### Why O(n) Time? ⏱️

**Seems like O(n²) with nested loops, but it's O(n)!**

Reason:
- Each element is swapped at most once to its final position
- Outer loop: O(n) iterations
- Inner swaps: O(n) total across ALL iterations
- Total: O(n) + O(n) = O(n)

Each number takes a "one-way trip" to its final position!

### Keywords Cheat Sheet 📝

**STRONG "Cyclic Sort" Keywords:**
- "1 to n"
- "0 to n"
- "range [1,n]"
- missing number
- duplicate number

**MODERATE Keywords:**
- corrupt
- mismatch
- first missing positive
- consecutive
- O(1) space

**ANTI-Keywords (probably NOT Cyclic Sort):**
- large range
- random values
- unsorted
- read-only array

### Red Flags 🚩

These suggest CYCLIC SORT might NOT be right:
- Range >> array size → Hash Set
- Negative numbers → Special handling needed
- Cannot modify array → Hash Set
- No clear range given → Other pattern
- Numbers not related to positions → Other sorting

### Green Flags 🟢

STRONG indicators for CYCLIC SORT:
- "Array of size n with numbers 1 to n"
- "Find missing number from 1 to n"
- "Find duplicate in [1,n]"
- "O(n) time and O(1) space"
- "First missing positive integer"
- "Set mismatch"
- Value-as-index relationship obvious

### Pro Tip 💡

**If you can map the value directly to an index without a hash table, that's Cyclic Sort!**

Example:
- Value 5 → Index 4 (for 1-indexed)
- Value 3 → Index 3 (for 0-indexed)

This direct mapping is the secret sauce!



## Implementation

### Problem 1: Cyclic Sort (Basic Template)

```python
from typing import List

def cyclic_sort(nums: List[int]) -> List[int]:
    """
    Sort array containing numbers from 1 to n using cyclic sort.
    
    Args:
        nums: Array with numbers in range [1, n]
    
    Returns:
        Sorted array
    
    Time Complexity: O(n) - each element moved at most once
    Space Complexity: O(1) - in-place sorting
    """
    i = 0
    
    while i < len(nums):
        # Calculate correct index for current number
        # For 1-indexed: number k goes to index k-1
        correct_index = nums[i] - 1
        
        # If number is not at correct position
        if nums[i] != nums[correct_index]:
            # Swap to correct position
            nums[i], nums[correct_index] = nums[correct_index], nums[i]
            # Don't increment i - check new number at position i
        else:
            # Number is at correct position, move to next
            i += 1
    
    return nums


# Usage Example
nums = [3, 1, 5, 4, 2]
print(cyclic_sort(nums))  # Output: [1, 2, 3, 4, 5]

nums = [2, 6, 4, 3, 1, 5]
print(cyclic_sort(nums))  # Output: [1, 2, 3, 4, 5, 6]
```

### Problem 2: Find the Missing Number (LeetCode #268)

```python
def findMissingNumber(nums: List[int]) -> int:
    """
    Find missing number in array containing 0 to n.
    
    Args:
        nums: Array with n numbers from range [0, n]
    
    Returns:
        The missing number
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    i = 0
    n = len(nums)
    
    # Place each number at its correct index
    while i < n:
        correct_index = nums[i]
        
        # Only swap if:
        # 1. Number is in valid range (< n)
        # 2. Number is not at correct position
        if nums[i] < n and nums[i] != nums[correct_index]:
            nums[i], nums[correct_index] = nums[correct_index], nums[i]
        else:
            i += 1
    
    # Find first index where nums[i] != i
    for i in range(n):
        if nums[i] != i:
            return i
    
    # If all indices match, missing number is n
    return n


# Usage Example
nums = [3, 0, 1]
print(findMissingNumber(nums))  # Output: 2

nums = [9,6,4,2,3,5,7,0,1]
print(findMissingNumber(nums))  # Output: 8
```

### Problem 3: Find All Missing Numbers (LeetCode #448)

```python
def findDisappearedNumbers(nums: List[int]) -> List[int]:
    """
    Find all missing numbers from 1 to n.
    
    Args:
        nums: Array with numbers in range [1, n]
    
    Returns:
        List of all missing numbers
    
    Time Complexity: O(n)
    Space Complexity: O(1) excluding output
    """
    i = 0
    
    # Cyclic sort
    while i < len(nums):
        correct_index = nums[i] - 1
        
        if nums[i] != nums[correct_index]:
            nums[i], nums[correct_index] = nums[correct_index], nums[i]
        else:
            i += 1
    
    # Find all positions where number doesn't match
    missing = []
    for i in range(len(nums)):
        if nums[i] != i + 1:
            missing.append(i + 1)
    
    return missing


# Usage Example
nums = [4,3,2,7,8,2,3,1]
print(findDisappearedNumbers(nums))  # Output: [5, 6]

nums = [1,1]
print(findDisappearedNumbers(nums))  # Output: [2]
```

### Problem 4: Find the Duplicate Number (LeetCode #287)

```python
def findDuplicate(nums: List[int]) -> int:
    """
    Find the duplicate number in array (read-only variation).
    Array has n+1 numbers in range [1, n].
    
    Args:
        nums: Array with one duplicate
    
    Returns:
        The duplicate number
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    # Using cyclic sort approach (modifies array)
    i = 0
    
    while i < len(nums):
        correct_index = nums[i] - 1
        
        if nums[i] != nums[correct_index]:
            nums[i], nums[correct_index] = nums[correct_index], nums[i]
        else:
            # Found duplicate: number at wrong position but
            # correct position already has same number
            if i != correct_index:
                return nums[i]
            i += 1
    
    return -1


# Alternative: Fast & Slow Pointers (doesn't modify array)
def findDuplicateFloyd(nums: List[int]) -> int:
    """
    Find duplicate using Floyd's cycle detection.
    Treats array as linked list where nums[i] points to nums[nums[i]].
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    # Phase 1: Find intersection point
    slow = fast = nums[0]
    
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break
    
    # Phase 2: Find entrance to cycle (duplicate)
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]
    
    return slow


# Usage Example
nums = [1,3,4,2,2]
print(findDuplicate(nums))  # Output: 2

nums = [3,1,3,4,2]
print(findDuplicate(nums))  # Output: 3
```

### Problem 5: Find All Duplicates (LeetCode #442)

```python
def findDuplicates(nums: List[int]) -> List[int]:
    """
    Find all numbers appearing twice in array.
    
    Args:
        nums: Array where some numbers appear twice
    
    Returns:
        List of all duplicates
    
    Time Complexity: O(n)
    Space Complexity: O(1) excluding output
    """
    i = 0
    
    # Cyclic sort
    while i < len(nums):
        correct_index = nums[i] - 1
        
        if nums[i] != nums[correct_index]:
            nums[i], nums[correct_index] = nums[correct_index], nums[i]
        else:
            i += 1
    
    # Find duplicates
    duplicates = []
    for i in range(len(nums)):
        if nums[i] != i + 1:
            duplicates.append(nums[i])
    
    return duplicates


# Usage Example
nums = [4,3,2,7,8,2,3,1]
print(findDuplicates(nums))  # Output: [2, 3]

nums = [1,1,2]
print(findDuplicates(nums))  # Output: [1]
```

### Problem 6: Find the Corrupt Pair (LeetCode #645 - Set Mismatch)

```python
def findErrorNums(nums: List[int]) -> List[int]:
    """
    Find the number that appears twice and the missing number.
    
    Args:
        nums: Array with one duplicate and one missing from [1, n]
    
    Returns:
        [duplicate, missing]
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    i = 0
    
    # Cyclic sort
    while i < len(nums):
        correct_index = nums[i] - 1
        
        if nums[i] != nums[correct_index]:
            nums[i], nums[correct_index] = nums[correct_index], nums[i]
        else:
            i += 1
    
    # Find corrupt pair
    for i in range(len(nums)):
        if nums[i] != i + 1:
            return [nums[i], i + 1]  # [duplicate, missing]
    
    return [-1, -1]


# Usage Example
nums = [1,2,2,4]
print(findErrorNums(nums))  # Output: [2, 3]

nums = [1,1]
print(findErrorNums(nums))  # Output: [1, 2]
```

### Problem 7: First Missing Positive (LeetCode #41)

```python
def firstMissingPositive(nums: List[int]) -> int:
    """
    Find the smallest missing positive integer.
    
    Args:
        nums: Array of integers (can have negatives, zeros, duplicates)
    
    Returns:
        Smallest missing positive integer
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    i = 0
    n = len(nums)
    
    # Cyclic sort (only for positive numbers in range [1, n])
    while i < n:
        correct_index = nums[i] - 1
        
        # Only swap if:
        # 1. Number is positive
        # 2. Number is in range [1, n]
        # 3. Number is not at correct position
        if 0 < nums[i] <= n and nums[i] != nums[correct_index]:
            nums[i], nums[correct_index] = nums[correct_index], nums[i]
        else:
            i += 1
    
    # Find first missing positive
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1
    
    # All numbers 1 to n are present
    return n + 1


# Usage Example
nums = [1,2,0]
print(firstMissingPositive(nums))  # Output: 3

nums = [3,4,-1,1]
print(firstMissingPositive(nums))  # Output: 2

nums = [7,8,9,11,12]
print(firstMissingPositive(nums))  # Output: 1
```

### Problem 8: Find First K Missing Positive Numbers

```python
def findKMissingPositive(nums: List[int], k: int) -> List[int]:
    """
    Find first k missing positive numbers.
    
    Args:
        nums: Array of positive integers
        k: Number of missing integers to find
    
    Returns:
        List of first k missing positive integers
    
    Time Complexity: O(n + k)
    Space Complexity: O(1) excluding output
    """
    i = 0
    n = len(nums)
    
    # Cyclic sort
    while i < n:
        correct_index = nums[i] - 1
        
        if 0 < nums[i] <= n and nums[i] != nums[correct_index]:
            nums[i], nums[correct_index] = nums[correct_index], nums[i]
        else:
            i += 1
    
    # Find missing numbers
    missing = []
    extra_numbers = set()
    
    # First pass: find missing in range [1, n]
    for i in range(n):
        if len(missing) < k and nums[i] != i + 1:
            missing.append(i + 1)
            extra_numbers.add(nums[i])
    
    # Second pass: if need more, add n+1, n+2, ...
    candidate = n + 1
    while len(missing) < k:
        if candidate not in extra_numbers:
            missing.append(candidate)
        candidate += 1
    
    return missing


# Usage Example
nums = [3, -1, 4, 5, 5]
k = 3
print(findKMissingPositive(nums, k))  # Output: [1, 2, 6]
```

## Complexity Analysis

### Time Complexity

**Cyclic Sort:**
- **Outer while loop:** Runs O(n) times
- **Inner swaps:** Each element moved to correct position at most once
- **Total swaps:** At most n swaps across entire execution
- **Overall:** O(n + n) = O(n)

**Why O(n) despite nested loops?**
Even though we have a while loop that might swap and not increment `i`, each swap places at least one element in its final position. Since we have n elements, we can have at most n swaps total. Therefore:
- Time = iterations without swap + total swaps
- Time = O(n) + O(n) = O(n)

**Finding Missing/Duplicates:**
- **Sorting phase:** O(n)
- **Finding phase:** O(n) single pass
- **Overall:** O(n)

### Space Complexity

**All Cyclic Sort Problems:**
- **Auxiliary space:** O(1) - only pointers used
- **Modifies input:** Yes (in-place)
- **Output space:** O(k) where k is number of results

### Comparison with Alternatives

| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| Cyclic Sort | O(n) | O(1) | Best for range [1,n] problems |
| Comparison Sort | O(n log n) | O(1) to O(n) | General purpose |
| Hash Set | O(n) | O(n) | Good for duplicates |
| XOR | O(n) | O(1) | Only for specific cases |
| Counting Sort | O(n + k) | O(k) | When range k is small |

## Examples

### Example 1: Basic Cyclic Sort

```
Input: [3, 1, 5, 4, 2]

Iteration by iteration:

i=0: [3, 1, 5, 4, 2]
     nums[0]=3, correct_index=2
     Swap with nums[2]=5
     [5, 1, 3, 4, 2]

i=0: [5, 1, 3, 4, 2]
     nums[0]=5, correct_index=4
     Swap with nums[4]=2
     [2, 1, 3, 4, 5]

i=0: [2, 1, 3, 4, 2]
     nums[0]=2, correct_index=1
     Swap with nums[1]=1
     [1, 2, 3, 4, 5]

i=0: [1, 2, 3, 4, 5]
     nums[0]=1, correct_index=0
     1 == nums[0] ✓ i++

i=1: nums[1]=2, correct ✓ i++
i=2: nums[2]=3, correct ✓ i++
i=3: nums[3]=4, correct ✓ i++
i=4: nums[4]=5, correct ✓ i++

Result: [1, 2, 3, 4, 5]
Total swaps: 3
```

### Example 2: Finding Missing Number

```
Input: [4, 0, 3, 1] (range 0 to n)

Cyclic sort:
i=0: [4, 0, 3, 1], nums[0]=4, skip (>= n), i++
i=1: [4, 0, 3, 1], nums[1]=0, swap with nums[0]
     [0, 4, 3, 1]
i=1: [0, 4, 3, 1], nums[1]=4, skip, i++
i=2: [0, 4, 3, 1], nums[2]=3, swap with nums[3]
     [0, 4, 1, 3]
i=2: [0, 4, 1, 3], nums[2]=1, swap with nums[1]
     [0, 1, 4, 3]
i=2: [0, 1, 4, 3], nums[2]=4, skip, i++
i=3: [0, 1, 4, 3], nums[3]=3, correct, i++

Find missing:
[0, 1, 4, 3]
 ✓  ✓  ✗  ✓
Index 2: nums[2]=4, should be 2
Missing = 2
```

### Example 3: Finding All Duplicates

```
Input: [4,3,2,7,8,2,3,1]

After cyclic sort: [1,2,3,4,3,2,7,8]

Check each position:
Index 0: nums[0]=1 ✓
Index 1: nums[1]=2 ✓
Index 2: nums[2]=3 ✓
Index 3: nums[3]=4 ✓
Index 4: nums[4]=3 ✗ (should be 5, found 3 duplicate)
Index 5: nums[5]=2 ✗ (should be 6, found 2 duplicate)
Index 6: nums[6]=7 ✓
Index 7: nums[7]=8 ✓

Duplicates: [3, 2]
```

### Example 4: First Missing Positive

```
Input: [3, 4, -1, 1]

Cyclic sort (only positive in range [1, n]):
i=0: [3, 4, -1, 1]
     nums[0]=3, correct_index=2
     Swap with nums[2]=-1
     [-1, 4, 3, 1]

i=0: [-1, 4, 3, 1]
     nums[0]=-1, not positive, i++

i=1: [-1, 4, 3, 1]
     nums[1]=4, correct_index=3
     Swap with nums[3]=1
     [-1, 1, 3, 4]

i=1: [-1, 1, 3, 4]
     nums[1]=1, correct_index=0
     Swap with nums[0]=-1
     [1, -1, 3, 4]

i=1: [1, -1, 3, 4]
     nums[1]=-1, not positive, i++

i=2: nums[2]=3, correct, i++
i=3: nums[3]=4, correct, i++

Result: [1, -1, 3, 4]

Find missing:
Index 1: should have 2, has -1
Missing = 2
```

## Edge Cases

### 1. Empty Array
**Scenario:** nums = []
**Challenge:** No elements to sort
**Solution:**
```python
if not nums:
    return []  # or 0, 1, depending on problem
```

### 2. Single Element
**Scenario:** nums = [1]
**Challenge:** Already sorted
**Solution:**
```python
# Loop executes once, number already at correct position
# Returns [1]
```

### 3. All Elements Same
**Scenario:** nums = [3, 3, 3, 3]
**Challenge:** Many duplicates
**Solution:**
```python
# Cyclic sort will try to place all 3s at index 2
# First 3 goes to index 2, others stay where they are
# Can identify duplicates
```

### 4. Numbers Out of Range
**Scenario:** nums = [10, 20, 30] for range [1, 3]
**Challenge:** All numbers invalid
**Solution:**
```python
# Skip all numbers outside range
if 0 < nums[i] <= n and nums[i] != nums[correct_index]:
    # Only swap if in valid range
```

### 5. Negative Numbers
**Scenario:** nums = [-1, -2, 0, 1]
**Challenge:** Negatives invalid for cyclic sort
**Solution:**
```python
# Skip negative numbers
if nums[i] > 0:  # Only process positive
```

### 6. Already Sorted
**Scenario:** nums = [1, 2, 3, 4, 5]
**Challenge:** No swaps needed
**Solution:**
```python
# Each iteration just increments i
# O(n) time with no swaps
```

### 7. Reverse Sorted
**Scenario:** nums = [5, 4, 3, 2, 1]
**Challenge:** Maximum number of swaps
**Solution:**
```python
# Still O(n) - each element moved once
# Different order of swaps but same total
```

## Common Pitfalls

### ❌ Pitfall 1: Incrementing i After Every Swap
**What happens:** Misses placing swapped element correctly
**Why it's wrong:**
```python
# Wrong
while i < len(nums):
    if nums[i] != nums[correct_index]:
        swap(nums[i], nums[correct_index])
        i += 1  # Wrong! New element at i needs checking
    else:
        i += 1
```
**Correct approach:**
```python
while i < len(nums):
    if nums[i] != nums[correct_index]:
        swap(nums[i], nums[correct_index])
        # Don't increment i - check new element
    else:
        i += 1  # Only increment when element is correct
```

### ❌ Pitfall 2: Wrong Correct Index Calculation
**What happens:** Elements placed at wrong positions
**Why it's wrong:**
```python
# Wrong for 1-indexed array
correct_index = nums[i]  # Should be nums[i] - 1

# Wrong for 0-indexed array
correct_index = nums[i] + 1  # Should be nums[i]
```
**Correct approach:**
```python
# For array [1, n]
correct_index = nums[i] - 1

# For array [0, n-1]
correct_index = nums[i]
```

### ❌ Pitfall 3: Not Handling Out-of-Range Numbers
**What happens:** Index out of bounds errors
**Why it's wrong:**
```python
# Wrong - no range check
correct_index = nums[i] - 1
swap(nums[i], nums[correct_index])  # Crash if nums[i] > n
```
**Correct approach:**
```python
if 0 < nums[i] <= n and nums[i] != nums[correct_index]:
    swap(nums[i], nums[correct_index])
```

### ❌ Pitfall 4: Infinite Loop on Duplicates
**What happens:** Loop never terminates
**Why it's wrong:**
```python
# Wrong - will loop forever on duplicates
while i < len(nums):
    if nums[i] != i + 1:  # Always true for duplicates
        correct_index = nums[i] - 1
        swap(nums[i], nums[correct_index])
```
**Correct approach:**
```python
# Check if swap would actually change anything
if nums[i] != nums[correct_index]:  # Prevents duplicate loop
    swap(nums[i], nums[correct_index])
```

### ❌ Pitfall 5: Modifying Read-Only Array
**What happens:** Problem constraints violated
**Why it's wrong:**
```python
# Some problems say "do not modify array"
# But cyclic sort modifies in-place
```
**Correct approach:**
```python
# Use alternative algorithm (Floyd's cycle detection)
# or make a copy first
```

## Variations and Extensions

### Variation 1: Non-Consecutive Range
**Description:** Numbers in range [k, k+n] instead of [1, n]
**Modification:**
```python
correct_index = nums[i] - k  # Adjust offset
```

### Variation 2: Partial Sort
**Description:** Only sort elements in certain range
**Implementation:**
```python
def partial_cyclic_sort(nums, low, high):
    i = 0
    while i < len(nums):
        if low <= nums[i] <= high:
            correct_index = nums[i] - low
            if nums[i] != nums[correct_index]:
                swap(nums[i], nums[correct_index])
            else:
                i += 1
        else:
            i += 1
```

### Variation 3: Find K Missing
**Description:** Find first k missing numbers
**Approach:** After cyclic sort, iterate and collect k missing values

### Variation 4: With Duplicates Allowed
**Description:** Handle arrays where duplicates are expected
**Modification:** Check for duplicate before swapping

## Practice Problems

### Beginner
1. **Missing Number (LeetCode #268)** - Single missing number
2. **Find All Numbers Disappeared in an Array (LeetCode #448)** - Multiple missing
3. **Single Number (LeetCode #136)** - Find non-duplicate (use XOR variant)

### Intermediate
1. **Find the Duplicate Number (LeetCode #287)** - One duplicate
2. **Find All Duplicates in an Array (LeetCode #442)** - Multiple duplicates
3. **Set Mismatch (LeetCode #645)** - Find corrupt pair
4. **First Missing Positive (LeetCode #41)** - Smallest missing positive
5. **Missing Element in Sorted Array (LeetCode #1060)** - Premium

### Advanced
1. **K Missing Positive Numbers** - First k missing
2. **Smallest Range from K Lists** - Combines cyclic sort concepts
3. **Array Nesting (LeetCode #565)** - Cycle detection variant

## Real-World Applications

### Industry Use Cases

1. **Data Validation:** Detecting corrupt or missing records in sequential ID systems
2. **Database Integrity:** Finding gaps in primary key sequences
3. **File System Checks:** Detecting missing or duplicate inodes
4. **Network Packet Analysis:** Finding lost packets in sequence numbers
5. **Inventory Management:** Detecting missing SKUs in sequential product codes

### Popular Implementations

- **Database Systems:** Gap detection in auto-increment IDs
- **File Systems:** Inode allocation and verification
- **Network Protocols:** TCP sequence number validation
- **Version Control:** Detecting missing commits in linear history

### Practical Scenarios

- **Student ID validation:** Finding which IDs in range 1-1000 are unassigned
- **Seat allocation:** Finding available seat numbers in theater
- **ISBN validation:** Checking for missing or duplicate book identifiers
- **Employee ID management:** Detecting gaps in sequential employee numbers

## Related Topics

### Prerequisites to Review
- **Arrays** - Basic operations and indexing
- **Swapping** - Understanding in-place modification
- **Range concepts** - Working with consecutive integers

### Next Steps
- **Hash Set** - Alternative for duplicate detection
- **Bit Manipulation** - XOR for single number problems
- **Fast & Slow Pointers** - Cycle detection in linked lists
- **Union-Find** - Another approach for grouping

### Similar Concepts
- **Counting Sort** - Also uses value-as-index concept
- **Bucket Sort** - Distributing elements to buckets
- **Radix Sort** - Multi-pass sorting for larger ranges

### Further Reading
- [LeetCode Cyclic Sort Pattern](https://leetcode.com/tag/cyclic-sort/)
- [Educative.io: Cyclic Sort Pattern](https://www.educative.io/courses/grokking-coding-interview)
- Algorithm Design Manual - Sorting algorithms chapter
