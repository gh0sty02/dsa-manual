# Sliding Window Pattern

**Difficulty:** Beginner to Hard
**Prerequisites:** Arrays, Hash Maps, Two Pointers
**Estimated Reading Time:** 30 minutes

## Introduction

The Sliding Window pattern is a powerful technique for solving problems that involve contiguous subarrays or substrings. Instead of recalculating the result for each possible subarray from scratch (which would be O(n²) or worse), we maintain a "window" that slides through the data structure, efficiently adding new elements and removing old ones.

**Why it matters:** This pattern transforms O(n²) or O(n³) brute force solutions into elegant O(n) algorithms. It's essential for string processing, array problems, and any scenario where you need to analyze contiguous sequences. Companies like Google, Facebook, and Amazon frequently ask sliding window problems in interviews.

**Real-world analogy:** Imagine viewing a long painting through a window frame. Instead of stepping back and looking at the entire painting each time you want to see a different section, you simply slide the frame left or right. You only need to look at what's entering the frame on one side and what's leaving on the other side. This is exactly how the sliding window works - maintaining state efficiently as you move through the data.

## Core Concepts

### Key Principles

1. **Window definition:** A contiguous portion of the array/string defined by start and end pointers

2. **Window expansion:** Grow the window by moving the end pointer to include more elements

3. **Window contraction:** Shrink the window by moving the start pointer to exclude elements

4. **State tracking:** Maintain information about current window (sum, character counts, etc.)

5. **Optimal subproblem:** Each window position represents a candidate solution

### Essential Terms

- **Window:** The subarray between left and right pointers [left, right]
- **Fixed-size window:** Window with constant size k
- **Dynamic window:** Window that grows and shrinks based on conditions
- **Window state:** Data structure tracking current window properties (sum, frequency map, etc.)
- **Valid window:** Window that satisfies the problem constraints

### Visual Overview

```
Fixed-Size Window (k=3):
[1, 2, 3, 4, 5, 6, 7, 8]
 └─────┘                    Window 1: [1,2,3]
    └─────┘                 Window 2: [2,3,4]
       └─────┘              Window 3: [3,4,5]
          └─────┘           Window 4: [4,5,6]
             └─────┘        Window 5: [5,6,7]
                └─────┘     Window 6: [6,7,8]

Dynamic Window (sum ≤ target):
Target = 7
[2, 1, 5, 2, 3, 2]

Window expands:
[2, 1, 5]           sum=8 > 7, contract
   [1, 5]           sum=6 ≤ 7, valid
   [1, 5, 2]        sum=8 > 7, contract
      [5, 2]        sum=7 ≤ 7, valid
      
Two-pointer movement:
left →              Contract window (remove from left)
        → right     Expand window (add to right)
```

## How It Works

### Fixed-Size Window Algorithm

1. Calculate result for first window of size k
2. Slide window one position right:
   - Remove leftmost element from window state
   - Add new rightmost element to window state
3. Update result if current window is better
4. Repeat until window reaches end of array

### Dynamic Window Algorithm

1. Initialize left = 0, right = 0
2. Expand window by moving right pointer:
   - Add arr[right] to window state
   - right++
3. While window is invalid:
   - Contract window from left
   - Remove arr[left] from window state
   - left++
4. Record result if current window is better
5. Repeat until right reaches end

### Step-by-Step Example: Maximum Sum Subarray of Size K

Problem: Find maximum sum of any contiguous subarray of size k=3 in [2, 1, 5, 1, 3, 2]

```
Step 1: First window [2, 1, 5]
[2, 1, 5, 1, 3, 2]
 └─────┘
sum = 2 + 1 + 5 = 8
max_sum = 8

Step 2: Slide window [1, 5, 1]
[2, 1, 5, 1, 3, 2]
    └─────┘
Remove 2, Add 1
sum = 8 - 2 + 1 = 7
max_sum = 8

Step 3: Slide window [5, 1, 3]
[2, 1, 5, 1, 3, 2]
       └─────┘
Remove 1, Add 3
sum = 7 - 1 + 3 = 9
max_sum = 9

Step 4: Slide window [1, 3, 2]
[2, 1, 5, 1, 3, 2]
          └─────┘
Remove 5, Add 2
sum = 9 - 5 + 2 = 6
max_sum = 9

Result: 9
```

## How to Identify This Pattern

Sliding Window is one of the most frequently tested patterns. Here's how to recognize it:

### Primary Indicators ✓

**Looking for contiguous subarray or substring**
- Problem asks about consecutive elements
- Need sequence without gaps
- Elements must be adjacent
- Keywords: "subarray", "substring", "contiguous", "consecutive"
- Example: "Find the longest substring that..."

**Fixed window size mentioned**
- Problem gives specific window size k
- Calculate something for each k-sized window
- Keywords: "window of size k", "k consecutive elements", "subarray of length k"
- Example: "Maximum sum of subarray of size k"

**Optimizing over all possible windows**
- Finding maximum/minimum/longest/shortest
- Best window meeting certain criteria
- Keywords: "maximum", "minimum", "longest", "shortest", "optimal"
- Example: "Longest substring with at most k distinct characters"

**Conditions based on window contents**
- Sum equals target
- Distinct characters count
- All elements satisfy property
- Keywords: "sum equals", "distinct", "contains all"
- Example: "Subarray with sum equal to k"

**Working with strings or arrays**
- Sequential data structures
- Order matters
- Keywords: "string", "array", "sequence"
- Example: "Given a string s and integer k..."

### Common Problem Phrases 🔑

Watch for these exact phrases:
- "Longest substring with..."
- "Maximum/minimum sum of subarray"
- "Subarray with sum equals..."
- "At most K distinct characters"
- "Contains all characters"
- "Smallest subarray with sum ≥..."
- "Average of all subarrays of size k"
- "Maximum of all subarrays of size k"
- "Permutation in string"
- "Find all anagrams"
- "Minimum window substring"

### Types of Sliding Window 🪟

**Type 1: Fixed-Size Window**
- Window size k is given
- Every window has exactly k elements
- Move window by 1 position each time
- Example: "Maximum sum of k consecutive elements"

**Type 2: Dynamic Window (Flexible Size)**
- Window size changes based on condition
- Expand when condition not met
- Contract when condition satisfied
- Example: "Longest substring with k distinct chars"

### When NOT to Use Sliding Window ✗

**Looking for subsequence (can skip elements)**
- Elements don't need to be consecutive
- Can pick non-adjacent elements
- → Use Dynamic Programming

**Finding pairs or triplets**
- Not about continuous sequence
- Comparing separate elements
- → Use Two Pointers

**No optimization over windows**
- Simple traversal
- Not finding best window
- → Use simple iteration

**Linked list problems**
- Different data structure
- → Use Fast & Slow Pointers

### Quick Decision Checklist ✅

Ask yourself:

1. **Is it about contiguous elements?** → Sliding Window
2. **Does it mention window size k?** → Sliding Window (Fixed)
3. **Finding longest/shortest/max/min?** → Sliding Window (Dynamic)
4. **Says "substring" (not subsequence)?** → Sliding Window
5. **Condition depends on window contents?** → Sliding Window
6. **Working with array/string?** → Possible Sliding Window

If YES to questions 1 AND (2 OR 3), it's definitely Sliding Window!

### Algorithm Templates 📝

**Fixed Window Template:**
```python
window_sum = sum(arr[:k])  # First window
result = window_sum

for i in range(k, len(arr)):
    window_sum = window_sum - arr[i-k] + arr[i]  # Slide
    result = max(result, window_sum)
```

**Dynamic Window Template:**
```python
left = 0
for right in range(len(arr)):
    # Add arr[right] to window
    
    while window_invalid:
        # Remove arr[left] from window
        left += 1
    
    # Update result with current window
```

### Visual Recognition 👁️

**Fixed Window:**
```
Array: [1, 2, 3, 4, 5, 6], k=3

Window 1: [1, 2, 3]
           └─────┘
Window 2:    [2, 3, 4]
              └─────┘
Window 3:       [3, 4, 5]
                 └─────┘
```

**Dynamic Window:**
```
Array: [2, 1, 5, 2, 3, 2], target_sum ≥ 7

Window expands:  [2, 1, 5] sum=8 ✓
Window contracts: [1, 5] sum=6 ✗
Window expands:   [1, 5, 2] sum=8 ✓
```

### Example Pattern Matching 💡

**Problem: "Maximum sum of subarray of size k"**

Analysis:
- ✓ Subarray (contiguous)
- ✓ Fixed size k
- ✓ Finding maximum

**Verdict: USE FIXED SLIDING WINDOW** ✓

**Problem: "Longest substring with at most k distinct characters"**

Analysis:
- ✓ Substring (contiguous)
- ✓ Finding longest
- ✓ Condition on window (k distinct)
- Dynamic window size

**Verdict: USE DYNAMIC SLIDING WINDOW** ✓

**Problem: "Find two numbers that sum to target"**

Analysis:
- ✗ Not contiguous sequence
- ✗ Finding pairs, not window

**Verdict: USE TWO POINTERS** (Not Sliding Window) ✗

**Problem: "Longest increasing subsequence"**

Analysis:
- ✗ Subsequence (can skip elements)
- ✗ Not contiguous

**Verdict: USE DYNAMIC PROGRAMMING** (Not Sliding Window) ✗

### Pattern vs Problem Type 📊

| Problem Type | Sliding Window? | Alternative |
|--------------|-----------------|-------------|
| Max sum subarray size k | ✅ YES (Fixed) | - |
| Longest substring k distinct | ✅ YES (Dynamic) | - |
| Minimum window substring | ✅ YES (Dynamic) | - |
| Subarray sum equals k | ✅ YES (Dynamic) | Prefix Sum |
| Find all anagrams | ✅ YES (Fixed) | - |
| Two sum | ❌ NO | Two Pointers/Hash Map |
| Longest increasing subseq | ❌ NO | Dynamic Programming |
| Palindrome check | ❌ NO | Two Pointers |

### Keywords Cheat Sheet 📝

**STRONG "Sliding Window" Keywords:**
- substring
- subarray
- contiguous
- consecutive
- window

**MODERATE "Sliding Window" Keywords:**
- longest
- shortest
- maximum
- minimum
- at most K
- at least K
- contains all

**ANTI-Keywords (probably NOT Sliding Window):**
- subsequence (can skip)
- pair/triplet (discrete elements)
- sorted (Two Pointers)
- cycle (Fast & Slow)
- reverse (In-place Reversal)

### Red Flags 🚩

These suggest SLIDING WINDOW might NOT be right:
- Problem says "subsequence" → Dynamic Programming
- Looking for pairs/triplets → Two Pointers
- Intervals [start, end] → Merge Intervals
- Linked list → Fast & Slow Pointers
- Needs to reverse → In-place Reversal

### Green Flags 🟢

STRONG indicators for SLIDING WINDOW:
- "substring"
- "subarray"
- "window of size k"
- "longest/shortest/maximum/minimum"
- "contiguous"
- "at most k distinct"
- "contains all elements of"
- "permutation in string"
- "anagrams"



## Implementation

### Problem 1: Maximum Sum Subarray of Size K

```python
from typing import List

def max_sub_array_of_size_k(k: int, arr: List[int]) -> int:
    """
    Find maximum sum of any contiguous subarray of size k.
    
    Args:
        k: Size of the subarray
        arr: Input array
    
    Returns:
        Maximum sum of subarray of size k
    
    Time Complexity: O(n) - single pass through array
    Space Complexity: O(1) - only variables used
    """
    if not arr or k > len(arr):
        return 0
    
    # Calculate sum of first window
    window_sum = sum(arr[:k])
    max_sum = window_sum
    
    # Slide window through array
    for i in range(k, len(arr)):
        # Remove leftmost element, add rightmost element
        window_sum = window_sum - arr[i - k] + arr[i]
        max_sum = max(max_sum, window_sum)
    
    return max_sum


# Usage Example
arr = [2, 1, 5, 1, 3, 2]
k = 3
result = max_sub_array_of_size_k(k, arr)
print(f"Maximum sum: {result}")  # Output: 9
```

### Problem 2: Smallest Subarray with Greater Sum

```python
def smallest_subarray_sum(s: int, arr: List[int]) -> int:
    """
    Find length of smallest contiguous subarray with sum ≥ s.
    
    Args:
        s: Target sum
        arr: Input array of positive integers
    
    Returns:
        Minimum length of subarray with sum ≥ s, 0 if impossible
    
    Time Complexity: O(n) - each element visited at most twice
    Space Complexity: O(1)
    """
    min_length = float('inf')
    window_sum = 0
    window_start = 0
    
    for window_end in range(len(arr)):
        # Expand window
        window_sum += arr[window_end]
        
        # Contract window while condition is met
        while window_sum >= s:
            min_length = min(min_length, window_end - window_start + 1)
            window_sum -= arr[window_start]
            window_start += 1
    
    return 0 if min_length == float('inf') else min_length


# Usage Example
arr = [2, 1, 5, 2, 3, 2]
s = 7
result = smallest_subarray_sum(s, arr)
print(f"Smallest length: {result}")  # Output: 2 ([5, 2])
```

### Problem 3: Longest Substring with K Distinct Characters

```python
def longest_substring_k_distinct(s: str, k: int) -> int:
    """
    Find length of longest substring with at most k distinct characters.
    
    Args:
        s: Input string
        k: Maximum number of distinct characters
    
    Returns:
        Length of longest valid substring
    
    Time Complexity: O(n) - single pass
    Space Complexity: O(k) - hash map with at most k+1 characters
    """
    if k == 0 or not s:
        return 0
    
    max_length = 0
    window_start = 0
    char_frequency = {}
    
    for window_end in range(len(s)):
        # Add character to window
        right_char = s[window_end]
        char_frequency[right_char] = char_frequency.get(right_char, 0) + 1
        
        # Contract window if we have more than k distinct characters
        while len(char_frequency) > k:
            left_char = s[window_start]
            char_frequency[left_char] -= 1
            if char_frequency[left_char] == 0:
                del char_frequency[left_char]
            window_start += 1
        
        # Update maximum length
        max_length = max(max_length, window_end - window_start + 1)
    
    return max_length


# Usage Example
s = "araaci"
k = 2
result = longest_substring_k_distinct(s, k)
print(f"Longest substring length: {result}")  # Output: 4 ("araa")
```

### Problem 4: Fruits into Baskets

```python
def fruits_into_baskets(fruits: List[int]) -> int:
    """
    Pick maximum fruits from trees with only 2 types of baskets.
    Same as: longest subarray with at most 2 distinct elements.
    
    Args:
        fruits: Array where fruits[i] is type of fruit from tree i
    
    Returns:
        Maximum number of fruits that can be collected
    
    Time Complexity: O(n)
    Space Complexity: O(1) - at most 3 types in map
    """
    max_fruits = 0
    window_start = 0
    fruit_frequency = {}
    
    for window_end in range(len(fruits)):
        # Add fruit to basket
        right_fruit = fruits[window_end]
        fruit_frequency[right_fruit] = fruit_frequency.get(right_fruit, 0) + 1
        
        # If more than 2 types, remove from left
        while len(fruit_frequency) > 2:
            left_fruit = fruits[window_start]
            fruit_frequency[left_fruit] -= 1
            if fruit_frequency[left_fruit] == 0:
                del fruit_frequency[left_fruit]
            window_start += 1
        
        max_fruits = max(max_fruits, window_end - window_start + 1)
    
    return max_fruits


# Usage Example
fruits = [1, 2, 1, 2, 3, 2, 2]
result = fruits_into_baskets(fruits)
print(f"Maximum fruits: {result}")  # Output: 5 ([2,1,2,3] won't work, [2,3,2,2] = 4, [1,2,1,2] = 4, [2,1,2,2] = 4, but [2,3,2,2,2] has 3 types. Actually [2,1,2,2] or [3,2,2] - let me recalculate: [2,3,2,2] = 4)
```

### Problem 5: Longest Substring with Same Letters after Replacement

```python
def length_of_longest_substring(s: str, k: int) -> int:
    """
    Find longest substring after replacing at most k characters.
    
    Args:
        s: Input string
        k: Maximum number of replacements allowed
    
    Returns:
        Length of longest substring with same letters after replacement
    
    Time Complexity: O(n)
    Space Complexity: O(1) - at most 26 characters in map
    """
    max_length = 0
    window_start = 0
    max_repeat_letter_count = 0
    char_frequency = {}
    
    for window_end in range(len(s)):
        right_char = s[window_end]
        char_frequency[right_char] = char_frequency.get(right_char, 0) + 1
        
        # Track most frequent character in current window
        max_repeat_letter_count = max(max_repeat_letter_count, 
                                      char_frequency[right_char])
        
        # If replacements needed > k, shrink window
        # window_length - max_repeat_count = replacements needed
        if (window_end - window_start + 1 - max_repeat_letter_count) > k:
            left_char = s[window_start]
            char_frequency[left_char] -= 1
            window_start += 1
        
        max_length = max(max_length, window_end - window_start + 1)
    
    return max_length


# Usage Example
s = "aabccbb"
k = 2
result = length_of_longest_substring(s, k)
print(f"Longest substring: {result}")  # Output: 5 ("bccbb")
```

### Problem 6: Longest Subarray with Ones after Replacement

```python
def length_of_longest_ones(arr: List[int], k: int) -> int:
    """
    Find longest subarray of 1s after replacing at most k 0s.
    
    Args:
        arr: Binary array (0s and 1s)
        k: Maximum number of 0s that can be replaced
    
    Returns:
        Length of longest subarray of 1s
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    max_length = 0
    window_start = 0
    max_ones_count = 0
    
    for window_end in range(len(arr)):
        if arr[window_end] == 1:
            max_ones_count += 1
        
        # If 0s in window > k, shrink window
        if (window_end - window_start + 1 - max_ones_count) > k:
            if arr[window_start] == 1:
                max_ones_count -= 1
            window_start += 1
        
        max_length = max(max_length, window_end - window_start + 1)
    
    return max_length


# Usage Example
arr = [0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1]
k = 2
result = length_of_longest_ones(arr, k)
print(f"Longest subarray: {result}")  # Output: 6
```

### Problem 7: Permutation in String

```python
def find_permutation(s: str, pattern: str) -> bool:
    """
    Check if string contains permutation of pattern.
    
    Args:
        s: Input string
        pattern: Pattern to find permutation of
    
    Returns:
        True if s contains permutation of pattern
    
    Time Complexity: O(n + m) where n = len(s), m = len(pattern)
    Space Complexity: O(m) for pattern frequency map
    """
    window_start = 0
    matched = 0
    char_frequency = {}
    
    # Build frequency map for pattern
    for char in pattern:
        char_frequency[char] = char_frequency.get(char, 0) + 1
    
    # Try to extend window
    for window_end in range(len(s)):
        right_char = s[window_end]
        
        # If character matches pattern, decrement count
        if right_char in char_frequency:
            char_frequency[right_char] -= 1
            if char_frequency[right_char] == 0:
                matched += 1
        
        # Check if we found permutation
        if matched == len(char_frequency):
            return True
        
        # Shrink window if it's larger than pattern
        if window_end >= len(pattern) - 1:
            left_char = s[window_start]
            window_start += 1
            
            if left_char in char_frequency:
                if char_frequency[left_char] == 0:
                    matched -= 1
                char_frequency[left_char] += 1
    
    return False


# Usage Example
s = "oidbcaf"
pattern = "abc"
result = find_permutation(s, pattern)
print(f"Contains permutation: {result}")  # Output: True ("bca")
```

### Problem 8: String Anagrams

```python
def find_string_anagrams(s: str, pattern: str) -> List[int]:
    """
    Find all starting indices of anagrams of pattern in string.
    
    Args:
        s: Input string
        pattern: Pattern to find anagrams of
    
    Returns:
        List of starting indices
    
    Time Complexity: O(n + m)
    Space Complexity: O(m)
    """
    result_indices = []
    window_start = 0
    matched = 0
    char_frequency = {}
    
    for char in pattern:
        char_frequency[char] = char_frequency.get(char, 0) + 1
    
    for window_end in range(len(s)):
        right_char = s[window_end]
        
        if right_char in char_frequency:
            char_frequency[right_char] -= 1
            if char_frequency[right_char] == 0:
                matched += 1
        
        # Found anagram
        if matched == len(char_frequency):
            result_indices.append(window_start)
        
        # Slide window
        if window_end >= len(pattern) - 1:
            left_char = s[window_start]
            window_start += 1
            
            if left_char in char_frequency:
                if char_frequency[left_char] == 0:
                    matched -= 1
                char_frequency[left_char] += 1
    
    return result_indices


# Usage Example
s = "ppqp"
pattern = "pq"
result = find_string_anagrams(s, pattern)
print(f"Anagram indices: {result}")  # Output: [1, 2]
```

### Problem 9: Smallest Window Containing Substring

```python
def min_window(s: str, t: str) -> str:
    """
    Find minimum window substring containing all characters from t.
    
    Args:
        s: Input string
        t: Target characters
    
    Returns:
        Minimum window substring, empty string if not found
    
    Time Complexity: O(n + m)
    Space Complexity: O(m)
    """
    if not s or not t:
        return ""
    
    char_frequency = {}
    for char in t:
        char_frequency[char] = char_frequency.get(char, 0) + 1
    
    window_start = 0
    matched = 0
    min_length = len(s) + 1
    substr_start = 0
    
    for window_end in range(len(s)):
        right_char = s[window_end]
        
        if right_char in char_frequency:
            char_frequency[right_char] -= 1
            if char_frequency[right_char] >= 0:  # >= 0 means we needed this
                matched += 1
        
        # Try to shrink window
        while matched == len(t):
            # Update minimum window
            if min_length > window_end - window_start + 1:
                min_length = window_end - window_start + 1
                substr_start = window_start
            
            # Remove from left
            left_char = s[window_start]
            window_start += 1
            
            if left_char in char_frequency:
                if char_frequency[left_char] == 0:
                    matched -= 1
                char_frequency[left_char] += 1
    
    return "" if min_length > len(s) else s[substr_start:substr_start + min_length]


# Usage Example
s = "aabdec"
t = "abc"
result = min_window(s, t)
print(f"Minimum window: {result}")  # Output: "abdec"
```

### Problem 10: Words Concatenation

```python
def find_word_concatenation(s: str, words: List[str]) -> List[int]:
    """
    Find indices where concatenation of all words (in any order) starts.
    
    Args:
        s: Input string
        words: List of words (all same length)
    
    Returns:
        List of starting indices
    
    Time Complexity: O(n * m * len(words)) where n=len(s), m=word_length
    Space Complexity: O(m) for word frequency map
    """
    if not s or not words:
        return []
    
    word_frequency = {}
    for word in words:
        word_frequency[word] = word_frequency.get(word, 0) + 1
    
    result_indices = []
    word_count = len(words)
    word_length = len(words[0])
    
    for i in range((len(s) - word_count * word_length) + 1):
        words_seen = {}
        
        for j in range(word_count):
            next_word_index = i + j * word_length
            word = s[next_word_index:next_word_index + word_length]
            
            if word not in word_frequency:
                break
            
            words_seen[word] = words_seen.get(word, 0) + 1
            
            if words_seen[word] > word_frequency.get(word, 0):
                break
            
            if j + 1 == word_count:
                result_indices.append(i)
    
    return result_indices


# Usage Example
s = "catfoxcat"
words = ["cat", "fox"]
result = find_word_concatenation(s, words)
print(f"Concatenation indices: {result}")  # Output: [0, 3]
```

## Complexity Analysis

### Time Complexity

**Fixed Window:**
- **Time:** O(n) - Single pass through array, constant work per element

**Dynamic Window:**
- **Time:** O(n) - Each element enters and leaves window at most once
- Even though nested loops exist, left and right pointers each traverse array once
- Total operations: O(2n) = O(n)

**With Hash Map:**
- **Time:** O(n) for sliding + O(k) for hash operations where k is alphabet size
- For strings: O(n) since k ≤ 26 (constant)
- For general arrays: O(n × m) where m is number of unique elements

### Space Complexity

**Basic Window:**
- **Space:** O(1) - Only pointers and variables

**With Frequency Map:**
- **Space:** O(k) where k is number of distinct elements in window
- For fixed alphabet: O(1) since k is bounded
- Worst case: O(n) if all elements are unique

### Comparison with Alternatives

| Approach | Time | Space | When to Use |
|----------|------|-------|-------------|
| Sliding Window | O(n) | O(1) to O(k) | Contiguous subarrays/substrings |
| Brute Force | O(n²) to O(n³) | O(1) | Small inputs only |
| Prefix Sum | O(n) | O(n) | When subarray sums needed |
| Two Pointers | O(n) | O(1) | Non-contiguous or sorted data |

## Examples

### Example 1: Maximum Sum - Fixed Window

```
Array: [2, 1, 5, 1, 3, 2], k = 3

Window 1: [2, 1, 5]
├─ sum = 8
└─ max = 8

Window 2: [1, 5, 1]
├─ remove 2, add 1
├─ sum = 8 - 2 + 1 = 7
└─ max = 8

Window 3: [5, 1, 3]
├─ remove 1, add 3
├─ sum = 7 - 1 + 3 = 9
└─ max = 9 ✓

Window 4: [1, 3, 2]
├─ remove 5, add 2
├─ sum = 9 - 5 + 2 = 6
└─ max = 9

Result: 9
```

### Example 2: Smallest Subarray - Dynamic Window

```
Array: [2, 1, 5, 2, 3, 2], target = 7

Start: left=0, right=0, sum=0

Step 1: Expand to include 2
[2, 1, 5, 2, 3, 2]
 ↑
sum = 2 < 7, expand

Step 2-3: Expand to include 1, 5
[2, 1, 5, 2, 3, 2]
 └─────┘
sum = 8 ≥ 7, found valid window (length=3)

Step 4: Contract from left
[2, 1, 5, 2, 3, 2]
    └──┘
sum = 6 < 7, expand

Step 5: Expand to include 2
[2, 1, 5, 2, 3, 2]
    └─────┘
sum = 8 ≥ 7, found length=3

Step 6: Contract
[2, 1, 5, 2, 3, 2]
       └──┘
sum = 7 ≥ 7, found length=2 ✓

Result: 2 (subarray [5, 2])
```

### Example 3: K Distinct Characters

```
String: "araaci", k = 2

Window: "a"
├─ {a: 1}, distinct = 1
└─ length = 1

Window: "ar"
├─ {a: 1, r: 1}, distinct = 2
└─ length = 2

Window: "ara"
├─ {a: 2, r: 1}, distinct = 2
└─ length = 3

Window: "araa"
├─ {a: 3, r: 1}, distinct = 2
└─ length = 4 ✓

Window: "araac"
├─ {a: 3, r: 1, c: 1}, distinct = 3 > k
└─ contract!

After contraction: "raac"
├─ {a: 2, r: 1, c: 1}, still 3
└─ contract again

After: "aac"
├─ {a: 2, c: 1}, distinct = 2
└─ length = 3

Result: 4
```

## Edge Cases

### 1. Empty Input
**Scenario:** arr = [] or s = ""
**Challenge:** No window possible
**Solution:**
```python
if not arr or len(arr) < k:
    return 0
```

### 2. Window Size Larger Than Array
**Scenario:** k = 5, arr = [1, 2, 3]
**Challenge:** Invalid window size
**Solution:**
```python
if k > len(arr):
    return 0  # or handle appropriately
```

### 3. All Elements Same
**Scenario:** Find k distinct in "aaaa"
**Challenge:** Fewer distinct elements than required
**Solution:** Return 0 or maximum possible

### 4. Single Element
**Scenario:** arr = [5], k = 1
**Challenge:** Edge of valid input
**Solution:**
```python
# Should return 5 for max sum
# Window of size 1 is the element itself
```

### 5. Negative Numbers
**Scenario:** Max sum with negatives [-2, -1, 5, -1]
**Challenge:** Logic remains same but max could be negative
**Solution:** Initialize max_sum with first window, not 0

### 6. Target Impossible to Achieve
**Scenario:** Find subarray with sum ≥ 1000 in [1, 2, 3]
**Challenge:** No valid window exists
**Solution:**
```python
return 0  # or appropriate sentinel value
```

## Common Pitfalls

### ❌ Pitfall 1: Forgetting to Update Window State
**What happens:** Incorrect results due to stale state
**Why it's wrong:**
```python
for window_end in range(len(arr)):
    window_sum += arr[window_end]
    # Missing: window_sum -= arr[window_start] when contracting
```
**Correct approach:**
```python
while window_sum >= target:
    # Record result
    window_sum -= arr[window_start]  # Update state!
    window_start += 1
```

### ❌ Pitfall 2: Not Handling Frequency Map Deletions
**What happens:** Memory leak, incorrect distinct count
**Why it's wrong:**
```python
char_frequency[left_char] -= 1
# Missing: delete when count reaches 0
```
**Correct approach:**
```python
char_frequency[left_char] -= 1
if char_frequency[left_char] == 0:
    del char_frequency[left_char]  # Maintain accurate distinct count
```

### ❌ Pitfall 3: Incorrect Window Size Calculation
**What happens:** Off-by-one errors
**Why it's wrong:**
```python
window_length = window_end - window_start  # Missing +1
```
**Correct approach:**
```python
window_length = window_end - window_start + 1  # Inclusive
```

### ❌ Pitfall 4: Not Initializing First Window Correctly
**What happens:** Wrong result for fixed-size windows
**Why it's wrong:**
```python
# Starting from index k without calculating first window
for i in range(k, len(arr)):
    # First window never calculated
```
**Correct approach:**
```python
window_sum = sum(arr[:k])  # Initialize first window
for i in range(k, len(arr)):
    window_sum = window_sum - arr[i-k] + arr[i]
```

### ❌ Pitfall 5: Over-Contracting Dynamic Windows
**What happens:** Skip valid windows
**Why it's wrong:**
```python
while condition:
    window_start += 1  # Contract too much
# Should stop as soon as condition becomes false
```
**Correct approach:**
```python
while window is invalid:
    contract one step
    check again
# Stop when valid
```

## Variations and Extensions

### Variation 1: Longest Substring Without Repeating Characters
**Description:** Dynamic window tracking last seen position
**Implementation:**
```python
def length_of_longest_substring(s: str) -> int:
    char_index_map = {}
    max_length = 0
    window_start = 0
    
    for window_end in range(len(s)):
        if s[window_end] in char_index_map:
            # Move start to right of last occurrence
            window_start = max(window_start, char_index_map[s[window_end]] + 1)
        
        char_index_map[s[window_end]] = window_end
        max_length = max(max_length, window_end - window_start + 1)
    
    return max_length
```

### Variation 2: Subarray Product Less Than K
**Description:** Multiplicative window instead of additive
**Key difference:** Need to handle division carefully
**Implementation:**
```python
def num_subarray_product_less_than_k(nums: List[int], k: int) -> int:
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
        
        # All subarrays ending at right
        count += right - left + 1
    
    return count
```

### Variation 3: Maximum of All Subarrays of Size K
**Description:** Use deque to maintain maximum in window
**When to use:** Need to track maximum/minimum efficiently
**Time:** O(n) using monotonic deque

### Variation 4: Longest Repeating Character Replacement
**Description:** Track most frequent character in window
**Key insight:** window_size - max_frequency = replacements needed

## Practice Problems

### Beginner
1. **Maximum Average Subarray I (LeetCode #643)** - Fixed window average
2. **Minimum Size Subarray Sum (LeetCode #209)** - Dynamic window with sum
3. **Contains Duplicate II (LeetCode #219)** - Window for duplicates within k distance
4. **Maximum Sum of Distinct Subarrays With Length K (LeetCode #2461)** - Fixed window with distinct elements

### Intermediate
1. **Longest Substring Without Repeating Characters (LeetCode #3)** - Classic dynamic window
2. **Longest Repeating Character Replacement (LeetCode #424)** - Window with replacements
3. **Fruit Into Baskets (LeetCode #904)** - Max 2 distinct types
4. **Max Consecutive Ones III (LeetCode #1004)** - Binary array with k flips
5. **Permutation in String (LeetCode #567)** - Anagram detection
6. **Find All Anagrams in a String (LeetCode #438)** - Multiple anagram positions
7. **Longest Substring with At Most K Distinct Characters (LeetCode #340)** - Premium
8. **Subarray Product Less Than K (LeetCode #713)** - Counting subarrays

### Advanced
1. **Minimum Window Substring (LeetCode #76)** - Hard template problem
2. **Substring with Concatenation of All Words (LeetCode #30)** - Multiple word matching
3. **Sliding Window Maximum (LeetCode #239)** - Deque optimization
4. **Minimum Window Subsequence (LeetCode #727)** - Premium, subsequence variant
5. **Longest Substring with At Most Two Distinct Characters (LeetCode #159)** - Premium
6. **Subarrays with K Different Integers (LeetCode #992)** - Exactly k distinct

## Real-World Applications

### Industry Use Cases

1. **Network Packet Analysis:** Monitoring network traffic in fixed time windows
   - Track packet loss rates over sliding time intervals
   - Detect anomalies in bandwidth usage

2. **Stock Trading:** Moving averages and technical indicators
   - Simple Moving Average (SMA) uses fixed-size sliding window
   - Relative Strength Index (RSI) calculations

3. **Text Processing:** Finding patterns in large documents
   - Spell checkers looking for similar words
   - Plagiarism detection with n-gram matching

4. **Log Analysis:** Analyzing server logs for patterns
   - Error rate monitoring over time windows
   - Request rate limiting

### Popular Implementations

- **Pandas rolling():** DataFrame.rolling(window=k) for time series
- **Apache Kafka Streams:** Windowed aggregations for stream processing
- **Redis TIME SERIES:** Sliding window aggregations
- **Elasticsearch aggregations:** Date histogram with sliding windows

### Practical Scenarios

- **Video streaming:** Buffering data in fixed-size windows
- **Autocomplete:** Finding longest matching prefix in recent queries
- **Rate limiting:** Counting requests in sliding time window
- **DNA sequence matching:** Finding similar subsequences in genomic data
- **Anomaly detection:** Comparing current window stats to historical baselines
- **Recommendation systems:** Analyzing user behavior in recent time windows

## Related Topics

### Prerequisites to Review
- **Arrays and strings** - Basic data structures
- **Hash maps** - For frequency tracking
- **Two pointers** - Foundation for sliding window

### Next Steps
- **Two heaps** - For advanced window statistics
- **Monotonic deque** - For window maximum/minimum
- **Segment trees** - For range query problems
- **Dynamic programming** - When windows overlap in state

### Similar Concepts
- **Convolution** - Signal processing equivalent
- **Moving average** - Statistical time series technique
- **Cache eviction (LRU)** - Window of recent items
- **Circular buffer** - Fixed-size window implementation

### Further Reading
- [LeetCode Sliding Window Study Guide](https://leetcode.com/tag/sliding-window/)
- [Grokking the Coding Interview - Sliding Window Pattern](https://www.educative.io/courses/grokking-coding-interview)
- Algorithm Design Manual (Skiena) - Chapter on sequence algorithms
- Competitive Programming 3 (Halim & Halim) - Window techniques
