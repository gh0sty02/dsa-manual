# Hash Maps Pattern

**Difficulty:** Easy to Medium
**Prerequisites:** Arrays, Basic data structures
**Estimated Reading Time:** 24 minutes

## Introduction

The Hash Map (also known as Hash Table, Dictionary, or Map) pattern is one of the most powerful and frequently used data structure patterns in programming. A hash map stores key-value pairs and provides average O(1) time complexity for insertions, deletions, and lookups. This makes it invaluable for solving problems involving counting, tracking, grouping, and finding duplicates or complements.

**Why it matters:** Hash maps are the Swiss Army knife of data structures - they appear in countless real-world applications from database indexing to caching systems. Understanding hash maps is essential for optimizing algorithms from O(n²) to O(n), and they're tested extensively in interviews because they demonstrate your ability to think about trade-offs between time and space complexity. Companies use hash maps everywhere - from user session management to recommendation systems.

**Real-world analogy:** Think of a library card catalog. Instead of searching through every book linearly to find one by title, you go directly to the card labeled with that title's first letter, then find the exact card. The title is the "key," the book's location is the "value," and the catalog provides instant lookup. Similarly, a hash map uses a key to instantly find the associated value, rather than searching through everything sequentially!

## Core Concepts

### Key Principles

1. **Key-value pairs:** Store associations between keys and values

2. **Hash function:** Converts keys to array indices for O(1) access

3. **Collision handling:** Multiple keys may hash to same index
   - **Chaining:** Store multiple items at same index (linked list)
   - **Open addressing:** Find next available slot

4. **Average O(1) operations:** Insert, delete, lookup (amortized)

5. **No guaranteed order:** Keys not stored in any particular sequence (except OrderedDict)

### Essential Terms

- **Hash Map/Table:** Data structure storing key-value pairs
- **Hash Function:** Function mapping keys to array indices
- **Collision:** Two keys hash to the same index
- **Load Factor:** Ratio of entries to capacity
- **Rehashing:** Resizing hash table when load factor exceeds threshold
- **Bucket:** Storage location for entries (may contain multiple items with chaining)

### Visual Overview

```
Hash Map Structure:

Key → Hash Function → Index → Value
"apple" → hash("apple") → 3 → 5
"banana" → hash("banana") → 7 → 3
"cherry" → hash("cherry") → 1 → 8

Internal Array:
Index:  0    1    2    3    4    5    6    7    8    9
Value: [ ] [8 ] [ ] [5 ] [ ] [ ] [ ] [3 ] [ ] [ ]
Key:        "cherry"  "apple"            "banana"

Collision Example:
"cat" → hash → 3
"dog" → hash → 3  (collision!)

Chaining:
Index 3: ["cat":10] → ["dog":20]  (linked list)

Common Operations:
- map[key] = value     # Insert/Update: O(1)
- value = map[key]     # Lookup: O(1)
- del map[key]         # Delete: O(1)
- key in map           # Contains: O(1)
- len(map)             # Size: O(1)
```

## How to Identify This Pattern

Recognizing when to use Hash Maps is crucial for optimizing solutions:

### Primary Indicators ✓

**Need fast lookup/access**
- Check if element exists
- Get value for a key
- O(1) lookup required
- Keywords: "find", "check if", "exists", "lookup"
- Example: "Check if number exists in array"

**Counting or frequency tracking**
- Count occurrences of elements
- Track how many times something appears
- Keywords: "count", "frequency", "occurrences", "how many times"
- Example: "Count frequency of each character"

**Finding pairs or complements**
- Two sum problem
- Finding elements that sum to target
- Keywords: "pair", "complement", "sum to target"
- Example: "Find two numbers that sum to target"

**Grouping or categorizing**
- Group items by some property
- Anagrams, patterns
- Keywords: "group by", "categorize", "anagrams", "patterns"
- Example: "Group anagrams together"

**Tracking seen/visited elements**
- Avoid duplicates
- Detect cycles
- Keywords: "duplicate", "unique", "first time", "seen before"
- Example: "Find first non-repeating character"

**Caching or memoization**
- Store computed results
- Avoid recomputation
- Keywords: "cache", "memoize", "remember", "store results"
- Example: "Fibonacci with memoization"

**Mapping relationships**
- One-to-one or one-to-many mappings
- Keywords: "map", "associate", "correspond", "relationship"
- Example: "Map each student to their grades"

### Common Problem Phrases 🔑

Watch for these exact phrases:
- "Two sum"
- "Contains duplicate"
- "First unique character"
- "Group anagrams"
- "Majority element"
- "Longest substring without repeating"
- "Subarray sum equals k"
- "Valid anagram"
- "Isomorphic strings"
- "Word pattern"
- "Top k frequent elements"

### When NOT to Use Hash Maps ✗

**Need ordered traversal**
- Elements in sorted order
- → Use TreeMap/Ordered structures

**Range queries needed**
- Find all elements in range
- → Use Segment Tree or BST

**Space is critical constraint**
- Cannot afford O(n) space
- → Look for O(1) space solutions

**Keys are continuous integers**
- Can use array indexing
- → Use array instead

### Quick Decision Checklist ✅

Ask yourself:

1. **Need O(1) lookup/insert/delete?** → Hash Map
2. **Counting frequency of elements?** → Hash Map
3. **Finding pairs/complements?** → Hash Map
4. **Need to track seen elements?** → Hash Map
5. **Grouping items by property?** → Hash Map
6. **Array unsorted and can't sort?** → Hash Map
7. **Caching results?** → Hash Map

If YES to any of these, Hash Map is likely the right choice!

### Decision Tree

```
Start
  ↓
Need fast lookup (O(1))?
  ↓ YES
Can afford O(n) space?
  ↓ YES
  → USE HASH MAP
  
  ↓ NO (if can't afford space)
Can sort the input?
  ↓ YES
  → USE TWO POINTERS
  ↓ NO
  → HASH MAP might still be best
```

### Algorithm Signatures

**Two Sum Pattern:**
```python
seen = {}
for i, num in enumerate(arr):
    complement = target - num
    if complement in seen:
        return [seen[complement], i]
    seen[num] = i
```

**Frequency Count Pattern:**
```python
freq = {}
for item in items:
    freq[item] = freq.get(item, 0) + 1
# Or use Counter: freq = Counter(items)
```

**Grouping Pattern:**
```python
groups = {}
for item in items:
    key = get_key(item)
    if key not in groups:
        groups[key] = []
    groups[key].append(item)
```

### Example Pattern Matching 💡

**Problem: "Find two numbers that sum to target"**

Analysis:
- ✓ Finding pairs/complements
- ✓ Need O(n) solution
- ✓ Hash map stores seen numbers

**Verdict: USE HASH MAP** ✓

**Problem: "Count frequency of each element"**

Analysis:
- ✓ Counting/frequency tracking
- ✓ O(n) time, O(n) space
- ✓ Perfect for hash map

**Verdict: USE HASH MAP** ✓

**Problem: "Group anagrams together"**

Analysis:
- ✓ Grouping by property (sorted letters)
- ✓ Need to categorize strings
- ✓ Hash map with sorted string as key

**Verdict: USE HASH MAP** ✓

**Problem: "Find pairs in sorted array"**

Analysis:
- ✓ Array is sorted
- ? Could use hash map
- ✓ But Two Pointers is more space-efficient

**Verdict: USE TWO POINTERS** (better choice) ✗

**Problem: "Find median of stream"**

Analysis:
- ✗ Need ordered elements
- ✗ Range-based operations
- ? Hash map doesn't help

**Verdict: USE HEAPS** (Not Hash Map) ✗

### Pattern vs Problem Type 📊

| Problem Type | Hash Map? | Alternative |
|--------------|-----------|-------------|
| Two sum (unsorted) | ✅ YES | Two Pointers (if sorted) |
| Count frequencies | ✅ YES | Array (if range known) |
| First unique char | ✅ YES | - |
| Group anagrams | ✅ YES | - |
| Contains duplicate | ✅ YES | Set |
| Longest substring | ✅ YES | - |
| Find median | ❌ NO | Heaps |
| Range sum query | ❌ NO | Prefix Sum/Segment Tree |
| Sorted operations | ❌ NO | TreeMap/BST |

### Keywords Cheat Sheet 📝

**STRONG "Hash Map" Keywords:**
- count
- frequency
- duplicate
- unique
- two sum
- group
- anagram

**MODERATE Keywords:**
- first
- map
- track
- seen
- cache
- lookup

**ANTI-Keywords (probably NOT Hash Map):**
- sorted
- range query
- median
- kth smallest/largest (unless "top k frequent")

### Red Flags 🚩

These suggest HASH MAP might NOT be best:
- "Sorted array" → Two Pointers might be better
- "Range query" → Segment Tree/BST
- "Median" or "Kth smallest" → Heaps
- "O(1) space required" → Look for in-place solution

### Green Flags 🟢

STRONG indicators for HASH MAP:
- "Two sum"
- "Contains duplicate"
- "First unique"
- "Group anagrams"
- "Count frequency"
- "Longest substring without repeating"
- "Subarray sum"
- "Isomorphic"
- "Valid anagram"

## How It Works

### Basic Hash Map Operations

1. **Insert/Update:** `map[key] = value`
   - Compute hash(key) → index
   - Store value at index
   - O(1) average time

2. **Lookup:** `value = map[key]`
   - Compute hash(key) → index
   - Return value at index
   - O(1) average time

3. **Delete:** `del map[key]`
   - Compute hash(key) → index
   - Remove entry
   - O(1) average time

4. **Contains:** `key in map`
   - Check if key exists
   - O(1) average time

### Solving Two Sum with Hash Map

1. **Initialize:** Create empty hash map
2. **For each number:**
   - Calculate complement = target - number
   - Check if complement in hash map
   - If yes: found pair, return indices
   - If no: store number with index in map
3. **Return:** Indices of the pair

### Step-by-Step Example: Two Sum

Problem: Find indices of two numbers that sum to 9
Array: [2, 7, 11, 15]

```
Target: 9
Map: {}

i=0, num=2:
  complement = 9 - 2 = 7
  7 in map? No
  map[2] = 0
  Map: {2: 0}

i=1, num=7:
  complement = 9 - 7 = 2
  2 in map? Yes! ✓
  Return [map[2], 1] = [0, 1]

Result: Indices [0, 1]
Numbers: arr[0]=2, arr[1]=7
Sum: 2 + 7 = 9 ✓
```

## Implementation

### Problem 1: Two Sum (LeetCode #1)

```python
from typing import List

def twoSum(nums: List[int], target: int) -> List[int]:
    """
    Find indices of two numbers that sum to target.
    
    Args:
        nums: Array of integers
        target: Target sum
    
    Returns:
        Indices of the two numbers
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    seen = {}  # Map: number → index
    
    for i, num in enumerate(nums):
        complement = target - num
        
        if complement in seen:
            return [seen[complement], i]
        
        seen[num] = i
    
    return []  # No solution found


# Usage Examples
print(twoSum([2, 7, 11, 15], 9))   # [0, 1]
print(twoSum([3, 2, 4], 6))        # [1, 2]
print(twoSum([3, 3], 6))           # [0, 1]
```

### Problem 2: Contains Duplicate (LeetCode #217)

```python
def containsDuplicate(nums: List[int]) -> bool:
    """
    Check if array contains duplicates.
    
    Args:
        nums: Array of integers
    
    Returns:
        True if duplicate exists
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    seen = set()
    
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    
    return False


# Alternative using dict
def containsDuplicateDict(nums: List[int]) -> bool:
    seen = {}
    for num in nums:
        if num in seen:
            return True
        seen[num] = True
    return False


# Usage Examples
print(containsDuplicate([1, 2, 3, 1]))    # True
print(containsDuplicate([1, 2, 3, 4]))    # False
```

### Problem 3: Group Anagrams (LeetCode #49)

```python
from collections import defaultdict

def groupAnagrams(strs: List[str]) -> List[List[str]]:
    """
    Group strings that are anagrams.
    
    Args:
        strs: List of strings
    
    Returns:
        Grouped anagrams
    
    Time Complexity: O(n * k log k) where k is max string length
    Space Complexity: O(n * k)
    """
    groups = defaultdict(list)
    
    for s in strs:
        # Sort string as key (anagrams have same sorted form)
        key = ''.join(sorted(s))
        groups[key].append(s)
    
    return list(groups.values())


# Alternative: Character count as key
def groupAnagramsCount(strs: List[str]) -> List[List[str]]:
    groups = defaultdict(list)
    
    for s in strs:
        # Count characters (a-z)
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        # Use tuple of counts as key
        key = tuple(count)
        groups[key].append(s)
    
    return list(groups.values())


# Usage Example
strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
print(groupAnagrams(strs))
# [["eat","tea","ate"], ["tan","nat"], ["bat"]]
```

### Problem 4: First Unique Character (LeetCode #387)

```python
from collections import Counter

def firstUniqChar(s: str) -> int:
    """
    Find index of first non-repeating character.
    
    Args:
        s: Input string
    
    Returns:
        Index of first unique char, -1 if none
    
    Time Complexity: O(n)
    Space Complexity: O(1) - at most 26 characters
    """
    # Count frequency of each character
    freq = Counter(s)
    
    # Find first character with frequency 1
    for i, char in enumerate(s):
        if freq[char] == 1:
            return i
    
    return -1


# Manual counting version
def firstUniqCharManual(s: str) -> int:
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    
    for i, char in enumerate(s):
        if freq[char] == 1:
            return i
    
    return -1


# Usage Examples
print(firstUniqChar("leetcode"))      # 0 ('l')
print(firstUniqChar("loveleetcode"))  # 2 ('v')
print(firstUniqChar("aabb"))          # -1
```

### Problem 5: Subarray Sum Equals K (LeetCode #560)

```python
def subarraySum(nums: List[int], k: int) -> int:
    """
    Count subarrays with sum equal to k.
    
    Args:
        nums: Array of integers
        k: Target sum
    
    Returns:
        Number of subarrays with sum k
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    # Map: prefix_sum → count
    prefix_sums = {0: 1}  # Empty subarray has sum 0
    current_sum = 0
    count = 0
    
    for num in nums:
        current_sum += num
        
        # Check if (current_sum - k) exists
        # If yes, found subarray(s) with sum k
        if current_sum - k in prefix_sums:
            count += prefix_sums[current_sum - k]
        
        # Update prefix sum count
        prefix_sums[current_sum] = prefix_sums.get(current_sum, 0) + 1
    
    return count


# Usage Examples
print(subarraySum([1, 1, 1], 2))        # 2
print(subarraySum([1, 2, 3], 3))        # 2
print(subarraySum([1, -1, 1, -1], 0))   # 4
```

### Problem 6: Longest Substring Without Repeating (LeetCode #3)

```python
def lengthOfLongestSubstring(s: str) -> int:
    """
    Find length of longest substring without repeating characters.
    
    Args:
        s: Input string
    
    Returns:
        Length of longest substring
    
    Time Complexity: O(n)
    Space Complexity: O(min(n, m)) where m is charset size
    """
    char_index = {}  # Map: character → last seen index
    max_length = 0
    start = 0
    
    for i, char in enumerate(s):
        # If char seen and within current window
        if char in char_index and char_index[char] >= start:
            # Move start to after last occurrence
            start = char_index[char] + 1
        
        # Update character's last seen index
        char_index[char] = i
        
        # Update max length
        max_length = max(max_length, i - start + 1)
    
    return max_length


# Usage Examples
print(lengthOfLongestSubstring("abcabcbb"))  # 3 ("abc")
print(lengthOfLongestSubstring("bbbbb"))     # 1 ("b")
print(lengthOfLongestSubstring("pwwkew"))    # 3 ("wke")
```

### Problem 7: Isomorphic Strings (LeetCode #205)

```python
def isIsomorphic(s: str, t: str) -> bool:
    """
    Check if two strings are isomorphic.
    
    Args:
        s: First string
        t: Second string
    
    Returns:
        True if isomorphic
    
    Time Complexity: O(n)
    Space Complexity: O(1) - at most 256 characters
    """
    if len(s) != len(t):
        return False
    
    # Two mappings needed (bijection)
    s_to_t = {}
    t_to_s = {}
    
    for char_s, char_t in zip(s, t):
        # Check s → t mapping
        if char_s in s_to_t:
            if s_to_t[char_s] != char_t:
                return False
        else:
            s_to_t[char_s] = char_t
        
        # Check t → s mapping
        if char_t in t_to_s:
            if t_to_s[char_t] != char_s:
                return False
        else:
            t_to_s[char_t] = char_s
    
    return True


# Usage Examples
print(isIsomorphic("egg", "add"))     # True
print(isIsomorphic("foo", "bar"))     # False
print(isIsomorphic("paper", "title")) # True
```

### Problem 8: Top K Frequent Elements (LeetCode #347)

```python
from collections import Counter
import heapq

def topKFrequent(nums: List[int], k: int) -> List[int]:
    """
    Find k most frequent elements.
    
    Args:
        nums: Array of integers
        k: Number of top elements
    
    Returns:
        K most frequent elements
    
    Time Complexity: O(n log k) with heap, O(n) with bucket sort
    Space Complexity: O(n)
    """
    # Count frequencies
    freq = Counter(nums)
    
    # Use heap to find top k
    # Python heapq is min heap, so negate frequencies
    return [num for num, _ in freq.most_common(k)]


# Alternative: Bucket sort (O(n))
def topKFrequentBucket(nums: List[int], k: int) -> List[int]:
    freq = Counter(nums)
    
    # Bucket sort by frequency
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, count in freq.items():
        buckets[count].append(num)
    
    # Collect top k
    result = []
    for i in range(len(buckets) - 1, 0, -1):
        result.extend(buckets[i])
        if len(result) >= k:
            return result[:k]
    
    return result


# Usage Examples
print(topKFrequent([1,1,1,2,2,3], 2))  # [1, 2]
print(topKFrequent([1], 1))            # [1]
```

## Complexity Analysis

### Time Complexity

**Average Case:**
- **Insert:** O(1)
- **Lookup:** O(1)
- **Delete:** O(1)
- **Contains:** O(1)

**Worst Case (all collisions):**
- **All operations:** O(n)
- Rare in practice with good hash function

**Problem Solutions:**
- **Two Sum:** O(n) - single pass
- **Group Anagrams:** O(n * k log k) - sorting keys
- **Subarray Sum:** O(n) - single pass

### Space Complexity

**Hash Map Storage:** O(n)
- Stores n key-value pairs

**Additional Space:**
- Load factor typically kept < 0.75
- May need ~1.3n space for efficiency

### Comparison with Alternatives

| Operation | Hash Map | Array | Balanced BST | Sorted Array |
|-----------|----------|-------|--------------|--------------|
| Insert | O(1) avg | O(1) append | O(log n) | O(n) |
| Lookup | O(1) avg | O(n) | O(log n) | O(log n) |
| Delete | O(1) avg | O(n) | O(log n) | O(n) |
| Ordered traversal | ✗ | ✗ | ✓ | ✓ |
| Space | O(n) | O(n) | O(n) | O(n) |

## Examples

### Example 1: Two Sum Trace

```
nums = [3, 2, 4], target = 6

i=0, num=3:
  complement = 6 - 3 = 3
  3 in seen? No
  seen[3] = 0
  seen = {3: 0}

i=1, num=2:
  complement = 6 - 2 = 4
  4 in seen? No
  seen[2] = 1
  seen = {3: 0, 2: 1}

i=2, num=4:
  complement = 6 - 4 = 2
  2 in seen? Yes! ✓
  return [seen[2], 2] = [1, 2]

Result: [1, 2]
```

### Example 2: Group Anagrams

```
strs = ["eat", "tea", "bat", "tab"]

"eat": sorted = "aet"
  groups["aet"] = ["eat"]

"tea": sorted = "aet"
  groups["aet"] = ["eat", "tea"]

"bat": sorted = "abt"
  groups["abt"] = ["bat"]

"tab": sorted = "abt"
  groups["abt"] = ["bat", "tab"]

Result: [["eat", "tea"], ["bat", "tab"]]
```

### Example 3: Subarray Sum

```
nums = [1, 2, 3], k = 3

prefix_sums = {0: 1}
current_sum = 0
count = 0

i=0, num=1:
  current_sum = 1
  1 - 3 = -2 in prefix_sums? No
  prefix_sums[1] = 1
  prefix_sums = {0: 1, 1: 1}

i=1, num=2:
  current_sum = 3
  3 - 3 = 0 in prefix_sums? Yes!
  count += 1 (subarray [1,2])
  prefix_sums[3] = 1
  prefix_sums = {0: 1, 1: 1, 3: 1}

i=2, num=3:
  current_sum = 6
  6 - 3 = 3 in prefix_sums? Yes!
  count += 1 (subarray [3])
  prefix_sums[6] = 1

Result: count = 2
Subarrays: [1,2] and [3]
```

## Edge Cases

### 1. Empty Input
**Scenario:** nums = []
**Return:** [] or 0 depending on problem

### 2. Single Element
**Scenario:** nums = [5]
**Handle:** Check if valid solution exists

### 3. All Same Elements
**Scenario:** nums = [1, 1, 1, 1]
**Frequency:** {1: 4}

### 4. No Solution
**Scenario:** Two sum with no valid pair
**Return:** []

### 5. Duplicate Keys
**Scenario:** Inserting same key twice
**Behavior:** Overwrites previous value

### 6. Large Numbers
**Scenario:** Very large integers as keys
**Handle:** Hash function should handle

## Common Pitfalls

### ❌ Pitfall 1: Modifying Dict While Iterating
**What happens:** Runtime error
**Why it's wrong:**
```python
# Wrong
for key in dict:
    if condition:
        del dict[key]  # Error!
```
**Correct:**
```python
keys_to_delete = [k for k, v in dict.items() if condition]
for key in keys_to_delete:
    del dict[key]
```

### ❌ Pitfall 2: Not Checking Key Existence
**What happens:** KeyError
**Why it's wrong:**
```python
# Wrong
value = dict[key]  # Error if key not in dict
```
**Correct:**
```python
value = dict.get(key, default_value)
# Or
if key in dict:
    value = dict[key]
```

### ❌ Pitfall 3: Using Mutable Keys
**What happens:** TypeError
**Why it's wrong:**
```python
# Wrong
dict[list] = value  # Lists are unhashable!
```
**Correct:**
```python
dict[tuple] = value  # Use tuple instead
```

### ❌ Pitfall 4: Forgetting Bidirectional Mapping
**What happens:** Incorrect isomorphic check
**Why it's wrong:**
```python
# Wrong - only checks one direction
s_to_t = {}
# Forgot t_to_s mapping
```
**Correct:**
```python
s_to_t = {}
t_to_s = {}
# Check both directions
```

## Variations and Extensions

### Variation 1: OrderedDict
**Description:** Maintains insertion order
**Use case:** LRU Cache

### Variation 2: DefaultDict
**Description:** Provides default value
**Use case:** Grouping, counting

### Variation 3: Counter
**Description:** Specialized for counting
**Use case:** Frequency problems

### Variation 4: Set
**Description:** Keys only, no values
**Use case:** Duplicate detection

## Practice Problems

### Beginner
1. **Two Sum (LeetCode #1)**
2. **Contains Duplicate (LeetCode #217)**
3. **Valid Anagram (LeetCode #242)**
4. **Single Number (LeetCode #136)**

### Intermediate
1. **Group Anagrams (LeetCode #49)**
2. **First Unique Character (LeetCode #387)**
3. **Subarray Sum Equals K (LeetCode #560)**
4. **Longest Substring Without Repeating (LeetCode #3)**
5. **Isomorphic Strings (LeetCode #205)**
6. **Top K Frequent Elements (LeetCode #347)**
7. **4Sum II (LeetCode #454)**

### Advanced
1. **LRU Cache (LeetCode #146)**
2. **LFU Cache (LeetCode #460)**
3. **Substring with Concatenation (LeetCode #30)**
4. **Minimum Window Substring (LeetCode #76)**
5. **Find All Anagrams (LeetCode #438)**

## Real-World Applications

### Industry Use Cases

1. **Database Indexing:** Hash indexes for O(1) lookup
2. **Caching Systems:** Redis, Memcached
3. **Session Management:** User session storage
4. **Routing Tables:** Network packet routing
5. **Symbol Tables:** Compiler/interpreter implementations

### Popular Implementations

- **Python:** dict, set, Counter, defaultdict
- **Java:** HashMap, HashSet
- **JavaScript:** Object, Map, Set
- **C++:** unordered_map, unordered_set

### Practical Scenarios

- **User authentication:** Username → password hash
- **DNS resolution:** Domain → IP address
- **Cache systems:** URL → cached response
- **Autocomplete:** Prefix → suggestions
- **Recommendation:** User → preferences

## Related Topics

### Prerequisites
- **Arrays** - Understanding indexing
- **Hashing** - Hash functions

### Next Steps
- **TreeMap** - Ordered hash map
- **Trie** - Prefix-based structure
- **Bloom Filter** - Probabilistic membership

### Similar Concepts
- **Set** - Keys without values
- **Array** - Direct indexing
- **BST** - Ordered structure

### Further Reading
- [Hash Table - Wikipedia](https://en.wikipedia.org/wiki/Hash_table)
- [LeetCode Hash Table Problems](https://leetcode.com/tag/hash-table/)
- Introduction to Algorithms (CLRS) - Hash Tables chapter
