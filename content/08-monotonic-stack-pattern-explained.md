# Monotonic Stack Pattern

**Difficulty:** Medium
**Prerequisites:** Stacks, Arrays
**Estimated Reading Time:** 24 minutes

## Introduction

The Monotonic Stack is a specialized stack data structure that maintains elements in a monotonically increasing or decreasing order. This elegant pattern is particularly powerful for solving problems involving finding the next/previous greater or smaller element, calculating spans, and dealing with visibility or histogram-based challenges. It's an optimization of the basic stack pattern that leverages the ordering property to achieve O(n) solutions for problems that might otherwise require O(n²).

**Why it matters:** Monotonic Stack transforms complex iteration problems into elegant linear-time solutions. Companies like Google, Amazon, and Facebook frequently test this pattern because it demonstrates advanced problem-solving skills and the ability to optimize brute-force solutions. Understanding monotonic stacks is crucial for stock price analysis, histogram problems, and many real-world scenarios involving ranges and comparisons.

**Real-world analogy:** Imagine you're standing in a line of people of different heights, and you want to know, for each person, who is the next taller person they can see ahead. A monotonic decreasing stack (heights decrease from bottom to top) helps track this efficiently. As you process each person, you pop everyone shorter than them from the stack (they found their "next taller"), then add the current person. This way, everyone in the stack is still waiting to find someone taller, and they're ordered by height!

## Core Concepts

### Key Principles

1. **Monotonic property:** Elements maintain increasing or decreasing order

2. **Two types:**
   - **Monotonic Increasing:** Elements increase from bottom to top
   - **Monotonic Decreasing:** Elements decrease from bottom to top

3. **Element removal:** Pop elements that violate monotonic property

4. **Next greater/smaller:** Natural use case for monotonic stacks

5. **Amortized O(1):** Each element pushed and popped at most once

### Essential Terms

- **Monotonic Increasing Stack:** Each element is greater than the one below it
- **Monotonic Decreasing Stack:** Each element is smaller than the one below it
- **Next Greater Element (NGE):** First element to the right that is greater
- **Next Smaller Element (NSE):** First element to the right that is smaller
- **Previous Greater/Smaller:** Same concept but looking left
- **Span:** Distance to previous greater/smaller element

### Visual Overview

```
Monotonic Increasing Stack (finding Next Greater Element):

Array: [2, 1, 2, 4, 3]

Process 2: Stack: [2]
Process 1: Stack: [1]          (pop 2, 2's NGE is none so far)
Process 2: Stack: [1, 2]       (1's NGE is 2!)
Process 4: Stack: [4]          (pop 2, then 1; their NGE is 4)
Process 3: Stack: [4, 3]

Monotonic Decreasing Stack (finding Next Smaller Element):

Array: [4, 2, 1, 5, 3]

Process 4: Stack: [4]
Process 2: Stack: [4, 2]       (4's NSE hasn't been found)
Process 1: Stack: [4, 2, 1]    (2's NSE is 1!)
Process 5: Stack: [5]          (pop all; their NSE is none)
Process 3: Stack: [5, 3]

Visual Representation:
Increasing Stack    Decreasing Stack
(bottom to top)     (bottom to top)
    5                   1
    3                   2
    2                   3
    1                   5
```

## How to Identify This Pattern

Recognizing Monotonic Stack problems is key to solving them efficiently:

### Primary Indicators ✓

**Finding next/previous greater element**
- Need the next larger value to the right
- Need the previous larger value to the left
- Keywords: "next greater", "next larger", "previous greater"
- Example: "Find next greater element for each element"

**Finding next/previous smaller element**
- Need the next smaller value to the right
- Need the previous smaller value to the left
- Keywords: "next smaller", "previous smaller", "next less"
- Example: "Find previous smaller element"

**Calculating spans or distances**
- Stock span problem
- Distance to previous greater/smaller
- Keywords: "span", "distance to", "how far"
- Example: "Calculate stock span for each day"

**Histogram or skyline problems**
- Largest rectangle in histogram
- Trapping rain water
- Skyline problems
- Keywords: "histogram", "rectangle", "area", "trapped water"
- Example: "Largest rectangle in histogram"

**Visibility or line-of-sight problems**
- Buildings you can see
- People who can see over others
- Keywords: "can see", "visible", "view"
- Example: "How many buildings can you see"

**Maintaining order while processing**
- Need to track elements in specific order
- Remove elements that won't be useful
- Keywords: "in order", "maintaining", "tracking"
- Example: "Remove k digits to get smallest number"

### Common Problem Phrases 🔑

Watch for these exact phrases:
- "Next greater element"
- "Next smaller element"
- "Previous greater/smaller"
- "Stock span"
- "Daily temperatures"
- "Largest rectangle in histogram"
- "Trapping rain water"
- "Remove k digits"
- "132 pattern"
- "Online stock span"
- "Sum of subarray minimums"

### When NOT to Use Monotonic Stack ✗

**Simple next/previous without order**
- Just need immediate neighbors
- → Use array indexing

**Need all greater/smaller elements**
- Not just the next/previous one
- → May need different approach

**Random access required**
- Need to access arbitrary elements
- → Use array or other structure

**Already have sorted data**
- Monotonic property already exists
- → May not need stack

### Quick Decision Checklist ✅

Ask yourself:

1. **Need next/previous greater/smaller element?** → Monotonic Stack
2. **Working with histograms or heights?** → Likely Monotonic Stack
3. **Stock span or temperature problems?** → Monotonic Stack
4. **Need to maintain increasing/decreasing order?** → Monotonic Stack
5. **Looking for visibility/line-of-sight?** → Monotonic Stack
6. **Brute force is O(n²) with nested loops?** → Try Monotonic Stack for O(n)

If YES to question 1, it's almost certainly Monotonic Stack!

### Decision Tree

```
Start
  ↓
Does problem mention "next greater" or "next smaller"?
  ↓ YES
  → USE MONOTONIC STACK
  
  ↓ NO
Is it about histogram, heights, or areas?
  ↓ YES
  → USE MONOTONIC STACK
  
  ↓ NO
Need to find spans or distances to prev element?
  ↓ YES
  → USE MONOTONIC STACK
```

### Algorithm Signatures

**Next Greater Element (Increasing Stack):**
```python
stack = []
result = [-1] * n

for i in range(n):
    while stack and arr[stack[-1]] < arr[i]:
        idx = stack.pop()
        result[idx] = arr[i]  # Found NGE!
    stack.append(i)
```

**Next Smaller Element (Decreasing Stack):**
```python
stack = []
result = [-1] * n

for i in range(n):
    while stack and arr[stack[-1]] > arr[i]:
        idx = stack.pop()
        result[idx] = arr[i]  # Found NSE!
    stack.append(i)
```

### Example Pattern Matching 💡

**Problem: "Find next greater element for each element"**

Analysis:
- ✓ Explicitly says "next greater"
- ✓ Classic monotonic stack problem
- ✓ Use increasing stack

**Verdict: USE MONOTONIC STACK** ✓

**Problem: "Daily temperatures - days until warmer"**

Analysis:
- ✓ Finding next greater (warmer temperature)
- ✓ Return distance (span)
- ✓ Monotonic stack perfect

**Verdict: USE MONOTONIC STACK** ✓

**Problem: "Largest rectangle in histogram"**

Analysis:
- ✓ Histogram problem
- ✓ Need to find boundaries
- ✓ Classic monotonic stack

**Verdict: USE MONOTONIC STACK** ✓

**Problem: "Find maximum in each subarray of size k"**

Analysis:
- ✗ Not about next greater
- ✗ Need sliding window maximum
- ? Could use monotonic deque

**Verdict: USE MONOTONIC DEQUE/SLIDING WINDOW** ✗

### Pattern vs Problem Type 📊

| Problem Type | Monotonic Stack? | Alternative |
|--------------|------------------|-------------|
| Next greater element | ✅ YES | Brute force O(n²) |
| Next smaller element | ✅ YES | Brute force O(n²) |
| Stock span | ✅ YES | - |
| Daily temperatures | ✅ YES | - |
| Largest rectangle histogram | ✅ YES | - |
| Trapping rain water | ✅ YES | Two Pointers |
| Sliding window maximum | ⚠️ Monotonic Deque | - |
| Valid parentheses | ❌ NO | Regular Stack |
| Expression evaluation | ❌ NO | Regular Stack |

### Keywords Cheat Sheet 📝

**STRONG "Monotonic Stack" Keywords:**
- next greater
- next smaller
- previous greater
- previous smaller
- stock span
- histogram

**MODERATE Keywords:**
- daily temperatures
- visibility
- trapped water
- largest rectangle
- remove k digits

**ANTI-Keywords (probably NOT Monotonic Stack):**
- matching pairs (Regular Stack)
- expression evaluation (Regular Stack)
- first/last element (Array access)

### Red Flags 🚩

These suggest MONOTONIC STACK might NOT be right:
- "Match brackets" → Regular Stack
- "Evaluate expression" → Regular Stack
- "Sliding window max" → Monotonic Deque
- No ordering needed → Regular Stack

### Green Flags 🟢

STRONG indicators for MONOTONIC STACK:
- "Next greater element"
- "Next smaller element"
- "Stock span problem"
- "Daily temperatures"
- "Largest rectangle in histogram"
- "Trapping rain water"
- "Remove k digits"
- Any "next/previous greater/smaller"

## How It Works

### Monotonic Increasing Stack (for Next Greater Element)

1. **Initialize:** Empty stack to store indices
2. **For each element:**
   - While stack not empty AND current > stack top:
     - Pop from stack (found its next greater!)
     - Record current as NGE for popped element
   - Push current index to stack
3. **Elements remaining in stack:** Have no next greater element

### Monotonic Decreasing Stack (for Next Smaller Element)

1. **Initialize:** Empty stack to store indices
2. **For each element:**
   - While stack not empty AND current < stack top:
     - Pop from stack (found its next smaller!)
     - Record current as NSE for popped element
   - Push current index to stack
3. **Elements remaining in stack:** Have no next smaller element

### Why O(n) Time?

Each element is:
- Pushed exactly once: O(n) total
- Popped at most once: O(n) total
- Total: O(n) + O(n) = O(n)

### Step-by-Step Example: Next Greater Element

Array: [2, 1, 2, 4, 3]

```
i=0, val=2:
Stack: []
Action: Push index 0
Stack: [0] → [2]

i=1, val=1:
Stack: [0] → [2]
1 < 2, no pop
Action: Push index 1
Stack: [0, 1] → [2, 1]

i=2, val=2:
Stack: [0, 1] → [2, 1]
2 > 1, pop index 1
Result[1] = 2 (NGE of 1 is 2)
2 == 2, no more pops
Action: Push index 2
Stack: [0, 2] → [2, 2]

i=3, val=4:
Stack: [0, 2] → [2, 2]
4 > 2, pop index 2
Result[2] = 4 (NGE of 2 is 4)
4 > 2, pop index 0
Result[0] = 4 (NGE of 2 is 4)
Stack: []
Action: Push index 3
Stack: [3] → [4]

i=4, val=3:
Stack: [3] → [4]
3 < 4, no pop
Action: Push index 4
Stack: [3, 4] → [4, 3]

Final:
Result: [4, 2, 4, -1, -1]
NGE of 2 is 4
NGE of 1 is 2
NGE of 2 is 4
NGE of 4 is -1 (none)
NGE of 3 is -1 (none)
```

## Implementation

### Problem 1: Next Greater Element I (LeetCode #496)

```python
from typing import List

def nextGreaterElement(nums1: List[int], nums2: List[int]) -> List[int]:
    """
    Find next greater element for nums1 elements in nums2.
    
    Args:
        nums1: Subset of nums2
        nums2: Array to find NGE in
    
    Returns:
        NGE for each element in nums1
    
    Time Complexity: O(n + m)
    Space Complexity: O(n)
    """
    # Build NGE map for nums2 using monotonic stack
    nge_map = {}
    stack = []
    
    for num in nums2:
        # Pop smaller elements (they found their NGE)
        while stack and stack[-1] < num:
            nge_map[stack.pop()] = num
        stack.append(num)
    
    # Elements in stack have no NGE
    for num in stack:
        nge_map[num] = -1
    
    # Build result for nums1
    return [nge_map[num] for num in nums1]


# Usage Example
nums1 = [4, 1, 2]
nums2 = [1, 3, 4, 2]
print(nextGreaterElement(nums1, nums2))  # [-1, 3, -1]
```

### Problem 2: Daily Temperatures (LeetCode #739)

```python
def dailyTemperatures(temperatures: List[int]) -> List[int]:
    """
    Find how many days until warmer temperature.
    
    Args:
        temperatures: Daily temperatures
    
    Returns:
        Days to wait for warmer temp
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    n = len(temperatures)
    result = [0] * n
    stack = []  # Store indices
    
    for i in range(n):
        # Pop days with colder temps (found warmer day)
        while stack and temperatures[stack[-1]] < temperatures[i]:
            prev_day = stack.pop()
            result[prev_day] = i - prev_day
        stack.append(i)
    
    # Days in stack never get warmer (result already 0)
    return result


# Usage Example
temps = [73, 74, 75, 71, 69, 72, 76, 73]
print(dailyTemperatures(temps))  # [1, 1, 4, 2, 1, 1, 0, 0]
```

### Problem 3: Largest Rectangle in Histogram (LeetCode #84)

```python
def largestRectangleArea(heights: List[int]) -> int:
    """
    Find largest rectangle area in histogram.
    
    Args:
        heights: Bar heights
    
    Returns:
        Maximum rectangle area
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    stack = []  # Monotonic increasing stack (indices)
    max_area = 0
    heights.append(0)  # Sentinel to pop all remaining
    
    for i in range(len(heights)):
        # Pop taller bars (calculate their max area)
        while stack and heights[stack[-1]] > heights[i]:
            h_idx = stack.pop()
            h = heights[h_idx]
            # Width: from previous in stack to current
            w = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * w)
        stack.append(i)
    
    return max_area


# Usage Example
heights = [2, 1, 5, 6, 2, 3]
print(largestRectangleArea(heights))  # 10
```

### Problem 4: Trapping Rain Water (LeetCode #42)

```python
def trap(height: List[int]) -> int:
    """
    Calculate trapped rain water.
    
    Args:
        height: Elevation map
    
    Returns:
        Units of trapped water
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    stack = []  # Monotonic decreasing (indices)
    water = 0
    
    for i in range(len(height)):
        # Found higher bar, can trap water
        while stack and height[i] > height[stack[-1]]:
            bottom = stack.pop()
            
            if not stack:
                break
            
            # Calculate trapped water
            distance = i - stack[-1] - 1
            bounded_height = min(height[i], height[stack[-1]]) - height[bottom]
            water += distance * bounded_height
        
        stack.append(i)
    
    return water


# Usage Example
height = [0,1,0,2,1,0,1,3,2,1,2,1]
print(trap(height))  # 6
```

### Problem 5: Remove K Digits (LeetCode #402)

```python
def removeKdigits(num: str, k: int) -> str:
    """
    Remove k digits to get smallest number.
    
    Args:
        num: Number as string
        k: Digits to remove
    
    Returns:
        Smallest possible number
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    stack = []  # Monotonic increasing
    
    for digit in num:
        # Remove larger digits (make number smaller)
        while stack and k > 0 and stack[-1] > digit:
            stack.pop()
            k -= 1
        stack.append(digit)
    
    # Remove remaining k digits from end
    if k > 0:
        stack = stack[:-k]
    
    # Convert to string, remove leading zeros
    result = ''.join(stack).lstrip('0')
    return result if result else '0'


# Usage Example
print(removeKdigits("1432219", 3))  # "1219"
print(removeKdigits("10200", 1))    # "200"
```

### Problem 6: Online Stock Span (LeetCode #901)

```python
class StockSpanner:
    """
    Calculate stock price span using monotonic stack.
    
    Span: Number of consecutive days where price ≤ today's price.
    """
    
    def __init__(self):
        self.stack = []  # (price, span) pairs
    
    def next(self, price: int) -> int:
        """
        Get span for today's price.
        
        Time: Amortized O(1)
        """
        span = 1
        
        # Pop days with lower/equal prices
        while self.stack and self.stack[-1][0] <= price:
            span += self.stack.pop()[1]
        
        self.stack.append((price, span))
        return span


# Usage Example
spanner = StockSpanner()
print(spanner.next(100))  # 1
print(spanner.next(80))   # 1
print(spanner.next(60))   # 1
print(spanner.next(70))   # 2
print(spanner.next(60))   # 1
print(spanner.next(75))   # 4
print(spanner.next(85))   # 6
```

### Problem 7: Sum of Subarray Minimums (LeetCode #907)

```python
def sumSubarrayMins(arr: List[int]) -> int:
    """
    Sum of minimum of all subarrays.
    
    Args:
        arr: Input array
    
    Returns:
        Sum of all subarray minimums
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    MOD = 10**9 + 7
    n = len(arr)
    
    # Find previous smaller and next smaller
    left = [0] * n   # Distance to previous smaller
    right = [0] * n  # Distance to next smaller
    
    # Previous smaller using monotonic increasing stack
    stack = []
    for i in range(n):
        while stack and arr[stack[-1]] > arr[i]:
            stack.pop()
        left[i] = i + 1 if not stack else i - stack[-1]
        stack.append(i)
    
    # Next smaller using monotonic increasing stack
    stack = []
    for i in range(n - 1, -1, -1):
        while stack and arr[stack[-1]] >= arr[i]:
            stack.pop()
        right[i] = n - i if not stack else stack[-1] - i
        stack.append(i)
    
    # Calculate result
    result = 0
    for i in range(n):
        result += arr[i] * left[i] * right[i]
        result %= MOD
    
    return result


# Usage Example
arr = [3, 1, 2, 4]
print(sumSubarrayMins(arr))  # 17
```

## Complexity Analysis

### Time Complexity

**Monotonic Stack Operations:**
- Each element pushed once: O(n)
- Each element popped at most once: O(n)
- **Total: O(n)** amortized

**Why not O(n²)?**
Even though we have while loop inside for loop, each element is processed at most twice (one push, one pop).

**Problem Complexities:**
- Next Greater Element: O(n)
- Daily Temperatures: O(n)
- Histogram: O(n)
- Trapping Water: O(n)

### Space Complexity

**Stack Storage:** O(n) worst case
- All elements in increasing/decreasing order
- Stack holds all indices

**Optimization:** Store indices instead of values to save space

### Comparison with Alternatives

| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| Monotonic Stack | O(n) | O(n) | Optimal |
| Brute Force (nested loops) | O(n²) | O(1) | TLE on large inputs |
| Binary Search | O(n log n) | O(n) | Not applicable here |

## Examples

### Example 1: Next Greater Element

```
Array: [4, 5, 2, 10, 8]

Process:
i=0, val=4:  stack=[] → [0]
i=1, val=5:  5>4, pop 0, NGE[0]=5
             stack=[] → [1]
i=2, val=2:  2<5, stack=[1,2]
i=3, val=10: 10>2, pop 2, NGE[2]=10
             10>5, pop 1, NGE[1]=10
             stack=[] → [3]
i=4, val=8:  8<10, stack=[3,4]

Result: [5, 10, 10, -1, -1]
```

### Example 2: Stock Span

```
Prices: [100, 80, 60, 70, 60, 75, 85]

Day 1: price=100
  stack=[]
  span=1
  stack=[(100,1)]

Day 2: price=80
  80 < 100, no pop
  span=1
  stack=[(100,1), (80,1)]

Day 3: price=60
  60 < 80, no pop
  span=1
  stack=[(100,1), (80,1), (60,1)]

Day 4: price=70
  70 > 60, pop (60,1), span += 1
  70 < 80, no more pops
  span=2
  stack=[(100,1), (80,1), (70,2)]

Day 5: price=60
  60 < 70, no pop
  span=1
  stack=[(100,1), (80,1), (70,2), (60,1)]

Day 6: price=75
  75 > 60, pop (60,1), span += 1
  75 > 70, pop (70,2), span += 2
  75 < 80, no more pops
  span=4
  stack=[(100,1), (80,1), (75,4)]

Day 7: price=85
  85 > 75, pop (75,4), span += 4
  85 > 80, pop (80,1), span += 1
  85 < 100, no more pops
  span=6
  stack=[(100,1), (85,6)]

Spans: [1, 1, 1, 2, 1, 4, 6]
```

## Edge Cases

### 1. All Increasing
**Scenario:** [1, 2, 3, 4, 5]
**Stack:** Grows to full size
**NGE:** All -1 (no greater to right)

### 2. All Decreasing
**Scenario:** [5, 4, 3, 2, 1]
**Stack:** Each element popped immediately
**NGE:** Each element's NGE is next element

### 3. Single Element
**Scenario:** [5]
**NGE:** [-1]

### 4. Duplicates
**Scenario:** [3, 3, 3]
**Handle:** Use >= or > carefully

### 5. Empty Array
**Scenario:** []
**Return:** []

## Common Pitfalls

### ❌ Pitfall 1: Wrong Comparison Direction
**What happens:** Gets next smaller instead of next greater
**Why it's wrong:**
```python
# Wrong for NGE
while stack and arr[stack[-1]] > arr[i]:  # Should be <
```
**Correct:**
```python
while stack and arr[stack[-1]] < arr[i]:  # For NGE
```

### ❌ Pitfall 2: Storing Values Instead of Indices
**What happens:** Can't calculate distances
**Why it's wrong:**
```python
# Wrong if you need distances
stack.append(arr[i])  # Lose index information
```
**Correct:**
```python
stack.append(i)  # Store index
```

### ❌ Pitfall 3: Forgetting Remaining Elements
**What happens:** Incomplete results
**Why it's wrong:**
```python
# After loop, stack may not be empty
# Forgot to handle remaining elements
```
**Correct:**
```python
# After loop
for idx in stack:
    result[idx] = -1  # No NGE found
```

### ❌ Pitfall 4: Off-by-One in Distance
**What happens:** Wrong span calculation
**Why it's wrong:**
```python
# Wrong
distance = i - stack[-1]  # Should be i - stack[-1] - 1
```
**Correct:**
```python
distance = i - stack[-1] - 1  # Exclude both endpoints
```

## Variations and Extensions

### Variation 1: Circular Array
**Description:** Array wraps around
**Solution:** Process array twice

### Variation 2: Previous Greater/Smaller
**Description:** Look left instead of right
**Solution:** Process array right to left

### Variation 3: Monotonic Queue/Deque
**Description:** Sliding window maximum
**Solution:** Use deque for both ends

## Practice Problems

### Beginner
1. **Next Greater Element I (LeetCode #496)**
2. **Next Greater Element II (LeetCode #503)** - Circular
3. **Daily Temperatures (LeetCode #739)**

### Intermediate
1. **Online Stock Span (LeetCode #901)**
2. **Remove K Digits (LeetCode #402)**
3. **Remove Duplicate Letters (LeetCode #316)**
4. **Car Fleet (LeetCode #853)**
5. **132 Pattern (LeetCode #456)**

### Advanced
1. **Largest Rectangle in Histogram (LeetCode #84)**
2. **Maximal Rectangle (LeetCode #85)**
3. **Trapping Rain Water (LeetCode #42)**
4. **Sum of Subarray Minimums (LeetCode #907)**
5. **Sum of Subarray Ranges (LeetCode #2104)**

## Real-World Applications

### Industry Use Cases

1. **Stock Market Analysis:** Price trends and patterns
2. **Weather Forecasting:** Temperature trends
3. **Network Monitoring:** Traffic patterns
4. **Data Visualization:** Histogram rendering
5. **Resource Allocation:** Span calculation

### Practical Scenarios

- **Trading algorithms:** Finding support/resistance
- **Climate analysis:** Temperature patterns
- **Building design:** Skyline visibility
- **Water management:** Reservoir capacity

## Related Topics

### Prerequisites
- **Stacks** - Basic stack operations
- **Arrays** - Sequential access

### Next Steps
- **Segment Tree** - Range queries
- **Binary Indexed Tree** - Prefix operations

### Similar Concepts
- **Monotonic Queue** - For sliding window
- **Priority Queue** - For general ordering

### Further Reading
- [Monotonic Stack - LeetCode](https://leetcode.com/tag/monotonic-stack/)
- [Stack Optimization Techniques](https://cp-algorithms.com/)
