# Fast & Slow Pointers Pattern

**Difficulty:** Beginner to Medium
**Prerequisites:** Linked Lists, Basic pointer concepts, Understanding of cycles
**Estimated Reading Time:** 20 minutes

## Introduction

The Fast & Slow Pointers pattern, also known as the "Tortoise and Hare" algorithm, uses two pointers that move through a data structure at different speeds. This elegant technique is primarily used to detect cycles in linked lists and to find middle elements, but it has applications in many other scenarios as well.

**Why it matters:** This pattern can detect cycles in O(n) time with O(1) space, making it incredibly efficient for problems where you might otherwise need additional data structures. It's a favorite in technical interviews because it tests your understanding of pointer manipulation and algorithm design.

**Real-world analogy:** Imagine two runners on a circular track. The fast runner runs twice as fast as the slow runner. If the track is circular (has a cycle), the fast runner will eventually lap the slow runner and they'll meet. If the track has an end (no cycle), the fast runner will finish first. This is exactly how the algorithm works!

## Core Concepts

### Key Principles

1. **Different speeds:** Slow pointer moves one step at a time, fast pointer moves two (or more) steps at a time

2. **Cycle detection:** If there's a cycle, fast and slow pointers will eventually meet inside the cycle

3. **Meeting point mathematics:** The meeting point has special mathematical properties useful for finding cycle start

4. **Finding middle:** When fast pointer reaches end, slow pointer is at middle

### Essential Terms

- **Slow pointer (tortoise):** Moves one node at a time
- **Fast pointer (hare):** Moves two nodes at a time
- **Cycle:** A loop in the linked list where a node points back to a previous node
- **Cycle start:** The first node where the cycle begins
- **Meeting point:** Where fast and slow pointers first meet (if cycle exists)

### Visual Overview

```
Cycle Detection:
1 → 2 → 3 → 4 → 5
         ↑       ↓
         8 ← 7 ← 6

Step 1: S F
        1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 3...

Step 2:   S     F
        1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 3...

Step 3:       S         F
        1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 3...

Eventually, they meet inside the cycle!

Finding Middle:
1 → 2 → 3 → 4 → 5 → None
S                    
    F

S       
        F

    S       
                F (None)
    
Slow is at 3 (middle)
```

## How It Works

### Cycle Detection Algorithm

1. Initialize slow and fast pointers to head
2. Move slow pointer one step forward
3. Move fast pointer two steps forward
4. If fast pointer becomes null → no cycle exists
5. If slow == fast → cycle detected
6. To find cycle start:
   - Reset one pointer to head
   - Move both one step at a time
   - When they meet → that's the cycle start

### Finding Middle Element

1. Initialize slow and fast pointers to head
2. While fast and fast.next are not null:
   - Move slow one step
   - Move fast two steps
3. When fast reaches end, slow is at middle

### Mathematical Proof for Cycle Detection

```
Let's say:
- Distance from head to cycle start = k
- Cycle length = C
- Distance from cycle start to meeting point = m

When they meet:
- Slow traveled: k + m
- Fast traveled: k + m + nC (n complete cycles)

Since fast moves twice as fast:
2(k + m) = k + m + nC
2k + 2m = k + m + nC
k + m = nC
k = nC - m

This means: distance from head to start = 
            (n cycles - distance from start to meeting point)
```

## How to Identify This Pattern

Recognizing Fast & Slow Pointers problems is crucial for linked list mastery. Here are the key indicators:

### Primary Indicators ✓

**Working with a Linked List**
- Problem explicitly mentions "linked list"
- Data structure with nodes and next pointers
- Cannot use random access/indexing
- Keywords: "linked list", "nodes", "next pointer"
- Example: "Given the head of a linked list..."

**Need to detect a cycle**
- Checking if list has a loop
- Finding where cycle begins
- Detecting circular references
- Keywords: "cycle", "loop", "circular"
- Example: "Detect if the linked list has a cycle"

**Finding middle element**
- Need middle node of linked list
- Single pass requirement
- Cannot count length first (or need O(1) space)
- Keywords: "middle node", "median"
- Example: "Find the middle of the linked list"

**Finding nth element from end**
- Kth node from tail
- One-pass solution required
- Keywords: "from end", "kth from last", "remove nth from end"
- Example: "Remove the nth node from the end"

**Palindrome checking with O(1) space**
- Verify if linked list is palindrome
- Cannot use array or stack
- Must use constant space
- Keywords: "palindrome", "reads same forwards and backwards"
- Example: "Check if linked list is a palindrome"

**Problems involving repeated patterns or sequences**
- Happy number problem
- Circular array traversal
- Sequence detection
- Keywords: "happy number", "circular", "repeating"
- Example: "Determine if a number is happy"

### Common Problem Phrases 🔑

Watch for these exact phrases:
- "Linked list cycle" / "Detect cycle"
- "Find the middle of linked list"
- "Linked list cycle II" (find start of cycle)
- "Remove nth node from end"
- "Happy number"
- "Palindrome linked list"
- "Reorder list"
- "Find duplicate number" (array as linked list)
- "Circular array loop"

### When NOT to Use Fast & Slow Pointers ✗

**Working with Arrays (not linked lists)**
- Use Two Pointers for sorted arrays
- Use Sliding Window for subarrays
- → Fast & Slow is for linked structures

**Simple traversal only**
- Just need to visit each node once
- No cycle detection or middle finding
- → Use single pointer

**Need to reverse the list**
- Reversing node connections
- → Use In-place Reversal pattern

**Need to modify links or structure**
- Reversing, inserting, deleting
- → Use In-place Reversal or standard linked list operations

### Quick Decision Checklist ✅

Ask yourself:

1. **Is it a linked list problem?** → Consider Fast & Slow
2. **Need to detect a cycle?** → Fast & Slow Pointers
3. **Finding middle without counting?** → Fast & Slow Pointers
4. **Kth from end in one pass?** → Fast & Slow Pointers
5. **Palindrome check with O(1) space?** → Fast & Slow Pointers
6. **Problem mentions "tortoise and hare"?** → Fast & Slow Pointers (that's this pattern!)

If YES to any of these, Fast & Slow is your answer!

### Algorithm Signatures 🔍

**Cycle Detection:**
```python
slow = fast = head
while fast and fast.next:
    slow = slow.next      # 1 step
    fast = fast.next.next # 2 steps
    if slow == fast:
        # Cycle detected!
```

**Find Middle:**
```python
slow = fast = head
while fast and fast.next:
    slow = slow.next      # 1 step
    fast = fast.next.next # 2 steps
# slow is at middle
```

**Kth from End:**
```python
fast = head
# Move fast k steps ahead
for _ in range(k):
    fast = fast.next

# Move both together
slow = head
while fast:
    slow = slow.next
    fast = fast.next
# slow is kth from end
```

### Visual Recognition 👁️

**Fast & Slow Pattern Looks Like:**
```
Linked List: 1 → 2 → 3 → 4 → 5 → 6 → None
             ↑       ↑
           slow    fast

After one iteration:
             1 → 2 → 3 → 4 → 5 → 6 → None
                 ↑           ↑
               slow        fast
```

**With Cycle:**
```
1 → 2 → 3 → 4 → 5
         ↑         ↓
         8 ← 7 ← 6

Fast catches Slow inside the cycle!
```

### Example Pattern Matching 💡

**Problem: "Detect if linked list has a cycle"**

Analysis:
- ✓ Linked list problem
- ✓ Cycle detection explicitly asked
- ✓ Classic Floyd's algorithm

**Verdict: USE FAST & SLOW POINTERS** ✓

**Problem: "Find middle of linked list"**

Analysis:
- ✓ Linked list
- ✓ Find middle in one pass
- ✓ Fast reaches end when slow at middle

**Verdict: USE FAST & SLOW POINTERS** ✓

**Problem: "Reverse a linked list"**

Analysis:
- ✓ Linked list
- ✗ Not finding middle or cycle
- ✗ Reversing connections

**Verdict: USE IN-PLACE REVERSAL** (Not Fast & Slow) ✗

**Problem: "Remove duplicates from sorted array"**

Analysis:
- ✗ Array, not linked list
- ✗ Sorted array problem

**Verdict: USE TWO POINTERS** (Not Fast & Slow) ✗

### Pattern vs Problem Type 📊

| Problem Type | Fast & Slow? | Alternative |
|--------------|--------------|-------------|
| Detect cycle in linked list | ✅ YES | - |
| Find middle of linked list | ✅ YES | - |
| Nth from end (linked list) | ✅ YES | - |
| Palindrome linked list | ✅ YES | - |
| Happy number | ✅ YES | Hash Set |
| Circular array loop | ✅ YES | - |
| Reverse linked list | ❌ NO | In-place Reversal |
| Sorted array operations | ❌ NO | Two Pointers |
| Substring problems | ❌ NO | Sliding Window |

### Red Flags 🚩

These suggest FAST & SLOW might NOT be right:
- Problem is about arrays (not linked lists) → Two Pointers
- Need to reverse or modify structure → In-place Reversal
- Simple single pass traversal → One pointer sufficient
- Need exact indices or positions → Arrays better

### Green Flags 🟢

STRONG indicators for FAST & SLOW:
- "Linked list cycle"
- "Find the middle"
- "Nth from end of list"
- "Palindrome linked list"
- "Happy number"
- "Tortoise and hare"
- "Floyd's algorithm"
- "Circular" anything with linked structures

### Special Case: Happy Number 🎯

**Why Happy Number uses Fast & Slow:**

Even though it's not a linked list, the sequence of numbers forms a cycle:
```
Example: n = 19
19 → 82 → 68 → 100 → 1 (happy!)

Example: n = 2
2 → 4 → 16 → 37 → 58 → 89 → 145 → 42 → 20 → 4 (cycle!)
```

We treat it like a linked list where each number "points to" its next square-sum!



## Implementation

### Problem 1: LinkedList Cycle Detection

```python
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def has_cycle(head: Optional[ListNode]) -> bool:
    """
    Detect if linked list has a cycle using fast & slow pointers.
    
    Args:
        head: Head of the linked list
    
    Returns:
        True if cycle exists, False otherwise
    
    Time Complexity: O(n) - visit each node at most once
    Space Complexity: O(1) - only two pointers used
    """
    if not head or not head.next:
        return False
    
    slow = head
    fast = head
    
    # Move pointers until they meet or fast reaches end
    while fast and fast.next:
        slow = slow.next          # Move slow one step
        fast = fast.next.next     # Move fast two steps
        
        if slow == fast:          # Cycle detected
            return True
    
    return False  # Fast reached end, no cycle


# Usage Example
def create_cycle_list():
    # Creating: 1 → 2 → 3 → 4 → 2 (cycle back to 2)
    head = ListNode(1)
    second = ListNode(2)
    third = ListNode(3)
    fourth = ListNode(4)
    
    head.next = second
    second.next = third
    third.next = fourth
    fourth.next = second  # Create cycle
    
    return head


head_with_cycle = create_cycle_list()
print(f"Has cycle: {has_cycle(head_with_cycle)}")  # Output: True

# List without cycle
head_no_cycle = ListNode(1, ListNode(2, ListNode(3)))
print(f"Has cycle: {has_cycle(head_no_cycle)}")  # Output: False
```

### Problem 2: Find Middle of LinkedList

```python
def find_middle(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Find the middle node of a linked list.
    For even-length lists, returns the second middle node.
    
    Args:
        head: Head of the linked list
    
    Returns:
        Middle node of the list
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not head:
        return None
    
    slow = head
    fast = head
    
    # When fast reaches end, slow will be at middle
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow


# Variation: Find first middle for even-length lists
def find_first_middle(head: Optional[ListNode]) -> Optional[ListNode]:
    """For even-length lists, returns the first middle node."""
    if not head or not head.next:
        return head
    
    slow = head
    fast = head.next  # Start fast one step ahead
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    return slow


# Usage Example
def create_list(values):
    if not values:
        return None
    head = ListNode(values[0])
    current = head
    for val in values[1:]:
        current.next = ListNode(val)
        current = current.next
    return head


# Odd length: 1 → 2 → 3 → 4 → 5
head = create_list([1, 2, 3, 4, 5])
middle = find_middle(head)
print(f"Middle value (odd): {middle.val}")  # Output: 3

# Even length: 1 → 2 → 3 → 4 → 5 → 6
head = create_list([1, 2, 3, 4, 5, 6])
middle = find_middle(head)
print(f"Middle value (even): {middle.val}")  # Output: 4
```

### Problem 3: Start of LinkedList Cycle

```python
def find_cycle_start(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Find the start of the cycle in a linked list.
    
    Args:
        head: Head of the linked list
    
    Returns:
        Node where cycle starts, None if no cycle
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not head or not head.next:
        return None
    
    # Phase 1: Detect cycle
    slow = head
    fast = head
    has_cycle_flag = False
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            has_cycle_flag = True
            break
    
    if not has_cycle_flag:
        return None
    
    # Phase 2: Find cycle start
    # Reset slow to head, move both one step at a time
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    
    return slow  # This is the start of the cycle


# Calculate cycle length (bonus)
def find_cycle_length(head: Optional[ListNode]) -> int:
    """Find the length of the cycle."""
    if not head or not head.next:
        return 0
    
    slow = head
    fast = head
    
    # Detect cycle
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        
        if slow == fast:
            # Count nodes in cycle
            current = slow
            cycle_length = 0
            while True:
                current = current.next
                cycle_length += 1
                if current == slow:
                    break
            return cycle_length
    
    return 0  # No cycle


# Usage Example
def create_cycle_at_position(values, pos):
    """Create list with cycle starting at position pos."""
    if not values:
        return None
    
    head = ListNode(values[0])
    current = head
    cycle_node = None
    
    if pos == 0:
        cycle_node = head
    
    for i, val in enumerate(values[1:], 1):
        current.next = ListNode(val)
        current = current.next
        if i == pos:
            cycle_node = current
    
    # Create cycle
    if cycle_node:
        current.next = cycle_node
    
    return head


# Create: 1 → 2 → 3 → 4 → 2 (cycle at position 1)
head = create_cycle_at_position([1, 2, 3, 4], 1)
start = find_cycle_start(head)
print(f"Cycle starts at value: {start.val if start else None}")  # Output: 2
```

### Problem 4: Happy Number

```python
def is_happy(n: int) -> bool:
    """
    Determine if a number is happy.
    A happy number is defined by: starting with any positive integer,
    replace the number by the sum of the squares of its digits.
    Repeat until the number equals 1 (happy) or loops in a cycle (not happy).
    
    Args:
        n: Positive integer to check
    
    Returns:
        True if n is happy, False otherwise
    
    Time Complexity: O(log n) - number of digits determines iterations
    Space Complexity: O(1) - only two pointers
    """
    def get_next(num: int) -> int:
        """Calculate sum of squares of digits."""
        total_sum = 0
        while num > 0:
            digit = num % 10
            total_sum += digit ** 2
            num //= 10
        return total_sum
    
    # Use fast & slow pointers to detect cycle
    slow = n
    fast = n
    
    while True:
        slow = get_next(slow)           # Move slow one step
        fast = get_next(get_next(fast)) # Move fast two steps
        
        if fast == 1:  # Happy number found
            return True
        
        if slow == fast:  # Cycle detected (not happy)
            return False


# Alternative: Using set (easier to understand but uses O(n) space)
def is_happy_set(n: int) -> bool:
    """Alternative implementation using a set to track seen numbers."""
    def get_next(num: int) -> int:
        total_sum = 0
        while num > 0:
            digit = num % 10
            total_sum += digit ** 2
            num //= 10
        return total_sum
    
    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = get_next(n)
    
    return n == 1


# Usage Example
print(f"Is 19 happy? {is_happy(19)}")  # Output: True
# Process: 19 → 82 → 68 → 100 → 1

print(f"Is 2 happy? {is_happy(2)}")    # Output: False
# Process: 2 → 4 → 16 → 37 → 58 → 89 → 145 → 42 → 20 → 4 (cycle!)
```

### Problem 5: Palindrome LinkedList

```python
def is_palindrome(head: Optional[ListNode]) -> bool:
    """
    Check if linked list is a palindrome.
    
    Args:
        head: Head of the linked list
    
    Returns:
        True if palindrome, False otherwise
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not head or not head.next:
        return True
    
    # Step 1: Find middle using fast & slow pointers
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    # Step 2: Reverse second half
    second_half = reverse_list(slow)
    
    # Step 3: Compare first and second half
    first_half = head
    result = True
    
    while result and second_half:
        if first_half.val != second_half.val:
            result = False
        first_half = first_half.next
        second_half = second_half.next
    
    # Step 4: (Optional) Restore list by reversing second half again
    reverse_list(slow)
    
    return result


def reverse_list(head: Optional[ListNode]) -> Optional[ListNode]:
    """Reverse a linked list in-place."""
    prev = None
    current = head
    
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    
    return prev


# Usage Example
# Palindrome: 1 → 2 → 3 → 2 → 1
head = create_list([1, 2, 3, 2, 1])
print(f"Is palindrome: {is_palindrome(head)}")  # Output: True

# Not palindrome: 1 → 2 → 3 → 4 → 5
head = create_list([1, 2, 3, 4, 5])
print(f"Is palindrome: {is_palindrome(head)}")  # Output: False
```

### Problem 6: Rearrange LinkedList

```python
def reorder_list(head: Optional[ListNode]) -> None:
    """
    Rearrange list from L0 → L1 → ... → Ln-1 → Ln to:
    L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → ...
    
    Modifies list in-place.
    
    Args:
        head: Head of the linked list
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not head or not head.next:
        return
    
    # Step 1: Find middle
    slow = head
    fast = head
    
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    # Step 2: Reverse second half
    second_half = reverse_list(slow)
    first_half = head
    
    # Step 3: Merge two halves
    while second_half.next:  # second_half will be longer or equal
        # Save next pointers
        first_next = first_half.next
        second_next = second_half.next
        
        # Reorder
        first_half.next = second_half
        second_half.next = first_next
        
        # Move to next pair
        first_half = first_next
        second_half = second_next


def print_list(head: Optional[ListNode]) -> None:
    """Helper to print list."""
    values = []
    current = head
    while current:
        values.append(str(current.val))
        current = current.next
    print(" → ".join(values))


# Usage Example
# Original: 1 → 2 → 3 → 4 → 5
head = create_list([1, 2, 3, 4, 5])
print("Original:")
print_list(head)

reorder_list(head)
print("Reordered:")
print_list(head)  # Output: 1 → 5 → 2 → 4 → 3
```

### Problem 7: Cycle in Circular Array

```python
def circular_array_loop(nums: List[int]) -> bool:
    """
    Check if there's a cycle in circular array.
    You can jump forward/backward based on value at current index.
    All values in cycle must have same direction (all positive or all negative).
    
    Args:
        nums: Array of integers
    
    Returns:
        True if valid cycle exists
    
    Time Complexity: O(n²) worst case
    Space Complexity: O(1)
    """
    def get_next_index(index: int) -> int:
        """Get next index in circular manner."""
        n = len(nums)
        return (index + nums[index]) % n
    
    def is_not_one_element_cycle(index: int) -> bool:
        """Check if cycle is more than one element."""
        return get_next_index(index) != index
    
    def is_same_direction(index1: int, index2: int) -> bool:
        """Check if two indices have same direction (both +ve or -ve)."""
        return (nums[index1] > 0) == (nums[index2] > 0)
    
    for i in range(len(nums)):
        if nums[i] == 0:
            continue
        
        # Use fast & slow pointers
        slow = i
        fast = i
        is_forward = nums[i] > 0
        
        # Move until we find a cycle or detect invalid state
        while True:
            # Move slow one step
            slow = get_next_index(slow)
            
            # Check if slow is valid
            if not is_same_direction(slow, i) or not is_not_one_element_cycle(slow):
                break
            
            # Move fast two steps
            fast = get_next_index(fast)
            if not is_same_direction(fast, i) or not is_not_one_element_cycle(fast):
                break
            
            fast = get_next_index(fast)
            if not is_same_direction(fast, i) or not is_not_one_element_cycle(fast):
                break
            
            # Check if they meet
            if slow == fast:
                return True
        
        # Mark current path as visited by setting to 0
        slow = i
        old_value = nums[i]
        while nums[slow] * old_value > 0:  # Same direction
            next_index = get_next_index(slow)
            nums[slow] = 0
            slow = next_index
    
    return False


# Usage Example
nums = [2, -1, 1, 2, 2]
print(f"Has cycle: {circular_array_loop(nums)}")  # Output: True
# Explanation: 0 → 2 → 3 → 0 (cycle of length 3)

nums = [-1, 2]
print(f"Has cycle: {circular_array_loop(nums)}")  # Output: False
# Explanation: Directions change
```

## Complexity Analysis

### Time Complexity

**Cycle Detection:**
- **Best Case:** O(1) - Empty list or single node
- **Average Case:** O(n) - Fast pointer travels at most 2n steps
- **Worst Case:** O(n) - Complete traversal

**Finding Middle:**
- **All Cases:** O(n) - Single pass through list

**Why O(n) for cycle detection?**
Even though fast pointer moves twice as fast:
- In worst case, fast pointer travels entire list (n nodes) plus partial cycle
- Maximum steps for fast: n + cycle_length ≤ 2n
- Total operations: O(2n) = O(n)

### Space Complexity

- **Space:** O(1) - Only two pointers regardless of input size
- No recursion, no additional data structures
- Constants space even for cycle detection

### Comparison with Alternatives

| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| Fast & Slow | O(n) | O(1) | Optimal for cycle detection |
| Hash Set | O(n) | O(n) | Simpler but uses extra space |
| Marking Nodes | O(n) | O(1) | Modifies original structure |
| Recursion | O(n) | O(n) | Call stack space |

## Examples

### Example 1: Cycle Detection

```
List: 1 → 2 → 3 → 4 → 5
               ↑       ↓
               8 ← 7 ← 6

Step-by-step execution:

Step 1: slow=1, fast=1
Step 2: slow=2, fast=3
Step 3: slow=3, fast=5
Step 4: slow=4, fast=7
Step 5: slow=5, fast=4
Step 6: slow=6, fast=6  ← They meet! Cycle detected

Finding cycle start:
Reset slow to 1
Step 1: slow=1, fast=6
Step 2: slow=2, fast=7
Step 3: slow=3, fast=8
Step 4: slow=4, fast=4  ← They meet at cycle start (node 4)
```

### Example 2: Finding Middle

```
Odd length: 1 → 2 → 3 → 4 → 5

Step 1: slow=1, fast=1
Step 2: slow=2, fast=3
Step 3: slow=3, fast=5
Step 4: fast.next=None, stop
Result: slow=3 (middle)

Even length: 1 → 2 → 3 → 4 → 5 → 6

Step 1: slow=1, fast=1
Step 2: slow=2, fast=3
Step 3: slow=3, fast=5
Step 4: slow=4, fast=None, stop
Result: slow=4 (second middle)
```

### Example 3: Happy Number

```
n = 19

Iteration 1: slow = 82, fast = 68
  19 → 1²+9² = 82
  19 → 82 → 8²+2² = 68

Iteration 2: slow = 68, fast = 100
  82 → 68
  68 → 100 → 1

Iteration 3: slow = 100, fast = 1
  68 → 100
  1 → 1 → 1

fast == 1, so 19 is happy!

n = 2 (not happy)

Will eventually cycle: 4 → 16 → 37 → 58 → 89 → 145 → 42 → 20 → 4
slow and fast will meet at some point in this cycle
```

### Example 4: Palindrome Check

```
List: 1 → 2 → 3 → 2 → 1

Step 1: Find middle
slow=3, fast=None

Step 2: Reverse second half
Original: 1 → 2 → 3 → 2 → 1
After:    1 → 2 → 3 ← 2 ← 1
                ↓
         first_half   second_half

Step 3: Compare
1 == 1 ✓
2 == 2 ✓
3 != None (second_half ends)

Result: True (palindrome)
```

## Edge Cases

### 1. Empty or Single Node List
**Scenario:** head = None or head.next = None
**Challenge:** No cycle possible, no middle to find
**Solution:** Return early
```python
if not head or not head.next:
    return False  # for cycle detection
    return head   # for finding middle
```

### 2. Two-Node List with Cycle
**Scenario:** 1 ⇄ 2 (pointing to each other)
**Challenge:** Smallest possible cycle
**Solution:** Algorithm handles naturally, pointers meet on second iteration

### 3. Cycle at Head
**Scenario:** Head points to itself
**Challenge:** Entire list is the cycle
**Solution:**
```python
# Detected immediately
slow = head
fast = head
slow = slow.next  # Points to head
fast = fast.next.next  # Points to head
# slow == fast == head
```

### 4. Very Long List Without Cycle
**Scenario:** 100,000 nodes, no cycle
**Challenge:** Fast pointer must reach null efficiently
**Solution:** Fast pointer will reach null in ~50,000 iterations

### 5. Cycle Length Equals List Length
**Scenario:** Every node is part of the cycle
**Challenge:** No "tail" before cycle
**Solution:** Works normally, cycle start is the head

### 6. Even vs Odd Length for Middle
**Scenario:** Finding middle with different list lengths
**Challenge:** Which middle for even length?
**Solution:** Adjust fast pointer starting position
```python
# Second middle (default)
fast = head

# First middle
fast = head.next
```

## Common Pitfalls

### ❌ Pitfall 1: Not Checking for Null Pointers
**What happens:** NullPointerException/AttributeError
**Why it's wrong:**
```python
# Missing null checks
while fast:  # Wrong! Should check fast.next too
    fast = fast.next.next  # Crashes if fast.next is None
```
**Correct approach:**
```python
while fast and fast.next:  # Check both conditions
    slow = slow.next
    fast = fast.next.next
```

### ❌ Pitfall 2: Forgetting Fast Moves Twice
**What happens:** Algorithm doesn't work correctly
**Why it's wrong:**
```python
# Both moving at same speed
slow = slow.next
fast = fast.next  # Should be fast.next.next
# They'll never meet in a cycle!
```
**Correct approach:**
```python
slow = slow.next
fast = fast.next.next  # Fast moves 2 steps
```

### ❌ Pitfall 3: Incorrect Cycle Start Detection
**What happens:** Returns wrong node
**Why it's wrong:**
```python
# Not resetting pointer to head
# slow and fast both still at meeting point
while slow != fast:
    slow = slow.next
    fast = fast.next
# This won't find the start!
```
**Correct approach:**
```python
# Reset ONE pointer to head
slow = head  # Reset to head
while slow != fast:
    slow = slow.next
    fast = fast.next  # Both move one step now
```

### ❌ Pitfall 4: Modifying List Without Restoration
**What happens:** Original list structure is lost
**Why it's wrong:**
```python
# Reversing second half for palindrome check
second_half = reverse_list(slow)
# ... comparison ...
# Forgot to reverse back!
```
**Correct approach:**
```python
# Save reference to restore later
second_half = reverse_list(slow)
# ... comparison ...
reverse_list(slow)  # Restore original structure
```

### ❌ Pitfall 5: Off-by-One Errors in Middle Finding
**What happens:** Returns wrong middle node
**Why it's wrong:**
```python
# Wrong condition
while fast.next:  # Should be: fast and fast.next
    slow = slow.next
    fast = fast.next.next
```
**Correct approach:**
```python
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
```

## Variations and Extensions

### Variation 1: Finding Cycle Length
**Description:** After detecting cycle, count nodes in it
**Implementation:**
```python
def find_cycle_length(head):
    # Detect cycle first (standard fast & slow)
    # ... (omitted for brevity)
    
    # Count nodes in cycle
    current = slow
    length = 0
    while True:
        current = current.next
        length += 1
        if current == slow:
            break
    return length
```

### Variation 2: Finding Kth Node from End
**Description:** Use fast pointer with k-step head start
**When to use:** Finding nth node from end in single pass
**Implementation:**
```python
def find_kth_from_end(head, k):
    fast = head
    slow = head
    
    # Move fast k steps ahead
    for _ in range(k):
        if not fast:
            return None
        fast = fast.next
    
    # Move both until fast reaches end
    while fast:
        slow = slow.next
        fast = fast.next
    
    return slow
```

### Variation 3: Triple Speed (Three Pointers)
**Description:** Three pointers at different speeds (1x, 2x, 3x)
**When to use:** More complex cycle detection scenarios
**Note:** Rarely needed, typically fast & slow is sufficient

### Variation 4: Finding Intersection of Two Lists
**Description:** Use two pointers switching between lists
**When to use:** Finding where two linked lists merge
**Implementation:**
```python
def find_intersection(head1, head2):
    if not head1 or not head2:
        return None
    
    ptr1, ptr2 = head1, head2
    
    # Pointers will meet at intersection or None
    while ptr1 != ptr2:
        ptr1 = ptr1.next if ptr1 else head2
        ptr2 = ptr2.next if ptr2 else head1
    
    return ptr1  # Intersection node or None
```

## Practice Problems

### Beginner
1. **Linked List Cycle (LeetCode #141)** - Detect if cycle exists
2. **Middle of the Linked List (LeetCode #876)** - Find middle node
3. **Remove Nth Node From End (LeetCode #19)** - Use fast pointer head start

### Intermediate
1. **Linked List Cycle II (LeetCode #142)** - Find cycle start
2. **Happy Number (LeetCode #202)** - Detect cycle in number sequence
3. **Palindrome Linked List (LeetCode #234)** - Check palindrome in O(1) space
4. **Reorder List (LeetCode #143)** - Rearrange list pattern
5. **Intersection of Two Linked Lists (LeetCode #160)** - Find merge point

### Advanced
1. **Circular Array Loop (LeetCode #457)** - Complex cycle detection
2. **Find Duplicate Number (LeetCode #287)** - Array as linked list
3. **Split Linked List in Parts (LeetCode #725)** - Multiple middle finding

## Real-World Applications

### Industry Use Cases

1. **Memory Leak Detection:** Detecting circular references in garbage collection
   - Modern garbage collectors use cycle detection algorithms

2. **Distributed Systems:** Detecting loops in network topology
   - Routing protocols use similar algorithms to detect routing loops

3. **Music Playlist:** Loop detection in circular playlists
   - Streaming services detect infinite loops in user-created playlists

4. **File System:** Detecting symbolic link cycles
   - Operating systems prevent infinite loops from circular symlinks

### Popular Implementations

- **Python garbage collector:** Uses cycle detection for reference counting
- **Git commit history:** Finding merge bases uses similar pointer techniques
- **Network protocols:** OSPF and other routing protocols detect loops
- **Browser back/forward:** Cycle detection in browser history

### Practical Scenarios

- **Train scheduling:** Detecting loops in circular train routes
- **Game development:** Detecting infinite loops in game state machines
- **Data validation:** Checking for circular dependencies in build systems
- **Social networks:** Finding loops in follower relationships (circular following)

## Related Topics

### Prerequisites to Review
- **Linked List basics** - Node structure, traversal, insertion/deletion
- **Pointer manipulation** - Understanding references and memory
- **Mathematical proof** - Understanding why the algorithm works

### Next Steps
- **Two Pointers Pattern** - Similar concept for arrays
- **Cycle Detection in Graphs** - DFS-based cycle detection
- **Floyd's Cycle Detection** - More formal study of the algorithm
- **Tortoise and Hare variants** - Advanced applications

### Similar Concepts
- **Binary search** - Another divide and conquer with pointers
- **Sliding window** - Two pointers moving together
- **Union-Find** - Another cycle detection approach for graphs

### Further Reading
- [Floyd's Cycle Detection Algorithm](https://en.wikipedia.org/wiki/Cycle_detection#Floyd's_Tortoise_and_Hare) - Original algorithm
- [LeetCode Linked List Study Guide](https://leetcode.com/tag/linked-list/)
- Knuth's "The Art of Computer Programming" - Vol 2, Semi-numerical Algorithms
- [GeeksforGeeks: Detect Loop in Linked List](https://www.geeksforgeeks.org/detect-loop-in-a-linked-list/)
