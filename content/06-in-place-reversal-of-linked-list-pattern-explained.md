# In-place Reversal of Linked List Pattern

**Difficulty:** Medium
**Prerequisites:** Linked Lists, Pointers
**Estimated Reading Time:** 25 minutes

## Introduction

The In-place Reversal of Linked List pattern is a fundamental technique that involves reversing the links between nodes in a linked list without using extra space for another data structure. This pattern is crucial for many linked list manipulation problems and demonstrates deep understanding of pointer manipulation.

**Why it matters:** This pattern appears frequently in technical interviews because it tests your ability to manipulate pointers carefully - a skill that's essential for systems programming, operating systems development, and low-level data structure implementation. Companies value this because it shows you can work with memory efficiently and understand how data structures actually work under the hood. Mastering this pattern gives you confidence to tackle any linked list problem.

**Real-world analogy:** Imagine a line of people holding hands in a chain, all facing forward. To reverse the line, you don't make them run to opposite ends - instead, you have each person turn around and hold the hand of the person who was behind them. Person by person, you reverse the direction each person is facing (their "next" pointer) until the entire chain is reversed. The line stays in place (in-place reversal), but everyone's now facing and holding hands in the opposite direction!

## Core Concepts

### Key Principles

1. **Three-pointer technique:** Use `previous`, `current`, and `next` pointers to reverse links one by one.

2. **Break and remake links:** Save the next node before breaking the current link, then reverse the current link.

3. **Iterative traversal:** Move through the list once, reversing each link as you go.

4. **New head discovery:** After reversal, the original tail becomes the new head.

5. **In-place modification:** O(1) space complexity - only using a few pointers, no additional data structures.

### Essential Terms

- **Node:** Single element in linked list containing data and next pointer
- **Head:** First node in the linked list
- **Tail:** Last node (points to None/null)
- **Reverse:** Change direction of next pointers
- **In-place:** Without using extra space for duplication
- **Sentinel/Dummy node:** Extra node before head to simplify edge cases

### Visual Overview

```
Original List:  1 → 2 → 3 → 4 → 5 → None

Reversal Process:

Step 0: Initial State
prev = None
curr = 1 → 2 → 3 → 4 → 5 → None

Step 1: Reverse first link
None ← 1    2 → 3 → 4 → 5 → None
      prev curr

Step 2: Move pointers, reverse second link  
None ← 1 ← 2    3 → 4 → 5 → None
           prev curr

Step 3: Continue...
None ← 1 ← 2 ← 3    4 → 5 → None
                prev curr

Step 4:
None ← 1 ← 2 ← 3 ← 4    5 → None
                    prev curr

Step 5: Final
None ← 1 ← 2 ← 3 ← 4 ← 5
                        prev
                        
Result: 5 → 4 → 3 → 2 → 1 → None
New head = prev (which is 5)
```

## How It Works

### Basic Reversal Algorithm

1. **Initialize pointers:**
   - `prev = None` (will become new tail)
   - `current = head` (node we're processing)

2. **While current is not None:**
   - Save next: `next_node = current.next`
   - Reverse link: `current.next = prev`
   - Move prev forward: `prev = current`
   - Move current forward: `current = next_node`

3. **Return prev** (new head)

### Why It's O(1) Space

We only use three pointers (`prev`, `current`, `next_node`) regardless of list size. No additional data structure scales with input size.

### Step-by-Step Example: Reverse Entire List

Problem: Reverse 1 → 2 → 3 → None

```
Initial State:
head → 1 → 2 → 3 → None
prev = None
curr = 1

Iteration 1:
┌─────────────────────────────────┐
│ next_node = curr.next (save 2)  │
│ curr.next = prev (1 → None)     │
│ prev = curr (prev is now 1)     │
│ curr = next_node (curr is now 2)│
└─────────────────────────────────┘

State after iteration 1:
None ← 1    2 → 3 → None
       prev curr

Iteration 2:
next_node = 3
curr.next = 1
prev = 2  
curr = 3

State after iteration 2:
None ← 1 ← 2    3 → None
           prev curr

Iteration 3:
next_node = None
curr.next = 2
prev = 3
curr = None

State after iteration 3:
None ← 1 ← 2 ← 3
               prev
               curr = None

Loop terminates (curr is None)

Return prev as new head:
3 → 2 → 1 → None
```

## How to Identify This Pattern

In-place Linked List Reversal has distinct characteristics. Here's how to recognize it:

### Primary Indicators ✓

**Problem explicitly mentions "reverse" and "linked list"**
- Need to reverse entire linked list
- Reverse portion of linked list
- Keywords: "reverse", "linked list", "reverse order"
- Example: "Reverse a singly linked list"

**Reversing nodes in groups or patterns**
- Reverse every k nodes
- Reverse alternate nodes
- Swap adjacent pairs
- Keywords: "groups", "k nodes", "pairs", "alternate"
- Example: "Reverse nodes in k-group"

**Rearranging linked list in specific pattern**
- Reorder list (L0→Ln→L1→Ln-1...)
- Odd-even reordering
- Zigzag patterns
- Keywords: "reorder", "rearrange", "odd-even"
- Example: "Reorder list alternating first and last"

**Checking palindrome with O(1) space**
- Verify if linked list reads same forwards/backwards
- Cannot use array or stack
- Must use constant space
- Keywords: "palindrome", "O(1) space", "constant space"
- Example: "Check if linked list is palindrome"

**Rotating or shifting linked list**
- Rotate list by k positions
- Circular shifts
- Keywords: "rotate", "shift", "move by k"
- Example: "Rotate list right by k places"

**O(1) space constraint with linked list manipulation**
- Cannot create new list
- Must modify in-place
- Keywords: "in-place", "constant space", "no extra list"
- Example: "Reverse in-place without extra space"

### Common Problem Phrases 🔑

Watch for these exact phrases:
- "Reverse linked list"
- "Reverse sublist from position m to n"
- "Reverse nodes in k-group"
- "Swap nodes in pairs"
- "Swap adjacent nodes"
- "Reverse alternate k nodes"
- "Palindrome linked list"
- "Reorder list"
- "Rotate list"
- "Rotate linked list"
- "Odd-even linked list"

### When NOT to Use In-place Reversal ✗

**Simple traversal only**
- Just need to visit nodes
- No restructuring needed
- → Use single pointer

**Detecting cycles**
- Finding if list has loop
- → Use Fast & Slow Pointers

**Finding middle or kth element**
- Location finding without reversal
- → Use Fast & Slow Pointers

**Can use extra space easily**
- Stack or array allowed
- Easier implementation available
- → Consider stack-based approach

**Not a linked list**
- Array reversal
- → Use Two Pointers on array

### Quick Decision Checklist ✅

Ask yourself:

1. **Is it a linked list?** → Consider this pattern
2. **Does it mention "reverse"?** → Strong indicator
3. **Reverse entire list or part?** → In-place Reversal
4. **Reversing in groups (k nodes)?** → In-place Reversal
5. **Swapping adjacent pairs?** → In-place Reversal
6. **Reordering with specific pattern?** → In-place Reversal
7. **Palindrome check with O(1) space?** → In-place Reversal
8. **Rotating linked list?** → In-place Reversal

If YES to question 1 AND question 2, it's likely In-place Reversal!

### Core Algorithm Pattern 📝

**Basic Reversal Template:**
```python
prev = None
current = head

while current:
    next_node = current.next    # Save next
    current.next = prev          # Reverse pointer
    prev = current               # Move prev forward
    current = next_node          # Move current forward

return prev  # New head
```

**This 3-pointer technique is the foundation of ALL reversal problems!**

### Visual Recognition 👁️

**If you see arrows reversing direction, it's In-place Reversal:**

```
Before: 1 → 2 → 3 → 4 → 5 → None

After:  None ← 1 ← 2 ← 3 ← 4 ← 5
         or
        5 → 4 → 3 → 2 → 1 → None
```

**Partial Reversal:**
```
Before: 1 → 2 → 3 → 4 → 5 → None
                ↑       ↑
                m       n

After:  1 → 4 → 3 → 2 → 5 → None
        └───────────────────┘
        Reversed m to n only
```

### Example Pattern Matching 💡

**Problem: "Reverse a singly linked list"**

Analysis:
- ✓ Linked list explicitly mentioned
- ✓ Says "reverse"
- ✓ Entire list reversal
- ✓ Classic in-place reversal

**Verdict: USE IN-PLACE REVERSAL** ✓

**Problem: "Reverse nodes from position m to n"**

Analysis:
- ✓ Linked list
- ✓ Reverse operation
- ✓ Partial reversal
- ✓ Use dummy node + reversal

**Verdict: USE IN-PLACE REVERSAL** ✓

**Problem: "Reverse nodes in groups of k"**

Analysis:
- ✓ Linked list
- ✓ Reverse in groups
- ✓ Multiple reversals needed

**Verdict: USE IN-PLACE REVERSAL** ✓

**Problem: "Check if linked list is palindrome (O(1) space)"**

Analysis:
- ✓ Linked list
- ? Needs comparison from both ends
- ✓ Find middle (Fast & Slow) + Reverse second half
- ✓ O(1) space requirement

**Verdict: USE IN-PLACE REVERSAL + FAST & SLOW** ✓

**Problem: "Detect cycle in linked list"**

Analysis:
- ✓ Linked list
- ✗ Not about reversing
- ✗ Cycle detection

**Verdict: USE FAST & SLOW POINTERS** (Not In-place Reversal) ✗

**Problem: "Reverse an array"**

Analysis:
- ✗ Not a linked list
- ✗ Array reversal

**Verdict: USE TWO POINTERS** (Not In-place Reversal) ✗

### Pattern vs Problem Type 📊

| Problem Type | In-place Reversal? | Alternative |
|--------------|--------------------| ------------|
| Reverse linked list | ✅ YES | Stack (O(n) space) |
| Reverse sublist | ✅ YES | - |
| Reverse k-group | ✅ YES | - |
| Swap pairs | ✅ YES | - |
| Palindrome (O(1) space) | ✅ YES | Array (O(n) space) |
| Reorder list | ✅ YES | - |
| Rotate list | ✅ YES | - |
| Detect cycle | ❌ NO | Fast & Slow Pointers |
| Find middle | ❌ NO | Fast & Slow Pointers |
| Reverse array | ❌ NO | Two Pointers |

### Problem Variants 🔀

**Variant 1: Reverse Entire List**
- Basic reversal start to end
- Template pattern applies directly

**Variant 2: Reverse Sublist (m to n)**
- Reverse only middle portion
- Need to connect before/after parts
- **Extra complexity:** Dummy node, reconnection

**Variant 3: Reverse in Groups (k nodes)**
- Reverse every k consecutive nodes
- More complex connection logic
- **Extra complexity:** Check if k nodes available

**Variant 4: Conditional/Alternating Reversal**
- Reverse alternate groups
- Reverse based on condition
- **Extra complexity:** Skip logic

**Variant 5: Reordering**
- Combine reversal with other operations
- Find middle + reverse + merge
- **Extra complexity:** Multiple steps

### Three-Pointer Technique 🔧

**Why three pointers?**

```
prev    current    next_node
         ↓           ↓
None ← 1 → 2 → 3 → 4 → None

1. Save next_node (don't lose rest of list)
2. Reverse current.next to prev
3. Move prev and current forward
```

**All reversal problems use this technique!**

### Keywords Cheat Sheet 📝

**STRONG "In-place Reversal" Keywords:**
- reverse linked list
- reverse in k-group
- swap pairs
- swap adjacent

**MODERATE Keywords:**
- reorder list
- rotate list
- palindrome (with O(1) space)
- in-place (with linked list)

**DOMAIN-Specific:**
- "reverse from position m to n"
- "reverse alternate k"
- "odd-even linked list"

### Red Flags 🚩

These suggest IN-PLACE REVERSAL might NOT be right:
- Not a linked list → Different reversal
- "detect cycle" → Fast & Slow Pointers
- "find middle" → Fast & Slow Pointers
- Can use extra space → Stack easier
- Array reversal → Two Pointers

### Green Flags 🟢

STRONG indicators for IN-PLACE REVERSAL:
- "Reverse linked list"
- "Reverse sublist"
- "Reverse in k-group"
- "Swap pairs"
- "Reorder list"
- "Rotate linked list"
- "Palindrome" + "O(1) space" + "linked list"
- Any pointer direction flipping

### Key Insight 💡

**The moment you need to FLIP the direction of "next" pointers, think In-place Reversal!**

**Pattern Recognition:**
```
If: LinkedList + Reverse → In-place Reversal
If: LinkedList + Cycle → Fast & Slow
If: LinkedList + Middle → Fast & Slow
If: Array + Reverse → Two Pointers
```

### Common Combinations 🔄

This pattern often combines with others:

**Palindrome Check:**
1. Fast & Slow to find middle
2. **Reverse second half** ← In-place Reversal
3. Compare both halves

**Reorder List:**
1. Fast & Slow to find middle
2. **Reverse second half** ← In-place Reversal
3. Merge alternately

These combinations are powerful!



## Implementation

### Problem 1: Reverse Linked List (LeetCode #206)

```python
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverseList(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Reverse a singly linked list.
    
    Args:
        head: Head of the linked list
    
    Returns:
        Head of reversed list
    
    Time Complexity: O(n) - visit each node once
    Space Complexity: O(1) - only three pointers used
    """
    prev = None
    current = head
    
    while current:
        # Step 1: Save the next node
        next_node = current.next
        
        # Step 2: Reverse the link
        current.next = prev
        
        # Step 3: Move prev and current one step forward
        prev = current
        current = next_node
    
    # prev is now pointing to the new head
    return prev


# Recursive Solution
def reverseListRecursive(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Reverse linked list using recursion.
    
    Time Complexity: O(n)
    Space Complexity: O(n) - recursion stack
    """
    # Base case: empty list or single node
    if not head or not head.next:
        return head
    
    # Recursive case: reverse rest of list
    new_head = reverseListRecursive(head.next)
    
    # Make next node point back to current
    head.next.next = head
    
    # Current node becomes tail
    head.next = None
    
    return new_head


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


def print_list(head):
    values = []
    while head:
        values.append(str(head.val))
        head = head.next
    print(" → ".join(values) + " → None")


# Test
head = create_list([1, 2, 3, 4, 5])
print("Original:")
print_list(head)  # 1 → 2 → 3 → 4 → 5 → None

reversed_head = reverseList(head)
print("Reversed:")
print_list(reversed_head)  # 5 → 4 → 3 → 2 → 1 → None
```

### Problem 2: Reverse Linked List II (LeetCode #92)

```python
def reverseBetween(head: Optional[ListNode], left: int, 
                   right: int) -> Optional[ListNode]:
    """
    Reverse nodes from position left to right (1-indexed).
    
    Args:
        head: Head of linked list
        left: Starting position (1-indexed)
        right: Ending position (1-indexed)
    
    Returns:
        Head of modified list
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not head or left == right:
        return head
    
    # Create dummy node to handle edge case where left=1
    dummy = ListNode(0)
    dummy.next = head
    
    # Step 1: Find node before 'left' position
    prev_left = dummy
    for _ in range(left - 1):
        prev_left = prev_left.next
    
    # Step 2: Reverse the sublist
    # prev_left → left_node → ... → right_node → after_right
    current = prev_left.next
    
    for _ in range(right - left):
        # Move node after current to front of reversed section
        next_node = current.next
        current.next = next_node.next
        next_node.next = prev_left.next
        prev_left.next = next_node
    
    return dummy.next


# Alternative: Three-step approach (easier to understand)
def reverseBetweenClear(head: Optional[ListNode], left: int, 
                        right: int) -> Optional[ListNode]:
    """
    Clearer implementation with explicit steps.
    """
    if not head or left == right:
        return head
    
    dummy = ListNode(0)
    dummy.next = head
    
    # Phase 1: Navigate to position left-1
    prev_sublist = dummy
    for _ in range(left - 1):
        prev_sublist = prev_sublist.next
    
    # Phase 2: Reverse the sublist from left to right
    sublist_head = prev_sublist.next
    prev = None
    current = sublist_head
    
    for _ in range(right - left + 1):
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    
    # Phase 3: Connect reversed sublist back
    prev_sublist.next = prev  # Connect to new head of reversed part
    sublist_head.next = current  # Connect tail to rest of list
    
    return dummy.next


# Usage Example
head = create_list([1, 2, 3, 4, 5])
print("Original:")
print_list(head)  # 1 → 2 → 3 → 4 → 5 → None

result = reverseBetween(head, 2, 4)
print("Reversed from position 2 to 4:")
print_list(result)  # 1 → 4 → 3 → 2 → 5 → None
```

### Problem 3: Reverse Nodes in k-Group (LeetCode #25)

```python
def reverseKGroup(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    Reverse nodes in groups of k. If remaining nodes < k, leave as is.
    
    Args:
        head: Head of linked list
        k: Group size
    
    Returns:
        Head of modified list
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    # Helper function to reverse k nodes starting from head
    def reverse_k_nodes(head, k):
        prev = None
        current = head
        
        for _ in range(k):
            if not current:  # Less than k nodes remaining
                return None, None
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        # prev is new head of reversed group
        # head is new tail of reversed group
        # current is first node of next group
        return prev, current
    
    # Check if we have at least k nodes
    count = 0
    check = head
    while count < k and check:
        check = check.next
        count += 1
    
    if count < k:
        return head  # Not enough nodes to reverse
    
    # Reverse first k nodes
    new_head, next_group_head = reverse_k_nodes(head, k)
    
    # Recursively reverse remaining groups
    # head is now tail of first reversed group
    head.next = reverseKGroup(next_group_head, k)
    
    return new_head


# Iterative version (more space efficient)
def reverseKGroupIterative(head: Optional[ListNode], 
                           k: int) -> Optional[ListNode]:
    """
    Iterative version of reverse in k groups.
    """
    dummy = ListNode(0)
    dummy.next = head
    group_prev = dummy
    
    while True:
        # Check if we have k nodes remaining
        kth = group_prev
        for _ in range(k):
            kth = kth.next
            if not kth:
                return dummy.next  # Less than k nodes left
        
        # Reverse k nodes
        group_next = kth.next
        prev, current = group_next, group_prev.next
        
        for _ in range(k):
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        
        # Connect reversed group
        temp = group_prev.next
        group_prev.next = kth
        group_prev = temp
    
    return dummy.next


# Usage Example
head = create_list([1, 2, 3, 4, 5])
print("Original:")
print_list(head)  # 1 → 2 → 3 → 4 → 5 → None

result = reverseKGroup(head, 2)
print("Reversed in groups of 2:")
print_list(result)  # 2 → 1 → 4 → 3 → 5 → None

head = create_list([1, 2, 3, 4, 5])
result = reverseKGroup(head, 3)
print("Reversed in groups of 3:")
print_list(result)  # 3 → 2 → 1 → 4 → 5 → None
```

### Problem 4: Swap Nodes in Pairs (LeetCode #24)

```python
def swapPairs(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Swap every two adjacent nodes.
    
    Args:
        head: Head of linked list
    
    Returns:
        Head of modified list
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    dummy = ListNode(0)
    dummy.next = head
    current = dummy
    
    while current.next and current.next.next:
        # Nodes to swap
        first = current.next
        second = current.next.next
        
        # Perform swap
        first.next = second.next
        second.next = first
        current.next = second
        
        # Move to next pair
        current = first
    
    return dummy.next


# Recursive version
def swapPairsRecursive(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Recursive approach to swap pairs.
    """
    # Base case: 0 or 1 node
    if not head or not head.next:
        return head
    
    # Nodes to swap
    first = head
    second = head.next
    
    # Swap
    first.next = swapPairsRecursive(second.next)
    second.next = first
    
    return second


# Usage Example
head = create_list([1, 2, 3, 4])
print("Original:")
print_list(head)  # 1 → 2 → 3 → 4 → None

result = swapPairs(head)
print("After swapping pairs:")
print_list(result)  # 2 → 1 → 4 → 3 → None
```

### Problem 5: Palindrome Linked List (LeetCode #234)

```python
def isPalindrome(head: Optional[ListNode]) -> bool:
    """
    Check if linked list is a palindrome.
    
    Args:
        head: Head of linked list
    
    Returns:
        True if palindrome, False otherwise
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not head or not head.next:
        return True
    
    # Step 1: Find middle using slow/fast pointers
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    # Step 2: Reverse second half
    second_half = reverse_list_helper(slow)
    
    # Step 3: Compare first and second half
    first_half = head
    result = True
    
    while result and second_half:
        if first_half.val != second_half.val:
            result = False
        first_half = first_half.next
        second_half = second_half.next
    
    # Step 4: (Optional) Restore list
    reverse_list_helper(slow)
    
    return result


def reverse_list_helper(head):
    """Helper to reverse list in-place."""
    prev = None
    current = head
    
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node
    
    return prev


# Usage Example
head = create_list([1, 2, 3, 2, 1])
print("List:", end=" ")
print_list(head)
print(f"Is palindrome: {isPalindrome(head)}")  # True

head = create_list([1, 2, 3, 4, 5])
print("List:", end=" ")
print_list(head)
print(f"Is palindrome: {isPalindrome(head)}")  # False
```

### Problem 6: Rotate List (LeetCode #61)

```python
def rotateRight(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    Rotate list to the right by k places.
    
    Args:
        head: Head of linked list
        k: Number of rotations
    
    Returns:
        Head of rotated list
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not head or not head.next or k == 0:
        return head
    
    # Step 1: Find length and connect tail to head (make circular)
    length = 1
    tail = head
    while tail.next:
        tail = tail.next
        length += 1
    
    tail.next = head  # Make circular
    
    # Step 2: Find new tail (length - k % length - 1 steps from head)
    k = k % length
    steps_to_new_tail = length - k - 1
    
    new_tail = head
    for _ in range(steps_to_new_tail):
        new_tail = new_tail.next
    
    # Step 3: Break circle and return new head
    new_head = new_tail.next
    new_tail.next = None
    
    return new_head


# Usage Example
head = create_list([1, 2, 3, 4, 5])
print("Original:")
print_list(head)

result = rotateRight(head, 2)
print("Rotated right by 2:")
print_list(result)  # 4 → 5 → 1 → 2 → 3 → None
```

### Problem 7: Reorder List (LeetCode #143)

```python
def reorderList(head: Optional[ListNode]) -> None:
    """
    Reorder list: L0 → L1 → ... → Ln-1 → Ln to:
    L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → ...
    
    Modifies list in-place.
    
    Time Complexity: O(n)
    Space Complexity: O(1)
    """
    if not head or not head.next:
        return
    
    # Step 1: Find middle
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    # Step 2: Reverse second half
    second = reverse_list_helper(slow.next)
    slow.next = None  # Split into two halves
    
    # Step 3: Merge two halves
    first = head
    while second:
        temp1 = first.next
        temp2 = second.next
        
        first.next = second
        second.next = temp1
        
        first = temp1
        second = temp2


# Usage Example
head = create_list([1, 2, 3, 4])
print("Original:")
print_list(head)

reorderList(head)
print("Reordered:")
print_list(head)  # 1 → 4 → 2 → 3 → None

head = create_list([1, 2, 3, 4, 5])
reorderList(head)
print("Reordered (odd length):")
print_list(head)  # 1 → 5 → 2 → 4 → 3 → None
```

## Complexity Analysis

### Time Complexity

**Basic Reversal:**
- **Single pass:** O(n) - visit each node once
- **Each operation:** O(1) - pointer updates
- **Total:** O(n)

**Reverse Sublist:**
- **Navigate to position:** O(left)
- **Reverse k nodes:** O(right - left)
- **Total:** O(n)

**Reverse in k-Groups:**
- **Each node reversed once:** O(n)
- **Group checking:** O(n/k) groups
- **Total:** O(n)

### Space Complexity

**Iterative Solutions:**
- **Pointers:** O(1) - only `prev`, `current`, `next`
- **Total:** O(1)

**Recursive Solutions:**
- **Call stack:** O(n) worst case
- **Total:** O(n)

### Comparison with Alternatives

| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| In-place Reversal | O(n) | O(1) | Optimal |
| Using Stack | O(n) | O(n) | Simple but uses extra space |
| Recursion | O(n) | O(n) | Elegant but stack overhead |
| New List | O(n) | O(n) | Easy but not in-place |

## Examples

### Example 1: Basic Reversal Detail

```
Input: 1 → 2 → 3 → None

Detailed Trace:

Initial:
prev = None
curr = 1 (pointing to 2)

─────────── Iteration 1 ───────────
Before:
None    1 → 2 → 3 → None
prev  curr

Actions:
1. next_node = curr.next    → next_node = 2
2. curr.next = prev         → 1.next = None
3. prev = curr              → prev = 1
4. curr = next_node         → curr = 2

After:
None ← 1    2 → 3 → None
      prev curr

─────────── Iteration 2 ───────────
Before:
None ← 1    2 → 3 → None
      prev curr

Actions:
1. next_node = 3
2. 2.next = 1
3. prev = 2
4. curr = 3

After:
None ← 1 ← 2    3 → None
           prev curr

─────────── Iteration 3 ───────────
Before:
None ← 1 ← 2    3 → None
           prev curr

Actions:
1. next_node = None
2. 3.next = 2
3. prev = 3
4. curr = None

After:
None ← 1 ← 2 ← 3
               prev
curr = None (loop ends)

Result: prev (which is 3) is new head
Final: 3 → 2 → 1 → None
```

### Example 2: Reverse Between Positions

```
Input: 1 → 2 → 3 → 4 → 5 → None, left=2, right=4
Output: 1 → 4 → 3 → 2 → 5 → None

Visual Process:

Original:
dummy → 1 → 2 → 3 → 4 → 5 → None
             ↑       ↑
           left    right

Step 1: Find prev_left (node before position left)
dummy → 1 → 2 → 3 → 4 → 5 → None
        ↑
    prev_left

Step 2: Reverse from position 2 to 4
Before:  1 → 2 → 3 → 4 → 5
After:   1 → 4 → 3 → 2 → 5

How reversal works:
- Move 3 to front: 1 → 3 → 2 → 4 → 5
- Move 4 to front: 1 → 4 → 3 → 2 → 5

Final: 1 → 4 → 3 → 2 → 5 → None
```

### Example 3: Reverse in K-Groups

```
Input: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8, k=3
Output: 3 → 2 → 1 → 6 → 5 → 4 → 7 → 8

Process:

Group 1: [1, 2, 3]
Reverse: 3 → 2 → 1

Group 2: [4, 5, 6]
Reverse: 6 → 5 → 4

Group 3: [7, 8]
Only 2 nodes, less than k=3, leave as is

Connect: 3 → 2 → 1 → 6 → 5 → 4 → 7 → 8 → None
```

### Example 4: Palindrome Check

```
Input: 1 → 2 → 3 → 2 → 1 → None

Step 1: Find middle
slow/fast pointers → middle = 3

Step 2: Reverse second half
Original: 1 → 2 → 3 → 2 → 1 → None
Split:    1 → 2 → 3   2 → 1 → None
Reverse second: 1 → 2 → 3   1 → 2 → None (in reverse)

Step 3: Compare
first:  1 → 2 → 3
second: 1 → 2 → (None)
        ✓   ✓    

All match → Palindrome!
```

## Edge Cases

### 1. Empty List
**Scenario:** head = None
**Challenge:** No nodes to reverse
**Solution:**
```python
if not head:
    return None
```

### 2. Single Node
**Scenario:** head = 1 → None
**Challenge:** Already "reversed"
**Solution:**
```python
if not head.next:
    return head
```

### 3. Two Nodes
**Scenario:** head = 1 → 2 → None
**Challenge:** Simple swap
**Solution:**
```python
# Algorithm handles naturally
# Result: 2 → 1 → None
```

### 4. Reverse from Position 1
**Scenario:** reverseBetween(head, 1, 3)
**Challenge:** New head of list changes
**Solution:**
```python
# Use dummy node to handle uniformly
dummy = ListNode(0)
dummy.next = head
```

### 5. Reverse Entire List via Position
**Scenario:** reverseBetween(head, 1, n)
**Challenge:** Same as full reversal
**Solution:**
```python
# Algorithm works same as basic reversal
```

### 6. k Larger Than List Length
**Scenario:** reverseKGroup with k > n
**Challenge:** Can't form even one group
**Solution:**
```python
# Check if k nodes exist before reversing
count = 0
check = head
while count < k and check:
    check = check.next
    count += 1
if count < k:
    return head  # Not enough nodes
```

### 7. k = 1
**Scenario:** reverseKGroup(head, 1)
**Challenge:** No actual reversal needed
**Solution:**
```python
# Each group of 1 stays same
# Return head as is
```

## Common Pitfalls

### ❌ Pitfall 1: Losing Reference to Next Node
**What happens:** Lose rest of list when reversing link
**Why it's wrong:**
```python
# Wrong - lose reference!
current.next = prev  # Lost curr.next!
current = current.next  # This is prev now, not next!
```
**Correct approach:**
```python
# Save next BEFORE changing current.next
next_node = current.next
current.next = prev
current = next_node  # Use saved reference
```

### ❌ Pitfall 2: Not Updating Head
**What happens:** Return wrong head after reversal
**Why it's wrong:**
```python
# Wrong
return head  # This is old head (now tail)!
```
**Correct approach:**
```python
# Correct
return prev  # prev is new head after loop
```

### ❌ Pitfall 3: Off-by-One in Position-Based Reversal
**What happens:** Reverse wrong section
**Why it's wrong:**
```python
# Wrong iteration count
for _ in range(right - left):  # Missing +1
```
**Correct approach:**
```python
for _ in range(right - left + 1):  # Include both endpoints
```

### ❌ Pitfall 4: Forgetting Dummy Node
**What happens:** Can't handle reversing from head
**Why it's wrong:**
```python
# When left=1, prev_left doesn't exist!
```
**Correct approach:**
```python
dummy = ListNode(0)
dummy.next = head
# Now prev_left can be dummy when left=1
```

### ❌ Pitfall 5: Not Reconnecting After Partial Reversal
**What happens:** Lose part of list
**Why it's wrong:**
```python
# Reversed middle section but didn't connect ends
```
**Correct approach:**
```python
# After reversing middle:
prev_sublist.next = reversed_head
tail_of_reversed.next = remaining_list
```

## Variations and Extensions

### Variation 1: Reverse in Place with Stack
**Description:** Use stack for reversal (trades space for simplicity)
**Space:** O(n)
**When:** Quick prototyping, not production

### Variation 2: Reverse Using Recursion
**Description:** Recursive reversal
**Space:** O(n) call stack
**When:** Cleaner code, not worried about stack

### Variation 3: Reverse Doubly Linked List
**Description:** Also swap prev pointers
**Implementation:**
```python
def reverse_doubly(head):
    current = head
    while current:
        current.next, current.prev = current.prev, current.next
        if not current.prev:  # Was next, now prev
            return current
        current = current.prev
```

### Variation 4: Reverse in Even/Odd Positions
**Description:** Reverse only even or odd positioned nodes
**Approach:** Extract, reverse, merge back

## Practice Problems

### Beginner
1. **Reverse Linked List (LeetCode #206)** - Basic reversal
2. **Palindrome Linked List (LeetCode #234)** - Use reversal
3. **Middle of Linked List (LeetCode #876)** - Finding middle

### Intermediate
1. **Reverse Linked List II (LeetCode #92)** - Partial reversal
2. **Swap Nodes in Pairs (LeetCode #24)** - Group of 2
3. **Odd Even Linked List (LeetCode #328)** - Reorder pattern
4. **Reorder List (LeetCode #143)** - Complex reordering
5. **Rotate List (LeetCode #61)** - Rotation using reversal

### Advanced
1. **Reverse Nodes in k-Group (LeetCode #25)** - Group reversal
2. **Reverse Alternate K Nodes** - Variation
3. **Split Linked List in Parts (LeetCode #725)** - Multiple reversals

## Real-World Applications

### Industry Use Cases

1. **Undo/Redo Functionality:** Reversing operation history
2. **Browser History:** Back button navigation (reverse traversal)
3. **Music Playlist:** Reverse playback order
4. **Text Editors:** Line reversal operations
5. **Network Packet Reordering:** Reverse packet sequences

### Popular Implementations

- **Git:** Commit history traversal (forward/backward)
- **Database Undo Logs:** Reverse transaction chains
- **File Systems:** Directory traversal (up and down)
- **Compilers:** AST traversal in different orders

### Practical Scenarios

- **Task scheduler:** Reverse priority queue
- **Assembly line:** Reverse product flow
- **Document versions:** Navigate version history
- **Cache implementation:** LRU cache list manipulation

## Related Topics

### Prerequisites to Review
- **Linked List Basics** - Node structure, traversal
- **Pointers** - Reference manipulation
- **Fast & Slow Pointers** - Finding middle

### Next Steps
- **Linked List Cycle Detection** - Fast & slow pointers
- **Merge Sorted Lists** - Combining lists
- **Linked List Sorting** - Merge sort on linked list

### Similar Concepts
- **Array Reversal** - Similar concept, different implementation
- **String Reversal** - Character array reversal
- **Stack-based Reversal** - Alternative approach

### Further Reading
- [LeetCode Linked List Study Guide](https://leetcode.com/tag/linked-list/)
- [Linked List Visualization Tool](https://visualgo.net/en/list)
- Introduction to Algorithms (CLRS) - Linked Lists chapter
- [GeeksforGeeks Linked List Reversal](https://www.geeksforgeeks.org/reverse-a-linked-list/)
