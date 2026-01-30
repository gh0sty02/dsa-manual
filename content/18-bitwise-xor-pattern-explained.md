# Bitwise XOR Pattern

**Difficulty:** Intermediate
**Prerequisites:** Binary numbers, Bit manipulation, Arrays, Basic mathematics
**Estimated Reading Time:** 45 minutes

## Introduction

The Bitwise XOR Pattern leverages the unique mathematical properties of the XOR (exclusive OR) operation to solve problems that would otherwise require additional space or complex logic. XOR has special properties that make it perfect for finding missing numbers, detecting duplicates, swapping values without temporary variables, and many other elegant solutions.

**Why it matters:** XOR operations execute in constant time at the hardware level, making them incredibly fast. Problems that might require O(n) space with hash sets can often be solved in O(1) space using XOR properties. Understanding XOR unlocks elegant solutions to seemingly complex problems: finding the single non-duplicate number among millions, swapping variables without extra memory, and even error detection in data transmission. These techniques are used in cryptography, data compression, error correction, and low-level system programming.

**Real-world analogy:** Imagine a light switch connected to two buttons in different rooms. Each button toggles the light: press once to turn it on (or off), press again to toggle back. If you press both buttons, the light returns to its original state—they cancel each other out. This is exactly how XOR works: combining something with itself cancels out (A XOR A = 0), and combining with zero leaves it unchanged (A XOR 0 = A). This "cancellation" property is the foundation of the XOR pattern!

## Core Concepts

### Key Principles

1. **Self-Cancellation:** Any number XORed with itself equals zero
   - `A ⊕ A = 0`
   - This is the most important property for the XOR pattern

2. **Identity Element:** Any number XORed with zero equals itself
   - `A ⊕ 0 = A`
   - Zero acts as the identity for XOR

3. **Commutative Property:** Order doesn't matter
   - `A ⊕ B = B ⊕ A`
   - Can rearrange XOR operations freely

4. **Associative Property:** Grouping doesn't matter
   - `(A ⊕ B) ⊕ C = A ⊕ (B ⊕ C)`
   - Can compute XOR in any order

5. **Complement Detection:** `A ⊕ B = ~A` when B is all 1s
   - XOR with all 1s flips all bits (bitwise NOT)

### Essential Terms

- **XOR (Exclusive OR):** Binary operation that returns 1 when bits are different, 0 when same
- **Bit:** Single binary digit (0 or 1)
- **Bitwise Operation:** Operation performed on individual bits
- **Complement:** Flipped version of binary number (0→1, 1→0)
- **Mask:** Binary pattern used to set, clear, or toggle specific bits
- **Parity:** Count of 1-bits (even or odd)
- **LSB (Least Significant Bit):** Rightmost bit
- **MSB (Most Significant Bit):** Leftmost bit

### Visual Overview

```
XOR Truth Table:
A  B  | A⊕B
0  0  |  0   (Same → 0)
0  1  |  1   (Different → 1)
1  0  |  1   (Different → 1)
1  1  |  0   (Same → 0)

XOR Properties Visualization:

1. Self-Cancellation: A ⊕ A = 0
   5 ⊕ 5 = 0
   101 ⊕ 101 = 000

2. Identity: A ⊕ 0 = A
   5 ⊕ 0 = 5
   101 ⊕ 000 = 101

3. Finding Single Number in Duplicates:
   Array: [2, 3, 4, 2, 3]
   
   2 ⊕ 3 ⊕ 4 ⊕ 2 ⊕ 3
   = (2 ⊕ 2) ⊕ (3 ⊕ 3) ⊕ 4   (commutative/associative)
   = 0 ⊕ 0 ⊕ 4                 (self-cancellation)
   = 4                          (identity)

4. Swapping Without Temp Variable:
   a = 5 (101), b = 3 (011)
   
   a = a ⊕ b  →  a = 101 ⊕ 011 = 110  (a=6)
   b = a ⊕ b  →  b = 110 ⊕ 011 = 101  (b=5) 
   a = a ⊕ b  →  a = 110 ⊕ 101 = 011  (a=3)
   
   Result: a=3, b=5 (swapped!)

5. Bit Flipping with Mask:
   Number: 10110 (22)
   Mask:   11111 (all 1s)
   XOR:    01001 (9) - all bits flipped

Complement of Base 10:
Number: 5 = 101
Find bit length: 3 bits
All 1s mask: 111 = 7
Complement: 5 ⊕ 7 = 101 ⊕ 111 = 010 = 2
```

## How It Works

### Single Number - Step by Step

**Problem:** Every element appears twice except one. Find the single one.

**Key Insight:** XOR all numbers. Duplicates cancel out, leaving only the single number.

**Algorithm:**

1. Initialize `result = 0`
2. XOR result with each number
3. Return result

**Detailed Walkthrough:**

```
Array: [4, 1, 2, 1, 2]

Step 1: result = 0
        result = 0 ⊕ 4 = 4
        Binary: 000 ⊕ 100 = 100

Step 2: result = 4
        result = 4 ⊕ 1 = 5
        Binary: 100 ⊕ 001 = 101

Step 3: result = 5
        result = 5 ⊕ 2 = 7
        Binary: 101 ⊕ 010 = 111

Step 4: result = 7
        result = 7 ⊕ 1 = 6
        Binary: 111 ⊕ 001 = 110

Step 5: result = 6
        result = 6 ⊕ 2 = 4
        Binary: 110 ⊕ 010 = 100

Final result: 4

Why this works:
4 ⊕ 1 ⊕ 2 ⊕ 1 ⊕ 2
= 4 ⊕ (1 ⊕ 1) ⊕ (2 ⊕ 2)   (rearrange using commutative/associative)
= 4 ⊕ 0 ⊕ 0                 (duplicates cancel: A ⊕ A = 0)
= 4                          (identity: A ⊕ 0 = A)
```

### Two Single Numbers - Step by Step

**Problem:** Every element appears twice except two. Find both unique numbers.

**Key Insight:** XOR all numbers gives `num1 ⊕ num2`. Use a differing bit to separate them.

**Algorithm:**

1. XOR all numbers to get `num1 ⊕ num2`
2. Find any bit where num1 and num2 differ (rightmost set bit)
3. Partition numbers into two groups based on this bit
4. XOR each group separately to find num1 and num2

**Detailed Walkthrough:**

```
Array: [1, 2, 3, 1, 2, 5]
The two single numbers are 3 and 5

Step 1: XOR all numbers
        xor_result = 1 ⊕ 2 ⊕ 3 ⊕ 1 ⊕ 2 ⊕ 5
                   = (1 ⊕ 1) ⊕ (2 ⊕ 2) ⊕ 3 ⊕ 5
                   = 0 ⊕ 0 ⊕ 3 ⊕ 5
                   = 3 ⊕ 5
                   = 011 ⊕ 101
                   = 110 (6 in decimal)

Step 2: Find rightmost set bit in xor_result
        110 in binary
        Rightmost set bit is at position 1 (from right, 0-indexed)
        Mask = 010 (only this bit set)

Step 3: Partition numbers based on this bit
        Numbers with bit 1 set: [2, 3, 2] → bit pattern: x1x
        Numbers with bit 1 clear: [1, 1, 5] → bit pattern: x0x

Step 4: XOR each group
        Group 1: 2 ⊕ 3 ⊕ 2 = (2 ⊕ 2) ⊕ 3 = 0 ⊕ 3 = 3 ✓
        Group 2: 1 ⊕ 1 ⊕ 5 = (1 ⊕ 1) ⊕ 5 = 0 ⊕ 5 = 5 ✓

Result: [3, 5]

Why this works:
- 3 = 011, 5 = 101
- They differ at bit position 1
- 3 has bit 1 set, 5 has bit 1 clear
- Partitioning by this bit separates them into different groups
- All duplicates are in the same group (same bit pattern)
- XORing each group cancels duplicates, leaving only the single number
```

### Complement of Base 10 Number - Step by Step

**Problem:** Find complement of a number (flip all bits).

**Key Insight:** XOR with a mask of all 1s (of same bit length) flips all bits.

**Algorithm:**

1. Find bit length of number
2. Create mask with all 1s of that length: `mask = (1 << bit_length) - 1`
3. XOR number with mask

**Detailed Walkthrough:**

```
Number: 5

Step 1: Find bit length
        5 in binary: 101
        Bit length: 3
        
        How to find bit length:
        bit_length = 0
        temp = 5
        while temp > 0:
            bit_length += 1
            temp >>= 1
        
        Iterations:
        temp=5 (101), bit_length=1, temp>>=1 → temp=2 (10)
        temp=2 (10), bit_length=2, temp>>=1 → temp=1 (1)
        temp=1 (1), bit_length=3, temp>>=1 → temp=0
        Result: bit_length = 3

Step 2: Create all-1s mask
        mask = (1 << 3) - 1
        1 << 3 = 1000 (8 in decimal)
        8 - 1 = 0111 (7 in decimal)
        mask = 111 in binary

Step 3: XOR with mask
        5 ⊕ 7 = 101 ⊕ 111
        
        Bit-by-bit:
        1 ⊕ 1 = 0
        0 ⊕ 1 = 1
        1 ⊕ 1 = 0
        
        Result: 010 = 2

Verification:
Original: 101 (5)
Complement: 010 (2)
All bits flipped ✓

Another Example:
Number: 10 = 1010 (4 bits)
Mask: (1 << 4) - 1 = 10000 - 1 = 1111 = 15
Complement: 10 ⊕ 15 = 1010 ⊕ 1111 = 0101 = 5
```

## Implementation

### Single Number

```python
from typing import List

def single_number(nums: List[int]) -> int:
    """
    Find the single number that appears once while others appear twice.
    
    Uses XOR property: a ⊕ a = 0 and a ⊕ 0 = a
    All duplicates cancel out, leaving only the single number.
    
    Args:
        nums: Array where every element appears twice except one
        
    Returns:
        The single number that appears only once
        
    Time Complexity: O(n) - single pass through array
    Space Complexity: O(1) - only one variable
    """
    result = 0
    for num in nums:
        result ^= num  # XOR with each number
    return result


# Example usage
nums = [4, 1, 2, 1, 2]
print(single_number(nums))  # Output: 4

# Why it works:
# 4 ^ 1 ^ 2 ^ 1 ^ 2 = 4 ^ (1 ^ 1) ^ (2 ^ 2) = 4 ^ 0 ^ 0 = 4
```

### Two Single Numbers

```python
def single_number_III(nums: List[int]) -> List[int]:
    """
    Find two numbers that appear once while others appear twice.
    
    Strategy:
    1. XOR all numbers to get num1 ⊕ num2
    2. Find a bit where num1 and num2 differ
    3. Partition array by this bit
    4. XOR each partition to find the two numbers
    
    Args:
        nums: Array where every element appears twice except two
        
    Returns:
        List of two numbers that appear only once
        
    Time Complexity: O(n) - two passes through array
    Space Complexity: O(1) - only a few variables
    """
    # Step 1: XOR all numbers
    xor_all = 0
    for num in nums:
        xor_all ^= num
    # xor_all = num1 ⊕ num2
    
    # Step 2: Find rightmost set bit (where num1 and num2 differ)
    # Method: xor_all & -xor_all isolates rightmost set bit
    rightmost_bit = xor_all & -xor_all
    
    # Alternative method to find rightmost set bit:
    # rightmost_bit = 1
    # while (xor_all & rightmost_bit) == 0:
    #     rightmost_bit <<= 1
    
    # Step 3 & 4: Partition and XOR
    num1, num2 = 0, 0
    for num in nums:
        if num & rightmost_bit:
            # Bit is set - group 1
            num1 ^= num
        else:
            # Bit is not set - group 2
            num2 ^= num
    
    return [num1, num2]


# Example usage
nums = [1, 2, 1, 3, 2, 5]
print(single_number_III(nums))  # Output: [3, 5] or [5, 3]
```

### Complement of Base 10 Number

```python
def find_complement(num: int) -> int:
    """
    Find the complement by flipping all bits.
    
    Strategy: XOR with a mask of all 1s (same bit length as num).
    
    Args:
        num: Positive integer
        
    Returns:
        Bitwise complement of the number
        
    Time Complexity: O(log n) - counting bits
    Space Complexity: O(1)
    """
    if num == 0:
        return 1  # Special case: complement of 0 is 1
    
    # Method 1: Count bits and create mask
    bit_count = 0
    temp = num
    while temp:
        bit_count += 1
        temp >>= 1
    
    # Create mask: all 1s with bit_count bits
    # (1 << bit_count) gives 10...0 (bit_count zeros)
    # Subtracting 1 gives 01...1 (bit_count ones)
    mask = (1 << bit_count) - 1
    
    # XOR with mask flips all bits
    return num ^ mask


# Alternative implementation using bit_length()
def find_complement_v2(num: int) -> int:
    """
    More Pythonic version using bit_length().
    
    Time Complexity: O(1) in Python (bit_length is O(1))
    Space Complexity: O(1)
    """
    if num == 0:
        return 1
    
    # Python's bit_length() returns number of bits
    bit_length = num.bit_length()
    mask = (1 << bit_length) - 1
    
    return num ^ mask


# Example usage
print(find_complement(5))   # Output: 2 (101 → 010)
print(find_complement(1))   # Output: 0 (1 → 0)
print(find_complement(7))   # Output: 0 (111 → 000)
print(find_complement(10))  # Output: 5 (1010 → 0101)
```

### Flip and Invert Image

```python
def flip_and_invert_image(image: List[List[int]]) -> List[List[int]]:
    """
    Flip image horizontally, then invert (0→1, 1→0).
    
    For binary matrix (only 0s and 1s), inverting is XOR with 1.
    
    Args:
        image: 2D binary matrix
        
    Returns:
        Flipped and inverted image
        
    Time Complexity: O(n²) where n is image dimension
    Space Complexity: O(1) if modifying in-place
    """
    n = len(image)
    
    for row in image:
        # Two pointers to flip horizontally
        left, right = 0, n - 1
        
        while left <= right:
            # Flip and invert in one step
            # After flip: row[left] ↔ row[right]
            # After invert: 0→1, 1→0 (XOR with 1)
            
            if left == right:
                # Middle element: just invert
                row[left] ^= 1
            else:
                # Swap and invert both
                row[left], row[right] = row[right] ^ 1, row[left] ^ 1
            
            left += 1
            right -= 1
    
    return image


# Cleaner implementation
def flip_and_invert_image_v2(image: List[List[int]]) -> List[List[int]]:
    """
    More elegant version.
    
    Key insight: if values are same, XOR^1 gives same value
                 if values differ, XOR^1 gives swapped values
    """
    n = len(image)
    
    for row in image:
        for i in range((n + 1) // 2):  # Process first half (including middle)
            # Swap positions i and n-1-i, and invert both
            row[i], row[n - 1 - i] = row[n - 1 - i] ^ 1, row[i] ^ 1
    
    return image


# Example usage
image = [[1,1,0],
         [1,0,1],
         [0,0,0]]

result = flip_and_invert_image(image)
# Output: [[1,0,0],
#          [0,1,0],
#          [1,1,1]]

# Explanation:
# Row 1: [1,1,0] → flip → [0,1,1] → invert → [1,0,0]
# Row 2: [1,0,1] → flip → [1,0,1] → invert → [0,1,0]
# Row 3: [0,0,0] → flip → [0,0,0] → invert → [1,1,1]
```

### Additional XOR Operations

```python
def swap_without_temp(a: int, b: int) -> tuple:
    """
    Swap two numbers without temporary variable using XOR.
    
    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    print(f"Before: a={a}, b={b}")
    
    a = a ^ b  # a now holds XOR of original a and b
    b = a ^ b  # b = (a^b) ^ b = a (original a)
    a = a ^ b  # a = (a^b) ^ a = b (original b)
    
    print(f"After: a={a}, b={b}")
    return a, b


def is_power_of_two(n: int) -> bool:
    """
    Check if number is power of 2 using bit manipulation.
    
    Power of 2 has only one bit set: 1000, 0100, 0010, etc.
    n-1 flips all bits after the set bit: 0111, 0011, 0001
    n & (n-1) removes rightmost set bit, giving 0 for power of 2.
    
    Time Complexity: O(1)
    """
    return n > 0 and (n & (n - 1)) == 0


def count_set_bits(n: int) -> int:
    """
    Count number of 1 bits (population count).
    
    Uses Brian Kernighan's algorithm: n & (n-1) removes rightmost set bit.
    
    Time Complexity: O(k) where k is number of set bits
    """
    count = 0
    while n:
        n &= n - 1  # Remove rightmost set bit
        count += 1
    return count


def get_bit(num: int, i: int) -> int:
    """Get i-th bit (0-indexed from right)."""
    return (num >> i) & 1


def set_bit(num: int, i: int) -> int:
    """Set i-th bit to 1."""
    return num | (1 << i)


def clear_bit(num: int, i: int) -> int:
    """Clear i-th bit (set to 0)."""
    return num & ~(1 << i)


def toggle_bit(num: int, i: int) -> int:
    """Toggle i-th bit (0→1, 1→0)."""
    return num ^ (1 << i)


# Example usage
print(swap_without_temp(5, 3))      # (3, 5)
print(is_power_of_two(16))          # True
print(is_power_of_two(18))          # False
print(count_set_bits(7))            # 3 (111 has three 1s)
print(f"Bit 2 of 5: {get_bit(5, 2)}")  # 1 (101, bit 2 is 1)
```

### Code Explanation

**Key Techniques:**

1. **Finding Rightmost Set Bit:** `n & -n` or `n & (~n + 1)`
   - `-n` is two's complement: flip all bits and add 1
   - Only the rightmost set bit survives the AND operation

2. **Creating All-1s Mask:** `(1 << bit_length) - 1`
   - Shift 1 left by bit_length: gives 10...0
   - Subtract 1: gives 01...1 (all 1s)

3. **XOR for Cancellation:**
   - Duplicates cancel: `a ^ a = 0`
   - Identity preserved: `a ^ 0 = a`
   - Chain all numbers: duplicates disappear

4. **Bit Manipulation Fundamentals:**
   - `num | (1 << i)`: Set bit i
   - `num & ~(1 << i)`: Clear bit i
   - `num ^ (1 << i)`: Toggle bit i
   - `(num >> i) & 1`: Get bit i

## Complexity Analysis

### Time Complexity

**All XOR Pattern Problems: O(n)** where n is array length

**Why O(n)?**
- Single pass through array
- Each XOR operation is O(1) - constant time at hardware level
- No nested loops or recursive calls
- Total: O(n) × O(1) = O(n)

**Specific Analysis:**
- **Single Number:** O(n) - one pass, XOR each element
- **Two Single Numbers:** O(n) - two passes (first to get XOR, second to partition)
- **Complement:** O(log k) where k is the number value (counting bits)
- **Flip and Invert Image:** O(n²) where n is image dimension

### Space Complexity

**All XOR Pattern Problems: O(1)**

**Why O(1)?**
- Only use a fixed number of variables (result, mask, etc.)
- No additional data structures (arrays, hash maps)
- No recursion (no call stack)
- Space doesn't grow with input size

**Advantage over Alternatives:**
- Hash set approach: O(n) space
- Sorting approach: O(n) space (or O(log n) with in-place sort)
- XOR approach: O(1) space

### Comparison with Alternatives

| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| XOR | O(n) | O(1) | Optimal for finding unique numbers |
| Hash Set | O(n) | O(n) | More intuitive but uses extra space |
| Sorting | O(n log n) | O(1) or O(log n) | Slower, modifies input |
| Counting | O(n) | O(1) | Only works for specific cases |

## Examples

### Example 1: Find Single Number

**Problem:** Array where every element appears twice except one.

**Input:** [2, 2, 1, 3, 3]

**Solution:**

```
XOR all numbers:
2 ^ 2 ^ 1 ^ 3 ^ 3

Rearrange (commutative/associative):
(2 ^ 2) ^ (3 ^ 3) ^ 1

Simplify (self-cancellation):
0 ^ 0 ^ 1

Result (identity):
1

Answer: 1
```

### Example 2: Find Two Single Numbers

**Problem:** Array where every element appears twice except two.

**Input:** [1, 2, 1, 3, 2, 5]

**Solution:**

```
Step 1: XOR all numbers
1 ^ 2 ^ 1 ^ 3 ^ 2 ^ 5 = 3 ^ 5 = 6
Binary: 011 ^ 101 = 110

Step 2: Find rightmost set bit of 6
110 in binary
Rightmost set bit at position 1
Mask = 010 (2 in decimal)

Step 3: Partition by bit 1
Numbers with bit 1 set:   [2, 3, 2] (x1x pattern)
Numbers with bit 1 clear: [1, 1, 5] (x0x pattern)

Step 4: XOR each group
Group 1: 2 ^ 3 ^ 2 = 3
Group 2: 1 ^ 1 ^ 5 = 5

Answer: [3, 5]
```

### Example 3: Number Complement

**Problem:** Find bitwise complement of 5.

**Input:** 5

**Solution:**

```
5 in binary: 101 (3 bits)

Create mask with 3 ones:
mask = (1 << 3) - 1
     = 1000 - 1
     = 111
     = 7

XOR with mask:
5 ^ 7 = 101 ^ 111 = 010 = 2

Answer: 2

Verification:
101 (5)
010 (2) - all bits flipped ✓
```

### Example 4: Flip and Invert Image

**Problem:** Flip matrix horizontally then invert all values.

**Input:** [[1,1,0],[1,0,1],[0,0,0]]

**Solution:**

```
Process each row:

Row 1: [1,1,0]
  Flip: [0,1,1]
  Invert (XOR with 1): [1,0,0]

Row 2: [1,0,1]
  Flip: [1,0,1]
  Invert: [0,1,0]

Row 3: [0,0,0]
  Flip: [0,0,0]
  Invert: [1,1,1]

Result: [[1,0,0],[0,1,0],[1,1,1]]

XOR operation for invert:
0 ^ 1 = 1
1 ^ 1 = 0
```

## Edge Cases

### 1. Single Element Array

**Scenario:** Array with only one element.

**Challenge:** Is this the answer or empty case?

**Solution:**

```python
def single_number(nums):
    # Works correctly: 0 ^ nums[0] = nums[0]
    result = 0
    for num in nums:
        result ^= num
    return result

# [5] → Returns 5 ✓
```

### 2. Empty Array

**Scenario:** Empty input array.

**Challenge:** What to return?

**Solution:**

```python
def single_number(nums):
    if not nums:
        return 0  # Or raise exception
    
    result = 0
    for num in nums:
        result ^= num
    return result
```

### 3. Number is Zero

**Scenario:** Finding complement of 0.

**Challenge:** Binary representation has no bits.

**Solution:**

```python
def find_complement(num):
    if num == 0:
        return 1  # Special case: complement of 0 is 1
    
    # Regular logic for positive numbers...
```

### 4. Negative Numbers

**Scenario:** Array contains negative numbers.

**Challenge:** XOR still works, but interpretation differs.

**Solution:**

```python
# XOR works with negative numbers in Python
# Python uses arbitrary precision integers
nums = [-1, -1, 2, 2, 3]
result = 0
for num in nums:
    result ^= num
# Returns 3 correctly

# -1 in two's complement (32-bit): 11111111111111111111111111111111
# XOR still cancels duplicates
```

### 5. Very Large Numbers

**Scenario:** Numbers exceed standard integer range.

**Challenge:** Some languages have overflow issues.

**Solution:**

```python
# Python handles arbitrary precision automatically
# No overflow concerns

# In Java/C++, might need long or BigInteger
# XOR still works but be aware of bit width
```

### 6. Two Numbers with All Same Bits Set

**Scenario:** In two single numbers problem, both numbers have same bits.

**Challenge:** No differing bit to partition on.

**Solution:**

```python
# This can't happen!
# If num1 == num2, they're not two different singles
# Problem guarantees two DIFFERENT single numbers

# xor_all = num1 ^ num2
# If num1 == num2, xor_all = 0
# But we'd have duplicates, contradicting problem
```

### 7. All Elements are Zero

**Scenario:** Array of all zeros except one non-zero.

**Challenge:** Does XOR work?

**Solution:**

```python
# nums = [0, 0, 0, 5, 0]
# 0 ^ 0 ^ 0 ^ 5 ^ 0 = 5
# Works perfectly! 0 is XOR identity ✓
```

## Common Pitfalls

### ❌ Pitfall 1: Forgetting XOR is Commutative

**What happens:** Confusion about order of operations.

```python
# All these give same result:
a ^ b ^ c
c ^ a ^ b
b ^ c ^ a

# Order doesn't matter with XOR
```

**Key insight:** Use this property to rearrange and group duplicates mentally.

### ❌ Pitfall 2: Using XOR for Non-Pairs

**What happens:** Wrong answer if elements appear odd number of times (not 2).

```python
# WRONG if elements appear 3 times
# nums = [1, 1, 1, 2, 2, 2, 3]
# 1^1^1 = 1 (not 0!)

# XOR cancellation requires pairs (even occurrences)
```

**Correct approach:** XOR only works when elements appear in pairs (or even counts).

### ❌ Pitfall 3: Incorrect Mask Creation

**What happens:** Wrong complement calculated.

```python
# WRONG - Using fixed mask
def find_complement_wrong(num):
    return num ^ 0xFFFFFFFF  # Wrong! Assumes 32 bits

# 5 ^ 0xFFFFFFFF = very large number, not 2

# CORRECT - Mask matches number's bit length
def find_complement_correct(num):
    bit_length = num.bit_length()
    mask = (1 << bit_length) - 1
    return num ^ mask
```

### ❌ Pitfall 4: Not Handling Rightmost Bit Correctly

**What happens:** Wrong partitioning in two singles problem.

```python
# WRONG - Incorrect rightmost bit extraction
rightmost_bit = xor_all & (xor_all - 1)  # This CLEARS rightmost bit!

# CORRECT - Isolate rightmost bit
rightmost_bit = xor_all & -xor_all       # Keeps only rightmost bit
# Or:
rightmost_bit = xor_all & (~xor_all + 1)
```

### ❌ Pitfall 5: Confusing XOR with OR

**What happens:** Completely wrong results.

```python
# XOR (^): Different bits → 1
# OR (|):  Any 1 → 1

# They're DIFFERENT!
5 ^ 3 = 101 ^ 011 = 110 = 6
5 | 3 = 101 | 011 = 111 = 7

# Don't use | when you mean ^
```

## Variations and Extensions

### Variation 1: Single Number II (Appears 3 Times)

**Description:** Every element appears 3 times except one.

**When to use:** When duplicates appear in triplets.

**Implementation:**

```python
def single_number_II(nums: List[int]) -> int:
    """
    Find single number when others appear 3 times.
    
    Strategy: Count bits. If bit appears 3k+1 times, it belongs to answer.
    """
    result = 0
    
    # Check each bit position (32 bits for integers)
    for i in range(32):
        bit_sum = 0
        for num in nums:
            # Count how many numbers have this bit set
            if num & (1 << i):
                bit_sum += 1
        
        # If count is not multiple of 3, bit is in answer
        if bit_sum % 3:
            result |= (1 << i)
    
    # Handle negative numbers (Python specific)
    if result >= 2**31:
        result -= 2**32
    
    return result
```

### Variation 2: Missing Number

**Description:** Find missing number in array [0, 1, 2, ..., n] with one missing.

**When to use:** Finding missing element in sequence.

**Implementation:**

```python
def missing_number(nums: List[int]) -> int:
    """
    Find missing number using XOR.
    
    Strategy: XOR all numbers from 0 to n with all array elements.
    All present numbers cancel out, leaving missing number.
    """
    result = len(nums)  # Start with n
    
    for i, num in enumerate(nums):
        result ^= i ^ num
    
    return result

# Alternative: XOR approach
def missing_number_v2(nums: List[int]) -> int:
    """More explicit version."""
    xor_complete = 0
    xor_array = 0
    
    # XOR all numbers from 0 to n
    for i in range(len(nums) + 1):
        xor_complete ^= i
    
    # XOR all array elements
    for num in nums:
        xor_array ^= num
    
    # Missing number is the difference
    return xor_complete ^ xor_array
```

### Variation 3: Find Duplicate Number

**Description:** Array contains n+1 numbers from 1 to n, one duplicate.

**When to use:** Finding duplicate in constrained range.

**Implementation:**

```python
def find_duplicate(nums: List[int]) -> int:
    """
    Find duplicate using XOR (works if all others appear once).
    
    Strategy: XOR [1..n] with array elements.
    Single elements cancel, duplicate remains.
    """
    xor_result = 0
    
    # XOR all array elements
    for num in nums:
        xor_result ^= num
    
    # XOR numbers from 1 to n
    for i in range(1, len(nums)):
        xor_result ^= i
    
    return xor_result

# Note: This only works if duplicate appears exactly twice
# For multiple duplicates, use cycle detection (Floyd's algorithm)
```

### Variation 4: Hamming Distance

**Description:** Count positions where bits differ between two numbers.

**When to use:** Measuring bit difference, error detection.

**Implementation:**

```python
def hamming_distance(x: int, y: int) -> int:
    """
    Count number of positions where bits differ.
    
    Strategy: XOR gives 1 where bits differ, count set bits.
    """
    xor_result = x ^ y
    
    # Count set bits
    count = 0
    while xor_result:
        count += 1
        xor_result &= xor_result - 1  # Remove rightmost set bit
    
    return count

# Example: 1 ^ 4 = 001 ^ 100 = 101 → 2 bits differ
```

### Variation 5: Total Hamming Distance

**Description:** Sum of hamming distances between all pairs in array.

**When to use:** Pairwise bit difference analysis.

**Implementation:**

```python
def total_hamming_distance(nums: List[int]) -> int:
    """
    Sum of hamming distances for all pairs.
    
    Strategy: For each bit position, count 0s and 1s.
    Contribution = count_0s × count_1s
    """
    total = 0
    
    for i in range(32):  # 32 bits
        count_ones = 0
        for num in nums:
            if num & (1 << i):
                count_ones += 1
        
        count_zeros = len(nums) - count_ones
        # Each 1 pairs with each 0, contributing 1 to distance
        total += count_ones * count_zeros
    
    return total
```

## Practice Problems

### Beginner

1. **Single Number** - Find single among duplicates
   - LeetCode #136

2. **Number Complement** - Flip all bits
   - LeetCode #476

3. **Hamming Distance** - Count differing bits
   - LeetCode #461

4. **Missing Number** - Find missing in sequence
   - LeetCode #268

### Intermediate

1. **Single Number III** - Find two singles
   - LeetCode #260

2. **Flip and Invert Image** - XOR-based transformation
   - LeetCode #832

3. **Single Number II** - Single among triplets
   - LeetCode #137

4. **Total Hamming Distance** - Pairwise bit differences
   - LeetCode #477

5. **Find the Duplicate Number** - One duplicate in range
   - LeetCode #287

### Advanced

1. **Maximum XOR of Two Numbers** - Maximum XOR in array
   - LeetCode #421

2. **Count Triplets XOR** - XOR-based counting
   - LeetCode #1442

3. **Bitwise ORs of Subarrays** - Distinct bitwise ORs
   - LeetCode #898

4. **Maximum XOR With Element** - XOR with constraints
   - LeetCode #1707

## Real-World Applications

### Industry Use Cases

1. **Error Detection:** Parity bits in RAM and data transmission use XOR to detect single-bit errors. RAID storage systems use XOR for data recovery when drives fail.

2. **Cryptography:** XOR is fundamental to stream ciphers and one-time pads. The Vernam cipher (provably secure when used correctly) relies entirely on XOR operations.

3. **Data Compression:** Differencing algorithms use XOR to store only changes between versions, saving massive amounts of storage in version control systems and backup software.

4. **Graphics Programming:** Fast pixel manipulation, alpha blending, and color inversion use XOR. Game developers use XOR for quick sprite swapping and animation.

5. **Networking:** Checksums and CRC error detection use XOR operations to verify data integrity during transmission across networks.

### Popular Implementations

- **RAID Systems:** XOR-based parity for disk redundancy
  - Used in enterprise storage, data centers

- **Git Version Control:** Diff algorithms use XOR for efficiency
  - Powers GitHub, GitLab, all git repositories

- **Network Protocols:** TCP/IP checksum uses XOR
  - Foundation of internet communication

- **Encryption Libraries:** OpenSSL, libsodium use XOR in ciphers
  - Secure communications worldwide

### Practical Scenarios

- **File Systems:** Filesystem journaling uses XOR for crash recovery
- **Embedded Systems:** XOR for memory-efficient state tracking in IoT devices
- **Database Systems:** XOR for fast duplicate detection in deduplication
- **Hardware Design:** XOR gates are fundamental building blocks of CPUs
- **Signal Processing:** XOR for noise cancellation and signal mixing

## Related Topics

### Prerequisites to Review

- **Binary Numbers** - Understanding binary representation
- **Bit Manipulation** - Basic bitwise operations (AND, OR, NOT)
- **Boolean Algebra** - Logical operations and properties
- **Two's Complement** - Negative number representation
- **Arrays and Lists** - Basic data structures

### Next Steps

- **Advanced Bit Manipulation** - Bit tricks, masks, and hacks
- **Gray Code** - Binary encoding using XOR
- **Bloom Filters** - Probabilistic data structures using XOR
- **Reed-Solomon Codes** - Error correction using XOR algebra
- **Cryptography Fundamentals** - XOR in encryption algorithms

### Similar Concepts

- **Parity Checking** - XOR for error detection
- **Hash Functions** - XOR in hash computation
- **Checksums** - XOR-based data verification
- **Bitboards** - Chess engines using bit manipulation

### Further Reading

- "Hacker's Delight" by Henry S. Warren - Bible of bit manipulation
- "The Art of Computer Programming" (TAOCP) - Knuth's coverage of bitwise ops
- "Bit Twiddling Hacks" - Stanford bit manipulation guide
- [XOR Properties](https://en.wikipedia.org/wiki/Exclusive_or) - Mathematical properties
- [Bitwise Operations Guide](https://graphics.stanford.edu/~seander/bithacks.html) - Comprehensive tricks
