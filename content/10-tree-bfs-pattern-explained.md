# Tree BFS (Breadth-First Search) Pattern

**Difficulty:** Medium
**Prerequisites:** Trees, Queues, Basic recursion
**Estimated Reading Time:** 25 minutes

## Introduction

Tree Breadth-First Search (BFS) is a fundamental tree traversal pattern that explores nodes level by level, from left to right. Starting from the root, BFS visits all nodes at depth 0, then all nodes at depth 1, then depth 2, and so on. This level-order traversal is implemented using a queue data structure and is essential for solving problems involving tree levels, shortest paths in trees, and layer-by-layer processing.

**Why it matters:** BFS is crucial for tree-related interview questions and appears in real-world applications like file system traversal, organizational hierarchy processing, and UI component rendering. Companies frequently test BFS because it demonstrates understanding of queue-based algorithms, tree structures, and level-wise processing. Mastering BFS opens the door to solving a wide variety of tree problems efficiently.

**Real-world analogy:** Imagine you're the CEO of a company exploring your organizational chart. With BFS, you first meet with all your direct reports (level 1), then you meet with all their direct reports (level 2), then their reports (level 3), and so on. You process the entire organization level by level, never skipping ahead to lower levels until the current level is complete. This is exactly how BFS traverses a tree!

## Core Concepts

### Key Principles

1. **Level-by-level traversal:** Process all nodes at current depth before moving to next depth

2. **Queue-based:** Uses queue (FIFO) to maintain nodes to visit

3. **Left-to-right order:** Within each level, process nodes from left to right

4. **Shortest path in trees:** BFS finds shortest path from root to any node (in terms of edges)

5. **Iterative implementation:** Typically implemented iteratively with a queue (unlike DFS which is often recursive)

### Essential Terms

- **Level:** All nodes at the same depth from root
- **Level-order traversal:** BFS traversal pattern
- **Queue:** FIFO data structure used for BFS
- **Depth/Level:** Distance from root (root is level 0)
- **Sibling nodes:** Nodes with the same parent
- **Leaf node:** Node with no children

### Visual Overview

```
Tree Structure:
        1
       / \
      2   3
     / \   \
    4   5   6

BFS Traversal Order: 1 → 2 → 3 → 4 → 5 → 6

Level by Level:
Level 0: [1]
Level 1: [2, 3]
Level 2: [4, 5, 6]

Queue Evolution:
Initial:  [1]
After 1:  [2, 3]
After 2:  [3, 4, 5]
After 3:  [4, 5, 6]
After 4:  [5, 6]
After 5:  [6]
After 6:  []

Visual Process:
        1          ← Process level 0
       / \
      2   3        ← Process level 1
     / \   \
    4   5   6      ← Process level 2

Each level processed left to right!
```

## How to Identify This Pattern

Recognizing when to use Tree BFS is crucial for efficient problem solving:

### Primary Indicators ✓

**Level-by-level processing required**
- Need to process tree level by level
- Nodes at same depth need handling together
- Keywords: "level order", "level by level", "each level"
- Example: "Return level order traversal of tree"

**Finding shortest path in tree**
- Shortest distance from root to node
- Minimum depth/height
- Keywords: "shortest", "minimum depth", "closest"
- Example: "Find minimum depth of binary tree"

**Zigzag or reverse level order**
- Alternating left-to-right and right-to-left
- Bottom-up traversal
- Keywords: "zigzag", "reverse level order", "bottom up"
- Example: "Zigzag level order traversal"

**Level-based calculations**
- Average/sum/max of each level
- Count nodes at each level
- Keywords: "average of levels", "sum of levels", "level statistics"
- Example: "Find average value at each level"

**Finding nodes at specific level**
- Rightmost/leftmost node at each level
- Nodes at depth k
- Keywords: "rightmost", "leftmost", "at depth k", "at level k"
- Example: "Right side view of tree"

**Serialization/Deserialization**
- Converting tree to/from string representation
- Level-order based encoding
- Keywords: "serialize", "deserialize", "encode", "decode"
- Example: "Serialize and deserialize binary tree"

**Connecting nodes at same level**
- Link nodes horizontally
- Next pointers between siblings
- Keywords: "connect", "next pointer", "same level"
- Example: "Populate next right pointers"

### Common Problem Phrases 🔑

Watch for these exact phrases:
- "Level order traversal"
- "Zigzag traversal"
- "Reverse level order"
- "Minimum depth"
- "Maximum depth"
- "Right side view"
- "Left side view"
- "Average of levels"
- "Largest value in each row"
- "Populating next right pointers"
- "Connect level order siblings"
- "Bottom-up level order"

### When NOT to Use Tree BFS ✗

**Need to process root-to-leaf paths**
- Path sum problems
- All paths from root to leaf
- → Use Tree DFS

**Need to visit nodes in specific order (inorder, preorder, postorder)**
- Binary search tree operations
- Expression tree evaluation
- → Use Tree DFS

**Working with parent pointers or going up the tree**
- Finding ancestors
- Lowest common ancestor
- → Use different approach

**Graph cycle detection**
- Detecting cycles in graphs
- → Use Union-Find or DFS

### Quick Decision Checklist ✅

Ask yourself:

1. **Need to process tree level by level?** → Tree BFS
2. **Finding shortest path/minimum depth?** → Tree BFS
3. **Need nodes at same level together?** → Tree BFS
4. **Right/left side view of tree?** → Tree BFS
5. **Zigzag or reverse level order?** → Tree BFS
6. **Average/sum of each level?** → Tree BFS
7. **Connecting nodes horizontally?** → Tree BFS

If YES to any of these, Tree BFS is the right choice!

### Decision Tree

```
Start
  ↓
Is it a tree problem?
  ↓ YES
Need level-by-level processing?
  ↓ YES
  → USE TREE BFS
  
  ↓ NO
Need to explore all paths or specific order?
  ↓ YES
  → USE TREE DFS
```

### Algorithm Signature

**Basic Level Order Traversal:**
```python
from collections import deque

def levelOrder(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result
```

### Example Pattern Matching 💡

**Problem: "Return level order traversal of binary tree"**

Analysis:
- ✓ Explicitly says "level order"
- ✓ Need to process level by level
- ✓ Classic BFS problem

**Verdict: USE TREE BFS** ✓

**Problem: "Find minimum depth of binary tree"**

Analysis:
- ✓ Finding shortest path from root to leaf
- ✓ BFS finds shortest path efficiently
- ✓ Stop when first leaf found

**Verdict: USE TREE BFS** ✓

**Problem: "Find all root-to-leaf paths"**

Analysis:
- ✗ Need complete paths from root to leaf
- ✗ Not level-based processing
- ✓ DFS more natural

**Verdict: USE TREE DFS** (Not BFS) ✗

**Problem: "Binary tree right side view"**

Analysis:
- ✓ Need rightmost node at each level
- ✓ Level-based processing
- ✓ BFS perfect for this

**Verdict: USE TREE BFS** ✓

### Pattern vs Problem Type 📊

| Problem Type | Tree BFS? | Alternative |
|--------------|-----------|-------------|
| Level order traversal | ✅ YES | - |
| Minimum depth | ✅ YES | DFS (but less efficient) |
| Right/left side view | ✅ YES | - |
| Average of levels | ✅ YES | - |
| Zigzag traversal | ✅ YES | - |
| Path sum | ❌ NO | Tree DFS |
| Inorder traversal | ❌ NO | Tree DFS |
| Lowest common ancestor | ❌ NO | Tree DFS |
| Maximum path sum | ❌ NO | Tree DFS |

### Keywords Cheat Sheet 📝

**STRONG "Tree BFS" Keywords:**
- level order
- level by level
- each level
- zigzag
- minimum depth
- right side view

**MODERATE Keywords:**
- breadth-first
- shortest path (in tree)
- average of levels
- connect level order
- bottom-up

**ANTI-Keywords (probably NOT Tree BFS):**
- path sum (Tree DFS)
- inorder/preorder/postorder (Tree DFS)
- ancestor (Tree DFS)
- validate BST (Tree DFS)

### Red Flags 🚩

These suggest TREE BFS might NOT be right:
- "All paths from root to leaf" → Tree DFS
- "Inorder traversal" → Tree DFS
- "Validate BST" → Tree DFS  
- "Path sum" → Tree DFS
- "Diameter of tree" → Tree DFS

### Green Flags 🟢

STRONG indicators for TREE BFS:
- "Level order traversal"
- "Zigzag"
- "Minimum depth"
- "Right side view"
- "Average of levels"
- "Each level"
- "Connect next pointers"
- "Bottom-up level order"

## How It Works

### Basic BFS Algorithm

1. **Initialize:** Create empty queue, add root
2. **While queue not empty:**
   - Get current level size
   - Process all nodes at current level
   - Add children of current level to queue
3. **Return:** Accumulated results

### Level-by-Level Processing

1. **Capture level size** before processing
2. **Process exactly that many nodes**
3. **Add children to queue** for next level
4. **Repeat** until queue empty

### Step-by-Step Example: Level Order Traversal

Tree:
```
      3
     / \
    9  20
      /  \
     15   7
```

Process:
```
Initial: queue = [3], result = []

Level 0:
  level_size = 1
  Process node 3: value=3
  Add children: 9, 20
  level = [3]
  queue = [9, 20]
  result = [[3]]

Level 1:
  level_size = 2
  Process node 9: value=9, no children
  Process node 20: value=20
  Add children: 15, 7
  level = [9, 20]
  queue = [15, 7]
  result = [[3], [9, 20]]

Level 2:
  level_size = 2
  Process node 15: value=15, no children
  Process node 7: value=7, no children
  level = [15, 7]
  queue = []
  result = [[3], [9, 20], [15, 7]]

Queue empty, done!
Result: [[3], [9, 20], [15, 7]]
```

## Implementation

### Tree Node Definition

```python
from typing import Optional, List
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

### Problem 1: Binary Tree Level Order Traversal (LeetCode #102)

```python
def levelOrder(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Return level order traversal of binary tree.
    
    Args:
        root: Root of binary tree
    
    Returns:
        List of levels, each level is list of values
    
    Time Complexity: O(n) - visit each node once
    Space Complexity: O(n) - queue can hold up to n/2 nodes
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        # Process all nodes at current level
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            # Add children for next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    return result


# Usage Example
#       3
#      / \
#     9  20
#       /  \
#      15   7
root = TreeNode(3)
root.left = TreeNode(9)
root.right = TreeNode(20, TreeNode(15), TreeNode(7))
print(levelOrder(root))  # [[3], [9, 20], [15, 7]]
```

### Problem 2: Binary Tree Zigzag Level Order (LeetCode #103)

```python
def zigzagLevelOrder(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Return zigzag level order traversal.
    Level 0: left to right
    Level 1: right to left
    Level 2: left to right, etc.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    left_to_right = True
    
    while queue:
        level_size = len(queue)
        level = deque()
        
        for _ in range(level_size):
            node = queue.popleft()
            
            # Add to level based on direction
            if left_to_right:
                level.append(node.val)
            else:
                level.appendleft(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(list(level))
        left_to_right = not left_to_right
    
    return result


# Usage Example
print(zigzagLevelOrder(root))  # [[3], [20, 9], [15, 7]]
```

### Problem 3: Minimum Depth of Binary Tree (LeetCode #111)

```python
def minDepth(root: Optional[TreeNode]) -> int:
    """
    Find minimum depth (shortest path to leaf).
    
    Args:
        root: Root of binary tree
    
    Returns:
        Minimum depth
    
    Time Complexity: O(n) worst case, but often better
    Space Complexity: O(n)
    """
    if not root:
        return 0
    
    queue = deque([(root, 1)])  # (node, depth)
    
    while queue:
        node, depth = queue.popleft()
        
        # First leaf found is at minimum depth (BFS property!)
        if not node.left and not node.right:
            return depth
        
        if node.left:
            queue.append((node.left, depth + 1))
        if node.right:
            queue.append((node.right, depth + 1))
    
    return 0


# Usage Example
print(minDepth(root))  # 2
```

### Problem 4: Maximum Depth of Binary Tree (LeetCode #104)

```python
def maxDepth(root: Optional[TreeNode]) -> int:
    """
    Find maximum depth of binary tree.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not root:
        return 0
    
    queue = deque([root])
    depth = 0
    
    while queue:
        level_size = len(queue)
        depth += 1
        
        for _ in range(level_size):
            node = queue.popleft()
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return depth


# Alternative: DFS solution (often simpler for max depth)
def maxDepthDFS(root: Optional[TreeNode]) -> int:
    if not root:
        return 0
    return 1 + max(maxDepthDFS(root.left), maxDepthDFS(root.right))
```

### Problem 5: Binary Tree Right Side View (LeetCode #199)

```python
def rightSideView(root: Optional[TreeNode]) -> List[int]:
    """
    Return values of nodes seen from right side.
    (Rightmost node at each level)
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        
        for i in range(level_size):
            node = queue.popleft()
            
            # Last node in level is rightmost
            if i == level_size - 1:
                result.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return result


# Usage Example
print(rightSideView(root))  # [3, 20, 7]
```

### Problem 6: Average of Levels (LeetCode #637)

```python
def averageOfLevels(root: Optional[TreeNode]) -> List[float]:
    """
    Calculate average value of nodes at each level.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level_sum = 0
        
        for _ in range(level_size):
            node = queue.popleft()
            level_sum += node.val
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level_sum / level_size)
    
    return result


# Usage Example
print(averageOfLevels(root))  # [3.0, 14.5, 11.0]
```

### Problem 7: Populating Next Right Pointers (LeetCode #116)

```python
class Node:
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


def connect(root: Optional[Node]) -> Optional[Node]:
    """
    Populate next right pointers in each node.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not root:
        return None
    
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        
        for i in range(level_size):
            node = queue.popleft()
            
            # Connect to next node in level (if not last)
            if i < level_size - 1:
                node.next = queue[0]  # Next in queue
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return root


# O(1) space solution using next pointers
def connectOptimal(root: Optional[Node]) -> Optional[Node]:
    """
    Connect using O(1) extra space.
    
    Time: O(n), Space: O(1)
    """
    if not root:
        return None
    
    leftmost = root
    
    while leftmost.left:  # While not at leaf level
        head = leftmost
        
        while head:
            # Connect children
            head.left.next = head.right
            
            # Connect across parent boundary
            if head.next:
                head.right.next = head.next.left
            
            head = head.next
        
        leftmost = leftmost.left
    
    return root
```

### Problem 8: Level Order Bottom (LeetCode #107)

```python
def levelOrderBottom(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Return bottom-up level order traversal.
    
    Time Complexity: O(n)
    Space Complexity: O(n)
    """
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level)
    
    # Reverse to get bottom-up
    return result[::-1]


# Alternative: Use insert(0, level) to avoid final reverse
def levelOrderBottomInsert(root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.insert(0, level)  # Insert at beginning
    
    return result
```

## Complexity Analysis

### Time Complexity

**Level Order Traversal:** O(n)
- Visit each node exactly once
- Each node processed in constant time

**Minimum Depth:** O(n) worst case
- May stop early when first leaf found
- Best case: O(h) where h is height (balanced tree)
- Worst case: O(n) (skewed tree)

**All BFS Operations:** O(n)
- Must visit all nodes to process levels

### Space Complexity

**Queue Size:** O(w) where w is maximum width
- Maximum nodes in queue = widest level
- For complete binary tree: O(n/2) = O(n)
- For skewed tree: O(1)

**Result Storage:** O(n)
- Storing all node values

**Total Space:** O(n)
- Dominated by queue or result storage

### Why BFS for Minimum Depth is Efficient

- **BFS:** Finds first leaf, can stop early - O(w) where w is width
- **DFS:** Must explore all paths - O(h) where h is height
- For balanced tree: BFS is better (stops at first complete level)
- For skewed tree: Both are similar

### Comparison with Alternatives

| Approach | Time | Space | When to Use |
|----------|------|-------|-------------|
| BFS (iterative) | O(n) | O(w) | Level-based problems |
| DFS (recursive) | O(n) | O(h) | Path-based problems |
| DFS (iterative) | O(n) | O(h) | Avoid recursion |
| Morris Traversal | O(n) | O(1) | Minimize space |

## Examples

### Example 1: Level Order Traversal

```
Tree:
      1
     / \
    2   3
   / \
  4   5

Queue evolution:
Start:     [1]
After 1:   [2, 3]
After 2:   [3, 4, 5]
After 3:   [4, 5]
After 4:   [5]
After 5:   []

Result: [[1], [2, 3], [4, 5]]
```

### Example 2: Zigzag Traversal

```
Tree:
      1
     / \
    2   3
   / \   \
  4   5   6

Level 0 (L→R): [1]
Level 1 (R→L): [3, 2]
Level 2 (L→R): [4, 5, 6]

Result: [[1], [3, 2], [4, 5, 6]]
```

### Example 3: Minimum Depth

```
Tree:
      1
     / \
    2   3
   /
  4

BFS Process:
Level 0: Process node 1 (not leaf)
Level 1: Process node 2 (not leaf), node 3 (LEAF!)
Stop! Minimum depth = 2

DFS would need to explore all paths:
Path 1-2-4: depth 3
Path 1-3: depth 2
Result: min(3, 2) = 2

BFS is more efficient here!
```

## Edge Cases

### 1. Empty Tree
**Scenario:** root = None
**Return:** [] or 0 depending on problem

### 2. Single Node
**Scenario:** root = TreeNode(1)
**Return:** [[1]] or depth = 1

### 3. Only Left Children
**Scenario:** Skewed left tree
**Queue:** Never grows beyond size 1

### 4. Only Right Children
**Scenario:** Skewed right tree
**Queue:** Never grows beyond size 1

### 5. Complete Binary Tree
**Scenario:** All levels filled
**Queue:** Can reach size n/2 at last level

### 6. All Nodes at Same Level Have Same Value
**Scenario:** [1, 1, 1, 1, 1]
**Handle:** Works normally, duplicates OK

## Common Pitfalls

### ❌ Pitfall 1: Not Capturing Level Size
**What happens:** Process nodes from next level in current level
**Why it's wrong:**
```python
# Wrong
while queue:
    node = queue.popleft()  # No level tracking!
    # Children added mix with current level
```
**Correct:**
```python
while queue:
    level_size = len(queue)  # Capture before processing
    for _ in range(level_size):
        node = queue.popleft()
```

### ❌ Pitfall 2: Forgetting to Check for None Children
**What happens:** Adding None to queue causes errors
**Why it's wrong:**
```python
# Wrong
queue.append(node.left)   # What if None?
queue.append(node.right)  # What if None?
```
**Correct:**
```python
if node.left:
    queue.append(node.left)
if node.right:
    queue.append(node.right)
```

### ❌ Pitfall 3: Using List Instead of Deque
**What happens:** O(n) popleft operation
**Why it's wrong:**
```python
# Wrong - list.pop(0) is O(n)
queue = []
node = queue.pop(0)  # Slow!
```
**Correct:**
```python
from collections import deque
queue = deque()
node = queue.popleft()  # O(1)
```

### ❌ Pitfall 4: Modifying Queue Size During Iteration
**What happens:** Incorrect level boundaries
**Why it's wrong:**
```python
# Wrong
for _ in range(len(queue)):  # Size changes during loop!
    node = queue.popleft()
    queue.append(node.left)
```
**Correct:**
```python
level_size = len(queue)  # Capture before loop
for _ in range(level_size):
    node = queue.popleft()
```

## Variations and Extensions

### Variation 1: Level Order with Depth Info
**Description:** Include depth with each node
**Implementation:**
```python
queue = deque([(root, 0)])  # (node, depth)
```

### Variation 2: Process Levels Separately
**Description:** Apply different logic to different levels
**Use case:** Alternate processing

### Variation 3: Multi-way Tree BFS
**Description:** Nodes can have multiple children
**Implementation:**
```python
for child in node.children:
    queue.append(child)
```

### Variation 4: Bidirectional BFS
**Description:** Search from both ends
**Use case:** Shortest path in graph

## Practice Problems

### Beginner
1. **Binary Tree Level Order Traversal (LeetCode #102)**
2. **Maximum Depth of Binary Tree (LeetCode #104)**
3. **Minimum Depth of Binary Tree (LeetCode #111)**
4. **Average of Levels (LeetCode #637)**

### Intermediate
1. **Binary Tree Zigzag Level Order (LeetCode #103)**
2. **Binary Tree Right Side View (LeetCode #199)**
3. **Populating Next Right Pointers (LeetCode #116)**
4. **Level Order Bottom (LeetCode #107)**
5. **Find Largest Value in Each Row (LeetCode #515)**
6. **Add One Row to Tree (LeetCode #623)**

### Advanced
1. **Serialize and Deserialize Binary Tree (LeetCode #297)**
2. **All Nodes Distance K (LeetCode #863)**
3. **Vertical Order Traversal (LeetCode #987)**
4. **Maximum Level Sum (LeetCode #1161)**

## Real-World Applications

### Industry Use Cases

1. **File System Traversal:** Directory browsing level by level
2. **Organizational Charts:** Processing hierarchy by management level
3. **Network Broadcasting:** Spreading messages level by level
4. **Web Crawling:** Crawling pages by depth
5. **Social Networks:** Finding friends at distance k

### Popular Implementations

- **DOM Tree Traversal:** Browser rendering engines
- **XML/JSON Parsing:** Level-order processing
- **Game Trees:** Board game AI (minimax)
- **Decision Trees:** ML model traversal

### Practical Scenarios

- **Company hierarchy:** Process by organizational level
- **Family tree:** Genealogy by generation
- **File browser:** Show directory contents level by level
- **Chat threads:** Display conversation hierarchy

## Related Topics

### Prerequisites
- **Trees** - Basic tree structure
- **Queues** - FIFO data structure
- **Recursion** - Understanding tree nature

### Next Steps
- **Tree DFS** - Depth-first traversal
- **Graph BFS** - BFS on graphs
- **Trie** - Prefix tree structure

### Similar Concepts
- **Graph BFS** - Same concept for graphs
- **Level Order in N-ary Tree** - Multiple children
- **Layer-by-layer processing** - General pattern

### Further Reading
- [Tree Traversals - GeeksforGeeks](https://www.geeksforgeeks.org/tree-traversals-inorder-preorder-and-postorder/)
- [BFS vs DFS](https://www.geeksforgeeks.org/difference-between-bfs-and-dfs/)
- [LeetCode Tree BFS Problems](https://leetcode.com/tag/breadth-first-search/)
