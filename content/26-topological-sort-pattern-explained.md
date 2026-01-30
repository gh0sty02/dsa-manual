# Topological Sort (Graph) Pattern

**Difficulty:** Intermediate to Advanced
**Prerequisites:** Graph theory basics, DFS/BFS, Queue/Stack data structures
**Estimated Reading Time:** 45 minutes

## Introduction

Topological Sort is an algorithm for ordering vertices in a Directed Acyclic Graph (DAG) such that for every directed edge from vertex A to vertex B, A comes before B in the ordering. It's essentially a linear arrangement of nodes that respects all dependency relationships.

**Why it matters:** Topological sorting is crucial for dependency resolution in real-world systems. It's used in build systems (like Make, Maven), package managers (npm, pip), task scheduling, course prerequisite planning, and compilation order determination. Understanding topological sort is essential for solving ordering and dependency problems efficiently, which appear frequently in system design interviews at companies like Google, Amazon, and Microsoft.

**Real-world analogy:** Imagine you're getting dressed in the morning. You can't put on your shoes before your socks, or your jacket before your shirt. There's a dependency order—some items must come before others. Topological sort is like figuring out a valid order to get dressed. You might put on underwear → pants → socks → shirt → shoes → jacket, or underwear → shirt → pants → socks → shoes → jacket. Both are valid orderings that respect all dependencies (you never put shoes on before socks). Topological sort finds such valid orderings for any dependency graph.

## Core Concepts

### Key Principles

1. **Directed Acyclic Graph (DAG):** Topological sort only works on DAGs. If the graph has cycles, no valid ordering exists (you can't put A before B before C before A).

2. **Multiple Valid Orderings:** Unlike sorting numbers, topological sort may have multiple valid solutions. Any ordering that respects all edges is correct.

3. **Dependency Resolution:** Each edge A→B represents "A must come before B" or "B depends on A".

4. **Two Main Approaches:**
   - **Kahn's Algorithm (BFS-based):** Uses in-degree counting and queue
   - **DFS-based:** Uses post-order traversal and stack

### Essential Terms

- **Directed Acyclic Graph (DAG):** A directed graph with no cycles
- **In-degree:** Number of incoming edges to a vertex (number of prerequisites)
- **Out-degree:** Number of outgoing edges from a vertex (number of dependents)
- **Source Vertex:** Vertex with in-degree 0 (no prerequisites)
- **Topological Order:** Linear ordering of vertices respecting all edges
- **Cycle Detection:** Determining if valid topological order exists

### Visual Overview

```
Task Dependencies:
A → B → D
↓   ↓
C → E

Adjacency representation:
A: [B, C]
B: [D, E]
C: [E]
D: []
E: []

In-degrees:
A: 0 (source)
B: 1
C: 1
D: 1
E: 2

Valid topological orders:
1. A → B → C → D → E
2. A → B → C → E → D
3. A → C → B → D → E
4. A → C → B → E → D

All respect the rule: if there's an edge X → Y, X comes before Y
```

## How It Works

### Kahn's Algorithm (BFS-based)

**Step 1: Calculate In-degrees**
- Count incoming edges for each vertex
- Vertices with in-degree 0 have no prerequisites

**Step 2: Initialize Queue**
- Add all vertices with in-degree 0 to queue
- These can be processed first (no dependencies)

**Step 3: Process Queue**
- While queue is not empty:
  - Dequeue a vertex, add to result
  - For each neighbor, reduce its in-degree by 1
  - If neighbor's in-degree becomes 0, enqueue it

**Step 4: Check for Cycles**
- If result has all vertices, topological sort succeeded
- If result is incomplete, graph has a cycle

### DFS-based Approach

**Step 1: Mark All Vertices Unvisited**
- Track three states: unvisited, visiting, visited

**Step 2: DFS from Each Vertex**
- For each unvisited vertex, start DFS
- Mark vertex as "visiting" when entering
- Recursively visit all neighbors
- Mark vertex as "visited" when leaving (post-order)
- Add to result stack when leaving

**Step 3: Detect Cycles**
- If we visit a "visiting" vertex, we found a cycle

**Step 4: Return Result**
- Reverse the stack (or use it backwards)

### Detailed Walkthrough Example

**Problem:** Course Schedule with prerequisites
**Input:** numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
**Meaning:** 
- Course 1 requires Course 0
- Course 2 requires Course 0
- Course 3 requires Course 1
- Course 3 requires Course 2

**Graph:**
```
0 → 1 → 3
↓     ↗
2 ────
```

**Kahn's Algorithm Trace:**

```
Step 1: Build graph and calculate in-degrees
graph = {
    0: [1, 2],
    1: [3],
    2: [3],
    3: []
}
in_degree = {
    0: 0,  ← source (no prerequisites)
    1: 1,
    2: 1,
    3: 2
}

Step 2: Initialize queue with sources
queue = [0]
result = []

Step 3: Process queue

Iteration 1:
  Dequeue: 0
  result = [0]
  Reduce in-degrees of neighbors [1, 2]:
    in_degree[1] = 1 - 1 = 0 → enqueue
    in_degree[2] = 1 - 1 = 0 → enqueue
  queue = [1, 2]

Iteration 2:
  Dequeue: 1
  result = [0, 1]
  Reduce in-degree of neighbor [3]:
    in_degree[3] = 2 - 1 = 1 (not 0, don't enqueue)
  queue = [2]

Iteration 3:
  Dequeue: 2
  result = [0, 1, 2]
  Reduce in-degree of neighbor [3]:
    in_degree[3] = 1 - 1 = 0 → enqueue
  queue = [3]

Iteration 4:
  Dequeue: 3
  result = [0, 1, 2, 3]
  No neighbors
  queue = []

Step 4: Check result
len(result) = 4 = numCourses ✓
Valid topological order: [0, 1, 2, 3]
This means: take course 0, then 1, then 2, then 3
```

## Implementation

### Python Implementation - Kahn's Algorithm

```python
from typing import List, Dict, Set
from collections import deque, defaultdict

def topological_sort_kahn(num_vertices: int, edges: List[List[int]]) -> List[int]:
    """
    Perform topological sort using Kahn's algorithm (BFS-based).
    
    Args:
        num_vertices: Number of vertices in graph
        edges: List of directed edges [from, to]
        
    Returns:
        Topological ordering of vertices, or empty list if cycle exists
        
    Time Complexity: O(V + E) where V is vertices, E is edges
    Space Complexity: O(V + E) for graph and in-degree storage
    
    Example:
        >>> topological_sort_kahn(4, [[0,1], [0,2], [1,3], [2,3]])
        [0, 1, 2, 3]  # or [0, 2, 1, 3]
    """
    # Build adjacency list
    graph = defaultdict(list)
    in_degree = [0] * num_vertices
    
    for src, dst in edges:
        graph[src].append(dst)
        in_degree[dst] += 1
    
    # Initialize queue with all sources (in-degree 0)
    queue = deque()
    for vertex in range(num_vertices):
        if in_degree[vertex] == 0:
            queue.append(vertex)
    
    result = []
    
    # Process vertices
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        
        # Reduce in-degree of neighbors
        for neighbor in graph[vertex]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check if all vertices processed (no cycle)
    if len(result) != num_vertices:
        return []  # Cycle detected
    
    return result


def topological_sort_dfs(num_vertices: int, edges: List[List[int]]) -> List[int]:
    """
    Perform topological sort using DFS.
    
    Args:
        num_vertices: Number of vertices in graph
        edges: List of directed edges [from, to]
        
    Returns:
        Topological ordering of vertices, or empty list if cycle exists
        
    Time Complexity: O(V + E)
    Space Complexity: O(V + E)
    
    Example:
        >>> topological_sort_dfs(4, [[0,1], [0,2], [1,3], [2,3]])
        [0, 2, 1, 3]  # or other valid ordering
    """
    # Build adjacency list
    graph = defaultdict(list)
    for src, dst in edges:
        graph[src].append(dst)
    
    # States: 0 = unvisited, 1 = visiting, 2 = visited
    state = [0] * num_vertices
    result = []
    has_cycle = [False]
    
    def dfs(vertex: int):
        """DFS helper function."""
        if has_cycle[0]:
            return
        
        if state[vertex] == 1:
            # Found back edge (cycle)
            has_cycle[0] = True
            return
        
        if state[vertex] == 2:
            # Already processed
            return
        
        # Mark as visiting
        state[vertex] = 1
        
        # Visit all neighbors
        for neighbor in graph[vertex]:
            dfs(neighbor)
        
        # Mark as visited (post-order)
        state[vertex] = 2
        result.append(vertex)
    
    # Start DFS from all unvisited vertices
    for vertex in range(num_vertices):
        if state[vertex] == 0:
            dfs(vertex)
    
    if has_cycle[0]:
        return []  # Cycle detected
    
    # Reverse result (post-order gives reverse topological order)
    return result[::-1]


def can_finish_courses(num_courses: int, prerequisites: List[List[int]]) -> bool:
    """
    Determine if all courses can be finished given prerequisites.
    
    LeetCode #207: Course Schedule
    
    Args:
        num_courses: Total number of courses
        prerequisites: [course, prerequisite] pairs
        
    Returns:
        True if all courses can be finished, False if impossible
        
    Time Complexity: O(V + E)
    Space Complexity: O(V + E)
    
    Example:
        >>> can_finish_courses(2, [[1,0]])
        True  # Take course 0, then course 1
        >>> can_finish_courses(2, [[1,0], [0,1]])
        False  # Circular dependency
    """
    # Build graph
    graph = defaultdict(list)
    in_degree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # Kahn's algorithm
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    courses_taken = 0
    
    while queue:
        course = queue.popleft()
        courses_taken += 1
        
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)
    
    return courses_taken == num_courses


def find_order(num_courses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    Return the ordering of courses to take given prerequisites.
    
    LeetCode #210: Course Schedule II
    
    Args:
        num_courses: Total number of courses
        prerequisites: [course, prerequisite] pairs
        
    Returns:
        Valid course ordering, or empty list if impossible
        
    Time Complexity: O(V + E)
    Space Complexity: O(V + E)
    
    Example:
        >>> find_order(4, [[1,0],[2,0],[3,1],[3,2]])
        [0, 1, 2, 3]  # or [0, 2, 1, 3]
    """
    graph = defaultdict(list)
    in_degree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    order = []
    
    while queue:
        course = queue.popleft()
        order.append(course)
        
        for next_course in graph[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                queue.append(next_course)
    
    return order if len(order) == num_courses else []


def find_all_orders(num_tasks: int, prerequisites: List[List[int]]) -> List[List[int]]:
    """
    Find all possible valid topological orderings.
    
    Uses backtracking to generate all valid orders.
    
    Args:
        num_tasks: Number of tasks
        prerequisites: [task, prerequisite] pairs
        
    Returns:
        List of all valid orderings
        
    Time Complexity: O(V! * E) - exponential
    Space Complexity: O(V + E)
    
    Example:
        >>> find_all_orders(3, [[0,1]])
        [[1, 0, 2], [1, 2, 0], [2, 1, 0]]
    """
    graph = defaultdict(list)
    in_degree = [0] * num_tasks
    
    for task, prereq in prerequisites:
        graph[prereq].append(task)
        in_degree[task] += 1
    
    all_orders = []
    
    def backtrack(current_order: List[int], remaining_in_degree: List[int]):
        """Generate all orderings using backtracking."""
        if len(current_order) == num_tasks:
            all_orders.append(current_order[:])
            return
        
        # Try all tasks with in-degree 0
        for task in range(num_tasks):
            if remaining_in_degree[task] == 0 and task not in current_order:
                # Choose task
                current_order.append(task)
                
                # Update in-degrees
                temp_in_degree = remaining_in_degree[:]
                for neighbor in graph[task]:
                    temp_in_degree[neighbor] -= 1
                
                # Recurse
                backtrack(current_order, temp_in_degree)
                
                # Backtrack
                current_order.pop()
    
    backtrack([], in_degree[:])
    return all_orders


def alien_dictionary(words: List[str]) -> str:
    """
    Derive alien alphabet order from sorted alien dictionary.
    
    LeetCode #269: Alien Dictionary
    
    Args:
        words: Words in sorted order in alien language
        
    Returns:
        Alien alphabet order, or empty string if invalid
        
    Time Complexity: O(C) where C is total characters in all words
    Space Complexity: O(1) since alphabet size is fixed (26 max)
    
    Example:
        >>> alien_dictionary(["wrt","wrf","er","ett","rftt"])
        "wertf"
    """
    # Build graph of character dependencies
    graph = defaultdict(set)
    in_degree = defaultdict(int)
    
    # Initialize all characters
    for word in words:
        for char in word:
            if char not in in_degree:
                in_degree[char] = 0
    
    # Compare adjacent words to find character order
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        min_len = min(len(word1), len(word2))
        
        # Check if word2 is prefix of word1 (invalid)
        if len(word1) > len(word2) and word1[:min_len] == word2[:min_len]:
            return ""
        
        # Find first different character
        for j in range(min_len):
            if word1[j] != word2[j]:
                # word1[j] comes before word2[j]
                if word2[j] not in graph[word1[j]]:
                    graph[word1[j]].add(word2[j])
                    in_degree[word2[j]] += 1
                break
    
    # Topological sort
    queue = deque([char for char in in_degree if in_degree[char] == 0])
    result = []
    
    while queue:
        char = queue.popleft()
        result.append(char)
        
        for neighbor in graph[char]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check if all characters processed
    if len(result) != len(in_degree):
        return ""  # Cycle detected
    
    return ''.join(result)


def min_height_trees(n: int, edges: List[List[int]]) -> List[int]:
    """
    Find roots that minimize tree height.
    
    LeetCode #310: Minimum Height Trees
    
    Approach: Repeatedly remove leaf nodes (like topological sort).
    The last remaining nodes are the centroids.
    
    Args:
        n: Number of nodes
        edges: Undirected edges
        
    Returns:
        List of roots that give minimum height
        
    Time Complexity: O(V + E)
    Space Complexity: O(V + E)
    
    Example:
        >>> min_height_trees(4, [[1,0],[1,2],[1,3]])
        [1]  # Node 1 as root gives minimum height
    """
    if n == 1:
        return [0]
    
    # Build adjacency list (undirected)
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    
    # Find all leaf nodes (degree 1)
    leaves = deque([i for i in range(n) if len(graph[i]) == 1])
    
    remaining = n
    
    # Remove leaves layer by layer
    while remaining > 2:
        leaf_count = len(leaves)
        remaining -= leaf_count
        
        for _ in range(leaf_count):
            leaf = leaves.popleft()
            
            # Remove leaf from its neighbor's adjacency list
            for neighbor in graph[leaf]:
                graph[neighbor].remove(leaf)
                
                # If neighbor becomes leaf, add to queue
                if len(graph[neighbor]) == 1:
                    leaves.append(neighbor)
    
    # Remaining nodes are the centroids
    return list(leaves)


# Example usage and testing
if __name__ == "__main__":
    print("=== Topological Sort - Kahn's Algorithm ===")
    edges = [[0, 1], [0, 2], [1, 3], [2, 3]]
    print(f"Edges: {edges}")
    print(f"Order: {topological_sort_kahn(4, edges)}")
    print()
    
    print("=== Topological Sort - DFS ===")
    print(f"Order: {topological_sort_dfs(4, edges)}")
    print()
    
    print("=== Course Schedule ===")
    print(f"Can finish [[1,0]]: {can_finish_courses(2, [[1, 0]])}")
    print(f"Can finish [[1,0],[0,1]]: {can_finish_courses(2, [[1, 0], [0, 1]])}")
    print()
    
    print("=== Course Schedule II ===")
    prereqs = [[1, 0], [2, 0], [3, 1], [3, 2]]
    print(f"Prerequisites: {prereqs}")
    print(f"Order: {find_order(4, prereqs)}")
    print()
    
    print("=== All Topological Orders ===")
    print(f"All orders for 3 tasks, [[0,1]]: {find_all_orders(3, [[0, 1]])}")
    print()
    
    print("=== Alien Dictionary ===")
    words = ["wrt", "wrf", "er", "ett", "rftt"]
    print(f"Words: {words}")
    print(f"Alphabet: {alien_dictionary(words)}")
    print()
    
    print("=== Minimum Height Trees ===")
    tree_edges = [[1, 0], [1, 2], [1, 3]]
    print(f"Edges: {tree_edges}")
    print(f"Roots: {min_height_trees(4, tree_edges)}")
```

### Code Explanation

**Kahn's Algorithm:**
- Maintains in-degree counts for all vertices
- Processes vertices with in-degree 0 first
- Decrements in-degrees as vertices are processed
- Detects cycles if not all vertices processed

**DFS Approach:**
- Uses three states: unvisited, visiting, visited
- Back edge to "visiting" vertex indicates cycle
- Post-order traversal gives reverse topological order
- More memory-efficient for sparse graphs

**Course Schedule:**
- Directly applies topological sort
- Returns boolean for whether sort succeeds

**Find All Orders:**
- Uses backtracking to explore all valid orderings
- Exponential time complexity
- Useful when you need all possible orderings

**Alien Dictionary:**
- Builds graph by comparing adjacent words
- First differing character creates edge
- Validates that longer word isn't prefix of shorter

**Minimum Height Trees:**
- Unique application: removes leaves iteratively
- Last remaining nodes are graph centroids
- Similar to topological sort but for undirected graph

## Complexity Analysis

### Time Complexity

**Kahn's Algorithm:**
- **Time:** O(V + E)
- **Why?** Visit each vertex once, examine each edge once

**DFS-based:**
- **Time:** O(V + E)
- **Why?** Each vertex visited once, each edge examined once

**Course Schedule:**
- **Time:** O(V + E)
- **Why?** Same as topological sort

**Find All Orders:**
- **Time:** O(V! * E)
- **Why?** Up to V! orderings, each takes O(E) to validate

**Alien Dictionary:**
- **Time:** O(C) where C is total characters
- **Why?** Compare each pair of words, then topological sort

**Minimum Height Trees:**
- **Time:** O(V)
- **Why?** Each node removed once

### Space Complexity

**All Algorithms:**
- **Space:** O(V + E)
- **Why?** Store graph (adjacency list) and auxiliary structures

**Additional Space:**
- Kahn's: O(V) for queue and in-degree array
- DFS: O(V) for recursion stack and state array
- Find All Orders: O(V! * V) to store all orderings

### Comparison with Alternatives

| Approach | Time | Space | Detects Cycles | All Orders | Best For |
|----------|------|-------|----------------|------------|----------|
| **Kahn's (BFS)** | O(V+E) | O(V+E) | Yes | No | General purpose, iterative |
| **DFS-based** | O(V+E) | O(V+E) | Yes | No | Recursive preference |
| **Backtracking** | O(V!*E) | O(V!*V) | Yes | Yes | Finding all orderings |
| **Simple DFS** | O(V+E) | O(V) | No | No | When no cycle guaranteed |

## Examples

### Example 1: Basic Course Schedule

**Input:** numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
**Output:** [0, 1, 2, 3] or [0, 2, 1, 3]

**Trace:**

```
Build graph:
0 → 1 → 3
↓     ↗
2 ────

In-degrees: {0:0, 1:1, 2:1, 3:2}

Queue: [0]
Result: []

Step 1: Process 0
  Result: [0]
  Update: in[1]=0, in[2]=0
  Queue: [1, 2]

Step 2: Process 1
  Result: [0, 1]
  Update: in[3]=1
  Queue: [2]

Step 3: Process 2
  Result: [0, 1, 2]
  Update: in[3]=0
  Queue: [3]

Step 4: Process 3
  Result: [0, 1, 2, 3] ✓
  Queue: []

Valid order found!
```

### Example 2: Detecting Cycle

**Input:** numCourses = 2, prerequisites = [[1,0],[0,1]]
**Output:** [] (impossible)

**Trace:**

```
Build graph:
0 ⇄ 1 (cycle!)

In-degrees: {0:1, 1:1}

Queue: [] (no source vertices!)

Since queue is empty but we haven't processed all vertices,
this means there's a cycle.

Result: [] (cannot complete courses)
```

### Example 3: Alien Dictionary

**Input:** words = ["wrt","wrf","er","ett","rftt"]
**Output:** "wertf"

**Detailed Trace:**

```
Compare adjacent words to build graph:

"wrt" vs "wrf":
  First diff at index 2: 't' != 'f'
  Edge: t → f

"wrf" vs "er":
  First diff at index 0: 'w' != 'e'
  Edge: w → e

"er" vs "ett":
  First diff at index 1: 'r' != 't'
  Edge: r → t

"ett" vs "rftt":
  First diff at index 0: 'e' != 'r'
  Edge: e → r

Graph:
w → e → r → t → f

In-degrees:
w: 0
e: 1
r: 1
t: 1
f: 1

Topological Sort:
Queue: [w]
Process w → e has in-degree 0
Queue: [e]
Process e → r has in-degree 0
Queue: [r]
Process r → t has in-degree 0
Queue: [t]
Process t → f has in-degree 0
Queue: [f]
Process f

Result: "wertf" ✓
```

### Example 4: All Possible Orders

**Input:** numTasks = 3, prerequisites = [[0,1]]
**Output:** [[1,0,2], [1,2,0], [2,1,0]]

**Backtracking Trace:**

```
Graph:
1 → 0
2 (independent)

In-degrees: {0:1, 1:0, 2:0}

Available: 1, 2 (both have in-degree 0)

Path 1: Choose 1
  Order: [1]
  Update in-degrees: {0:0, 2:0}
  Available: 0, 2
  
  Path 1a: Choose 0
    Order: [1, 0]
    Available: 2
    Choose 2
    Order: [1, 0, 2] ✓
  
  Path 1b: Choose 2
    Order: [1, 2]
    Available: 0
    Choose 0
    Order: [1, 2, 0] ✓

Path 2: Choose 2
  Order: [2]
  Update in-degrees: {0:1, 1:0}
  Available: 1
  
  Choose 1
  Order: [2, 1]
  Update: {0:0}
  Choose 0
  Order: [2, 1, 0] ✓

All orders: [[1,0,2], [1,2,0], [2,1,0]]
```

### Example 5: Minimum Height Trees

**Input:** n = 4, edges = [[1,0],[1,2],[1,3]]
**Output:** [1]

**Trace:**

```
Graph (undirected):
    0
    |
1 - 1 - 2
    |
    3

Degrees: {0:1, 1:3, 2:1, 3:1}
Leaves: [0, 2, 3]
Remaining: 4

Round 1: Remove leaves [0, 2, 3]
  Remove 0 from 1's neighbors
  Remove 2 from 1's neighbors
  Remove 3 from 1's neighbors
  Now 1 has degree 0
  Leaves: [1]
  Remaining: 1

Remaining ≤ 2, stop
Result: [1]

Node 1 is the centroid - choosing it as root gives minimum height
```

## Edge Cases

### 1. Empty Graph
**Scenario:** No vertices or no edges
**Challenge:** Trivial case
**Solution:** Return empty list or all vertices in any order
**Code example:**
```python
if num_vertices == 0:
    return []
if not edges:
    return list(range(num_vertices))
```

### 2. Single Vertex
**Scenario:** Graph with one vertex, no edges
**Challenge:** Simplest non-empty case
**Solution:** Return [0]
**Code example:**
```python
if num_vertices == 1:
    return [0]
```

### 3. Disconnected Components
**Scenario:** Multiple independent subgraphs
**Challenge:** Need to process all components
**Solution:** DFS/BFS visits all components automatically
**Code example:**
```python
# DFS handles this naturally:
for vertex in range(num_vertices):
    if state[vertex] == 0:
        dfs(vertex)  # Start new component
```

### 4. Complete DAG
**Scenario:** Every vertex connected to every other
**Challenge:** Many valid orderings exist
**Solution:** Any valid ordering works
**Code example:**
```python
# For vertices [0,1,2] with all edges:
# 0→1, 0→2, 1→2
# Valid orders: [0,1,2] only
```

### 5. Linear Chain
**Scenario:** Single path: 0→1→2→3
**Challenge:** Only one valid ordering
**Solution:** Return that single path
**Code example:**
```python
# Only valid order: [0,1,2,3]
```

### 6. Self-Loop
**Scenario:** Edge from vertex to itself
**Challenge:** Forms a cycle (invalid DAG)
**Solution:** Detect and return empty/error
**Code example:**
```python
# If edge [0,0] exists:
# This is a cycle, return []
```

### 7. Large Graph
**Scenario:** Millions of vertices/edges
**Challenge:** Memory and time constraints
**Solution:** Use efficient data structures, possibly streaming
**Code example:**
```python
# Use adjacency list (not matrix)
# Process in batches if needed
```

## Common Pitfalls

### ❌ Pitfall 1: Using Adjacency Matrix for Sparse Graphs
**What happens:** Wastes O(V²) space
**Why it's wrong:** Most graphs are sparse (E << V²)
**Correct approach:**
```python
# WRONG for sparse graphs:
adj_matrix = [[0] * n for _ in range(n)]  # O(V²) space

# CORRECT:
adj_list = defaultdict(list)  # O(V + E) space
```

### ❌ Pitfall 2: Not Checking for Cycles
**What happens:** Returns incomplete/invalid ordering
**Why it's wrong:** Topological sort only works on DAGs
**Correct approach:**
```python
# WRONG:
while queue:
    process_vertex()
return result  # Might be incomplete!

# CORRECT:
while queue:
    process_vertex()
if len(result) != num_vertices:
    return []  # Cycle detected
return result
```

### ❌ Pitfall 3: Modifying In-Degree Array During Iteration
**What happens:** Incorrect in-degree counts
**Why it's wrong:** Need original values for multiple uses
**Correct approach:**
```python
# WRONG:
for v in range(n):
    if in_degree[v] == 0:
        queue.append(v)
        in_degree[v] = -1  # Modifying during iteration!

# CORRECT:
queue = deque([v for v in range(n) if in_degree[v] == 0])
# Don't modify in_degree during initialization
```

### ❌ Pitfall 4: Reversing Edge Direction
**What happens:** Completely wrong ordering
**Why it's wrong:** Edge A→B means A before B, not B before A
**Correct approach:**
```python
# WRONG:
for course, prereq in prerequisites:
    graph[course].append(prereq)  # Reversed!

# CORRECT:
for course, prereq in prerequisites:
    graph[prereq].append(course)  # Prereq → Course
```

### ❌ Pitfall 5: Not Handling Multiple Sources
**What happens:** Miss some valid orderings
**Why it's wrong:** Can start from any source vertex
**Correct approach:**
```python
# WRONG:
queue = deque([0])  # Assumes vertex 0 is only source

# CORRECT:
queue = deque([v for v in range(n) if in_degree[v] == 0])
```

### ❌ Pitfall 6: Forgetting to Reset Visited State
**What happens:** Incorrect results when running multiple times
**Why it's wrong:** State persists across calls
**Correct approach:**
```python
# WRONG (reusing global state):
visited = [False] * n  # Global
def topological_sort():
    # Uses global visited
    
# CORRECT:
def topological_sort():
    visited = [False] * n  # Local
    # ... use visited
```

### ❌ Pitfall 7: Incorrect Cycle Detection in DFS
**What happens:** False positives or missing cycles
**Why it's wrong:** Need three states, not two
**Correct approach:**
```python
# WRONG (only two states):
visited = [False] * n
def dfs(v):
    visited[v] = True
    for neighbor in graph[v]:
        if visited[neighbor]:
            return True  # False positive!

# CORRECT (three states):
# 0 = unvisited, 1 = visiting, 2 = visited
state = [0] * n
def dfs(v):
    if state[v] == 1:
        return True  # Cycle!
    if state[v] == 2:
        return False  # Already done
    state[v] = 1
    for neighbor in graph[v]:
        if dfs(neighbor):
            return True
    state[v] = 2
    return False
```

## Variations and Extensions

### Variation 1: Longest Path in DAG
**Description:** Find longest path in directed acyclic graph
**When to use:** Critical path method, project scheduling
**Key differences:** Use topological sort + DP
**Implementation:**
```python
def longest_path_dag(n: int, edges: List[List[int]]) -> int:
    """
    Find longest path in DAG.
    
    Time: O(V + E)
    Space: O(V + E)
    """
    graph = defaultdict(list)
    in_degree = [0] * n
    
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    
    # Topological sort
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    dist = [0] * n
    
    while queue:
        u = queue.popleft()
        
        for v in graph[u]:
            dist[v] = max(dist[v], dist[u] + 1)
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
    
    return max(dist)
```

### Variation 2: Parallel Course Scheduling
**Description:** Minimum semesters needed to complete courses
**When to use:** When tasks can be done in parallel
**Key differences:** Track levels in BFS
**Implementation:**
```python
def min_semesters(n: int, relations: List[List[int]]) -> int:
    """
    Find minimum semesters to complete courses.
    
    Time: O(V + E)
    Space: O(V + E)
    """
    graph = defaultdict(list)
    in_degree = [0] * (n + 1)
    
    for prev, next in relations:
        graph[prev].append(next)
        in_degree[next] += 1
    
    queue = deque([i for i in range(1, n + 1) if in_degree[i] == 0])
    semesters = 0
    courses_taken = 0
    
    while queue:
        semesters += 1
        for _ in range(len(queue)):
            course = queue.popleft()
            courses_taken += 1
            
            for next_course in graph[course]:
                in_degree[next_course] -= 1
                if in_degree[next_course] == 0:
                    queue.append(next_course)
    
    return semesters if courses_taken == n else -1
```

### Variation 3: Sequence Reconstruction
**Description:** Verify if unique sequence can be reconstructed
**When to use:** LeetCode #444
**Key differences:** Check if topological order is unique
**Implementation:**
```python
def sequence_reconstruction(nums: List[int], sequences: List[List[int]]) -> bool:
    """
    Check if nums is unique topological sort of sequences.
    
    Time: O(V + E)
    Space: O(V + E)
    """
    graph = defaultdict(list)
    in_degree = {num: 0 for num in nums}
    
    for seq in sequences:
        for i in range(len(seq) - 1):
            if seq[i] not in in_degree or seq[i + 1] not in in_degree:
                return False
            graph[seq[i]].append(seq[i + 1])
            in_degree[seq[i + 1]] += 1
    
    queue = deque([num for num in nums if in_degree[num] == 0])
    result = []
    
    while queue:
        if len(queue) > 1:
            return False  # Not unique!
        
        num = queue.popleft()
        result.append(num)
        
        for next_num in graph[num]:
            in_degree[next_num] -= 1
            if in_degree[next_num] == 0:
                queue.append(next_num)
    
    return result == nums
```

### Variation 4: Build Order with Groups
**Description:** Dependencies between groups of tasks
**When to use:** Complex scheduling with batching
**Key differences:** Vertices represent groups
**Implementation:**
```python
def build_order_groups(groups: List[List[int]], 
                       dependencies: List[List[int]]) -> List[List[int]]:
    """
    Order groups considering internal and external dependencies.
    """
    # Build graph between groups
    # Topological sort on groups
    # Return ordered groups
```

## Practice Problems

### Beginner
1. **Course Schedule** - Can finish all courses?
   - LeetCode #207

2. **Course Schedule II** - Return valid course order
   - LeetCode #210

3. **Find Eventual Safe Nodes** - Nodes not in cycles
   - LeetCode #802

### Intermediate
1. **Alien Dictionary** - Derive alphabet order
   - LeetCode #269 (Premium)

2. **Minimum Height Trees** - Find tree centroids
   - LeetCode #310

3. **Sort Items by Groups Respecting Dependencies** - Complex topological sort
   - LeetCode #1203

4. **Sequence Reconstruction** - Verify unique topological order
   - LeetCode #444 (Premium)

5. **Parallel Courses** - Minimum semesters needed
   - LeetCode #1136 (Premium)

6. **Build Matrix** - Construct matrix with row/column conditions
   - LeetCode #2392

### Advanced
1. **Parallel Courses II** - Minimum semesters with k-course limit
   - LeetCode #1494

2. **Strange Printer II** - Determine if printing order exists
   - LeetCode #1591

3. **Minimum Number of Semesters** - Optimize course load
   - LeetCode #1494

4. **Build Array Where You Can Find Maximum Exactly K Comparisons** - Complex dependency modeling
   - LeetCode #1420

## Real-World Applications

### Industry Use Cases

1. **Build Systems:** Make, Maven, Gradle use topological sort to determine compilation order based on file dependencies.

2. **Package Managers:** npm, pip, apt resolve installation order for packages with dependencies.

3. **Task Scheduling:** Project management tools (Jira, Asana) order tasks based on dependencies.

4. **Course Planning:** University systems determine valid course sequences respecting prerequisites.

5. **Data Processing Pipelines:** ETL tools order transformations based on data dependencies.

### Popular Implementations

- **Make/CMake:** Build automation uses topological sort
- **Maven/Gradle:** Java build tools for dependency resolution
- **npm/yarn:** JavaScript package managers
- **Git:** Commit graph ordering
- **Apache Airflow:** Workflow scheduling

### Practical Scenarios

- **CI/CD Pipelines:** Order build stages based on dependencies
- **Compiler Design:** Determine symbol resolution order
- **Spreadsheet Calculation:** Excel formula evaluation order
- **Database Query Optimization:** Join order determination
- **Operating System:** Deadlock detection and prevention

## Related Topics

### Prerequisites to Review
- **Graph Representation** - Adjacency list vs matrix
- **DFS/BFS** - Foundation for topological sort
- **Queue/Stack** - Data structures used
- **Cycle Detection** - Essential for validation

### Next Steps
- **Shortest Path Algorithms** - Dijkstra, Bellman-Ford
- **Strongly Connected Components** - Kosaraju's, Tarjan's algorithms
- **Minimum Spanning Trees** - Kruskal's, Prim's
- **Network Flow** - Max-flow algorithms

### Similar Concepts
- **Dependency Resolution** - Similar ordering problems
- **Critical Path Method (CPM)** - Project scheduling
- **Partial Order** - Mathematical foundation
- **Directed Acyclic Graph (DAG)** - Graph theory concept

### Further Reading
- "Introduction to Algorithms" (CLRS) - Chapter on Graph Algorithms
- "Algorithm Design Manual" by Skiena - Topological Sorting section
- [Topological Sorting - GeeksforGeeks](https://www.geeksforgeeks.org/topological-sorting/)
- [Kahn's Algorithm](https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm)
- "Graph Algorithms" by Shimon Even
- "Network Flows" by Ahuja, Magnanti, and Orlin
