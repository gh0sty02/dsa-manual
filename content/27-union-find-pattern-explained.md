# Union Find (Disjoint Set Union) Pattern

**Difficulty:** Intermediate
**Prerequisites:** Arrays, Basic graph concepts, Recursion
**Estimated Reading Time:** 40 minutes

## Introduction

Union Find (also called Disjoint Set Union or DSU) is a data structure that efficiently tracks and merges disjoint sets. It supports two primary operations: finding which set an element belongs to, and uniting two sets into one. The structure maintains a collection of non-overlapping sets and can quickly determine if two elements are in the same set.

**Why it matters:** Union Find is essential for solving connectivity problems in graphs, including cycle detection, finding connected components, and implementing Kruskal's minimum spanning tree algorithm. It's used in network connectivity analysis, image processing (finding connected regions), social network analysis (friend circles), and game development (terrain generation). The data structure achieves near-constant time complexity for operations through clever optimizations, making it one of the most elegant and efficient structures in computer science.

**Real-world analogy:** Imagine organizing a massive networking event where people form groups based on connections. Initially, everyone is in their own group (set). When two people shake hands (union operation), their entire groups merge into one big group. To check if two people are in the same network (find operation), you trace each person up to their group leader and see if they have the same leader. The clever part: when tracing to find leaders, you can shortcut future lookups by making everyone point directly to the top leader (path compression). This is exactly how Union Find works!

## Core Concepts

### Key Principles

1. **Disjoint Sets:** Each element belongs to exactly one set. Sets don't overlap.

2. **Representative/Parent:** Each set has a representative element (root). All elements in a set ultimately point to this root.

3. **Path Compression:** When finding the root, make all nodes on the path point directly to the root for faster future lookups.

4. **Union by Rank/Size:** When merging sets, attach the smaller tree under the larger one to keep trees shallow.

5. **Near-Constant Time:** With optimizations, operations run in O(α(n)) time, where α is the inverse Ackermann function (essentially constant for practical purposes).

### Essential Terms

- **Parent Array:** Maps each element to its parent; root elements point to themselves
- **Rank Array:** Tracks approximate tree height for union by rank optimization
- **Size Array:** Tracks set size for union by size optimization
- **Path Compression:** Optimization that flattens tree during find operation
- **Union by Rank:** Attach shorter tree under taller tree
- **Union by Size:** Attach smaller set under larger set
- **Connected Components:** Groups of connected elements

### Visual Overview

```
Initial state (5 elements):
0  1  2  3  4
↑  ↑  ↑  ↑  ↑
Each element is its own parent (root)

After union(0, 1):
  0
  ↑
  1  2  3  4
  ↑  ↑  ↑  ↑

After union(2, 3):
  0    2
  ↑    ↑
  1    3  4
  ↑    ↑  ↑

After union(0, 2):
    0
   ↗ ↖
  1   2
      ↑
      3  4
         ↑

After path compression on find(3):
    0
   ↗|↖
  1 2 3  4
        ↑
All nodes in set point directly to root 0

Connected components: {0,1,2,3}, {4}
```

## How It Works

### Basic Operations

**Step 1: Initialization**
- Create parent array where parent[i] = i
- Optionally create rank or size arrays initialized to 0 or 1

**Step 2: Find Operation**
- Follow parent pointers until reaching root (where parent[x] = x)
- Apply path compression: make all nodes on path point directly to root
- Return the root

**Step 3: Union Operation**
- Find roots of both elements
- If roots are same, elements already in same set
- If different, merge by making one root point to the other
- Use rank/size to decide which root becomes parent

**Step 4: Connected Check**
- Find roots of both elements
- Return true if roots are the same

### Detailed Walkthrough Example

**Problem:** Determine number of connected components
**Operations:** union(0,1), union(1,2), union(3,4), find connected components

```
Initial Setup (5 nodes: 0,1,2,3,4):
parent = [0, 1, 2, 3, 4]  # Each is its own parent
rank =   [0, 0, 0, 0, 0]  # All have rank 0
Components: {0}, {1}, {2}, {3}, {4}  (5 components)

Operation 1: union(0, 1)
---------------------------
find(0): parent[0] = 0 → root is 0
find(1): parent[1] = 1 → root is 1

Roots are different, so merge:
rank[0] = 0, rank[1] = 0 → same rank
Attach 1 under 0 (arbitrary choice)
parent[1] = 0
rank[0] = 1 (increased because we attached same-rank tree)

parent = [0, 0, 2, 3, 4]
rank =   [1, 0, 0, 0, 0]

Tree structure:
  0
  ↑
  1

Components: {0,1}, {2}, {3}, {4}  (4 components)

Operation 2: union(1, 2)
---------------------------
find(1): 
  parent[1] = 0
  parent[0] = 0 → root is 0
  Path compression: parent[1] remains 0

find(2): parent[2] = 2 → root is 2

Roots are different (0 and 2), so merge:
rank[0] = 1, rank[2] = 0
rank[0] > rank[2], so attach 2 under 0
parent[2] = 0

parent = [0, 0, 0, 3, 4]
rank =   [1, 0, 0, 0, 0]

Tree structure:
    0
   ↗ ↖
  1   2

Components: {0,1,2}, {3}, {4}  (3 components)

Operation 3: union(3, 4)
---------------------------
find(3): parent[3] = 3 → root is 3
find(4): parent[4] = 4 → root is 4

Roots are different, so merge:
rank[3] = 0, rank[4] = 0 → same rank
Attach 4 under 3 (arbitrary)
parent[4] = 3
rank[3] = 1

parent = [0, 0, 0, 3, 3]
rank =   [1, 0, 0, 1, 0]

Tree structure:
    0         3
   ↗ ↖       ↑
  1   2      4

Components: {0,1,2}, {3,4}  (2 components)

Finding Connected Components:
------------------------------
Count unique roots:
find(0) = 0
find(1) = 0 (same as 0)
find(2) = 0 (same as 0)
find(3) = 3
find(4) = 3 (same as 3)

Unique roots: {0, 3}
Number of components: 2 ✓

With path compression, after find(2):
parent = [0, 0, 0, 3, 3]
All nodes already point directly to root!
```

## Implementation

### Python Implementation - Complete Union Find

```python
from typing import List, Dict, Set

class UnionFind:
    """
    Union Find (Disjoint Set Union) data structure.
    
    Supports efficient union and find operations with path compression
    and union by rank optimizations.
    
    Time Complexity (amortized):
        - find: O(α(n)) ≈ O(1)
        - union: O(α(n)) ≈ O(1)
        where α is the inverse Ackermann function
    
    Space Complexity: O(n)
    """
    
    def __init__(self, size: int):
        """
        Initialize Union Find structure.
        
        Args:
            size: Number of elements (0 to size-1)
        """
        self.parent = list(range(size))  # parent[i] = i initially
        self.rank = [0] * size  # All trees have rank 0
        self.count = size  # Number of disjoint sets
    
    def find(self, x: int) -> int:
        """
        Find the root of element x with path compression.
        
        Path compression: make all nodes on path point directly to root.
        
        Args:
            x: Element to find root of
            
        Returns:
            Root of the set containing x
            
        Time: O(α(n)) amortized
        """
        if self.parent[x] != x:
            # Path compression: recursively find root and update parent
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """
        Unite the sets containing x and y.
        
        Uses union by rank to keep trees balanced.
        
        Args:
            x: First element
            y: Second element
            
        Returns:
            True if sets were merged, False if already in same set
            
        Time: O(α(n)) amortized
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        # Already in same set
        if root_x == root_y:
            return False
        
        # Union by rank: attach smaller rank tree under larger rank tree
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            # Same rank: attach y under x and increment x's rank
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        self.count -= 1  # Merged two sets into one
        return True
    
    def connected(self, x: int, y: int) -> bool:
        """
        Check if x and y are in the same set.
        
        Args:
            x: First element
            y: Second element
            
        Returns:
            True if x and y are connected
            
        Time: O(α(n)) amortized
        """
        return self.find(x) == self.find(y)
    
    def get_count(self) -> int:
        """
        Get number of disjoint sets.
        
        Returns:
            Number of connected components
        """
        return self.count
    
    def get_components(self) -> Dict[int, List[int]]:
        """
        Get all connected components.
        
        Returns:
            Dictionary mapping root -> list of elements in that component
            
        Time: O(n)
        """
        components = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in components:
                components[root] = []
            components[root].append(i)
        return components


class UnionFindBySize:
    """
    Union Find using union by size instead of union by rank.
    
    Often preferred when you need to track component sizes.
    """
    
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.size = [1] * size  # Each set initially has size 1
        self.count = size
    
    def find(self, x: int) -> int:
        """Find with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        """Union by size: attach smaller set under larger set."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # Attach smaller set under larger set
        if self.size[root_x] < self.size[root_y]:
            self.parent[root_x] = root_y
            self.size[root_y] += self.size[root_x]
        else:
            self.parent[root_y] = root_x
            self.size[root_x] += self.size[root_y]
        
        self.count -= 1
        return True
    
    def get_size(self, x: int) -> int:
        """Get size of set containing x."""
        return self.size[self.find(x)]


def redundant_connection(edges: List[List[int]]) -> List[int]:
    """
    Find edge that creates a cycle in undirected graph.
    
    LeetCode #684: Redundant Connection
    
    Args:
        edges: List of undirected edges [u, v]
        
    Returns:
        Edge that creates cycle (last one in input)
        
    Time Complexity: O(n * α(n)) ≈ O(n)
    Space Complexity: O(n)
    
    Example:
        >>> redundant_connection([[1,2],[1,3],[2,3]])
        [2,3]  # This edge creates the cycle
    """
    n = len(edges)
    uf = UnionFind(n + 1)  # +1 because nodes are 1-indexed
    
    for u, v in edges:
        # If u and v are already connected, this edge creates a cycle
        if uf.connected(u, v):
            return [u, v]
        uf.union(u, v)
    
    return []


def num_provinces(isConnected: List[List[int]]) -> int:
    """
    Find number of provinces (connected components).
    
    LeetCode #547: Number of Provinces
    
    Args:
        isConnected: Adjacency matrix where isConnected[i][j] = 1 if connected
        
    Returns:
        Number of provinces
        
    Time Complexity: O(n² * α(n))
    Space Complexity: O(n)
    
    Example:
        >>> num_provinces([[1,1,0],[1,1,0],[0,0,1]])
        2  # Province {0,1} and province {2}
    """
    n = len(isConnected)
    uf = UnionFind(n)
    
    # Union all connected cities
    for i in range(n):
        for j in range(i + 1, n):
            if isConnected[i][j] == 1:
                uf.union(i, j)
    
    return uf.get_count()


def is_bipartite(graph: List[List[int]]) -> bool:
    """
    Determine if graph is bipartite using Union Find.
    
    LeetCode #785: Is Graph Bipartite?
    
    A graph is bipartite if nodes can be divided into two sets where
    no edge connects nodes in the same set.
    
    Args:
        graph: Adjacency list representation
        
    Returns:
        True if graph is bipartite
        
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    
    Example:
        >>> is_bipartite([[1,3],[0,2],[1,3],[0,2]])
        True  # Can color as {0,2} and {1,3}
    """
    n = len(graph)
    uf = UnionFind(2 * n)  # Create two copies of each node
    
    for node in range(n):
        for neighbor in graph[node]:
            # If node and neighbor in same set, not bipartite
            if uf.connected(node, neighbor):
                return False
            
            # Union node with neighbor's opposite
            # Union neighbor with node's opposite
            uf.union(node, neighbor + n)
            uf.union(neighbor, node + n)
    
    return True


def min_cost_connect_points(points: List[List[int]]) -> int:
    """
    Find minimum cost to connect all points (Minimum Spanning Tree).
    
    LeetCode #1584: Min Cost to Connect All Points
    
    Uses Kruskal's algorithm with Union Find.
    
    Args:
        points: List of [x, y] coordinates
        
    Returns:
        Minimum cost (sum of edge weights)
        
    Time Complexity: O(n² log n)
    Space Complexity: O(n²)
    
    Example:
        >>> min_cost_connect_points([[0,0],[2,2],[3,10],[5,2],[7,0]])
        20
    """
    n = len(points)
    
    # Build all edges with Manhattan distance
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dist = abs(x1 - x2) + abs(y1 - y2)
            edges.append((dist, i, j))
    
    # Sort edges by distance (Kruskal's algorithm)
    edges.sort()
    
    uf = UnionFind(n)
    total_cost = 0
    edges_used = 0
    
    for dist, u, v in edges:
        # If u and v not connected, add this edge
        if uf.union(u, v):
            total_cost += dist
            edges_used += 1
            
            # MST has n-1 edges
            if edges_used == n - 1:
                break
    
    return total_cost


def min_effort_path(heights: List[List[int]]) -> int:
    """
    Find path with minimum effort (minimum maximum edge weight).
    
    LeetCode #1631: Path With Minimum Effort
    
    Uses binary search + Union Find or Kruskal-like approach.
    
    Args:
        heights: 2D grid of heights
        
    Returns:
        Minimum effort needed to reach bottom-right from top-left
        
    Time Complexity: O(mn log(mn))
    Space Complexity: O(mn)
    
    Example:
        >>> min_effort_path([[1,2,2],[3,8,2],[5,3,5]])
        2  # Path: 1→3→5→3→5 with max diff of 2
    """
    if not heights or not heights[0]:
        return 0
    
    rows, cols = len(heights), len(heights[0])
    
    # Build all edges with effort (height difference)
    edges = []
    for i in range(rows):
        for j in range(cols):
            cell = i * cols + j
            
            # Right neighbor
            if j + 1 < cols:
                neighbor = i * cols + (j + 1)
                effort = abs(heights[i][j] - heights[i][j + 1])
                edges.append((effort, cell, neighbor))
            
            # Down neighbor
            if i + 1 < rows:
                neighbor = (i + 1) * cols + j
                effort = abs(heights[i][j] - heights[i + 1][j])
                edges.append((effort, cell, neighbor))
    
    # Sort edges by effort
    edges.sort()
    
    uf = UnionFind(rows * cols)
    start = 0
    end = (rows - 1) * cols + (cols - 1)
    
    for effort, u, v in edges:
        uf.union(u, v)
        
        # If start and end are connected, return current effort
        if uf.connected(start, end):
            return effort
    
    return 0


def accounts_merge(accounts: List[List[str]]) -> List[List[str]]:
    """
    Merge accounts belonging to same person.
    
    LeetCode #721: Accounts Merge
    
    Args:
        accounts: List of [name, email1, email2, ...] 
        
    Returns:
        Merged accounts
        
    Time Complexity: O(n * k * α(n)) where k is max emails per account
    Space Complexity: O(n * k)
    
    Example:
        >>> accounts_merge([["John","j1@com","j2@com"],["John","j3@com"],["John","j2@com","j3@com"]])
        [["John","j1@com","j2@com","j3@com"]]
    """
    email_to_id = {}
    email_to_name = {}
    
    # Assign ID to each unique email
    email_id = 0
    for account in accounts:
        name = account[0]
        for email in account[1:]:
            if email not in email_to_id:
                email_to_id[email] = email_id
                email_to_name[email] = name
                email_id += 1
    
    # Union emails in same account
    uf = UnionFind(email_id)
    for account in accounts:
        first_email_id = email_to_id[account[1]]
        for email in account[2:]:
            uf.union(first_email_id, email_to_id[email])
    
    # Group emails by root
    components = {}
    for email, email_id in email_to_id.items():
        root = uf.find(email_id)
        if root not in components:
            components[root] = []
        components[root].append(email)
    
    # Build result
    result = []
    for emails in components.values():
        name = email_to_name[emails[0]]
        result.append([name] + sorted(emails))
    
    return result


# Example usage and testing
if __name__ == "__main__":
    print("=== Basic Union Find Operations ===")
    uf = UnionFind(5)
    print(f"Initial components: {uf.get_count()}")
    
    uf.union(0, 1)
    print(f"After union(0,1): {uf.get_count()} components")
    
    uf.union(1, 2)
    print(f"After union(1,2): {uf.get_count()} components")
    
    uf.union(3, 4)
    print(f"After union(3,4): {uf.get_count()} components")
    
    print(f"Connected(0,2): {uf.connected(0, 2)}")
    print(f"Connected(0,3): {uf.connected(0, 3)}")
    print(f"Components: {uf.get_components()}")
    print()
    
    print("=== Redundant Connection ===")
    edges = [[1, 2], [1, 3], [2, 3]]
    print(f"Edges: {edges}")
    print(f"Redundant: {redundant_connection(edges)}")
    print()
    
    print("=== Number of Provinces ===")
    connected = [[1, 1, 0], [1, 1, 0], [0, 0, 1]]
    print(f"Matrix: {connected}")
    print(f"Provinces: {num_provinces(connected)}")
    print()
    
    print("=== Is Bipartite ===")
    graph = [[1, 3], [0, 2], [1, 3], [0, 2]]
    print(f"Graph: {graph}")
    print(f"Bipartite: {is_bipartite(graph)}")
```

### Code Explanation

**UnionFind Class:**
- `find()`: Uses path compression to flatten tree
- `union()`: Uses union by rank to keep trees balanced
- `connected()`: Checks if two elements share same root
- Tracks component count for efficiency

**UnionFindBySize:**
- Alternative implementation using size instead of rank
- Useful when you need component sizes
- Slightly different merging strategy

**Redundant Connection:**
- Processes edges in order
- First edge connecting already-connected nodes is the answer
- Classic cycle detection in undirected graph

**Number of Provinces:**
- Treats adjacency matrix as connection information
- Unions all directly connected cities
- Returns number of remaining components

**Is Bipartite:**
- Creative use: creates two copies of each node
- Connects node to opposite copy of neighbors
- If node connects to itself, not bipartite

**Min Cost Connect Points:**
- Implements Kruskal's MST algorithm
- Sorts all edges by weight
- Adds edges that don't create cycles

**Path With Minimum Effort:**
- Similar to Kruskal's but stops when start/end connected
- Processes edges by increasing effort
- Returns effort when endpoints become connected

## Complexity Analysis

### Time Complexity

**Find Operation:**
- **Without optimization:** O(n) - might traverse entire tree
- **With path compression:** O(α(n)) amortized
- **Why?** Path compression flattens tree over time

**Union Operation:**
- **Without optimization:** O(n) - find roots, then link
- **With both optimizations:** O(α(n)) amortized
- **Why?** Finds are fast, linking is O(1)

**Connected Operation:**
- **Time:** O(α(n)) - two finds
- **Same as find complexity**

**Overall Sequence:**
- **m operations on n elements:** O(m * α(n))
- **For practical purposes:** O(m) since α(n) ≤ 5 for any realistic n

### Space Complexity

**Basic Structure:**
- **Space:** O(n) for parent array
- **With rank:** O(n) for parent + rank arrays
- **With size:** O(n) for parent + size arrays

**Auxiliary Space:**
- **Recursion stack:** O(log n) for find with path compression
- **Iterative find:** O(1)

### Comparison with Alternatives

| Approach | Find | Union | Space | When to Use |
|----------|------|-------|-------|-------------|
| **Union Find (optimized)** | O(α(n)) | O(α(n)) | O(n) | Dynamic connectivity |
| **DFS/BFS** | O(V+E) | N/A | O(V+E) | Static connectivity, one-time |
| **Adjacency List** | O(1) | O(V+E) | O(V+E) | Need full graph structure |
| **Boolean Matrix** | O(1) | O(1) | O(n²) | Dense graphs, small n |

**α(n) values:**
- n = 10: α(n) = 2
- n = 1000: α(n) = 3
- n = 10^9: α(n) = 4
- n = 10^(10^19728): α(n) = 5

For all practical purposes, α(n) ≤ 5, so it's effectively constant!

## Examples

### Example 1: Basic Union and Find

**Operations:**
```
uf = UnionFind(5)
union(0, 1)
union(2, 3)
union(0, 2)
connected(1, 3)?
```

**Trace:**

```
Initial: [0] [1] [2] [3] [4]
parent: [0, 1, 2, 3, 4]
rank:   [0, 0, 0, 0, 0]

union(0, 1):
  find(0) = 0, find(1) = 1
  rank[0] = rank[1] = 0 (same)
  parent[1] = 0, rank[0] = 1
  
  [0, 1] [2] [3] [4]
  parent: [0, 0, 2, 3, 4]
  rank:   [1, 0, 0, 0, 0]

union(2, 3):
  find(2) = 2, find(3) = 3
  rank[2] = rank[3] = 0
  parent[3] = 2, rank[2] = 1
  
  [0, 1] [2, 3] [4]
  parent: [0, 0, 2, 2, 4]
  rank:   [1, 0, 1, 0, 0]

union(0, 2):
  find(0) = 0, find(2) = 2
  rank[0] = rank[2] = 1 (same)
  parent[2] = 0, rank[0] = 2
  
  [0, 1, 2, 3] [4]
  parent: [0, 0, 0, 2, 4]
  rank:   [2, 0, 1, 0, 0]

connected(1, 3):
  find(1): parent[1] = 0, parent[0] = 0 → root = 0
    Path compression: parent[1] = 0 (already)
  find(3): parent[3] = 2, parent[2] = 0, parent[0] = 0 → root = 0
    Path compression: parent[3] = 0
  
  Both roots are 0 → True ✓
  
  After path compression:
  parent: [0, 0, 0, 0, 4]
  All elements in component point directly to 0!
```

### Example 2: Redundant Connection

**Input:** edges = [[1,2],[1,3],[2,3]]
**Output:** [2,3]

**Trace:**

```
Process edges in order:

Edge [1,2]:
  find(1) = 1, find(2) = 2
  Different roots → union(1,2)
  parent: [_, 1, 1, 3]
  Components: {1,2}, {3}

Edge [1,3]:
  find(1) = 1, find(3) = 3
  Different roots → union(1,3)
  parent: [_, 1, 1, 1]
  Components: {1,2,3}

Edge [2,3]:
  find(2): parent[2] = 1, parent[1] = 1 → root = 1
  find(3): parent[3] = 1, parent[1] = 1 → root = 1
  Same root! Already connected ✗
  
  This edge creates a cycle!
  Return [2,3] ✓

Graph visualization:
1 - 2
|   |  ← Edge [2,3] creates triangle
3 --
```

### Example 3: Number of Provinces

**Input:**
```
isConnected = [
  [1, 1, 0],
  [1, 1, 0],
  [0, 0, 1]
]
```
**Output:** 2

**Trace:**

```
Initial: 3 cities, each in own province
parent: [0, 1, 2]
count: 3

Process connections:

isConnected[0][1] = 1:
  union(0, 1)
  parent: [0, 0, 2]
  count: 2

isConnected[0][2] = 0: skip

isConnected[1][2] = 0: skip

Final components:
  Component 1: {0, 1}
  Component 2: {2}
  
Answer: 2 provinces ✓
```

### Example 4: Is Graph Bipartite

**Input:** graph = [[1,3],[0,2],[1,3],[0,2]]
**Output:** True

**Trace:**

```
Graph:
0 - 1
|   |
3 - 2

Create 2 copies of each node (0-3 and 4-7):
uf = UnionFind(8)

Process node 0's edges [1, 3]:
  neighbor 1:
    Check connected(0, 1)? No
    union(0, 1+4) → union(0, 5)  # 0 with opposite of 1
    union(1, 0+4) → union(1, 4)  # 1 with opposite of 0
  
  neighbor 3:
    Check connected(0, 3)? No
    union(0, 3+4) → union(0, 7)
    union(3, 0+4) → union(3, 4)

Process node 1's edges [0, 2]:
  neighbor 0: already processed
  neighbor 2:
    Check connected(1, 2)? No
    union(1, 6)
    union(2, 5)

Process node 2's edges [1, 3]:
  neighbor 1: already processed
  neighbor 3:
    Check connected(2, 3)? No
    union(2, 7)
    union(3, 6)

Process node 3's edges [0, 2]:
  All already processed

No conflicts found → Bipartite ✓

Color assignment:
  Set A: {0, 2} (connected to copies 4, 6)
  Set B: {1, 3} (connected to copies 5, 7)
```

### Example 5: Path With Minimum Effort

**Input:**
```
heights = [
  [1, 2, 2],
  [3, 8, 2],
  [5, 3, 5]
]
```
**Output:** 2

**Trace:**

```
Flatten to 1D: cells 0-8
0  1  2
3  4  5
6  7  8

Build edges with effort (height difference):
Horizontal edges:
  (0,1): |1-2| = 1
  (1,2): |2-2| = 0
  (3,4): |3-8| = 5
  (4,5): |8-2| = 6
  (6,7): |5-3| = 2
  (7,8): |3-5| = 2

Vertical edges:
  (0,3): |1-3| = 2
  (1,4): |2-8| = 6
  (2,5): |2-2| = 0
  (3,6): |3-5| = 2
  (4,7): |8-3| = 5
  (5,8): |2-5| = 3

Sort edges: [0, 0, 1, 2, 2, 2, 2, 3, 5, 5, 6, 6]

Process edges:
  Edge (1,2), effort=0: union(1,2)
  Edge (2,5), effort=0: union(2,5) → now {1,2,5}
  Edge (0,1), effort=1: union(0,1) → now {0,1,2,5}
  Edge (0,3), effort=2: union(0,3) → now {0,1,2,3,5}
  Edge (3,6), effort=2: union(3,6) → now {0,1,2,3,5,6}
  Edge (6,7), effort=2: union(6,7) → now {0,1,2,3,5,6,7}
  Edge (7,8), effort=2: union(7,8) → now {0,1,2,3,5,6,7,8}
  
  Check: connected(0, 8)?
    find(0) = 0, find(8) = 0
    Connected! ✓
  
  Return effort = 2

Path: 0(1) → 3(3) → 6(5) → 7(3) → 8(5)
Max effort: max(2, 2, 2, 2) = 2 ✓
```

## Edge Cases

### 1. Single Element
**Scenario:** Union Find with n=1
**Challenge:** Trivial case
**Solution:** Element is in its own component
**Code example:**
```python
uf = UnionFind(1)
assert uf.get_count() == 1
assert uf.find(0) == 0
```

### 2. No Unions
**Scenario:** Initialize but never call union
**Challenge:** Each element is separate component
**Solution:** Count equals n
**Code example:**
```python
uf = UnionFind(5)
assert uf.get_count() == 5
```

### 3. All Elements United
**Scenario:** Union all elements into one component
**Challenge:** Should result in one component
**Solution:** Count equals 1
**Code example:**
```python
uf = UnionFind(5)
for i in range(4):
    uf.union(i, i+1)
assert uf.get_count() == 1
```

### 4. Redundant Unions
**Scenario:** Union same pair multiple times
**Challenge:** Should not affect result
**Solution:** Union returns False if already connected
**Code example:**
```python
uf = UnionFind(3)
assert uf.union(0, 1) == True
assert uf.union(0, 1) == False  # Already connected
assert uf.get_count() == 2
```

### 5. Self Union
**Scenario:** union(x, x)
**Challenge:** Element already in same set as itself
**Solution:** Returns False, no change
**Code example:**
```python
uf = UnionFind(3)
assert uf.union(0, 0) == False
assert uf.get_count() == 3
```

### 6. Disconnected Components
**Scenario:** Multiple separate components
**Challenge:** Find should not cross components
**Solution:** Each component has unique root
**Code example:**
```python
uf = UnionFind(6)
uf.union(0, 1)
uf.union(2, 3)
uf.union(4, 5)
assert uf.get_count() == 3
assert not uf.connected(0, 2)
```

### 7. Large Input
**Scenario:** Millions of elements
**Challenge:** Memory and performance
**Solution:** Optimizations keep it efficient
**Code example:**
```python
# Even with 10^6 elements, operations are near-constant time
uf = UnionFind(1000000)
# Operations remain O(α(n)) ≈ O(1)
```

## Common Pitfalls

### ❌ Pitfall 1: Not Using Path Compression
**What happens:** Find operations become O(n)
**Why it's wrong:** Loses the main optimization
**Correct approach:**
```python
# WRONG: No path compression
def find(self, x):
    while self.parent[x] != x:
        x = self.parent[x]
    return x

# CORRECT: With path compression
def find(self, x):
    if self.parent[x] != x:
        self.parent[x] = self.find(self.parent[x])  # Compress path
    return self.parent[x]
```

### ❌ Pitfall 2: Not Using Union by Rank/Size
**What happens:** Trees become unbalanced
**Why it's wrong:** Operations degrade to O(n)
**Correct approach:**
```python
# WRONG: Always attach second to first
def union(self, x, y):
    root_x, root_y = self.find(x), self.find(y)
    self.parent[root_y] = root_x  # Can create long chains!

# CORRECT: Union by rank
def union(self, x, y):
    root_x, root_y = self.find(x), self.find(y)
    if self.rank[root_x] < self.rank[root_y]:
        self.parent[root_x] = root_y
    elif self.rank[root_x] > self.rank[root_y]:
        self.parent[root_y] = root_x
    else:
        self.parent[root_y] = root_x
        self.rank[root_x] += 1
```

### ❌ Pitfall 3: Forgetting to Check Same Root
**What happens:** Incorrect union, wrong component count
**Why it's wrong:** Must check if already connected
**Correct approach:**
```python
# WRONG: Always union without checking
def union(self, x, y):
    root_x, root_y = self.find(x), self.find(y)
    self.parent[root_y] = root_x
    self.count -= 1  # Wrong if already connected!

# CORRECT: Check first
def union(self, x, y):
    root_x, root_y = self.find(x), self.find(y)
    if root_x == root_y:
        return False  # Already connected
    self.parent[root_y] = root_x
    self.count -= 1
    return True
```

### ❌ Pitfall 4: Modifying Parent During Find
**What happens:** Incorrect path compression
**Why it's wrong:** Parent array gets corrupted
**Correct approach:**
```python
# WRONG: Direct modification
def find(self, x):
    root = x
    while self.parent[root] != root:
        root = self.parent[root]
    self.parent[x] = root  # Partial compression, misses intermediate nodes
    return root

# CORRECT: Recursive compression
def find(self, x):
    if self.parent[x] != x:
        self.parent[x] = self.find(self.parent[x])  # Full compression
    return self.parent[x]
```

### ❌ Pitfall 5: Using 0-indexed When Problem is 1-indexed
**What happens:** Index out of bounds or wrong results
**Why it's wrong:** Many graph problems use 1-indexed nodes
**Correct approach:**
```python
# WRONG: For 1-indexed problem
uf = UnionFind(n)  # Will miss node n!

# CORRECT:
uf = UnionFind(n + 1)  # Accommodate 1 to n
# Or map to 0-indexed internally
```

### ❌ Pitfall 6: Not Updating Count on Union
**What happens:** get_count() returns wrong value
**Why it's wrong:** Count must decrease when merging
**Correct approach:**
```python
# WRONG: Forget to decrement
def union(self, x, y):
    root_x, root_y = self.find(x), self.find(y)
    if root_x == root_y:
        return False
    self.parent[root_y] = root_x
    # Forgot: self.count -= 1

# CORRECT:
def union(self, x, y):
    root_x, root_y = self.find(x), self.find(y)
    if root_x == root_y:
        return False
    self.parent[root_y] = root_x
    self.count -= 1  # Decrement ✓
    return True
```

### ❌ Pitfall 7: Using Find Without Storing Result
**What happens:** Multiple find calls, inefficient
**Why it's wrong:** Each find does work
**Correct approach:**
```python
# WRONG: Multiple finds
def union(self, x, y):
    if self.find(x) == self.find(y):  # Two finds
        return False
    if self.rank[self.find(x)] < self.rank[self.find(y)]:  # Two more finds!
        ...

# CORRECT: Store results
def union(self, x, y):
    root_x = self.find(x)  # Once
    root_y = self.find(y)  # Once
    if root_x == root_y:
        return False
    if self.rank[root_x] < self.rank[root_y]:
        ...
```

## Variations and Extensions

### Variation 1: Weighted Union Find
**Description:** Track values/weights on edges
**When to use:** Need to maintain relationships between elements
**Key differences:** Store relative values to parent
**Implementation:**
```python
class WeightedUnionFind:
    """Union Find with edge weights."""
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.weight = [0] * n  # Weight to parent
    
    def find(self, x):
        if self.parent[x] != x:
            original_parent = self.parent[x]
            self.parent[x] = self.find(self.parent[x])
            # Update weight: cumulative to root
            self.weight[x] += self.weight[original_parent]
        return self.parent[x]
    
    def union(self, x, y, w):
        """Union with weight: w represents weight from x to y."""
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return
        self.parent[root_x] = root_y
        # Set weight so that: weight[x] + weight[root_x] = weight[y] + w
        self.weight[root_x] = self.weight[y] - self.weight[x] + w
```

### Variation 2: Union Find with Rollback
**Description:** Undo union operations
**When to use:** When you need to try different configurations
**Key differences:** Store history of operations
**Implementation:**
```python
class UnionFindWithRollback:
    """Union Find supporting rollback."""
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.history = []  # Stack of operations
    
    def find(self, x):
        # Iterative to avoid recursion (needed for rollback)
        while self.parent[x] != x:
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            self.history.append(None)  # No-op
            return
        
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        
        # Save state before modification
        self.history.append((root_y, self.parent[root_y], self.rank[root_x]))
        
        self.parent[root_y] = root_x
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
    
    def rollback(self):
        """Undo last union operation."""
        if not self.history:
            return
        
        state = self.history.pop()
        if state is None:
            return
        
        node, old_parent, old_rank = state
        self.parent[node] = old_parent
        root = self.find(node)
        self.rank[root] = old_rank
```

### Variation 3: Dynamic Connectivity with Deletion
**Description:** Support removing elements
**When to use:** Elements can be added and removed
**Key differences:** More complex, often use link-cut trees
**Implementation:**
```python
# Simplified approach: mark as deleted
class UnionFindWithDeletion:
    def __init__(self, n):
        self.parent = list(range(n))
        self.deleted = [False] * n
        self.next_available = {}  # Map deleted -> next available
    
    def delete(self, x):
        """Mark element as deleted."""
        self.deleted[x] = True
        # Find next non-deleted element
        next_elem = x + 1
        while next_elem < len(self.parent) and self.deleted[next_elem]:
            next_elem += 1
        if next_elem < len(self.parent):
            self.next_available[x] = next_elem
```

### Variation 4: Persistent Union Find
**Description:** Multiple versions of structure
**When to use:** Need to query historical states
**Key differences:** Immutable structure, more memory
**Implementation:**
```python
# Each operation creates new version sharing structure
# Implementation is complex, typically uses persistent arrays
```

## Practice Problems

### Beginner
1. **Redundant Connection** - Find cycle-creating edge
   - LeetCode #684

2. **Number of Provinces** - Count connected components
   - LeetCode #547

3. **Find if Path Exists in Graph** - Simple connectivity check
   - LeetCode #1971

4. **Accounts Merge** - Merge accounts by email
   - LeetCode #721

### Intermediate
1. **Is Graph Bipartite?** - Check bipartiteness using UF
   - LeetCode #785

2. **Path With Minimum Effort** - Minimum bottleneck path
   - LeetCode #1631

3. **Min Cost to Connect All Points** - Minimum spanning tree
   - LeetCode #1584

4. **Regions Cut by Slashes** - Count regions in grid
   - LeetCode #959

5. **Satisfiability of Equality Equations** - Evaluate equations
   - LeetCode #990

6. **Smallest String With Swaps** - Lexicographically smallest
   - LeetCode #1202

### Advanced
1. **Number of Islands II** - Dynamic island formation
   - LeetCode #305 (Premium)

2. **Checking Existence of Edge Length Limited Paths** - Complex queries
   - LeetCode #1724

3. **Minimize Malware Spread** - Optimize node removal
   - LeetCode #924

4. **Bricks Falling When Hit** - Reverse time union find
   - LeetCode #803

## Real-World Applications

### Industry Use Cases

1. **Network Connectivity:** Determining if two computers are on the same network, handling dynamic connections.

2. **Image Processing:** Finding connected components in images (flood fill, region labeling).

3. **Social Networks:** Finding friend circles, detecting communities.

4. **Kruskal's MST:** Minimum spanning tree algorithm for network design.

5. **Compilation:** Detecting circular dependencies in build systems.

### Popular Implementations

- **NetworkX (Python):** Graph library uses Union Find for connected components
- **MATLAB:** Image processing functions use Union Find for region labeling
- **Game Engines:** Terrain generation, connectivity analysis
- **Databases:** Query optimization, join operations

### Practical Scenarios

- **LAN Management:** Tracking network segments and connections
- **Maze Generation:** Creating mazes with Union Find
- **Percolation:** Modeling physical systems (fluid flow, electrical conductivity)
- **Circuit Design:** Checking electrical connectivity
- **File Systems:** Tracking hard link relationships

## Related Topics

### Prerequisites to Review
- **Arrays** - Foundation for Union Find
- **Trees** - Understanding tree structures
- **Recursion** - For path compression
- **Graph Basics** - Connectivity concepts

### Next Steps
- **Minimum Spanning Trees** - Kruskal's and Prim's algorithms
- **Graph Connectivity** - Strongly connected components
- **Dynamic Connectivity** - More advanced structures
- **Persistent Data Structures** - Version control for data structures

### Similar Concepts
- **DFS/BFS** - Alternative for static connectivity
- **Segment Trees** - Another union-based structure
- **Link-Cut Trees** - Dynamic tree connectivity
- **Lowest Common Ancestor** - Related tree queries

### Further Reading
- "Introduction to Algorithms" (CLRS) - Chapter on Disjoint Sets
- "Algorithm Design" by Kleinberg & Tardos - Union Find applications
- [Union Find - CP-Algorithms](https://cp-algorithms.com/data_structures/disjoint_set_union.html)
- [Disjoint Set Union - Wikipedia](https://en.wikipedia.org/wiki/Disjoint-set_data_structure)
- "The Art of Computer Programming Vol 1" by Knuth - Analysis of Union Find
- Papers on path compression and union by rank optimizations
