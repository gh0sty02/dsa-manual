# Graphs Pattern

**Difficulty:** Intermediate to Advanced
**Prerequisites:** Arrays, Hash Maps, Queues, Stacks, Recursion, Basic data structures
**Estimated Reading Time:** 60 minutes

## Introduction

A graph is a non-linear data structure consisting of vertices (nodes) and edges (connections between nodes) that represent relationships and networks. Graphs are fundamental to computer science, modeling everything from social networks and road maps to web pages and molecule structures.

**Why it matters:** Graphs solve real-world problems that other data structures can't handle effectively. Need to find the shortest path between cities? That's a graph problem. Want to detect cycles in dependencies? Graph algorithm. Analyzing social network connections? Graph traversal. From GPS navigation to recommendation systems to compiler optimization, graphs are everywhere.

**Real-world analogy:** Think of a graph like a subway map. The stations are vertices (nodes), and the train lines connecting them are edges. You can travel from one station to another following different paths, some stations might be hubs with many connections, and you might need to find the quickest route from your starting point to your destination. This is exactly how graphs work in computer science!

## Core Concepts

### Key Principles

1. **Vertices and Edges:** A graph G = (V, E) consists of a set of vertices V and a set of edges E. Each edge connects two vertices, representing a relationship or path between them.

2. **Connectivity:** Graphs can be connected (you can reach any vertex from any other vertex) or disconnected (some vertices are isolated or in separate components).

3. **Traversal Strategies:** Two fundamental approaches exist for exploring graphs:
   - **Depth-First Search (DFS):** Go as deep as possible before backtracking
   - **Breadth-First Search (BFS):** Explore all neighbors before going deeper

4. **Paths and Cycles:** A path is a sequence of vertices connected by edges. A cycle is a path that starts and ends at the same vertex.

### Essential Terms

- **Vertex (Node):** A fundamental unit in a graph representing an entity
- **Edge:** A connection between two vertices
- **Degree:** The number of edges connected to a vertex
- **Adjacent:** Two vertices are adjacent if connected by an edge
- **Path:** A sequence of vertices where each adjacent pair is connected by an edge
- **Cycle:** A path that starts and ends at the same vertex
- **Connected Component:** A maximal set of vertices where each pair is connected by a path
- **Directed Graph (Digraph):** Edges have direction (one-way relationships)
- **Undirected Graph:** Edges have no direction (two-way relationships)
- **Weighted Graph:** Edges have associated values (costs, distances, etc.)
- **Neighbor:** A vertex directly connected to another vertex by an edge

### Visual Overview

```
Undirected Graph:          Directed Graph:           Weighted Graph:
    
    A --- B                   A --> B                  A --5-- B
    |     |                   |     ↓                  |       |
    |     |                   ↓     C                  3       7
    C --- D                   D <-- ↓                  |       |
                                                       C --2-- D

Connected Components:      Graph with Cycle:
    
    A --- B    E --- F         A --- B
    |     |                    |   / |
    C --- D    G               C----- D
    
    (2 components)            (Cycle: A-B-D-C-A)
```

## How It Works

### Graph Representations

Graphs can be represented in three main ways:

**1. Adjacency Matrix (2D Array)**

```
For graph:  A --- B
            |     |
            C --- D

Matrix:     A  B  C  D
         A [0  1  1  0]
         B [1  0  0  1]
         C [1  0  0  1]
         D [0  1  1  0]

1 = edge exists, 0 = no edge
```

**2. Adjacency List (Hash Map of Lists)**

```
A: [B, C]
B: [A, D]
C: [A, D]
D: [B, C]
```

**3. Edge List (List of Pairs)**

```
[(A, B), (A, C), (B, D), (C, D)]
```

### Depth-First Search (DFS) - How It Works

DFS explores a graph by going as deep as possible along each branch before backtracking.

**Step-by-step process:**

1. Start at a source vertex, mark it as visited
2. Explore one unvisited neighbor
3. Recursively apply DFS to that neighbor
4. When no unvisited neighbors exist, backtrack
5. Repeat until all reachable vertices are visited

**Visual walkthrough:**

```
Graph:      A --- B --- E
            |     |
            C --- D

DFS starting from A:

Step 1: Visit A, mark visited
Visited: {A}
Stack: [A]

Step 2: Go to neighbor B
Visited: {A, B}
Stack: [A, B]

Step 3: From B, go to E
Visited: {A, B, E}
Stack: [A, B, E]

Step 4: E has no unvisited neighbors, backtrack to B
Stack: [A, B]

Step 5: From B, go to D
Visited: {A, B, E, D}
Stack: [A, B, D]

Step 6: From D, go to C
Visited: {A, B, E, D, C}
Stack: [A, B, D, C]

Step 7: C has no unvisited neighbors, backtrack
Final visited order: A → B → E → D → C
```

### Breadth-First Search (BFS) - How It Works

BFS explores a graph level by level, visiting all neighbors before going deeper.

**Step-by-step process:**

1. Start at a source vertex, add to queue, mark as visited
2. Dequeue a vertex, process it
3. Enqueue all unvisited neighbors, mark them as visited
4. Repeat until queue is empty

**Visual walkthrough:**

```
Graph:      A --- B --- E
            |     |
            C --- D

BFS starting from A:

Step 1: Start with A
Queue: [A]
Visited: {A}

Step 2: Process A, add neighbors B, C
Queue: [B, C]
Visited: {A, B, C}

Step 3: Process B, add neighbors E, D (A already visited)
Queue: [C, E, D]
Visited: {A, B, C, E, D}

Step 4: Process C (all neighbors visited)
Queue: [E, D]

Step 5: Process E (all neighbors visited)
Queue: [D]

Step 6: Process D (all neighbors visited)
Queue: []

Final visited order: A → B → C → E → D
(Level 0: A, Level 1: B,C, Level 2: E,D)
```

## Implementation

### Graph Representation - Adjacency List

```python
from typing import List, Dict, Set
from collections import defaultdict, deque

class Graph:
    """
    Graph implementation using adjacency list representation.
    
    Attributes:
        graph: Dictionary mapping vertices to list of adjacent vertices
        directed: Boolean indicating if graph is directed
    """
    
    def __init__(self, directed: bool = False):
        """
        Initialize an empty graph.
        
        Args:
            directed: If True, creates directed graph; otherwise undirected
        """
        self.graph: Dict[int, List[int]] = defaultdict(list)
        self.directed = directed
    
    def add_edge(self, u: int, v: int) -> None:
        """
        Add an edge between vertices u and v.
        
        Args:
            u: Source vertex
            v: Destination vertex
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self.graph[u].append(v)
        # For undirected graph, add edge in both directions
        if not self.directed:
            self.graph[v].append(u)
    
    def get_neighbors(self, vertex: int) -> List[int]:
        """
        Get all neighbors of a vertex.
        
        Args:
            vertex: The vertex to get neighbors for
            
        Returns:
            List of adjacent vertices
            
        Time Complexity: O(1)
        """
        return self.graph.get(vertex, [])
    
    def get_vertices(self) -> List[int]:
        """
        Get all vertices in the graph.
        
        Returns:
            List of all vertices
            
        Time Complexity: O(V)
        """
        return list(self.graph.keys())
```

### Depth-First Search (DFS) Implementation

```python
def dfs_recursive(graph: Dict[int, List[int]], start: int, 
                  visited: Set[int] = None) -> List[int]:
    """
    Perform depth-first search recursively.
    
    Args:
        graph: Adjacency list representation of graph
        start: Starting vertex
        visited: Set of already visited vertices
        
    Returns:
        List of vertices in DFS order
        
    Time Complexity: O(V + E) where V is vertices, E is edges
    Space Complexity: O(V) for recursion stack and visited set
    """
    if visited is None:
        visited = set()
    
    result = []
    
    # Mark current vertex as visited
    visited.add(start)
    result.append(start)
    
    # Recursively visit all unvisited neighbors
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            result.extend(dfs_recursive(graph, neighbor, visited))
    
    return result


def dfs_iterative(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    Perform depth-first search iteratively using a stack.
    
    Args:
        graph: Adjacency list representation of graph
        start: Starting vertex
        
    Returns:
        List of vertices in DFS order
        
    Time Complexity: O(V + E)
    Space Complexity: O(V) for stack and visited set
    """
    visited = set()
    stack = [start]
    result = []
    
    while stack:
        # Pop from stack (LIFO - Last In First Out)
        vertex = stack.pop()
        
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            
            # Add all unvisited neighbors to stack
            # Reverse to maintain left-to-right order
            for neighbor in reversed(graph.get(vertex, [])):
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return result
```

### Breadth-First Search (BFS) Implementation

```python
def bfs(graph: Dict[int, List[int]], start: int) -> List[int]:
    """
    Perform breadth-first search using a queue.
    
    Args:
        graph: Adjacency list representation of graph
        start: Starting vertex
        
    Returns:
        List of vertices in BFS order (level by level)
        
    Time Complexity: O(V + E)
    Space Complexity: O(V) for queue and visited set
    """
    visited = set([start])
    queue = deque([start])
    result = []
    
    while queue:
        # Dequeue from front (FIFO - First In First Out)
        vertex = queue.popleft()
        result.append(vertex)
        
        # Enqueue all unvisited neighbors
        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return result


def bfs_with_levels(graph: Dict[int, List[int]], start: int) -> Dict[int, int]:
    """
    BFS that tracks the level/distance of each vertex from start.
    
    Args:
        graph: Adjacency list representation of graph
        start: Starting vertex
        
    Returns:
        Dictionary mapping vertex to its level/distance from start
        
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    """
    visited = {start}
    queue = deque([(start, 0)])  # (vertex, level)
    levels = {start: 0}
    
    while queue:
        vertex, level = queue.popleft()
        
        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                visited.add(neighbor)
                levels[neighbor] = level + 1
                queue.append((neighbor, level + 1))
    
    return levels
```

**Usage Examples:**

```python
# Create a graph
graph = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0, 4],
    3: [1],
    4: [1, 2]
}

# DFS from vertex 0
print(dfs_recursive(graph, 0))  # Output: [0, 1, 3, 4, 2]
print(dfs_iterative(graph, 0))  # Output: [0, 1, 3, 4, 2]

# BFS from vertex 0
print(bfs(graph, 0))  # Output: [0, 1, 2, 3, 4]

# BFS with levels
print(bfs_with_levels(graph, 0))  
# Output: {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
```

### Code Explanation

**DFS Recursive:**
- Uses the call stack for backtracking
- Marks vertex as visited immediately upon entry
- Explores each neighbor recursively before moving to the next
- Natural and concise, but limited by stack depth

**DFS Iterative:**
- Uses explicit stack instead of call stack
- Reverses neighbors to maintain consistent traversal order
- More control over memory usage
- Avoids stack overflow for deep graphs

**BFS:**
- Uses queue (FIFO) to process vertices level by level
- Marks vertices as visited when enqueued (not when dequeued) to avoid duplicates
- Guarantees shortest path in unweighted graphs
- Natural for finding distances and levels

## Complexity Analysis

### Time Complexity

- **Graph Construction (Adjacency List):** O(E) - where E is the number of edges
- **Graph Construction (Adjacency Matrix):** O(V²) - where V is the number of vertices
- **DFS/BFS Traversal:** O(V + E)
  - Visit each vertex once: O(V)
  - Examine each edge once: O(E)
  - Total: O(V + E)

**Why O(V + E)?**
In the worst case, we visit every vertex exactly once (contributing V to complexity) and examine every edge exactly once when checking neighbors (contributing E to complexity). For a connected graph with V vertices, we have at least V-1 edges, so O(V + E) ≥ O(V). For a complete graph, E = V²/2, making O(V + E) approach O(V²).

### Space Complexity

- **Adjacency List:** O(V + E)
  - Store V vertices and E edges
  - Space-efficient for sparse graphs (few edges)

- **Adjacency Matrix:** O(V²)
  - Always uses V×V space regardless of edge count
  - Space-efficient for dense graphs (many edges)

- **DFS Space:** O(V)
  - Visited set: O(V)
  - Recursion stack depth (worst case - linear graph): O(V)
  - Iterative stack (worst case): O(V)

- **BFS Space:** O(V)
  - Visited set: O(V)
  - Queue size (worst case - star graph): O(V)

### Comparison with Alternatives

| Algorithm | Time | Space | Use Case | Guarantees |
|-----------|------|-------|----------|------------|
| DFS | O(V+E) | O(V) | Topological sort, cycle detection, pathfinding | Finds a path (not shortest) |
| BFS | O(V+E) | O(V) | Shortest path (unweighted), level-order | Finds shortest path |
| Dijkstra | O((V+E)logV) | O(V) | Shortest path (weighted, non-negative) | Optimal shortest path |
| Bellman-Ford | O(VE) | O(V) | Shortest path (weighted, negative edges) | Handles negative weights |
| Floyd-Warshall | O(V³) | O(V²) | All-pairs shortest path | All shortest paths |

## Examples

### Example 1: Simple Graph Traversal

**Problem:** Given an undirected graph, traverse it using DFS and BFS.

**Input:**
```
Graph:  0 --- 1 --- 3
        |     |
        2 --- 4

Edges: [(0,1), (0,2), (1,3), (1,4), (2,4)]
Start: 0
```

**DFS Traversal:**
```
Step 1: Visit 0 → neighbors: [1, 2]
Step 2: Visit 1 → neighbors: [0✓, 3, 4]
Step 3: Visit 3 → neighbors: [1✓]
Step 4: Backtrack to 1, visit 4 → neighbors: [1✓, 2]
Step 5: Visit 2 → neighbors: [0✓, 4✓]

DFS Order: [0, 1, 3, 4, 2]
```

**BFS Traversal:**
```
Level 0: [0]
Level 1: [1, 2]         (neighbors of 0)
Level 2: [3, 4]         (neighbors of 1 and 2)

BFS Order: [0, 1, 2, 3, 4]
```

### Example 2: Detecting if Path Exists

**Problem:** Determine if a path exists between two vertices.

**Input:**
```
Graph: 0 → 1 → 2
       ↓   ↓
       3 → 4

Check: Does path exist from 0 to 4?
```

**Solution using BFS:**
```
Start: 0, Target: 4

Step 1: Queue = [0], Visited = {0}
Step 2: Process 0, add neighbors 1, 3
        Queue = [1, 3], Visited = {0, 1, 3}
Step 3: Process 1, add neighbor 2, 4
        Queue = [3, 2, 4], Visited = {0, 1, 3, 2, 4}
Step 4: Process 3 (4 already visited)
        Queue = [2, 4]
Step 5: Process 2 (all neighbors visited)
        Queue = [4]
Step 6: Process 4 → FOUND!

Answer: Yes, path exists (0 → 1 → 4)
```

### Example 3: Finding Connected Components

**Problem:** Count the number of connected components in an undirected graph.

**Input:**
```
Graph with 3 components:

Component 1:  0 --- 1     Component 2:  3 --- 4     Component 3:  5
              |                                      (isolated)
              2
```

**Solution:**
```
visited = set()
component_count = 0

Step 1: Start at vertex 0 (not visited)
        DFS from 0: visits {0, 1, 2}
        component_count = 1

Step 2: Try vertex 1 (already visited), skip
        Try vertex 2 (already visited), skip

Step 3: Try vertex 3 (not visited)
        DFS from 3: visits {3, 4}
        component_count = 2

Step 4: Try vertex 4 (already visited), skip

Step 5: Try vertex 5 (not visited)
        DFS from 5: visits {5}
        component_count = 3

Answer: 3 connected components
```

### Example 4: Cycle Detection in Directed Graph

**Problem:** Detect if a directed graph contains a cycle.

**Input:**
```
Graph: 0 → 1 → 2
       ↑       ↓
       └───────┘

Has cycle: 0 → 1 → 2 → 0
```

**Solution using DFS with recursion stack:**
```
visited = set()
rec_stack = set()  # Tracks current path

DFS from 0:
  rec_stack = {0}, visited = {0}
  
  Visit neighbor 1:
    rec_stack = {0, 1}, visited = {0, 1}
    
    Visit neighbor 2:
      rec_stack = {0, 1, 2}, visited = {0, 1, 2}
      
      Visit neighbor 0:
        0 is in rec_stack → CYCLE DETECTED!

Answer: Cycle exists
```

## Edge Cases

### 1. Empty Graph

**Scenario:** Graph with no vertices or edges.

**Challenge:** Operations on empty graph should return appropriate defaults.

**Solution:**
- DFS/BFS on empty graph returns empty list
- Path finding returns False
- Component count returns 0

```python
def dfs_safe(graph: Dict[int, List[int]], start: int) -> List[int]:
    """DFS that handles empty graph or invalid start vertex."""
    if not graph or start not in graph:
        return []
    # Regular DFS logic...
```

### 2. Single Vertex Graph

**Scenario:** Graph with one vertex and no edges.

**Challenge:** Edge case for many algorithms.

**Solution:**
- DFS/BFS returns [vertex]
- Path from vertex to itself: True
- Component count: 1

```python
# Graph: {0: []}
dfs(graph, 0)  # Returns: [0]
```

### 3. Disconnected Graph

**Scenario:** Graph with multiple components that aren't connected.

**Challenge:** Single traversal won't visit all vertices.

**Solution:** Loop through all vertices and start DFS/BFS from unvisited ones.

```python
def dfs_all_components(graph: Dict[int, List[int]]) -> List[List[int]]:
    """Find all connected components."""
    visited = set()
    components = []
    
    for vertex in graph:
        if vertex not in visited:
            component = []
            dfs_helper(graph, vertex, visited, component)
            components.append(component)
    
    return components
```

### 4. Graph with Self-Loops

**Scenario:** Edge from a vertex to itself.

**Challenge:** Can cause infinite loops if not handled.

**Solution:** Check if neighbor is current vertex before processing.

```python
def dfs_with_self_loops(graph: Dict[int, List[int]], start: int) -> List[int]:
    """Handle graphs with self-loops."""
    visited = set()
    result = []
    stack = [start]
    
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            
            for neighbor in graph[vertex]:
                # Skip self-loops and visited vertices
                if neighbor != vertex and neighbor not in visited:
                    stack.append(neighbor)
    
    return result
```

### 5. Very Large Graphs (Memory Constraints)

**Scenario:** Graph too large to fit in memory.

**Challenge:** Standard adjacency list might cause out-of-memory errors.

**Solution:**
- Use adjacency matrix only when needed (dense graphs)
- Stream edges from disk/database
- Use generators for lazy evaluation
- Implement external memory algorithms

```python
def dfs_generator(graph: Dict[int, List[int]], start: int):
    """Memory-efficient DFS using generator."""
    visited = set()
    stack = [start]
    
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            yield vertex  # Yield instead of storing
            
            for neighbor in graph.get(vertex, []):
                if neighbor not in visited:
                    stack.append(neighbor)
```

## Common Pitfalls

### ❌ Pitfall 1: Not Marking Vertices as Visited

**What happens:** Infinite loops in graphs with cycles.

```python
# WRONG - No visited tracking
def dfs_wrong(graph, start):
    result = [start]
    for neighbor in graph[start]:
        result.extend(dfs_wrong(graph, neighbor))  # Infinite recursion!
    return result
```

**Why it's wrong:** Without tracking visited vertices, DFS will revisit vertices infinitely in cyclic graphs.

**Correct approach:**

```python
# CORRECT - Track visited vertices
def dfs_correct(graph, start, visited=None):
    if visited is None:
        visited = set()
    
    if start in visited:
        return []
    
    visited.add(start)
    result = [start]
    
    for neighbor in graph[start]:
        result.extend(dfs_correct(graph, neighbor, visited))
    
    return result
```

### ❌ Pitfall 2: Marking Visited Too Late in BFS

**What happens:** Vertices get added to queue multiple times, causing duplicates and inefficiency.

```python
# WRONG - Mark visited when dequeuing
def bfs_wrong(graph, start):
    visited = set()
    queue = deque([start])
    result = []
    
    while queue:
        vertex = queue.popleft()
        visited.add(vertex)  # Too late!
        result.append(vertex)
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                queue.append(neighbor)  # Might be added multiple times
    
    return result
```

**Why it's wrong:** A vertex can be added to the queue multiple times before being processed.

**Correct approach:**

```python
# CORRECT - Mark visited when enqueuing
def bfs_correct(graph, start):
    visited = set([start])  # Mark start as visited immediately
    queue = deque([start])
    result = []
    
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)  # Mark before adding to queue
                queue.append(neighbor)
    
    return result
```

### ❌ Pitfall 3: Confusing Directed and Undirected Graphs

**What happens:** Adding edges incorrectly leads to wrong graph structure.

```python
# WRONG - Treating undirected as directed
def add_edge_wrong(graph, u, v):
    graph[u].append(v)  # Only one direction!
    # Missing: graph[v].append(u) for undirected
```

**Why it's wrong:** Undirected edges must be added in both directions.

**Correct approach:**

```python
# CORRECT - Specify graph type
def add_edge(graph, u, v, directed=False):
    graph[u].append(v)
    if not directed:
        graph[v].append(u)  # Add reverse edge for undirected
```

### ❌ Pitfall 4: Not Handling Disconnected Components

**What happens:** Only explores vertices reachable from starting vertex.

```python
# WRONG - Only explores from one start vertex
def count_components_wrong(graph):
    visited = set()
    dfs(graph, 0, visited)  # Only explores component containing 0
    return 1  # Wrong! Might be multiple components
```

**Correct approach:**

```python
# CORRECT - Explore all vertices
def count_components_correct(graph):
    visited = set()
    count = 0
    
    for vertex in graph:
        if vertex not in visited:
            dfs(graph, vertex, visited)
            count += 1
    
    return count
```

### ❌ Pitfall 5: Using Wrong Data Structure for Graph Representation

**What happens:** Inefficient memory usage or slow operations.

```python
# WRONG - Using adjacency matrix for sparse graph
# For 10,000 vertices with only 20,000 edges
matrix = [[0] * 10000 for _ in range(10000)]  # 100M entries for 20K edges!
```

**Why it's wrong:** Adjacency matrix uses O(V²) space even for sparse graphs with few edges.

**Correct approach:**

```python
# CORRECT - Use adjacency list for sparse graphs
from collections import defaultdict

graph = defaultdict(list)  # Only stores actual edges
# For 10,000 vertices and 20,000 edges: only ~20K entries
```

**Rule of thumb:**
- Sparse graph (E << V²): Use adjacency list
- Dense graph (E ≈ V²): Use adjacency matrix
- Need fast edge lookup: Use adjacency matrix or set-based adjacency list

## Variations and Extensions

### Variation 1: Weighted Graphs

**Description:** Edges have associated weights (costs, distances, times).

**When to use:** Finding shortest/cheapest paths, minimum spanning trees.

**Key differences:**
- Store weights with edges: `graph[u] = [(v1, weight1), (v2, weight2)]`
- Use Dijkstra or Bellman-Ford instead of simple BFS for shortest paths

**Implementation:**

```python
def add_weighted_edge(graph: Dict[int, List[tuple]], u: int, v: int, weight: float):
    """Add weighted edge to graph."""
    graph[u].append((v, weight))

# Example weighted graph
weighted_graph = {
    0: [(1, 4), (2, 1)],
    1: [(3, 1)],
    2: [(1, 2), (3, 5)],
    3: []
}
```

### Variation 2: Bidirectional Search

**Description:** Search from both start and end simultaneously until paths meet.

**When to use:** Finding path between two specific vertices in large graphs.

**Key differences:** Two BFS/DFS running simultaneously, terminate when they meet.

**Implementation:**

```python
def bidirectional_search(graph: Dict[int, List[int]], start: int, end: int) -> bool:
    """
    Search from both start and end to find path faster.
    
    Time Complexity: O(b^(d/2)) instead of O(b^d) where b is branching factor
    """
    if start == end:
        return True
    
    # BFS from both ends
    visited_from_start = {start}
    visited_from_end = {end}
    queue_start = deque([start])
    queue_end = deque([end])
    
    while queue_start and queue_end:
        # Expand from start
        if queue_start:
            current = queue_start.popleft()
            for neighbor in graph.get(current, []):
                if neighbor in visited_from_end:
                    return True  # Paths met!
                if neighbor not in visited_from_start:
                    visited_from_start.add(neighbor)
                    queue_start.append(neighbor)
        
        # Expand from end
        if queue_end:
            current = queue_end.popleft()
            for neighbor in graph.get(current, []):
                if neighbor in visited_from_start:
                    return True  # Paths met!
                if neighbor not in visited_from_end:
                    visited_from_end.add(neighbor)
                    queue_end.append(neighbor)
    
    return False
```

### Variation 3: Iterative Deepening DFS (IDDFS)

**Description:** Combines benefits of DFS (low space) and BFS (shortest path).

**When to use:** When you need BFS completeness but have memory constraints.

**Implementation:**

```python
def iddfs(graph: Dict[int, List[int]], start: int, max_depth: int) -> List[int]:
    """
    Iterative deepening DFS - runs DFS with increasing depth limits.
    
    Time Complexity: O(V + E) per depth level
    Space Complexity: O(d) where d is depth (better than BFS)
    """
    def dfs_limited(vertex: int, depth: int, visited: Set[int], result: List[int]):
        if depth == 0:
            return
        
        visited.add(vertex)
        result.append(vertex)
        
        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                dfs_limited(neighbor, depth - 1, visited, result)
    
    for depth in range(max_depth + 1):
        visited = set()
        result = []
        dfs_limited(start, depth, visited, result)
        if result:  # Found something at this depth
            return result
    
    return []
```

### Variation 4: Parallel Graph Traversal

**Description:** Use multiple threads/processes to explore graph simultaneously.

**When to use:** Very large graphs where single-threaded traversal is too slow.

**Key differences:** Requires thread-safe data structures and synchronization.

### Variation 5: A* Search (Informed Search)

**Description:** BFS/Dijkstra enhanced with heuristic function to guide search.

**When to use:** Finding shortest path when you have domain knowledge about target location.

**Implementation:**

```python
import heapq

def a_star(graph: Dict[int, List[tuple]], start: int, goal: int, 
           heuristic: callable) -> List[int]:
    """
    A* search using f(n) = g(n) + h(n)
    where g(n) is cost from start, h(n) is estimated cost to goal.
    """
    # Priority queue: (f_score, vertex, path)
    pq = [(heuristic(start, goal), start, [start])]
    visited = set()
    
    while pq:
        f_score, current, path = heapq.heappop(pq)
        
        if current == goal:
            return path
        
        if current in visited:
            continue
        
        visited.add(current)
        
        for neighbor, cost in graph.get(current, []):
            if neighbor not in visited:
                g_score = len(path)  # Cost from start
                h_score = heuristic(neighbor, goal)  # Estimated cost to goal
                f_score = g_score + h_score
                heapq.heappush(pq, (f_score, neighbor, path + [neighbor]))
    
    return []  # No path found
```

## Practice Problems

### Beginner

1. **Number of Islands** - Count connected components in a 2D grid
   - LeetCode #200

2. **Find if Path Exists in Graph** - Check if path exists between two nodes
   - LeetCode #1971

3. **Clone Graph** - Deep copy a graph
   - LeetCode #133

4. **All Paths From Source to Target** - Find all paths in DAG
   - LeetCode #797

### Intermediate

1. **Number of Provinces** - Count connected components in adjacency matrix
   - LeetCode #547

2. **Course Schedule** - Detect cycle in directed graph (topological sort)
   - LeetCode #207

3. **Pacific Atlantic Water Flow** - Multi-source BFS/DFS
   - LeetCode #417

4. **Find Eventual Safe States** - Find nodes not in cycles
   - LeetCode #802

5. **Minimum Number of Vertices to Reach All Nodes** - Graph analysis
   - LeetCode #1557

6. **Surrounded Regions** - Boundary DFS/BFS
   - LeetCode #130

### Advanced

1. **Bus Routes** - Shortest path in graph of buses
   - LeetCode #815

2. **Word Ladder** - BFS with transformations
   - LeetCode #127

3. **Alien Dictionary** - Topological sort with constraints
   - LeetCode #269 (Premium)

4. **Critical Connections in a Network** - Find bridges
   - LeetCode #1192

5. **Reconstruct Itinerary** - Eulerian path
   - LeetCode #332

## Real-World Applications

### Industry Use Cases

1. **Social Networks:** Graph algorithms power friend suggestions, influence detection, and community finding. Facebook uses graph traversal to suggest "People You May Know" and detect fake accounts through pattern analysis.

2. **Navigation Systems:** GPS and mapping applications use graphs where intersections are vertices and roads are edges. Dijkstra's algorithm finds shortest routes, while A* provides faster pathfinding with heuristics.

3. **Web Crawling:** Search engines use graph traversal to discover and index web pages. Pages are vertices, hyperlinks are edges. BFS ensures systematic coverage while avoiding duplicate crawling.

4. **Recommendation Systems:** Netflix, Amazon, and Spotify model user-item relationships as bipartite graphs. Graph algorithms find similar users, recommend products, and create personalized playlists.

5. **Compiler Optimization:** Compilers build control flow graphs and data dependency graphs to optimize code. Graph coloring assigns registers, and topological sort determines instruction order.

6. **Network Routing:** Internet routers use graph algorithms to find optimal paths for data packets. BGP (Border Gateway Protocol) uses modified Bellman-Ford for inter-domain routing.

### Popular Implementations

- **NetworkX (Python):** Comprehensive graph library for analysis and visualization
  - Uses: Research, data science, network analysis
  
- **Neo4j:** Graph database for storing and querying connected data
  - Uses: Fraud detection, knowledge graphs, recommendation engines

- **Apache Giraph:** Large-scale graph processing framework
  - Uses: Facebook's friend suggestions, LinkedIn's connection recommendations

- **Graph Neural Networks (GNNs):** PyTorch Geometric, DGL
  - Uses: Molecule property prediction, social network analysis, traffic prediction

### Practical Scenarios

- **Dependency Resolution:** Package managers (npm, pip) use topological sort to install dependencies in correct order
- **Deadlock Detection:** Operating systems model resource allocation as graphs to detect and prevent deadlocks
- **Circuit Design:** Electronic circuits are graphs where components are vertices and wires are edges
- **Epidemic Modeling:** Contact tracing applications use graphs to track disease spread and identify exposure chains

## Related Topics

### Prerequisites to Review

- **Arrays and Lists** - Understanding linear data structures helps with adjacency lists
- **Hash Maps** - Essential for efficient vertex lookups and visited tracking
- **Queues and Stacks** - Core data structures for BFS and DFS respectively
- **Recursion** - Understanding recursion is crucial for recursive DFS
- **Sets** - Used extensively for tracking visited vertices

### Next Steps

- **Topological Sort** - Ordering vertices in directed acyclic graphs (DAGs)
- **Shortest Path Algorithms** - Dijkstra's, Bellman-Ford, Floyd-Warshall
- **Minimum Spanning Trees** - Prim's and Kruskal's algorithms
- **Network Flow** - Max flow, min cut algorithms
- **Advanced Graph Algorithms** - Strongly connected components, articulation points, bridges
- **Dynamic Programming on Graphs** - Solving optimization problems on graphs

### Similar Concepts

- **Trees** - Special type of acyclic, connected graph
- **Tries** - Tree-like graph structure for string operations
- **Heaps** - Tree structure often represented as arrays
- **Disjoint Set (Union-Find)** - Efficient structure for tracking graph components

### Further Reading

- "Introduction to Algorithms" (CLRS) - Chapter 22: Elementary Graph Algorithms
- "Algorithm Design Manual" by Skiena - Comprehensive graph algorithm coverage
- "Grokking Algorithms" by Bhargava - Visual, beginner-friendly graph explanations
- Competitive Programming 4 - Advanced graph techniques and problem-solving
- [Graph Algorithms Visualizations](https://visualgo.net/en/dfsbfs) - Interactive DFS/BFS demonstrations
- [NetworkX Documentation](https://networkx.org/) - Python graph library reference
