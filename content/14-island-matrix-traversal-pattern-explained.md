# Island (Matrix Traversal) Pattern

**Difficulty:** Beginner to Intermediate
**Prerequisites:** 2D Arrays (Matrices), Graph Traversal (DFS/BFS), Recursion, Stacks and Queues
**Estimated Reading Time:** 45 minutes

## Introduction

The Island Pattern, also known as Matrix Traversal, is a technique for exploring connected regions in a 2D grid by treating the grid as an implicit graph. Each cell is a vertex, and adjacent cells (horizontally or vertically) are connected by edges. This pattern is fundamental for solving problems involving regions, connected components, and flood-fill algorithms.

**Why it matters:** Island pattern problems appear everywhere in real-world applications: image processing (flood fill in Paint), game development (pathfinding, terrain analysis), map analysis (identifying land masses), and cellular automata. Mastering this pattern gives you the tools to solve a wide variety of grid-based problems efficiently.

**Real-world analogy:** Imagine looking at a map of Earth from above. Each pixel represents water (0) or land (1). Islands are groups of connected land pixels. When you want to count islands, you're essentially doing what a cartographer does: starting at any land pixel, tracing all connected land pixels to mark one complete island, then moving to find the next uncharted island. This is exactly how the island pattern works!

## Core Concepts

### Key Principles

1. **Grid as Graph:** A 2D matrix can be treated as an implicit graph where:
   - Each cell is a vertex
   - Two cells are connected (neighbors) if they're adjacent (usually 4-directional: up, down, left, right)
   - Some problems use 8-directional connectivity (including diagonals)

2. **Connected Components:** An "island" is a maximal connected component of cells with the same value (typically 1 for land, 0 for water).

3. **Traversal Strategies:**
   - **DFS (Depth-First Search):** Explore as deep as possible before backtracking - natural for recursive solutions
   - **BFS (Breadth-First Search):** Explore level by level - useful when distance/layers matter

4. **Visited Tracking:** Must mark visited cells to avoid infinite loops and double-counting. Can use:
   - Separate visited matrix
   - Modify original matrix (if allowed)
   - Set to track visited coordinates

### Essential Terms

- **Cell:** A single element in the 2D matrix at position (row, col)
- **Island:** A connected group of cells with the same value (typically 1)
- **Adjacent/Neighbor:** Cells that are directly next to each other (usually 4-directional)
- **Boundary:** Edge cells of the matrix that connect to the "outside"
- **Flood Fill:** Technique to mark/color all connected cells starting from a point
- **4-Directional:** Up, down, left, right movements (most common)
- **8-Directional:** Includes diagonals in addition to 4-directional
- **In-bounds:** Cell coordinates that exist within the matrix dimensions

### Visual Overview

```
4-Directional Neighbors:          8-Directional Neighbors:
        ↑                                ↖  ↑  ↗
        |                                  \ | /
    ← [Cell] →                          ←[Cell]→
        |                                  / | \
        ↓                                ↙  ↓  ↘

Example Grid (0 = water, 1 = land):

    0  1  2  3  4
  ┌─────────────┐
0 │ 1  1  0  0  1 │     Islands visible:
1 │ 1  1  0  0  0 │     - Island A: cells (0,0), (0,1), (1,0), (1,1)
2 │ 0  0  1  0  0 │     - Island B: cells (2,2)
3 │ 0  0  0  1  1 │     - Island C: cells (3,3), (3,4), (4,4)
4 │ 0  0  0  0  1 │     - Island D: cells (0,4)
  └─────────────┘
Total: 4 islands

Flood Fill Example (starting from (0,0)):
Before:          After marking Island A:
1  1  0  0  1    X  X  0  0  1
1  1  0  0  0    X  X  0  0  0
0  0  1  0  0    0  0  1  0  0
0  0  0  1  1    0  0  0  1  1
0  0  0  0  1    0  0  0  0  1
```

## How It Works

### DFS Approach - Step by Step

The DFS approach explores an island by going as deep as possible along each branch before backtracking.

**Algorithm:**

1. Iterate through each cell in the matrix
2. When you find an unvisited land cell (value = 1):
   - Increment island counter
   - Start DFS from this cell to mark entire island
3. DFS marks current cell as visited and recursively visits all unvisited land neighbors
4. Continue until all cells are processed

**Visual Walkthrough:**

```
Grid:
    0  1  2  3
  ┌──────────┐
0 │ 1  1  0  1 │
1 │ 0  1  0  0 │
2 │ 0  0  1  1 │
  └──────────┘

Step 1: Start at (0,0), value = 1 (land!)
        Island count = 1
        Start DFS from (0,0)

DFS from (0,0):
  Visit (0,0), mark as visited
  Check neighbors:
    - Up: out of bounds
    - Down: (1,0) = 0 (water), skip
    - Left: out of bounds
    - Right: (0,1) = 1 (land!), recurse
    
  DFS from (0,1):
    Visit (0,1), mark as visited
    Check neighbors:
      - Up: out of bounds
      - Down: (1,1) = 1 (land!), recurse
      - Left: (0,0) = visited, skip
      - Right: (0,2) = 0 (water), skip
    
    DFS from (1,1):
      Visit (1,1), mark as visited
      Check neighbors:
        - All are water, visited, or out of bounds
      Backtrack
    
    Backtrack
  Backtrack

Visited after Island 1:
    0  1  2  3
  ┌──────────┐
0 │ X  X  0  1 │
1 │ 0  X  0  0 │
2 │ 0  0  1  1 │
  └──────────┘

Step 2: Continue scanning, find (0,3) = 1 (land!)
        Island count = 2
        Start DFS from (0,3)

DFS from (0,3):
  Visit (0,3), mark as visited
  All neighbors are water, visited, or out of bounds
  Backtrack

Visited after Island 2:
    0  1  2  3
  ┌──────────┐
0 │ X  X  0  X │
1 │ 0  X  0  0 │
2 │ 0  0  1  1 │
  └──────────┘

Step 3: Continue scanning, find (2,2) = 1 (land!)
        Island count = 3
        Start DFS from (2,2)

DFS from (2,2):
  Visit (2,2), mark as visited
  Check neighbors:
    - Right: (2,3) = 1 (land!), recurse
    
  DFS from (2,3):
    Visit (2,3), mark as visited
    All neighbors are water, visited, or out of bounds
    Backtrack
  
  Backtrack

Final state:
    0  1  2  3
  ┌──────────┐
0 │ X  X  0  X │
1 │ 0  X  0  0 │
2 │ 0  0  X  X │
  └──────────┘

Result: 3 islands found
```

### BFS Approach - Step by Step

The BFS approach explores an island level by level using a queue.

**Algorithm:**

1. Iterate through each cell in the matrix
2. When you find an unvisited land cell:
   - Increment island counter
   - Add cell to queue and mark as visited
   - While queue is not empty:
     - Dequeue a cell
     - Add all unvisited land neighbors to queue
3. Continue until all cells are processed

**Visual Walkthrough:**

```
Grid:
    0  1  2
  ┌────────┐
0 │ 1  1  0 │
1 │ 1  0  0 │
  └────────┘

Step 1: Find (0,0) = 1 (land!)
        Island count = 1
        Queue = [(0,0)]
        Mark (0,0) as visited

BFS Level 0:
  Process (0,0)
  Add neighbors:
    - Right (0,1) = 1 → Queue = [(0,1)]
    - Down (1,0) = 1 → Queue = [(0,1), (1,0)]
  
  Visited: {(0,0), (0,1), (1,0)}

BFS Level 1:
  Process (0,1)
  Add neighbors:
    - Right (0,2) = 0 (water)
    - Down (1,1) = 0 (water)
    - Left (0,0) = visited
  No new cells added
  
  Process (1,0)
  Add neighbors:
    - Right (1,1) = 0 (water)
    - Up (0,0) = visited
    - Down: out of bounds
  No new cells added
  
  Queue = [] (empty)

Result: 1 island found, containing cells (0,0), (0,1), (1,0)
```

## Implementation

### DFS Recursive Implementation

```python
from typing import List

def num_islands_dfs(grid: List[List[str]]) -> int:
    """
    Count number of islands using DFS (recursive).
    
    An island is a group of connected 1s (land) surrounded by 0s (water).
    Connection is 4-directional (up, down, left, right).
    
    Args:
        grid: 2D list where '1' represents land and '0' represents water
        
    Returns:
        Number of islands found
        
    Time Complexity: O(m * n) where m = rows, n = cols
        - Visit each cell once
    Space Complexity: O(m * n)
        - Recursion stack in worst case (entire grid is one island)
        - If modifying grid in place: O(m * n) for recursion stack only
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    island_count = 0
    
    def dfs(r: int, c: int) -> None:
        """
        Perform DFS to mark all cells in current island.
        
        Args:
            r: Current row
            c: Current column
        """
        # Base cases: out of bounds or water or already visited
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            grid[r][c] == '0'):
            return
        
        # Mark current cell as visited by changing to '0'
        grid[r][c] = '0'
        
        # Explore all 4 directions
        dfs(r + 1, c)  # Down
        dfs(r - 1, c)  # Up
        dfs(r, c + 1)  # Right
        dfs(r, c - 1)  # Left
    
    # Iterate through every cell in the grid
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':  # Found unvisited land
                island_count += 1
                dfs(r, c)  # Mark entire island
    
    return island_count


# Example usage
grid1 = [
    ["1", "1", "0", "0", "0"],
    ["1", "1", "0", "0", "0"],
    ["0", "0", "1", "0", "0"],
    ["0", "0", "0", "1", "1"]
]
print(num_islands_dfs(grid1))  # Output: 3
```

### DFS Iterative Implementation (Using Stack)

```python
def num_islands_dfs_iterative(grid: List[List[str]]) -> int:
    """
    Count islands using iterative DFS with explicit stack.
    
    Args:
        grid: 2D list where '1' represents land and '0' represents water
        
    Returns:
        Number of islands
        
    Time Complexity: O(m * n)
    Space Complexity: O(m * n) for stack in worst case
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    island_count = 0
    
    # 4 directions: down, up, right, left
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                island_count += 1
                
                # Use stack for iterative DFS
                stack = [(r, c)]
                grid[r][c] = '0'  # Mark as visited
                
                while stack:
                    curr_r, curr_c = stack.pop()
                    
                    # Check all 4 directions
                    for dr, dc in directions:
                        new_r, new_c = curr_r + dr, curr_c + dc
                        
                        # If valid land cell, add to stack
                        if (0 <= new_r < rows and 
                            0 <= new_c < cols and 
                            grid[new_r][new_c] == '1'):
                            stack.append((new_r, new_c))
                            grid[new_r][new_c] = '0'  # Mark as visited
    
    return island_count
```

### BFS Implementation (Using Queue)

```python
from collections import deque

def num_islands_bfs(grid: List[List[str]]) -> int:
    """
    Count islands using BFS with queue.
    
    Args:
        grid: 2D list where '1' represents land and '0' represents water
        
    Returns:
        Number of islands
        
    Time Complexity: O(m * n)
    Space Complexity: O(min(m, n))
        - Queue size is bounded by the perimeter of an island
        - In worst case (entire grid is one island), queue size is O(min(m,n))
    """
    if not grid or not grid[0]:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    island_count = 0
    
    # 4 directions: down, up, right, left
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                island_count += 1
                
                # BFS using queue
                queue = deque([(r, c)])
                grid[r][c] = '0'  # Mark as visited
                
                while queue:
                    curr_r, curr_c = queue.popleft()
                    
                    # Check all 4 directions
                    for dr, dc in directions:
                        new_r, new_c = curr_r + dr, curr_c + dc
                        
                        # If valid land cell, add to queue
                        if (0 <= new_r < rows and 
                            0 <= new_c < cols and 
                            grid[new_r][new_c] == '1'):
                            queue.append((new_r, new_c))
                            grid[new_r][new_c] = '0'  # Mark as visited
    
    return island_count
```

### Helper Function for Bounds Checking

```python
def is_valid_cell(grid: List[List[any]], r: int, c: int, 
                  target_value: any = None) -> bool:
    """
    Check if a cell is within bounds and optionally matches a value.
    
    Args:
        grid: 2D matrix
        r: Row index
        c: Column index
        target_value: Optional value to check against
        
    Returns:
        True if cell is valid (and matches value if specified)
        
    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    rows, cols = len(grid), len(grid[0])
    
    # Check bounds
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return False
    
    # Check value if specified
    if target_value is not None and grid[r][c] != target_value:
        return False
    
    return True
```

### Code Explanation

**Key Design Decisions:**

1. **Direction Arrays:** Using `[(1,0), (-1,0), (0,1), (0,-1)]` makes it easy to iterate through neighbors without repeating code.

2. **In-place Modification:** Modifying `grid[r][c] = '0'` directly saves space for a separate visited matrix. If you can't modify the input, use a `Set` to track visited cells.

3. **When to Mark Visited:**
   - DFS Recursive: Mark at the start of function call
   - DFS/BFS Iterative: Mark when adding to stack/queue (not when popping)
   - This prevents adding the same cell multiple times

4. **Stack vs Queue:**
   - Stack (DFS): LIFO - explores depth first, uses less memory on average
   - Queue (BFS): FIFO - explores breadth first, better for shortest path problems

## Complexity Analysis

### Time Complexity

**All Approaches: O(m × n)** where m = number of rows, n = number of columns

**Why?**
- We visit each cell exactly once during the iteration: O(m × n)
- Each cell can be added to stack/queue at most once: O(m × n)
- Even though DFS/BFS explores neighbors, each cell's neighbors are checked exactly once
- Total: O(m × n) + O(m × n) = O(m × n)

**Detailed breakdown:**
```
Outer loops: m × n iterations
For each cell:
  If land (worst case all cells are land):
    - DFS/BFS visits this cell and all connected cells
    - But each cell marked visited immediately
    - Each cell processed exactly once across all DFS/BFS calls
    
Total: Each cell processed once = O(m × n)
```

### Space Complexity

**DFS Recursive: O(m × n)** in worst case
- Recursion call stack can go as deep as m × n (if entire grid is one island forming a snake pattern)
- Best case: O(1) if grid has no land
- Average case: O(min(m, n)) for balanced islands

**DFS Iterative: O(m × n)** in worst case
- Explicit stack can hold up to m × n elements
- Same analysis as recursive version

**BFS: O(min(m, n))** in worst case
- Queue holds the "perimeter" of an island
- For a square island, perimeter is roughly 4√(area)
- Maximum queue size is bounded by min(m, n) when island spans entire grid
- More space-efficient than DFS for wide, flat islands

**Memory optimization:** If you can modify the input grid, you don't need a separate visited structure. Otherwise, add O(m × n) for visited set.

### Comparison with Alternatives

| Approach | Time | Space | Pros | Cons |
|----------|------|-------|------|------|
| DFS Recursive | O(m×n) | O(m×n) | Clean code, easy to understand | Stack overflow risk, more memory |
| DFS Iterative | O(m×n) | O(m×n) | No stack overflow risk | More verbose code |
| BFS | O(m×n) | O(min(m,n)) | Better space efficiency, finds distances | Slightly more complex |
| Union-Find | O(m×n×α(m×n)) | O(m×n) | Good for dynamic connectivity | Overkill for static grids |

## Examples

### Example 1: Basic Island Counting

**Problem:** Count the number of islands in a grid.

**Input:**
```
grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
```

**Step-by-step trace:**

```
Initial grid:
  0 1 2 3 4
0 1 1 0 0 0
1 1 1 0 0 0
2 0 0 1 0 0
3 0 0 0 1 1

Scan (0,0): value='1' → Island #1 found
  DFS marks: (0,0), (0,1), (1,0), (1,1)
  
Grid after Island 1:
  0 1 2 3 4
0 0 0 0 0 0
1 0 0 0 0 0
2 0 0 1 0 0
3 0 0 0 1 1

Scan (2,2): value='1' → Island #2 found
  DFS marks: (2,2)
  
Grid after Island 2:
  0 1 2 3 4
0 0 0 0 0 0
1 0 0 0 0 0
2 0 0 0 0 0
3 0 0 0 1 1

Scan (3,3): value='1' → Island #3 found
  DFS marks: (3,3), (3,4)
  
Final grid (all islands marked):
  0 1 2 3 4
0 0 0 0 0 0
1 0 0 0 0 0
2 0 0 0 0 0
3 0 0 0 0 0

Answer: 3 islands
```

### Example 2: Largest Island (Max Area)

**Problem:** Find the size of the largest island.

**Input:**
```
grid = [
  [1,0,0,1,0],
  [1,0,0,0,0],
  [0,0,1,0,1],
  [0,0,0,1,1]
]
```

**Solution:**

```python
def max_area_island(grid: List[List[int]]) -> int:
    """
    Find the area of the largest island.
    
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    """
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    max_area = 0
    
    def dfs(r: int, c: int) -> int:
        """Returns area of island starting from (r, c)."""
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            grid[r][c] == 0):
            return 0
        
        grid[r][c] = 0  # Mark visited
        
        # Count current cell + all connected cells
        area = 1
        area += dfs(r + 1, c)
        area += dfs(r - 1, c)
        area += dfs(r, c + 1)
        area += dfs(r, c - 1)
        
        return area
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                max_area = max(max_area, dfs(r, c))
    
    return max_area

# Trace:
# Island at (0,0): size = 2 (cells (0,0), (1,0))
# Island at (0,3): size = 1 (cell (0,3))
# Island at (2,2): size = 1 (cell (2,2))
# Island at (2,4): size = 3 (cells (2,4), (3,3), (3,4))
# Answer: 3 (largest island)
```

### Example 3: Flood Fill

**Problem:** Change color of a region starting from a point.

**Input:**
```
image = [[1,1,1],
         [1,1,0],
         [1,0,1]]
sr = 1, sc = 1 (starting row, col)
new_color = 2
```

**Solution:**

```python
def flood_fill(image: List[List[int]], sr: int, sc: int, 
               new_color: int) -> List[List[int]]:
    """
    Flood fill starting from (sr, sc) with new_color.
    
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    """
    if not image or image[sr][sc] == new_color:
        return image
    
    rows, cols = len(image), len(image[0])
    original_color = image[sr][sc]
    
    def dfs(r: int, c: int) -> None:
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            image[r][c] != original_color):
            return
        
        image[r][c] = new_color
        
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    dfs(sr, sc)
    return image

# Trace:
# Start at (1,1), original_color = 1
# DFS visits: (1,1) → (0,1) → (0,0) → (0,2) → (1,0) → (2,0)
# All cells with value 1 connected to (1,1) become 2
# 
# Result:
# [[2,2,2],
#  [2,2,0],
#  [2,0,1]]
```

### Example 4: Number of Closed Islands

**Problem:** Count islands not touching the boundary.

**Input:**
```
grid = [[1,1,1,1,1,1,1,0],
        [1,0,0,0,0,1,1,0],
        [1,0,1,0,1,1,1,0],
        [1,0,0,0,0,1,0,1],
        [1,1,1,1,1,1,1,0]]
```

**Solution:**

```python
def closed_island(grid: List[List[int]]) -> int:
    """
    Count islands that don't touch the boundary.
    
    Strategy:
    1. First, eliminate all islands touching the boundary
    2. Then count remaining islands
    
    Time Complexity: O(m * n)
    Space Complexity: O(m * n)
    """
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    
    def dfs(r: int, c: int) -> None:
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            grid[r][c] == 1):
            return
        
        grid[r][c] = 1  # Mark as water
        
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    # Eliminate islands touching boundaries
    # Top and bottom rows
    for c in range(cols):
        if grid[0][c] == 0:
            dfs(0, c)
        if grid[rows-1][c] == 0:
            dfs(rows-1, c)
    
    # Left and right columns
    for r in range(rows):
        if grid[r][0] == 0:
            dfs(r, 0)
        if grid[r][cols-1] == 0:
            dfs(r, cols-1)
    
    # Count remaining islands (closed islands)
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                count += 1
                dfs(r, c)
    
    return count

# Answer: 2 (only internal islands counted)
```

## Edge Cases

### 1. Empty Grid

**Scenario:** Grid is None, empty list, or has no rows/columns.

**Challenge:** Causes index errors or infinite loops.

**Solution:**

```python
def num_islands_safe(grid: List[List[str]]) -> int:
    """Handle empty grid gracefully."""
    # Check for None or empty grid
    if not grid or not grid[0]:
        return 0
    
    # Regular algorithm...
```

### 2. Single Cell Grid

**Scenario:** 1×1 grid with single cell.

**Challenge:** Boundary conditions must work correctly.

**Solution:**
```python
# Grid = [[1]]
# Expected output: 1 island

# Grid = [[0]]
# Expected output: 0 islands

# Both cases handled naturally by the algorithm
```

### 3. All Water or All Land

**Scenario:** Every cell is the same value.

**Challenge:** Edge case for counting logic.

**Solution:**
```python
# All water: [[0,0], [0,0]]
# Expected: 0 islands

# All land: [[1,1], [1,1]]
# Expected: 1 island (entire grid is one connected component)
```

### 4. Diagonal Connections

**Scenario:** Problem asks for 8-directional connectivity.

**Challenge:** Must check diagonal neighbors in addition to orthogonal.

**Solution:**

```python
def num_islands_8_directional(grid: List[List[int]]) -> int:
    """Count islands with 8-directional connectivity."""
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    
    # 8 directions: 4 orthogonal + 4 diagonal
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),      # orthogonal
        (1, 1), (1, -1), (-1, 1), (-1, -1)     # diagonal
    ]
    
    def dfs(r: int, c: int) -> None:
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            grid[r][c] == 0):
            return
        
        grid[r][c] = 0
        
        for dr, dc in directions:
            dfs(r + dr, c + dc)
    
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                count += 1
                dfs(r, c)
    
    return count
```

### 5. Cannot Modify Input Grid

**Scenario:** Must preserve original grid.

**Challenge:** Need separate visited tracking.

**Solution:**

```python
def num_islands_preserve_grid(grid: List[List[str]]) -> int:
    """Count islands without modifying the input grid."""
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    visited = set()  # Track visited cells
    count = 0
    
    def dfs(r: int, c: int) -> None:
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            grid[r][c] == '0' or (r, c) in visited):
            return
        
        visited.add((r, c))
        
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1' and (r, c) not in visited:
                count += 1
                dfs(r, c)
    
    return count
```

## Common Pitfalls

### ❌ Pitfall 1: Not Checking Bounds Before Accessing Cell

**What happens:** IndexError or accessing wrong cells.

```python
# WRONG - Check value before bounds
def dfs_wrong(r, c):
    if grid[r][c] == '0':  # Error if r,c out of bounds!
        return
```

**Why it's wrong:** Accessing `grid[r][c]` before checking if r,c are valid causes crashes.

**Correct approach:**

```python
# CORRECT - Check bounds first
def dfs_correct(r, c):
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return
    if grid[r][c] == '0':
        return
    # Rest of logic...
```

### ❌ Pitfall 2: Not Marking Cells as Visited

**What happens:** Infinite recursion/loops, counting same island multiple times.

```python
# WRONG - No visited tracking
def dfs_wrong(r, c):
    if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
        return
    # Missing: grid[r][c] = '0'
    dfs_wrong(r+1, c)  # Can revisit same cell infinitely!
```

**Correct approach:**

```python
# CORRECT - Mark as visited immediately
def dfs_correct(r, c):
    if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
        return
    grid[r][c] = '0'  # Mark visited BEFORE recursing
    dfs_correct(r+1, c)
```

### ❌ Pitfall 3: Marking Visited at Wrong Time in BFS

**What happens:** Same cell added to queue multiple times.

```python
# WRONG - Mark when processing
def bfs_wrong(start_r, start_c):
    queue = deque([(start_r, start_c)])
    
    while queue:
        r, c = queue.popleft()
        grid[r][c] = '0'  # Too late! Already in queue multiple times
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc) and grid[nr][nc] == '1':
                queue.append((nr, nc))
```

**Correct approach:**

```python
# CORRECT - Mark when adding to queue
def bfs_correct(start_r, start_c):
    queue = deque([(start_r, start_c)])
    grid[start_r][start_c] = '0'  # Mark immediately
    
    while queue:
        r, c = queue.popleft()
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc) and grid[nr][nc] == '1':
                grid[nr][nc] = '0'  # Mark before adding
                queue.append((nr, nc))
```

### ❌ Pitfall 4: Counting Islands Before DFS/BFS

**What happens:** Count incremented too many times.

```python
# WRONG - Count for every land cell
count = 0
for r in range(rows):
    for c in range(cols):
        if grid[r][c] == '1':
            count += 1  # Counts every cell, not islands!
```

**Correct approach:**

```python
# CORRECT - Count only when starting new island
count = 0
for r in range(rows):
    for c in range(cols):
        if grid[r][c] == '1':  # Found new island
            count += 1
            dfs(r, c)  # Mark entire island
```

### ❌ Pitfall 5: Forgetting to Handle Boundary Islands (Closed Islands Problem)

**What happens:** Counting islands that touch boundary when shouldn't.

```python
# WRONG - Counts all islands
def closed_island_wrong(grid):
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                count += 1  # Counts boundary islands too!
                dfs(r, c)
    return count
```

**Correct approach:**

```python
# CORRECT - Eliminate boundary islands first
def closed_island_correct(grid):
    # First pass: eliminate boundary islands
    for c in range(cols):
        if grid[0][c] == 0: dfs(0, c)
        if grid[rows-1][c] == 0: dfs(rows-1, c)
    for r in range(rows):
        if grid[r][0] == 0: dfs(r, 0)
        if grid[r][cols-1] == 0: dfs(r, cols-1)
    
    # Second pass: count remaining islands
    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                count += 1
                dfs(r, c)
    return count
```

## Variations and Extensions

### Variation 1: Count Distinct Islands (Shape Matters)

**Description:** Two islands are the same if one can be translated to match the other.

**When to use:** When island shape/pattern matters, not just connectivity.

**Implementation:**

```python
def num_distinct_islands(grid: List[List[int]]) -> int:
    """
    Count islands with unique shapes.
    
    Strategy: Record relative path taken during DFS as a signature.
    """
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    unique_shapes = set()
    
    def dfs(r: int, c: int, r0: int, c0: int, shape: List[tuple]) -> None:
        """Record relative coordinates from origin (r0, c0)."""
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            grid[r][c] == 0):
            return
        
        grid[r][c] = 0
        shape.append((r - r0, c - c0))  # Relative position
        
        dfs(r+1, c, r0, c0, shape)
        dfs(r-1, c, r0, c0, shape)
        dfs(r, c+1, r0, c0, shape)
        dfs(r, c-1, r0, c0, shape)
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                shape = []
                dfs(r, c, r, c, shape)
                unique_shapes.add(tuple(sorted(shape)))
    
    return len(unique_shapes)
```

### Variation 2: Island Perimeter

**Description:** Calculate the total perimeter of all islands.

**When to use:** Need boundary measurements, not just counts.

**Implementation:**

```python
def island_perimeter(grid: List[List[int]]) -> int:
    """
    Calculate total perimeter of islands.
    
    Key insight: Each land cell contributes 4 to perimeter,
    but subtract 2 for each shared edge with another land cell.
    
    Time Complexity: O(m * n)
    Space Complexity: O(1)
    """
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    perimeter = 0
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                # Start with 4 sides
                perimeter += 4
                
                # Subtract for each adjacent land cell
                if r > 0 and grid[r-1][c] == 1:  # Top neighbor
                    perimeter -= 2
                if c > 0 and grid[r][c-1] == 1:  # Left neighbor
                    perimeter -= 2
    
    return perimeter
```

### Variation 3: Making Island (Minimum Days to Disconnect)

**Description:** Find minimum number of days (cell removals) to disconnect island.

**When to use:** Graph disconnection problems, critical points.

**Key insight:** Check if removing each land cell disconnects the island.

### Variation 4: Largest Region by Color

**Description:** Find largest connected region of same color in multi-color grid.

**When to use:** Image segmentation, region analysis.

**Implementation:**

```python
def largest_color_region(grid: List[List[int]]) -> int:
    """
    Find size of largest region with same color.
    
    Works with multiple colors (not just 0 and 1).
    """
    if not grid:
        return 0
    
    rows, cols = len(grid), len(grid[0])
    visited = set()
    max_size = 0
    
    def dfs(r: int, c: int, color: int) -> int:
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            (r, c) in visited or grid[r][c] != color):
            return 0
        
        visited.add((r, c))
        size = 1
        
        size += dfs(r+1, c, color)
        size += dfs(r-1, c, color)
        size += dfs(r, c+1, color)
        size += dfs(r, c-1, color)
        
        return size
    
    for r in range(rows):
        for c in range(cols):
            if (r, c) not in visited:
                max_size = max(max_size, dfs(r, c, grid[r][c]))
    
    return max_size
```

## Practice Problems

### Beginner

1. **Number of Islands** - Classic island counting problem
   - LeetCode #200

2. **Flood Fill** - Change region color from starting point
   - LeetCode #733

3. **Max Area of Island** - Find largest island size
   - LeetCode #695

4. **Island Perimeter** - Calculate total perimeter
   - LeetCode #463

### Intermediate

1. **Number of Closed Islands** - Islands not touching boundary
   - LeetCode #1254

2. **Number of Enclaves** - Land cells that can't reach boundary
   - LeetCode #1020

3. **Surrounded Regions** - Capture regions surrounded by X
   - LeetCode #130

4. **Pacific Atlantic Water Flow** - Cells reaching both oceans
   - LeetCode #417

5. **Making A Large Island** - Maximum island by adding one land cell
   - LeetCode #827

### Advanced

1. **Number of Distinct Islands** - Count unique island shapes
   - LeetCode #694 (Premium)

2. **Number of Distinct Islands II** - With rotations/reflections
   - LeetCode #711 (Premium)

3. **Shortest Bridge** - Minimum cells to connect two islands
   - LeetCode #934

4. **Minimum Days to Disconnect Island** - Find articulation points
   - LeetCode #1568

## Real-World Applications

### Industry Use Cases

1. **Image Processing:** Flood fill algorithms power the "bucket fill" tool in image editors like Photoshop and GIMP. Connected component analysis identifies objects in computer vision.

2. **Geographic Information Systems (GIS):** Identifying land masses, water bodies, and terrain features on maps. Used in environmental monitoring and urban planning.

3. **Game Development:** Pathfinding in tile-based games, terrain generation, fog of war systems, and region detection for strategy games.

4. **Medical Imaging:** Tumor detection in MRI/CT scans by identifying connected regions of abnormal tissue. Organ segmentation in diagnostic software.

5. **Circuit Board Design:** Identifying connected components in PCB layouts, checking for shorts and open circuits.

### Popular Implementations

- **OpenCV:** Connected component labeling for image analysis
  - `cv2.connectedComponents()` uses optimized union-find

- **scikit-image:** Region analysis and segmentation
  - `skimage.measure.label()` for connected components

- **Unity/Unreal Engine:** Tile-based map systems and navmesh generation
  - Flood fill for area detection and pathfinding

- **GDAL (Geospatial Data Abstraction Library):** Raster analysis
  - Identifying contiguous regions in satellite imagery

### Practical Scenarios

- **Pixel Art Tools:** Fill bucket tool uses flood fill algorithm
- **Minesweeper Game:** Revealing connected empty cells
- **Cellular Automata:** Conway's Game of Life uses neighbor counting on grids
- **Forest Fire Simulation:** Modeling fire spread through connected vegetation
- **Network Analysis:** Identifying disconnected network segments

## Related Topics

### Prerequisites to Review

- **2D Arrays (Matrices)** - Understanding how to navigate and manipulate grids
- **Graph Traversal** - DFS and BFS form the foundation of island pattern
- **Recursion** - Essential for recursive DFS implementations
- **Stacks and Queues** - Data structures for iterative traversal

### Next Steps

- **Union-Find (Disjoint Set)** - Alternative approach for connected components
- **Topological Sort** - Ordering elements in directed graphs
- **Shortest Path Algorithms** - Dijkstra, A* for weighted graphs
- **Dynamic Programming on Grids** - Optimizing path problems
- **Advanced Graph Algorithms** - Articulation points, strongly connected components

### Similar Concepts

- **Flood Fill Algorithm** - Specific application of island pattern
- **Connected Components in Graphs** - Same concept, different representation
- **Region Growing** - Image segmentation technique
- **Maze Solving** - Path finding in grids with obstacles

### Further Reading

- "Introduction to Algorithms" (CLRS) - Chapter 20: Connected Components
- "The Algorithm Design Manual" - Grid and graph problems
- [GeeksforGeeks: Islands in a Graph](https://www.geeksforgeeks.org/find-number-of-islands/) - Comprehensive examples
- [VisuAlgo: Graph Traversal](https://visualgo.net/en/dfsbfs) - Interactive visualization
- [LeetCode Explore: Graph](https://leetcode.com/explore/learn/card/graph/) - Structured learning path
