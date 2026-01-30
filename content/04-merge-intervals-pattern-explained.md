# Merge Intervals Pattern

**Difficulty:** Medium
**Prerequisites:** Arrays, Sorting algorithms
**Estimated Reading Time:** 25 minutes

## Introduction

The Merge Intervals pattern is a powerful technique for solving problems that involve overlapping intervals, time ranges, or any scenarios where you need to combine or process ranges of values. This pattern works by sorting intervals and then systematically merging overlapping ones.

**Why it matters:** This pattern appears everywhere in real-world systems - calendar applications, meeting schedulers, resource allocation systems, network bandwidth management, and task scheduling. Companies like Google Calendar, Microsoft Outlook, and scheduling platforms rely heavily on these algorithms. Understanding this pattern is crucial for system design interviews and building production scheduling systems.

**Real-world analogy:** Imagine you're a conference room manager with multiple booking requests throughout the day. Someone books the room from 2PM-3PM, another from 2:30PM-4PM, and another from 5PM-6PM. Instead of tracking each individual booking, you merge the overlapping ones: the 2PM-3PM and 2:30PM-4PM bookings become one continuous 2PM-4PM block. This way, you know the room is occupied 2PM-4PM and 5PM-6PM, with a gap from 4PM-5PM when it's available. That's exactly what the merge intervals pattern does!

## Core Concepts

### Key Principles

1. **Sort by start time:** Always begin by sorting intervals based on their start time. This ensures we process intervals in chronological order.

2. **Merge overlapping intervals:** Two intervals overlap if one starts before the other ends. When they overlap, we merge them by extending the end time.

3. **Track current merged interval:** We maintain the "last merged" interval and compare each new interval against it.

4. **Mathematical overlap condition:** Intervals [a, b] and [c, d] overlap if c ≤ b (assuming a ≤ c after sorting).

### Essential Terms

- **Interval:** A pair [start, end] representing a time range or value range
- **Overlapping intervals:** Two intervals where one starts before the other ends
- **Merging:** Combining two overlapping intervals into one continuous interval
- **Non-overlapping:** Intervals that are completely separate with no overlap
- **Start time:** The beginning point of an interval
- **End time:** The ending point of an interval

### Visual Overview

```
Problem: Merge [[1,3], [2,6], [8,10], [15,18]]

Step 1: Sort by start time (already sorted)
[1,3]  [2,6]  [8,10]  [15,18]
  |      |       |        |
  └──────┘       |        |  (overlap: merge to [1,6])
         └───────┘        |  (no overlap: keep separate)
                 └────────┘  (no overlap: keep separate)

Visual Timeline:
0   2   4   6   8   10  12  14  16  18  20
|===|           |===|                       Before merge
    |=======|                               
|===========|   |===|       |===|           After merge

Result: [[1,6], [8,10], [15,18]]

Overlap Detection:
[1,3] and [2,6]:  3 ≥ 2 → overlap! → merge to [1, max(3,6)] = [1,6]
[1,6] and [8,10]: 6 < 8 → no overlap → keep both
[8,10] and [15,18]: 10 < 15 → no overlap → keep both
```

## How It Works

### Standard Merge Algorithm

1. **Sort intervals by start time**
   - Use any sorting algorithm (typically O(n log n))
   - After sorting, intervals are in chronological order

2. **Initialize result with first interval**
   - The first interval becomes our starting point for merging

3. **Iterate through remaining intervals**
   - For each interval, check if it overlaps with the last merged interval
   
4. **Check for overlap**
   - If current.start ≤ lastMerged.end → overlap exists
   - If no overlap → add current as new separate interval

5. **Merge when overlapping**
   - Extend lastMerged.end to max(lastMerged.end, current.end)
   - This handles cases where one interval completely contains another

6. **Return merged result**
   - All overlapping intervals have been combined

### Step-by-Step Example: Merging Intervals

Problem: Merge [[1,4], [2,5], [7,9], [8,10], [12,15]]

```
Initial: [[1,4], [2,5], [7,9], [8,10], [12,15]]

Step 1: Sort (already sorted by start time)
[[1,4], [2,5], [7,9], [8,10], [12,15]]

Step 2: Initialize
merged = [[1,4]]

Step 3: Process [2,5]
Current: [2,5]
Last merged: [1,4]
Check: 2 ≤ 4? YES → overlap
Merge: [1, max(4,5)] = [1,5]
merged = [[1,5]]

Step 4: Process [7,9]
Current: [7,9]
Last merged: [1,5]
Check: 7 ≤ 5? NO → no overlap
Add as new interval
merged = [[1,5], [7,9]]

Step 5: Process [8,10]
Current: [8,10]
Last merged: [7,9]
Check: 8 ≤ 9? YES → overlap
Merge: [7, max(9,10)] = [7,10]
merged = [[1,5], [7,10]]

Step 6: Process [12,15]
Current: [12,15]
Last merged: [7,10]
Check: 12 ≤ 10? NO → no overlap
Add as new interval
merged = [[1,5], [7,10], [12,15]]

Final Result: [[1,5], [7,10], [12,15]]
```

## How to Identify This Pattern

Merge Intervals is a common pattern in scheduling and interval problems. Here's how to spot it:

### Primary Indicators ✓

**Problem involves intervals or ranges**
- Intervals given as [start, end] pairs
- Time ranges, date ranges, number ranges
- Keywords: "intervals", "ranges", "periods", "time slots"
- Example: "Given intervals [[1,3], [2,6], [8,10]]..."

**Need to merge overlapping intervals**
- Combining intervals that overlap
- Consolidating time ranges
- Keywords: "merge", "combine", "overlapping"
- Example: "Merge all overlapping intervals"

**Finding conflicts or intersections**
- Checking if intervals overlap
- Finding common available times
- Detecting scheduling conflicts
- Keywords: "conflict", "intersection", "overlap", "available"
- Example: "Can you attend all meetings?"

**Counting concurrent events**
- How many events happen simultaneously
- Maximum overlapping intervals
- Resource allocation
- Keywords: "concurrent", "simultaneous", "at the same time", "maximum overlap"
- Example: "Minimum meeting rooms required"

**Inserting new interval**
- Adding interval to existing set
- Maintaining sorted, non-overlapping property
- Keywords: "insert", "add interval"
- Example: "Insert interval and merge if necessary"

**Scheduling or calendar problems**
- Meeting rooms
- Event bookings
- Resource scheduling
- Keywords: "meeting", "schedule", "calendar", "booking", "reservation"
- Example: "Book conference room"

### Common Problem Phrases 🔑

Watch for these exact phrases:
- "Merge intervals"
- "Merge overlapping intervals"
- "Insert interval"
- "Meeting rooms" / "Meeting rooms II"
- "Can attend all meetings"
- "Minimum number of rooms/resources"
- "Employee free time"
- "Interval intersection"
- "Non-overlapping intervals"
- "My Calendar I/II/III"
- "Remove intervals"
- "Minimum time to finish tasks"

### When NOT to Use Merge Intervals ✗

**Working with points, not ranges**
- Single values, not [start, end]
- → Use other appropriate pattern

**Contiguous subarray problems**
- Looking for consecutive elements
- → Use Sliding Window

**Sorted array problems without intervals**
- Finding pairs/triplets
- → Use Two Pointers

**Tree or graph interval problems**
- Complex hierarchical intervals
- → Use Interval Tree (advanced)

### Quick Decision Checklist ✅

Ask yourself:

1. **Are intervals [start, end] explicitly given?** → Merge Intervals
2. **Need to merge overlapping ranges?** → Merge Intervals
3. **Checking for scheduling conflicts?** → Merge Intervals
4. **Finding max concurrent events?** → Merge Intervals
5. **Problem mentions "meeting/booking/calendar"?** → Merge Intervals
6. **Need intersection of ranges?** → Merge Intervals

If YES to question 1 AND any other, it's Merge Intervals!

### Key Overlap Detection 🔍

**Two intervals [a, b] and [c, d] overlap if:**
```
c ≤ b (assuming a ≤ c after sorting by start time)
```

**Visual:**
```
Overlap:
[a========b]
      [c========d]
      ↑ c ≤ b, so they overlap

No Overlap:
[a====b]
           [c====d]
           ↑ c > b, no overlap
```

### Algorithm Pattern 📝

**Basic Merge Template:**
```python
intervals.sort(key=lambda x: x[0])  # Sort by start
merged = [intervals[0]]

for current in intervals[1:]:
    last = merged[-1]
    if current[0] <= last[1]:  # Overlap
        last[1] = max(last[1], current[1])  # Merge
    else:
        merged.append(current)  # No overlap
```

**Meeting Rooms (Min Heap):**
```python
intervals.sort(key=lambda x: x[0])
heap = [intervals[0][1]]  # End times

for interval in intervals[1:]:
    if interval[0] >= heap[0]:  # Room free
        heappop(heap)
    heappush(heap, interval[1])

return len(heap)  # Min rooms needed
```

### Visual Recognition 👁️

**If you can draw intervals on a timeline, it's likely Merge Intervals:**
```
Timeline: 0   2   4   6   8   10  12  14  16
          |===|
              |=======|
                      |===|
                                  |======|
          
These are intervals! → Use Merge Intervals pattern
```

### Example Pattern Matching 💡

**Problem: "Merge all overlapping intervals"**

Analysis:
- ✓ Intervals [start, end] given
- ✓ Explicitly asks to merge
- ✓ Classic merge intervals

**Verdict: USE MERGE INTERVALS** ✓

**Problem: "Minimum meeting rooms needed for meetings"**

Analysis:
- ✓ Intervals represent meetings
- ✓ Need to count concurrent meetings
- ✓ Use heap to track end times

**Verdict: USE MERGE INTERVALS (with heap)** ✓

**Problem: "Find longest substring with k distinct characters"**

Analysis:
- ✗ Not about intervals
- ✗ About contiguous substring

**Verdict: USE SLIDING WINDOW** (Not Merge Intervals) ✗

**Problem: "Can attend all meetings (no conflicts)?"**

Analysis:
- ✓ Meeting intervals given
- ✓ Check for overlaps
- ✓ Scheduling problem

**Verdict: USE MERGE INTERVALS** ✓

### Pattern vs Problem Type 📊

| Problem Type | Merge Intervals? | Notes |
|--------------|------------------|-------|
| Merge overlapping intervals | ✅ YES | Classic problem |
| Insert interval | ✅ YES | Merge after insert |
| Meeting rooms | ✅ YES | Check conflicts |
| Meeting rooms II | ✅ YES | Count concurrent (use heap) |
| Interval intersection | ✅ YES | Two pointers on sorted |
| Employee free time | ✅ YES | Merge then find gaps |
| My Calendar problems | ✅ YES | Overlap detection |
| Longest substring | ❌ NO | Use Sliding Window |
| Find pairs | ❌ NO | Use Two Pointers |

### Problem Variants 🔀

**Variant 1: Direct Merging**
- Merge all overlapping intervals
- Return non-overlapping list
- **Approach:** Sort by start, merge overlaps

**Variant 2: Conflict Detection (Boolean)**
- Can attend all meetings?
- Any conflicts?
- **Approach:** Sort, check consecutive pairs

**Variant 3: Count Concurrent Events**
- Minimum rooms needed
- Maximum simultaneous events
- **Approach:** Min heap of end times

**Variant 4: Find Gaps**
- Employee free time
- Available time slots
- **Approach:** Merge busy times, gaps are free

**Variant 5: Insert and Merge**
- Insert new interval
- Merge if overlapping
- **Approach:** Three phases (before, merge, after)

### Keywords Cheat Sheet 📝

**STRONG "Merge Intervals" Keywords:**
- intervals
- ranges
- [start, end]
- merge
- overlapping

**MODERATE Keywords:**
- meeting
- schedule
- calendar
- booking
- conflict
- concurrent
- rooms

**DOMAIN-Specific:**
- "employee free time"
- "minimum rooms"
- "can attend"
- "My Calendar"

### Red Flags 🚩

These suggest MERGE INTERVALS might NOT be right:
- No intervals/ranges mentioned → Other pattern
- "substring" or "subarray" → Sliding Window
- "linked list" → Fast & Slow or Reversal
- Point values, not ranges → Other pattern
- "sorted array" (without intervals) → Two Pointers

### Green Flags 🟢

STRONG indicators for MERGE INTERVALS:
- Intervals [start, end] in input
- "Merge overlapping"
- "Meeting rooms"
- "Schedule" or "Calendar"
- "Can attend all"
- "Minimum rooms needed"
- "Interval intersection"
- "Insert interval"
- Timeline/time slot problems

### Special Note: Sweep Line ⚠️

Some interval problems can also use "Sweep Line" algorithm:
- Events at points in time
- Count active intervals at each point
- Alternative to merge intervals for some problems

**When to consider Sweep Line:**
- Need to know state at every point
- Many intervals with queries
- Event-based processing

But for most interview problems, Merge Intervals pattern is simpler!



## Implementation

### Problem 1: Merge Intervals (LeetCode #56)

```python
from typing import List

def merge(intervals: List[List[int]]) -> List[List[int]]:
    """
    Merge all overlapping intervals.
    
    Args:
        intervals: List of [start, end] intervals
    
    Returns:
        List of merged intervals with no overlaps
    
    Time Complexity: O(n log n) - dominated by sorting
    Space Complexity: O(n) - for the output list (O(log n) if using in-place sort)
    """
    if not intervals:
        return []
    
    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    
    # Initialize merged list with first interval
    merged = [intervals[0]]
    
    # Process each remaining interval
    for current in intervals[1:]:
        # Get the last merged interval for comparison
        last_merged = merged[-1]
        
        # Check if current interval overlaps with last merged
        if current[0] <= last_merged[1]:
            # Overlapping: extend the end time
            # Use max to handle cases where one interval contains another
            last_merged[1] = max(last_merged[1], current[1])
        else:
            # Non-overlapping: add as new interval
            merged.append(current)
    
    return merged


# Usage Examples
print(merge([[1,3],[2,6],[8,10],[15,18]]))  
# Output: [[1,6],[8,10],[15,18]]

print(merge([[1,4],[4,5]]))  
# Output: [[1,5]] (touching intervals are merged)

print(merge([[1,4],[2,3]]))  
# Output: [[1,4]] (one interval contains another)
```

### Problem 2: Insert Interval (LeetCode #57)

```python
def insert(intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    """
    Insert a new interval and merge if necessary.
    
    Args:
        intervals: Sorted list of non-overlapping intervals
        newInterval: New interval to insert
    
    Returns:
        Updated list with newInterval merged
    
    Time Complexity: O(n) - single pass through intervals
    Space Complexity: O(n) - for result list
    """
    result = []
    i = 0
    n = len(intervals)
    
    # Phase 1: Add all intervals that come before newInterval
    while i < n and intervals[i][1] < newInterval[0]:
        result.append(intervals[i])
        i += 1
    
    # Phase 2: Merge all overlapping intervals with newInterval
    while i < n and intervals[i][0] <= newInterval[1]:
        # Expand newInterval to encompass overlapping intervals
        newInterval[0] = min(newInterval[0], intervals[i][0])
        newInterval[1] = max(newInterval[1], intervals[i][1])
        i += 1
    
    # Add the merged interval
    result.append(newInterval)
    
    # Phase 3: Add all remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1
    
    return result


# Usage Example
intervals = [[1,3],[6,9]]
newInterval = [2,5]
print(insert(intervals, newInterval))  
# Output: [[1,5],[6,9]]

intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]]
newInterval = [4,8]
print(insert(intervals, newInterval))  
# Output: [[1,2],[3,10],[12,16]]
```

### Problem 3: Interval List Intersections (LeetCode #986)

```python
def intervalIntersection(firstList: List[List[int]], 
                        secondList: List[List[int]]) -> List[List[int]]:
    """
    Find intersection of two interval lists.
    
    Args:
        firstList: First sorted list of intervals
        secondList: Second sorted list of intervals
    
    Returns:
        List of intersecting intervals
    
    Time Complexity: O(m + n) where m, n are lengths of lists
    Space Complexity: O(min(m, n)) for result in worst case
    """
    result = []
    i = j = 0
    
    while i < len(firstList) and j < len(secondList):
        # Find the intersection
        # Intersection start is max of the two starts
        start = max(firstList[i][0], secondList[j][0])
        # Intersection end is min of the two ends
        end = min(firstList[i][1], secondList[j][1])
        
        # Check if intersection exists (start <= end)
        if start <= end:
            result.append([start, end])
        
        # Move pointer of interval that ends first
        # This ensures we don't miss any potential intersections
        if firstList[i][1] < secondList[j][1]:
            i += 1
        else:
            j += 1
    
    return result


# Usage Example
firstList = [[0,2],[5,10],[13,23],[24,25]]
secondList = [[1,5],[8,12],[15,24],[25,26]]
print(intervalIntersection(firstList, secondList))
# Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
```

### Problem 4: Meeting Rooms (LeetCode #252)

```python
def canAttendMeetings(intervals: List[List[int]]) -> bool:
    """
    Determine if a person can attend all meetings.
    
    Args:
        intervals: List of meeting time intervals
    
    Returns:
        True if can attend all meetings, False otherwise
    
    Time Complexity: O(n log n) - sorting
    Space Complexity: O(1) - excluding sort space
    """
    if not intervals:
        return True
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    
    # Check each consecutive pair for overlap
    for i in range(1, len(intervals)):
        # If current meeting starts before previous ends
        if intervals[i][0] < intervals[i-1][1]:
            return False  # Conflict found
    
    return True


# Usage Example
meetings = [[0,30],[5,10],[15,20]]
print(canAttendMeetings(meetings))  # Output: False

meetings = [[7,10],[2,4]]
print(canAttendMeetings(meetings))  # Output: True
```

### Problem 5: Meeting Rooms II (LeetCode #253)

```python
import heapq

def minMeetingRooms(intervals: List[List[int]]) -> int:
    """
    Find minimum number of meeting rooms required.
    
    Args:
        intervals: List of meeting time intervals
    
    Returns:
        Minimum number of rooms needed
    
    Time Complexity: O(n log n) - sorting and heap operations
    Space Complexity: O(n) - heap can contain all meetings
    """
    if not intervals:
        return 0
    
    # Sort meetings by start time
    intervals.sort(key=lambda x: x[0])
    
    # Min heap to track end times of ongoing meetings
    # heap[0] always contains earliest ending meeting
    rooms = []
    heapq.heappush(rooms, intervals[0][1])
    
    for i in range(1, len(intervals)):
        # If earliest ending meeting has finished
        # we can reuse that room
        if intervals[i][0] >= rooms[0]:
            heapq.heappop(rooms)
        
        # Add current meeting's end time
        # (either reusing a room or taking a new one)
        heapq.heappush(rooms, intervals[i][1])
    
    # Size of heap = number of rooms needed
    return len(rooms)


# Usage Example
meetings = [[0,30],[5,10],[15,20]]
print(minMeetingRooms(meetings))  # Output: 2

meetings = [[7,10],[2,4]]
print(minMeetingRooms(meetings))  # Output: 1
```

### Problem 6: Non-overlapping Intervals (LeetCode #435)

```python
def eraseOverlapIntervals(intervals: List[List[int]]) -> int:
    """
    Find minimum number of intervals to remove to make rest non-overlapping.
    
    Args:
        intervals: List of intervals
    
    Returns:
        Minimum number of intervals to remove
    
    Time Complexity: O(n log n)
    Space Complexity: O(1)
    """
    if not intervals:
        return 0
    
    # Sort by end time (greedy: keep intervals that end earliest)
    intervals.sort(key=lambda x: x[1])
    
    count = 0
    end = intervals[0][1]
    
    for i in range(1, len(intervals)):
        # If current interval starts before previous ends
        if intervals[i][0] < end:
            # Overlap found, need to remove one
            count += 1
            # Keep the one that ends earlier (already have that in 'end')
        else:
            # No overlap, update end time
            end = intervals[i][1]
    
    return count


# Usage Example
intervals = [[1,2],[2,3],[3,4],[1,3]]
print(eraseOverlapIntervals(intervals))  # Output: 1

intervals = [[1,2],[1,2],[1,2]]
print(eraseOverlapIntervals(intervals))  # Output: 2
```

### Problem 7: Employee Free Time (LeetCode #759)

```python
def employeeFreeTime(schedule: List[List[List[int]]]) -> List[List[int]]:
    """
    Find common free time for all employees.
    
    Args:
        schedule: List of employee schedules, each is list of intervals
    
    Returns:
        List of free time intervals
    
    Time Complexity: O(n log n) where n is total number of intervals
    Space Complexity: O(n)
    """
    # Flatten all intervals into one list
    all_intervals = []
    for employee in schedule:
        all_intervals.extend(employee)
    
    # Sort by start time
    all_intervals.sort(key=lambda x: x[0])
    
    # Merge overlapping intervals (busy times)
    merged = [all_intervals[0]]
    for interval in all_intervals[1:]:
        if interval[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], interval[1])
        else:
            merged.append(interval)
    
    # Find gaps between merged intervals (free times)
    free_time = []
    for i in range(1, len(merged)):
        free_time.append([merged[i-1][1], merged[i][0]])
    
    return free_time


# Usage Example
schedule = [[[1,3],[6,7]], [[2,4]], [[2,5],[9,12]]]
print(employeeFreeTime(schedule))  
# Output: [[5,6],[7,9]]
```

## Complexity Analysis

### Time Complexity

**Merge Intervals:**
- **Sorting:** O(n log n) - dominates the complexity
- **Merging:** O(n) - single pass through sorted intervals
- **Overall:** O(n log n)

**Insert Interval:**
- **No sorting needed:** Input is already sorted
- **Single pass:** O(n) to find position and merge
- **Overall:** O(n)

**Interval Intersection:**
- **Two pointers:** O(m + n) where m and n are list lengths
- **No sorting needed:** Both lists already sorted
- **Overall:** O(m + n)

**Meeting Rooms II:**
- **Sorting:** O(n log n)
- **Heap operations:** O(n log n) - n insertions/deletions
- **Overall:** O(n log n)

**Why O(n log n)?**
The sorting step dominates because:
- Comparison-based sorting requires O(n log n) comparisons
- The merge step is only O(n) which is absorbed by the sorting complexity
- Total: O(n log n) + O(n) = O(n log n)

### Space Complexity

**Basic Merge:**
- **Output list:** O(n) in worst case (no merges)
- **Sorting:** O(log n) for typical implementations
- **Overall:** O(n)

**With Heap (Meeting Rooms II):**
- **Heap size:** O(n) worst case (all meetings overlap)
- **Overall:** O(n)

**In-place Sorting:**
- If modifying input is allowed, can achieve O(1) auxiliary space
- But output still requires O(n) in general

### Comparison with Alternatives

| Approach | Time | Space | When to Use |
|----------|------|-------|-------------|
| Sort + Merge | O(n log n) | O(n) | Standard interval problems |
| Interval Tree | O(n log n) build, O(log n) query | O(n) | Multiple queries on same set |
| Sweep Line | O(n log n) | O(n) | Event-based interval problems |
| Brute Force | O(n²) | O(1) | Only for tiny inputs |

## Examples

### Example 1: Basic Merge

```
Input: [[1,3],[2,6],[8,10],[15,18]]

Visual Timeline:
0   2   4   6   8   10  12  14  16  18
|===|
    |=======|
                |===|
                            |===|

Step-by-step:
Start: merged = []

1. Add [1,3]
   merged = [[1,3]]

2. Process [2,6]
   2 ≤ 3? YES (overlap)
   Merge: [1, max(3,6)] = [1,6]
   merged = [[1,6]]

3. Process [8,10]
   8 ≤ 6? NO (no overlap)
   Add new: merged = [[1,6], [8,10]]

4. Process [15,18]
   15 ≤ 10? NO (no overlap)
   Add new: merged = [[1,6], [8,10], [15,18]]

Output: [[1,6],[8,10],[15,18]]
```

### Example 2: Nested Intervals

```
Input: [[1,10],[2,3],[4,5],[6,7],[8,9]]

Visual:
|========================|  [1,10]
    |=|                     [2,3]
        |=|                 [4,5]
            |=|             [6,7]
                |=|         [8,9]

All intervals are inside [1,10]!

Processing:
merged = [[1,10]]

[2,3]: 2 ≤ 10? YES → merge → [1,10] (no change, already contained)
[4,5]: 4 ≤ 10? YES → merge → [1,10]
[6,7]: 6 ≤ 10? YES → merge → [1,10]
[8,9]: 8 ≤ 10? YES → merge → [1,10]

Output: [[1,10]]
```

### Example 3: Touching Intervals

```
Input: [[1,2],[2,3],[3,4],[4,5]]

Visual:
|=||=||=||=|
1 2 3 4 5

These intervals touch at boundaries!

Processing:
[1,2] + [2,3]: 2 ≤ 2? YES → merge → [1,3]
[1,3] + [3,4]: 3 ≤ 3? YES → merge → [1,4]
[1,4] + [4,5]: 4 ≤ 4? YES → merge → [1,5]

Output: [[1,5]]
```

### Example 4: Insert Interval with Multiple Merges

```
Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]]
       newInterval = [4,8]

Visual Before:
|=| |==| |=| |==|     |====|
1 2 3  5 6 7 8 10    12  16

Insert [4,8]:
                [4,8]
|=| |==| |=| |==|     |====|

Overlaps with [3,5], [6,7], [8,10]!

Phase 1: Add intervals before [4,8]
result = [[1,2]]

Phase 2: Merge overlapping intervals
[3,5]: overlap → newInterval = [3,8]
[6,7]: overlap → newInterval = [3,8] (no change)
[8,10]: overlap → newInterval = [3,10]

Add merged: result = [[1,2],[3,10]]

Phase 3: Add remaining
result = [[1,2],[3,10],[12,16]]

Output: [[1,2],[3,10],[12,16]]
```

## Edge Cases

### 1. Empty Input
**Scenario:** intervals = []
**Challenge:** No intervals to merge
**Solution:**
```python
if not intervals:
    return []
```

### 2. Single Interval
**Scenario:** intervals = [[1,5]]
**Challenge:** Nothing to merge with
**Solution:**
```python
# After sorting and initialization, loop doesn't execute
# Returns [[1,5]]
```

### 3. No Overlaps
**Scenario:** intervals = [[1,2],[3,4],[5,6]]
**Challenge:** All intervals are separate
**Solution:**
```python
# Each interval gets added as new
# Result: [[1,2],[3,4],[5,6]]
```

### 4. All Overlapping
**Scenario:** intervals = [[1,5],[2,6],[3,7],[4,8]]
**Challenge:** Everything merges into one
**Solution:**
```python
# Continuous merging
# Result: [[1,8]]
```

### 5. Duplicate Intervals
**Scenario:** intervals = [[1,3],[1,3],[1,3]]
**Challenge:** Same interval repeated
**Solution:**
```python
# All merge into one: [[1,3]]
```

### 6. One Interval Contains Others
**Scenario:** intervals = [[1,10],[2,3],[4,5],[6,7]]
**Challenge:** Nested intervals
**Solution:**
```python
# Use max in merge: max(10, 3), max(10, 5), etc.
# Result: [[1,10]]
```

### 7. Touching at Boundaries
**Scenario:** intervals = [[1,2],[2,3]]
**Challenge:** Should these merge?
**Solution:**
```python
# 2 ≤ 2 is TRUE, so they merge
# Result: [[1,3]]
# If you want them separate, change condition to <
```

### 8. Negative Numbers
**Scenario:** intervals = [[-5,-2],[-3,0],[1,4]]
**Challenge:** Negative start/end times
**Solution:**
```python
# Algorithm works the same
# Result: [[-5,0],[1,4]]
```

## Common Pitfalls

### ❌ Pitfall 1: Forgetting to Sort
**What happens:** Algorithm fails completely
**Why it's wrong:**
```python
# Wrong: No sorting
intervals = [[8,10],[1,3],[2,6]]
merged = [intervals[0]]  # Start with [8,10]
# Process [1,3]: 1 ≤ 10? YES → wrong merge!
```
**Correct approach:**
```python
intervals.sort(key=lambda x: x[0])  # ALWAYS sort first
merged = [intervals[0]]
```

### ❌ Pitfall 2: Not Using Max When Merging
**What happens:** Lose information when one interval contains another
**Why it's wrong:**
```python
# Wrong: Direct assignment
if current[0] <= last[1]:
    last[1] = current[1]  # What if current[1] < last[1]?
    
# Example: [[1,10],[2,3]]
# Would incorrectly change [1,10] to [1,3]
```
**Correct approach:**
```python
if current[0] <= last[1]:
    last[1] = max(last[1], current[1])  # Preserve maximum end
```

### ❌ Pitfall 3: Wrong Overlap Condition
**What happens:** Misses overlaps or incorrectly merges
**Why it's wrong:**
```python
# Wrong: Using < instead of <=
if current[0] < last[1]:  # Misses touching intervals
    
# [[1,2],[2,3]] wouldn't merge (2 < 2 is False)
```
**Correct approach:**
```python
if current[0] <= last[1]:  # Correctly handles touching
```

### ❌ Pitfall 4: Modifying While Iterating
**What happens:** Index errors and unexpected behavior
**Why it's wrong:**
```python
# Wrong: Modifying list during iteration
for i in range(len(intervals)):
    intervals.append(...)  # Dangerous!
```
**Correct approach:**
```python
# Use separate result list
merged = []
for interval in intervals:
    # Safe to build new list
```

### ❌ Pitfall 5: Not Handling Empty Input
**What happens:** Index errors
**Why it's wrong:**
```python
# Wrong: Assuming non-empty
intervals.sort()
merged = [intervals[0]]  # IndexError if empty!
```
**Correct approach:**
```python
if not intervals:
    return []
intervals.sort()
merged = [intervals[0]]
```

### ❌ Pitfall 6: Sorting by End Instead of Start
**What happens:** Algorithm doesn't work correctly
**Why it's wrong:**
```python
# Wrong sorting key
intervals.sort(key=lambda x: x[1])  # Sorting by end
# Breaks the merge logic!
```
**Correct approach:**
```python
intervals.sort(key=lambda x: x[0])  # Always sort by start
```

## Variations and Extensions

### Variation 1: Count Overlapping Intervals
**Description:** Instead of merging, count how many intervals overlap at each point
**When to use:** Finding maximum concurrent meetings
**Implementation:**
```python
def maxConcurrentIntervals(intervals):
    events = []
    for start, end in intervals:
        events.append((start, 1))   # +1 for start
        events.append((end, -1))    # -1 for end
    
    events.sort()
    
    max_concurrent = 0
    current = 0
    for time, delta in events:
        current += delta
        max_concurrent = max(max_concurrent, current)
    
    return max_concurrent
```

### Variation 2: Find Gaps Between Intervals
**Description:** Find free time between merged intervals
**When to use:** Finding available time slots
**Implementation:**
```python
def findGaps(intervals):
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    gaps = []
    
    for i in range(1, len(intervals)):
        if intervals[i][0] > intervals[i-1][1]:
            gaps.append([intervals[i-1][1], intervals[i][0]])
    
    return gaps
```

### Variation 3: Merge with Minimum Gap
**Description:** Merge intervals if gap between them is ≤ k
**When to use:** Grouping nearby events
**Implementation:**
```python
def mergeWithGap(intervals, k):
    if not intervals:
        return []
    
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        # Merge if gap ≤ k
        if current[0] - last[1] <= k:
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)
    
    return merged
```

### Variation 4: Weighted Interval Scheduling
**Description:** Select non-overlapping intervals with maximum total weight
**When to use:** Optimization problems with weighted intervals
**Approach:** Dynamic programming + sorting by end time

### Variation 5: Interval Tree
**Description:** Data structure for efficient interval queries
**When to use:** Multiple queries on static interval set
**Complexity:** O(n log n) build, O(log n + k) query where k is output size

## Practice Problems

### Beginner
1. **Merge Intervals (LeetCode #56)** - Classic merge problem
2. **Meeting Rooms (LeetCode #252)** - Check if person can attend all meetings
3. **Summary Ranges (LeetCode #228)** - Find continuous ranges

### Intermediate
1. **Insert Interval (LeetCode #57)** - Insert and merge in sorted list
2. **Interval List Intersections (LeetCode #986)** - Find intersections of two lists
3. **Non-overlapping Intervals (LeetCode #435)** - Minimum removals
4. **Meeting Rooms II (LeetCode #253)** - Minimum meeting rooms needed
5. **Merge Similar Items (LeetCode #2363)** - Merge and sum values
6. **Remove Covered Intervals (LeetCode #1288)** - Remove intervals covered by others

### Advanced
1. **Employee Free Time (LeetCode #759)** - Find common free time
2. **My Calendar I (LeetCode #729)** - Booking system without conflicts
3. **My Calendar II (LeetCode #731)** - Allow at most 2 overlaps
4. **My Calendar III (LeetCode #732)** - Count maximum k-bookings
5. **Data Stream as Disjoint Intervals (LeetCode #352)** - Dynamic interval merging
6. **Range Module (LeetCode #715)** - Track and query ranges

## Real-World Applications

### Industry Use Cases

1. **Calendar Applications (Google Calendar, Outlook)**
   - Merging meeting invites with same time
   - Finding free time slots
   - Detecting scheduling conflicts
   - Optimizing calendar display

2. **Resource Allocation Systems**
   - Hotel room bookings
   - Conference room scheduling
   - Equipment rental management
   - Parking space allocation

3. **Network Traffic Management**
   - Bandwidth allocation over time
   - Packet scheduling
   - QoS (Quality of Service) management
   - Network utilization analysis

4. **Video Streaming**
   - Buffering time ranges
   - Downloaded video segments
   - Ad insertion points
   - Content delivery optimization

5. **Task Scheduling**
   - CPU time slices in operating systems
   - Job scheduling in distributed systems
   - Build pipeline scheduling
   - Database query optimization

### Popular Implementations

- **Google Calendar API:** Uses interval merging for event consolidation
- **Kubernetes:** Job scheduling with resource intervals
- **Database Systems:** Query plan optimization with time ranges
- **Video Players:** Buffering and segment management
- **Cloud Platforms:** VM scheduling and resource allocation

### Practical Scenarios

- **Meeting scheduler:** "Find a 1-hour slot when all 5 people are free"
- **Hotel booking:** "Check if room is available for these dates"
- **Delivery routing:** "Optimize delivery time windows"
- **Conference planning:** "Schedule talks with no conflicts"
- **Medical appointments:** "Find next available slot with overlap handling"
- **Batch processing:** "Schedule jobs without resource conflicts"

## Related Topics

### Prerequisites to Review
- **Sorting algorithms** - Understanding time complexity and stability
- **Arrays and lists** - Basic manipulation and iteration
- **Greedy algorithms** - For optimization variants

### Next Steps
- **Sweep Line Algorithm** - Related technique for interval problems
- **Interval Tree** - Advanced data structure for interval queries
- **Segment Tree** - For range query problems
- **Line Sweep** - 2D geometry problems

### Similar Concepts
- **Union-Find** - Merging disjoint sets (conceptually similar)
- **Graph Connected Components** - Finding connected regions
- **Range Queries** - Overlaps with segment trees
- **Event Scheduling** - Practical application domain

### Further Reading
- [LeetCode Merge Intervals Study Guide](https://leetcode.com/tag/interval/)
- [Interval Scheduling - Wikipedia](https://en.wikipedia.org/wiki/Interval_scheduling)
- Introduction to Algorithms (CLRS) - Chapter on Greedy Algorithms
- [Sweep Line Algorithm Tutorial](https://www.topcoder.com/community/competitive-programming/tutorials/line-sweep-algorithms/)
- [Interval Tree - GeeksforGeeks](https://www.geeksforgeeks.org/interval-tree/)
