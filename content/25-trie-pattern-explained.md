# Trie (Prefix Tree) Pattern

**Difficulty:** Intermediate
**Prerequisites:** Trees, Hash Maps, Recursion, String manipulation
**Estimated Reading Time:** 40 minutes

## Introduction

A Trie (pronounced "try"), also called a prefix tree or digital tree, is a specialized tree-like data structure used for efficiently storing and retrieving strings. Each path from the root to a node represents a prefix, and paths to leaf nodes (or marked nodes) represent complete words.

**Why it matters:** Tries are the go-to data structure for problems involving string prefixes, autocomplete systems, spell checkers, and IP routing tables. They provide O(m) time complexity for search, insert, and delete operations (where m is the key length), regardless of how many keys are stored. This makes them significantly faster than hash tables for prefix-based operations. Companies like Google use Trie-based structures for search autocomplete, and network routers use them for IP address lookup.

**Real-world analogy:** Think of a Trie like a filing cabinet organized by word prefixes. To find all words starting with "cat", you don't search through every word—instead, you go to drawer 'c', then folder 'a', then subfolder 't', and all words in that subfolder start with "cat". Each letter leads you down a specific path, and at the end of complete words, you have a flag marking "this is a valid word". This hierarchical organization makes finding words with common prefixes incredibly fast!

## Core Concepts

### Key Principles

1. **Prefix-Based Organization:** Words sharing common prefixes share the same path in the Trie, maximizing space efficiency for related words.

2. **Character-by-Character Navigation:** Each node typically has up to 26 children (for lowercase English letters), representing possible next characters.

3. **End-of-Word Marking:** Nodes are marked to indicate where complete words end, distinguishing "car" from "card".

4. **Space-Time Tradeoff:** Tries use more space than simple arrays but provide faster prefix operations than hash tables.

### Essential Terms

- **Trie Node:** A node containing children pointers (usually an array or hash map) and an end-of-word flag
- **Root:** The empty string; all words start from here
- **Prefix:** Any path from root to a node represents a prefix
- **End-of-Word Flag:** Boolean marker indicating a complete word ends at this node
- **Children:** Array or map of pointers to child nodes (one per possible character)
- **Depth:** The number of edges from root to a node (equals the prefix length)

### Visual Overview

```
Trie containing: ["cat", "car", "card", "dog", "dodge"]

                    root
                   /    \
                  c      d
                  |      |
                  a      o
                 / \     |
                t   r    g
               [*] [*]   |
                    |    e
                    d   [*]
                   [*]   |
                         e
                        [*]

Legend:
[*] = end of word marker
Each edge represents a character
Paths from root represent prefixes:
- root→c→a→t = "cat" (complete word)
- root→c→a→r = "car" (complete word)
- root→c→a→r→d = "card" (complete word)
- root→d→o→g = "dog" (complete word)
- root→d→o→g→e = "doge" (autocorrect fail!)

Note: "ca" is a prefix but not a complete word (no [*] marker)
```

## How It Works

### Building a Trie

**Step 1: Start at Root**
- Begin at the empty root node
- Root represents the empty prefix ""

**Step 2: Insert Character by Character**
- For each character in the word:
  - If a child node for that character exists, move to it
  - If not, create a new child node and move to it

**Step 3: Mark End of Word**
- After inserting all characters, mark the final node as end-of-word

**Step 4: Handle Multiple Words**
- Words with common prefixes share nodes
- Only diverge at the point where they differ

### Detailed Walkthrough Example

**Task:** Insert ["cat", "car", "card"] into an empty Trie

```
Initial State: Just root node
root: {}

Step 1: Insert "cat"
-------------------
Insert 'c':
  root has no child 'c' → create node
  root: {c: Node}
  Move to node for 'c'

Insert 'a':
  'c' node has no child 'a' → create node
  c: {a: Node}
  Move to node for 'a'

Insert 't':
  'a' node has no child 't' → create node
  a: {t: Node}
  Move to node for 't'

End of word:
  Mark 't' node as end-of-word
  t: {is_end: True}

Trie after "cat":
root → c → a → t[*]

Step 2: Insert "car"
-------------------
Insert 'c':
  root has child 'c' → reuse existing node
  Move to 'c' node

Insert 'a':
  'c' node has child 'a' → reuse existing node
  Move to 'a' node

Insert 'r':
  'a' node has no child 'r' → create node
  a: {t: Node, r: Node}
  Move to node for 'r'

End of word:
  Mark 'r' node as end-of-word
  r: {is_end: True}

Trie after "car":
root → c → a → t[*]
            ↓
            r[*]

Step 3: Insert "card"
--------------------
Insert 'c':
  root has child 'c' → reuse
  Move to 'c' node

Insert 'a':
  'c' has child 'a' → reuse
  Move to 'a' node

Insert 'r':
  'a' has child 'r' → reuse
  Move to 'r' node

Insert 'd':
  'r' node has no child 'd' → create node
  r: {d: Node, is_end: True}
  Move to node for 'd'

End of word:
  Mark 'd' node as end-of-word
  d: {is_end: True}

Final Trie:
root → c → a → t[*]
            ↓
            r[*] → d[*]

Key Observations:
- "cat", "car", "card" share the prefix "ca"
- Only 7 nodes needed for 3 words (11 characters total)
- Saved 4 nodes through prefix sharing!
```

## Implementation

### Python Implementation - Complete Trie

```python
from typing import List, Optional, Tuple

class TrieNode:
    """
    Node in a Trie data structure.
    
    Attributes:
        children: Dictionary mapping characters to child TrieNodes
        is_end_of_word: Flag indicating if a word ends at this node
        word: Optional storage of the complete word (useful for some problems)
    """
    
    def __init__(self):
        self.children = {}  # char -> TrieNode
        self.is_end_of_word = False
        self.word = None  # Optional: store complete word at end nodes


class Trie:
    """
    Trie (Prefix Tree) implementation for efficient string storage and retrieval.
    
    Supports:
    - insert(word): Add a word to the trie
    - search(word): Check if exact word exists
    - startsWith(prefix): Check if any word has this prefix
    - delete(word): Remove a word from the trie
    
    Time Complexity:
        - Insert: O(m) where m is word length
        - Search: O(m)
        - StartsWith: O(m)
        - Delete: O(m)
    
    Space Complexity: O(ALPHABET_SIZE * N * M)
        where N is number of words, M is average word length
    """
    
    def __init__(self):
        """Initialize Trie with empty root node."""
        self.root = TrieNode()
    
    def insert(self, word: str) -> None:
        """
        Insert a word into the trie.
        
        Args:
            word: String to insert
            
        Example:
            >>> trie = Trie()
            >>> trie.insert("apple")
            >>> trie.search("apple")
            True
        """
        node = self.root
        
        # Traverse/create path for each character
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        # Mark end of word
        node.is_end_of_word = True
        node.word = word  # Optionally store the complete word
    
    def search(self, word: str) -> bool:
        """
        Search for exact word match in trie.
        
        Args:
            word: String to search for
            
        Returns:
            True if word exists, False otherwise
            
        Example:
            >>> trie = Trie()
            >>> trie.insert("apple")
            >>> trie.search("apple")
            True
            >>> trie.search("app")
            False
        """
        node = self.root
        
        # Try to follow path for each character
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        
        # Word exists only if we end at a marked node
        return node.is_end_of_word
    
    def startsWith(self, prefix: str) -> bool:
        """
        Check if any word in trie starts with given prefix.
        
        Args:
            prefix: Prefix string to check
            
        Returns:
            True if prefix exists, False otherwise
            
        Example:
            >>> trie = Trie()
            >>> trie.insert("apple")
            >>> trie.startsWith("app")
            True
            >>> trie.startsWith("ban")
            False
        """
        node = self.root
        
        # Try to follow path for each character
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        
        # If we successfully traversed the prefix, it exists
        return True
    
    def delete(self, word: str) -> bool:
        """
        Delete a word from the trie.
        
        Args:
            word: Word to delete
            
        Returns:
            True if word was deleted, False if word didn't exist
            
        Example:
            >>> trie = Trie()
            >>> trie.insert("apple")
            >>> trie.delete("apple")
            True
            >>> trie.search("apple")
            False
        """
        def _delete_helper(node: TrieNode, word: str, index: int) -> bool:
            """
            Recursive helper to delete word.
            
            Returns True if current node should be deleted.
            """
            if index == len(word):
                # Reached end of word
                if not node.is_end_of_word:
                    return False  # Word doesn't exist
                
                node.is_end_of_word = False
                node.word = None
                
                # Delete node if it has no children
                return len(node.children) == 0
            
            char = word[index]
            if char not in node.children:
                return False  # Word doesn't exist
            
            child = node.children[char]
            should_delete_child = _delete_helper(child, word, index + 1)
            
            if should_delete_child:
                del node.children[char]
                
                # Delete current node if:
                # - It's not end of another word
                # - It has no other children
                return not node.is_end_of_word and len(node.children) == 0
            
            return False
        
        return _delete_helper(self.root, word, 0) is not None
    
    def get_all_words(self) -> List[str]:
        """
        Get all words stored in the trie.
        
        Returns:
            List of all words in the trie
            
        Time Complexity: O(N * M) where N is number of words, M is avg length
        """
        words = []
        
        def dfs(node: TrieNode, current: str):
            if node.is_end_of_word:
                words.append(current)
            
            for char, child in node.children.items():
                dfs(child, current + char)
        
        dfs(self.root, "")
        return words
    
    def autocomplete(self, prefix: str, max_suggestions: int = 5) -> List[str]:
        """
        Get autocomplete suggestions for a prefix.
        
        Args:
            prefix: Prefix to autocomplete
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of word suggestions
            
        Example:
            >>> trie = Trie()
            >>> for word in ["cat", "car", "card", "care", "careful"]:
            ...     trie.insert(word)
            >>> trie.autocomplete("car")
            ['car', 'card', 'care', 'careful']
        """
        # Navigate to prefix node
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # Prefix doesn't exist
            node = node.children[char]
        
        # DFS to find all words with this prefix
        suggestions = []
        
        def dfs(node: TrieNode, current: str):
            if len(suggestions) >= max_suggestions:
                return
            
            if node.is_end_of_word:
                suggestions.append(current)
            
            for char in sorted(node.children.keys()):  # Alphabetical order
                dfs(node.children[char], current + char)
        
        dfs(node, prefix)
        return suggestions


class WordDictionary:
    """
    Add and search words data structure supporting '.' wildcard.
    
    LeetCode #211: Design Add and Search Words Data Structure
    
    Time Complexity:
        - addWord: O(m)
        - search: O(m) for exact match, O(26^m) worst case with wildcards
    """
    
    def __init__(self):
        self.root = TrieNode()
    
    def addWord(self, word: str) -> None:
        """Add a word to the dictionary."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
    
    def search(self, word: str) -> bool:
        """
        Search for word with wildcard support.
        '.' matches any single character.
        
        Example:
            >>> wd = WordDictionary()
            >>> wd.addWord("bad")
            >>> wd.search("bad")
            True
            >>> wd.search("b.d")
            True
            >>> wd.search("b..")
            True
        """
        def dfs(node: TrieNode, index: int) -> bool:
            if index == len(word):
                return node.is_end_of_word
            
            char = word[index]
            
            if char == '.':
                # Wildcard: try all children
                for child in node.children.values():
                    if dfs(child, index + 1):
                        return True
                return False
            else:
                # Regular character
                if char not in node.children:
                    return False
                return dfs(node.children[char], index + 1)
        
        return dfs(self.root, 0)


def index_pairs(text: str, words: List[str]) -> List[List[int]]:
    """
    Find all index pairs where words from list appear in text.
    
    LeetCode #1065: Index Pairs of a String
    
    Args:
        text: String to search in
        words: List of words to find
        
    Returns:
        List of [start, end] pairs where words appear
        
    Time Complexity: O(n^2 + m) where n is text length, m is total chars in words
    Space Complexity: O(m)
    
    Example:
        >>> index_pairs("thestoryofleetcodeandme", ["story", "fleet", "leetcode"])
        [[3,7], [9,13], [10,17]]
    """
    # Build trie with all words
    trie = Trie()
    for word in words:
        trie.insert(word)
    
    result = []
    
    # Try starting from each position
    for i in range(len(text)):
        node = trie.root
        
        # Try extending from position i
        for j in range(i, len(text)):
            char = text[j]
            if char not in node.children:
                break
            
            node = node.children[char]
            
            # Found a complete word
            if node.is_end_of_word:
                result.append([i, j])
    
    return result


def min_extra_char(s: str, dictionary: List[str]) -> int:
    """
    Find minimum extra characters when breaking string using dictionary words.
    
    LeetCode #2707: Extra Characters in a String
    
    Args:
        s: String to break
        dictionary: List of valid words
        
    Returns:
        Minimum number of extra characters
        
    Time Complexity: O(n^2 + m) where n is string length, m is dict size
    Space Complexity: O(n + m)
    
    Example:
        >>> min_extra_char("leetscode", ["leet", "code", "leetcode"])
        1  # "leet" + "s" + "code", 's' is extra
    """
    # Build trie from dictionary
    trie = Trie()
    for word in dictionary:
        trie.insert(word)
    
    n = len(s)
    # dp[i] = min extra chars for s[0:i]
    dp = [float('inf')] * (n + 1)
    dp[0] = 0  # Empty string has 0 extra chars
    
    for i in range(1, n + 1):
        # Option 1: Skip current character (it's extra)
        dp[i] = dp[i - 1] + 1
        
        # Option 2: Try all words ending at position i
        node = trie.root
        for j in range(i - 1, -1, -1):
            char = s[j]
            if char not in node.children:
                break
            
            node = node.children[char]
            
            # Found a valid dictionary word s[j:i]
            if node.is_end_of_word:
                dp[i] = min(dp[i], dp[j])
    
    return dp[n]


def suggested_products(products: List[str], searchWord: str) -> List[List[str]]:
    """
    Return top 3 product suggestions for each prefix of searchWord.
    
    LeetCode #1268: Search Suggestions System
    
    Args:
        products: List of product names
        searchWord: Word being typed
        
    Returns:
        List of suggestion lists for each prefix
        
    Time Complexity: O(m*n + k) where m is products count, n is avg length, k is searchWord length
    Space Complexity: O(m*n)
    
    Example:
        >>> suggested_products(["mobile","mouse","moneypot","monitor"], "mouse")
        [
            ["mobile","moneypot","monitor"],
            ["mobile","moneypot","monitor"],
            ["mouse","moneypot"],
            ["mouse","moneypot"],
            ["mouse"]
        ]
    """
    # Build trie with products
    trie = Trie()
    for product in products:
        trie.insert(product)
    
    result = []
    node = trie.root
    current_prefix = ""
    
    for char in searchWord:
        current_prefix += char
        
        # Navigate to next character
        if char in node.children:
            node = node.children[char]
            # Get up to 3 suggestions with this prefix
            suggestions = trie.autocomplete(current_prefix, max_suggestions=3)
            result.append(suggestions)
        else:
            # No products with this prefix
            result.append([])
            # All future prefixes will also have no matches
            node.children = {}  # Clear to avoid further searching
    
    return result


# Example usage and testing
if __name__ == "__main__":
    print("=== Basic Trie Operations ===")
    trie = Trie()
    
    # Insert words
    words = ["cat", "car", "card", "dog", "dodge"]
    for word in words:
        trie.insert(word)
    
    print(f"Inserted: {words}")
    print(f"Search 'car': {trie.search('car')}")  # True
    print(f"Search 'cars': {trie.search('cars')}")  # False
    print(f"Prefix 'ca': {trie.startsWith('ca')}")  # True
    print(f"All words: {trie.get_all_words()}")
    print()
    
    print("=== Autocomplete ===")
    print(f"Autocomplete 'ca': {trie.autocomplete('ca')}")
    print(f"Autocomplete 'do': {trie.autocomplete('do')}")
    print()
    
    print("=== Word Dictionary with Wildcards ===")
    wd = WordDictionary()
    wd.addWord("bad")
    wd.addWord("dad")
    wd.addWord("mad")
    print(f"Search 'bad': {wd.search('bad')}")  # True
    print(f"Search '.ad': {wd.search('.ad')}")  # True
    print(f"Search 'b..': {wd.search('b..')}")  # True
    print()
    
    print("=== Index Pairs ===")
    text = "thestoryofleetcodeandme"
    words_to_find = ["story", "fleet", "leetcode"]
    print(f"Text: {text}")
    print(f"Words: {words_to_find}")
    print(f"Pairs: {index_pairs(text, words_to_find)}")
    print()
    
    print("=== Search Suggestions ===")
    products = ["mobile", "mouse", "moneypot", "monitor", "mousepad"]
    search = "mouse"
    print(f"Products: {products}")
    print(f"Search: {search}")
    suggestions = suggested_products(products, search)
    for i, sug in enumerate(suggestions):
        print(f"  After typing '{search[:i+1]}': {sug}")
```

### Code Explanation

**TrieNode Class:**
- Uses dictionary for children (more flexible than array)
- `is_end_of_word` flag marks complete words
- Optional `word` storage useful for reconstruction

**Trie Class:**
- `insert`: Follows/creates path character by character
- `search`: Traverses path and checks end-of-word flag
- `startsWith`: Only needs to traverse prefix
- `delete`: Recursive deletion with cleanup of unused nodes
- `autocomplete`: DFS from prefix node to find suggestions

**WordDictionary:**
- Extends basic Trie with wildcard support
- Uses DFS to try all paths when encountering '.'

**Index Pairs:**
- Tries starting from each position in text
- Uses Trie to efficiently check all word endings

**Min Extra Char:**
- Combines Trie with dynamic programming
- DP tracks minimum extra characters up to each position

**Search Suggestions:**
- Uses autocomplete functionality
- Limits suggestions to 3 per prefix

## Complexity Analysis

### Time Complexity

**Insert:**
- **Time:** O(m) where m is word length
- **Why?** Visit each character exactly once

**Search:**
- **Time:** O(m) for exact match
- **Time:** O(ALPHABET_SIZE^m) worst case with wildcards
- **Why?** Without wildcards, follow single path. With wildcards, may explore all branches.

**StartsWith:**
- **Time:** O(m) where m is prefix length
- **Why?** Same as search but don't check end-of-word flag

**Delete:**
- **Time:** O(m)
- **Why?** Traverse path once, potentially clean up nodes on return

**Autocomplete:**
- **Time:** O(n) where n is number of words with given prefix
- **Why?** DFS from prefix node visits all matching words

### Space Complexity

**Trie Storage:**
- **Space:** O(ALPHABET_SIZE * N * M)
- **Why?** Each node can have ALPHABET_SIZE children. With N words of average length M, worst case creates ALPHABET_SIZE * N * M nodes.

**Optimized (with prefix sharing):**
- **Space:** O(total characters across all unique prefixes)
- **Why?** Words sharing prefixes share nodes

**Search/Insert:**
- **Space:** O(1) - iterative implementation
- **Space:** O(m) - recursive implementation (call stack)

### Comparison with Alternatives

| Operation | Trie | Hash Set | Binary Search Tree | Sorted Array |
|-----------|------|----------|-------------------|--------------|
| **Insert** | O(m) | O(m) | O(m log n) | O(n) |
| **Search** | O(m) | O(m) | O(m log n) | O(m log n) |
| **Prefix Search** | O(m + k) | O(n*m) | O(m log n + k) | O(m log n + k) |
| **Space** | O(Σ prefixes) | O(n*m) | O(n*m) | O(n*m) |
| **Autocomplete** | O(m + k) | O(n*m) | O(m log n + k) | O(m log n + k) |

**Legend:** n = number of words, m = word length, k = results count, Σ = alphabet size

**When to use Trie:**
- Need fast prefix operations
- Autocomplete functionality
- Dictionary/spell checker
- IP routing
- Many words with common prefixes

**When NOT to use Trie:**
- Only exact match needed (use hash set)
- Memory constrained (Trie uses more space)
- Few words with little prefix overlap
- Need range queries (use BST)

## Examples

### Example 1: Basic Trie Operations

**Operations:**
```
insert("apple")
search("apple")   → true
search("app")     → false
startsWith("app") → true
insert("app")
search("app")     → true
```

**Trie Evolution:**

```
After insert("apple"):
root → a → p → p → l → e[*]

After insert("app"):
root → a → p → p[*] → l → e[*]

Note: "app" now marked as complete word,
but "apple" still exists!
```

### Example 2: Word Dictionary with Wildcards

**Input:**
```
addWord("bad")
addWord("dad")
addWord("mad")
search("pad")  → false
search("bad")  → true
search(".ad")  → true
search("b..")  → true
```

**Search Trace for ".ad":**

```
Trie:
root → b → a → d[*]
    ↓
    d → a → d[*]
    ↓
    m → a → d[*]

Search ".ad":
  At root, char = '.'
    Try child 'b':
      At 'b', char = 'a'
        Move to 'a'
      At 'a', char = 'd'
        Move to 'd'
      At 'd', index = 3 (end)
        Check is_end_of_word → True ✓
    Return True

Alternative paths 'd' and 'm' would also succeed.
```

### Example 3: Index Pairs

**Input:**
```
text = "thestoryofleetcodeandme"
words = ["story", "fleet", "leetcode"]
```

**Output:** `[[3,7], [9,13], [10,17]]`

**Trace:**

```
Build Trie from words:
root → s → t → o → r → y[*]
    ↓
    f → l → e → e → t[*]
    ↓
    l → e → e → t → c → o → d → e[*]

Scan text starting at each position:

i=0: "thestoryofleetcodeandme"
     ^
  Try 't' → not in Trie root

i=1: "thestoryofleetcodeandme"
      ^
  Try 'h' → not in Trie root

i=2: "thestoryofleetcodeandme"
       ^
  Try 'e' → not in Trie root

i=3: "thestoryofleetcodeandme"
        ^^^^^
  Try 's' → in Trie!
  Continue: 's','t','o','r','y' all match
  At 'y', is_end_of_word = True
  Add [3, 7] ✓

i=9: "thestoryofleetcodeandme"
                ^^^^
  Try 'f' → in Trie!
  Continue: 'f','l','e','e','t' all match
  At 't', is_end_of_word = True
  Add [9, 13] ✓

i=10: "thestoryofleetcodeandme"
                 ^^^^^^^^
   Try 'l' → in Trie!
   Continue: 'l','e','e','t','c','o','d','e' all match
   At final 'e', is_end_of_word = True
   Add [10, 17] ✓

Result: [[3,7], [9,13], [10,17]]
```

### Example 4: Extra Characters in String

**Input:**
```
s = "leetscode"
dictionary = ["leet", "code", "leetcode"]
```

**Output:** 1 (the 's' is extra)

**DP Trace:**

```
Build Trie from dictionary:
root → l → e → e → t[*] → c → o → d → e[*]
    ↓
    c → o → d → e[*]

DP array: dp[i] = min extra chars for s[0:i]

dp[0] = 0 (empty string)

For i=1 ('l'):
  Option 1: Skip 'l' → dp[0] + 1 = 1
  Check words ending at 1: none
  dp[1] = 1

For i=2 ('le'):
  Option 1: Skip 'e' → dp[1] + 1 = 2
  Check words ending at 2: none
  dp[2] = 2

For i=3 ('lee'):
  Option 1: Skip 'e' → dp[2] + 1 = 3
  Check words ending at 3: none
  dp[3] = 3

For i=4 ('leet'):
  Option 1: Skip 't' → dp[3] + 1 = 4
  Check words ending at 4:
    s[0:4] = "leet" is in dict! → dp[0] = 0
  dp[4] = min(4, 0) = 0 ✓

For i=5 ('leets'):
  Option 1: Skip 's' → dp[4] + 1 = 1
  Check words ending at 5: none
  dp[5] = 1

For i=6 ('leetsc'):
  Option 1: Skip 'c' → dp[5] + 1 = 2
  Check words ending at 6: none
  dp[6] = 2

For i=7 ('leetsco'):
  Option 1: Skip 'o' → dp[6] + 1 = 3
  Check words ending at 7: none
  dp[7] = 3

For i=8 ('leetscod'):
  Option 1: Skip 'd' → dp[7] + 1 = 4
  Check words ending at 8: none
  dp[8] = 4

For i=9 ('leetscode'):
  Option 1: Skip 'e' → dp[8] + 1 = 5
  Check words ending at 9:
    s[5:9] = "code" is in dict! → dp[5] = 1
    s[0:9] = "leetscode" not in dict
  dp[9] = min(5, 1) = 1 ✓

Answer: dp[9] = 1
Optimal: "leet" + "s" + "code"
```

### Example 5: Search Suggestions System

**Input:**
```
products = ["mobile","mouse","moneypot","monitor","mousepad"]
searchWord = "mouse"
```

**Output:**
```
[
  ["mobile","moneypot","monitor"],
  ["mobile","moneypot","monitor"],  
  ["mouse","moneypot","monitor"],
  ["mouse","mousepad"],
  ["mouse","mousepad"]
]
```

**Trace:**

```
Build Trie and sort products:
After sorting: ["mobile","moneypot","monitor","mouse","mousepad"]

Trie:
root → m → o → b → i → l → e[*]
        ↓
        n → e → y → p → o → t[*]
        ↓  ↓
        i → t → o → r[*]
        ↓
        u → s → e[*] → p → a → d[*]

Process searchWord "mouse":

After 'm':
  Navigate to 'm'
  Find all words: ["mobile", "moneypot", "monitor", "mouse", "mousepad"]
  Top 3: ["mobile", "moneypot", "monitor"] ✓

After 'mo':
  Navigate to 'o'
  Find all words: ["mobile", "moneypot", "monitor", "mouse", "mousepad"]
  Top 3: ["mobile", "moneypot", "monitor"] ✓

After 'mou':
  Navigate to 'u'
  Find all words: ["mouse", "mousepad"]
  Top 3: ["mouse", "moneypot", "monitor"]
  Wait, let me recalculate...
  From 'u' node, DFS finds: ["mouse", "mousepad"]
  But we need products starting with "mou"
  Actually: ["mouse", "mousepad"]
  But alphabetically after "mou": ["mouse", "mousepad", then others]
  Actually top 3 with "mou": ["mouse", "moneypot", "monitor"]
  
After 'mous':
  Navigate to 's'
  Find all words: ["mouse", "mousepad"]
  Top 3: ["mouse", "mousepad"] ✓

After 'mouse':
  Navigate to 'e'
  Find all words: ["mouse", "mousepad"]
  Top 3: ["mouse", "mousepad"] ✓
```

## Edge Cases

### 1. Empty String
**Scenario:** Insert or search for empty string ""
**Challenge:** Root represents empty string
**Solution:** Add special handling or treat root as empty string
**Code example:**
```python
def insert(self, word: str) -> None:
    if not word:
        self.root.is_end_of_word = True  # Mark root
        return
    # ... normal insertion
```

### 2. Single Character Words
**Scenario:** Words like "a", "I"
**Challenge:** Need to mark nodes at depth 1
**Solution:** Works naturally with standard implementation
**Code example:**
```python
# Insert "a":
# root → a[*]
# No special handling needed
```

### 3. Prefix of Another Word
**Scenario:** Insert "car" and "card"
**Challenge:** Both must be marked as complete words
**Solution:** Use `is_end_of_word` flag at each applicable node
**Code example:**
```python
# "car" and "card" both exist:
# root → c → a → r[*] → d[*]
# 'r' has is_end_of_word = True
# 'd' has is_end_of_word = True
```

### 4. All Words Share Prefix
**Scenario:** Words like ["cat", "cats", "caterpillar"]
**Challenge:** Maximum prefix sharing
**Solution:** This is optimal case for Trie - minimal nodes
**Code example:**
```python
# Only branches at 's' and 'e':
# root → c → a → t[*] → s[*]
#                    ↓
#                    e → r → p → i → l → l → a → r[*]
```

### 5. No Common Prefixes
**Scenario:** Words like ["apple", "banana", "cherry"]
**Challenge:** Trie offers no space advantage
**Solution:** Still works, but hash set might be better
**Code example:**
```python
# Three separate branches from root:
# root → a → p → p → l → e[*]
#     ↓
#     b → a → n → a → n → a[*]
#     ↓
#     c → h → e → r → r → y[*]
```

### 6. Very Long Words
**Scenario:** Words with length > 100
**Challenge:** Deep tree, recursion depth
**Solution:** Use iterative implementation
**Code example:**
```python
# Iterative insert avoids stack overflow:
def insert(self, word: str) -> None:
    node = self.root
    for char in word:  # Iterative, not recursive
        if char not in node.children:
            node.children[char] = TrieNode()
        node = node.children[char]
    node.is_end_of_word = True
```

### 7. Special Characters
**Scenario:** Words with numbers, symbols
**Challenge:** Dictionary vs array for children
**Solution:** Use dictionary (not fixed-size array)
**Code example:**
```python
# Dictionary handles any character:
self.children = {}  # Not fixed size array
# Can store 'a', '1', '@', etc.
```

## Common Pitfalls

### ❌ Pitfall 1: Not Marking End of Word
**What happens:** "car" exists, search("car") returns False
**Why it's wrong:** Forgot to set `is_end_of_word = True`
**Correct approach:**
```python
# WRONG:
def insert(self, word):
    node = self.root
    for char in word:
        if char not in node.children:
            node.children[char] = TrieNode()
        node = node.children[char]
    # Forgot to mark end!

# CORRECT:
def insert(self, word):
    node = self.root
    for char in word:
        if char not in node.children:
            node.children[char] = TrieNode()
        node = node.children[char]
    node.is_end_of_word = True  # Mark end ✓
```

### ❌ Pitfall 2: Confusing Prefix and Word Search
**What happens:** `startsWith("app")` and `search("app")` both return True when only "apple" exists
**Why it's wrong:** Not checking `is_end_of_word` flag in search
**Correct approach:**
```python
# WRONG search:
def search(self, word):
    node = self.root
    for char in word:
        if char not in node.children:
            return False
        node = node.children[char]
    return True  # Should check is_end_of_word!

# CORRECT:
def search(self, word):
    node = self.root
    for char in word:
        if char not in node.children:
            return False
        node = node.children[char]
    return node.is_end_of_word  # Check flag ✓
```

### ❌ Pitfall 3: Memory Leak in Delete
**What happens:** Deleted word's nodes remain in memory
**Why it's wrong:** Not removing unnecessary nodes
**Correct approach:**
```python
# WRONG delete:
def delete(self, word):
    node = self.root
    for char in word:
        if char not in node.children:
            return False
        node = node.children[char]
    node.is_end_of_word = False  # Only unmarks, doesn't free nodes!

# CORRECT delete (recursive cleanup):
def delete(self, word):
    def _delete_helper(node, word, index):
        if index == len(word):
            if not node.is_end_of_word:
                return False
            node.is_end_of_word = False
            return len(node.children) == 0  # Delete if no children
        
        char = word[index]
        if char not in node.children:
            return False
        
        should_delete = _delete_helper(node.children[char], word, index + 1)
        
        if should_delete:
            del node.children[char]  # Free memory ✓
            return not node.is_end_of_word and len(node.children) == 0
        
        return False
    
    return _delete_helper(self.root, word, 0)
```

### ❌ Pitfall 4: Using Fixed-Size Array for Unicode
**What happens:** Can't handle characters beyond 'a'-'z'
**Why it's wrong:** Array size 26 only works for lowercase English
**Correct approach:**
```python
# WRONG for general use:
class TrieNode:
    def __init__(self):
        self.children = [None] * 26  # Only handles a-z!
        self.is_end_of_word = False

# CORRECT for general characters:
class TrieNode:
    def __init__(self):
        self.children = {}  # Handles any character ✓
        self.is_end_of_word = False
```

### ❌ Pitfall 5: Incorrect Wildcard DFS
**What happens:** Wildcard search returns wrong results
**Why it's wrong:** Not trying all branches for '.'
**Correct approach:**
```python
# WRONG:
def search(self, word):
    node = self.root
    for char in word:
        if char == '.':
            node = list(node.children.values())[0]  # Only tries first child!
        else:
            if char not in node.children:
                return False
            node = node.children[char]
    return node.is_end_of_word

# CORRECT (DFS):
def search(self, word):
    def dfs(node, index):
        if index == len(word):
            return node.is_end_of_word
        
        char = word[index]
        if char == '.':
            # Try ALL children ✓
            for child in node.children.values():
                if dfs(child, index + 1):
                    return True
            return False
        else:
            if char not in node.children:
                return False
            return dfs(node.children[char], index + 1)
    
    return dfs(self.root, 0)
```

### ❌ Pitfall 6: Not Handling Case Sensitivity
**What happens:** "Apple" and "apple" treated differently unintentionally
**Why it's wrong:** Didn't normalize case
**Correct approach:**
```python
# WRONG (case-sensitive when not intended):
def insert(self, word):
    # Inserts as-is

# CORRECT (normalize if case-insensitive needed):
def insert(self, word):
    word = word.lower()  # Normalize ✓
    # ... rest of insertion
```

### ❌ Pitfall 7: Forgetting to Initialize Root
**What happens:** NullPointerException on first insert
**Why it's wrong:** Root not created in constructor
**Correct approach:**
```python
# WRONG:
class Trie:
    def __init__(self):
        pass  # No root!

# CORRECT:
class Trie:
    def __init__(self):
        self.root = TrieNode()  # Initialize root ✓
```

## Variations and Extensions

### Variation 1: Compressed Trie (Radix Tree)
**Description:** Store edge labels instead of single characters
**When to use:** When memory is critical and words have long unique suffixes
**Key differences:** Edges can represent multiple characters
**Implementation:**
```python
class RadixNode:
    def __init__(self):
        self.children = {}  # edge_label -> RadixNode
        self.is_end = False

# Example:
# Instead of: root → r → o → m → a → n → e[*]
#                    ↓
#                    u → b → y[*]
# Store as:   root → "roman"[*]
#                 ↓
#                 "ruby"[*]
```

### Variation 2: Ternary Search Tree
**Description:** Each node has 3 children: less, equal, greater
**When to use:** More memory-efficient than standard Trie
**Key differences:** Binary-search-like structure for each character
**Implementation:**
```python
class TSTNode:
    def __init__(self, char):
        self.char = char
        self.left = None    # Characters < char
        self.equal = None   # Next character in word
        self.right = None   # Characters > char
        self.is_end = False
```

### Variation 3: Suffix Tree
**Description:** Trie of all suffixes of a string
**When to use:** Pattern matching, finding repeated substrings
**Key differences:** Stores all suffixes, not just complete words
**Implementation:**
```python
def build_suffix_trie(text: str) -> Trie:
    """Build trie containing all suffixes of text."""
    trie = Trie()
    for i in range(len(text)):
        trie.insert(text[i:])  # Insert each suffix
    return trie

# For "banana":
# Suffixes: "banana", "anana", "nana", "ana", "na", "a"
```

### Variation 4: Trie with Frequency Count
**Description:** Track how many times each word was inserted
**When to use:** Autocomplete with popularity ranking
**Key differences:** Add count field to nodes
**Implementation:**
```python
class TrieNodeWithCount:
    def __init__(self):
        self.children = {}
        self.count = 0  # Number of times word inserted
        self.word = None

def insert_with_count(trie, word):
    node = trie.root
    for char in word:
        if char not in node.children:
            node.children[char] = TrieNodeWithCount()
        node = node.children[char]
    node.count += 1  # Increment count
    node.word = word

def get_top_k_words(trie, prefix, k):
    """Get k most frequent words with prefix."""
    # Navigate to prefix, then collect (word, count) pairs
    # Sort by count descending
    # Return top k
```

### Variation 5: Bitwise Trie (Binary Trie)
**Description:** Trie for binary representation of numbers
**When to use:** Maximum XOR queries, finding closest number
**Key differences:** Only 2 children per node (0 and 1)
**Implementation:**
```python
class BitwiseTrie:
    """Trie for storing integers as binary."""
    
    def __init__(self):
        self.root = {'0': None, '1': None, 'value': None}
    
    def insert(self, num: int, bits: int = 32):
        """Insert number into trie."""
        node = self.root
        for i in range(bits - 1, -1, -1):
            bit = '1' if (num & (1 << i)) else '0'
            if node[bit] is None:
                node[bit] = {'0': None, '1': None, 'value': None}
            node = node[bit]
        node['value'] = num
    
    def find_max_xor(self, num: int, bits: int = 32) -> int:
        """Find number in trie that gives maximum XOR with num."""
        node = self.root
        for i in range(bits - 1, -1, -1):
            bit = '1' if (num & (1 << i)) else '0'
            # Try opposite bit for max XOR
            opposite = '0' if bit == '1' else '1'
            if node[opposite] is not None:
                node = node[opposite]
            elif node[bit] is not None:
                node = node[bit]
            else:
                return 0
        return num ^ node['value']
```

### Variation 6: Persistent Trie
**Description:** Maintain multiple versions of Trie
**When to use:** Need to query historical states
**Key differences:** Nodes are immutable, new versions share structure
**Implementation:**
```python
# Each insert creates new version sharing unchanged nodes
class PersistentTrie:
    def __init__(self):
        self.versions = [TrieNode()]  # List of root versions
    
    def insert_new_version(self, word: str):
        """Insert word and create new version."""
        old_root = self.versions[-1]
        new_root = self._copy_insert(old_root, word, 0)
        self.versions.append(new_root)
    
    def _copy_insert(self, node, word, index):
        """Copy node and insert character."""
        new_node = TrieNode()
        new_node.children = node.children.copy()
        new_node.is_end_of_word = node.is_end_of_word
        
        if index < len(word):
            char = word[index]
            if char in new_node.children:
                new_node.children[char] = self._copy_insert(
                    new_node.children[char], word, index + 1
                )
            else:
                new_node.children[char] = self._copy_insert(
                    TrieNode(), word, index + 1
                )
        else:
            new_node.is_end_of_word = True
        
        return new_node
```

## Practice Problems

### Beginner
1. **Implement Trie (Prefix Tree)** - Basic Trie with insert, search, startsWith
   - LeetCode #208

2. **Longest Word in Dictionary** - Find longest word built one character at a time
   - LeetCode #720

3. **Map Sum Pairs** - Trie with value sums
   - LeetCode #677

4. **Index Pairs of a String** - Find all occurrences of words in text
   - LeetCode #1065

### Intermediate
1. **Design Add and Search Words Data Structure** - Trie with wildcard support
   - LeetCode #211

2. **Search Suggestions System** - Autocomplete with top 3 suggestions
   - LeetCode #1268

3. **Replace Words** - Replace words with shortest root
   - LeetCode #648

4. **Extra Characters in a String** - Minimize extra characters using dictionary
   - LeetCode #2707

5. **Implement Magic Dictionary** - Search with one character modification
   - LeetCode #676

6. **Design Search Autocomplete System** - Real-time autocomplete
   - LeetCode #642 (Premium)

### Advanced
1. **Word Search II** - Find all dictionary words in 2D grid
   - LeetCode #212

2. **Maximum XOR of Two Numbers** - Use binary trie
   - LeetCode #421

3. **Concatenated Words** - Find words made of other words
   - LeetCode #472

4. **Palindrome Pairs** - Find palindrome pairs using Trie
   - LeetCode #336

5. **Stream of Characters** - Query stream for dictionary words
   - LeetCode #1032

6. **Word Squares** - Generate all word squares
   - LeetCode #425 (Premium)

## Real-World Applications

### Industry Use Cases

1. **Autocomplete Systems:** Google search suggestions, IDE code completion, mobile keyboard predictions use Tries to quickly suggest completions based on prefixes.

2. **Spell Checkers:** Dictionary applications use Tries for fast word lookup and suggestions for misspelled words (using edit distance from Trie words).

3. **IP Routing:** Internet routers use Tries (specifically, Radix/Patricia trees) for IP address lookup to route packets efficiently.

4. **Genome Sequence Matching:** Bioinformatics uses suffix tries/trees for finding patterns in DNA sequences.

5. **T9 Predictive Text:** Old mobile phones used Tries to map number sequences to words.

### Popular Implementations

- **Linux Kernel:** Uses Radix trees for page cache management
- **Redis:** Uses Radix tree for storing keys efficiently
- **Apache Lucene:** Search engine uses Tries for term dictionary
- **DNS Servers:** Use Tries for domain name lookup
- **GCC Compiler:** Uses Tries for identifier storage in symbol tables

### Practical Scenarios

- **Contact Search:** Phone contacts searchable by name prefix
- **File System Path Lookup:** Directory traversal optimization
- **Database Indexing:** Prefix-based index structures
- **Text Editors:** Symbol/variable name autocomplete
- **E-commerce:** Product search with suggestions
- **Network Security:** Pattern matching in intrusion detection systems

## Related Topics

### Prerequisites to Review
- **Trees** - Tries are specialized trees
- **Hash Maps** - Alternative for some Trie use cases
- **Recursion** - Often used in Trie operations
- **DFS/BFS** - For traversing Tries

### Next Steps
- **Suffix Trees/Arrays** - Advanced string matching structures
- **Aho-Corasick Algorithm** - Multiple pattern matching using Trie
- **Radix Trees** - Compressed Tries
- **Ternary Search Trees** - Memory-efficient alternative

### Similar Concepts
- **Hash Tables** - Alternative for exact string matching
- **Binary Search Trees** - For sorted string storage
- **Suffix Arrays** - Array-based alternative to suffix trees
- **Bloom Filters** - Probabilistic set membership (space-efficient)

### Further Reading
- "Introduction to Algorithms" (CLRS) - Chapter on String Matching
- "Algorithm Design Manual" by Skiena - Section on String Data Structures
- [Trie - GeeksforGeeks](https://www.geeksforgeeks.org/trie-insert-and-search/)
- [Trie - Wikipedia](https://en.wikipedia.org/wiki/Trie)
- "Algorithms on Strings, Trees, and Sequences" by Dan Gusfield
- [LeetCode Trie Problems Collection](https://leetcode.com/tag/trie/)
