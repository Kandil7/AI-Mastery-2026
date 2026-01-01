# ML Data Structures & Algorithms - Coding Interview

Common data structures and algorithms questions tailored for ML interviews.

---

## Table of Contents
1. [Array Manipulation](#1-array-manipulation)
2. [Matrix Operations](#2-matrix-operations)
3. [Graph Algorithms](#3-graph-algorithms)
4. [Dynamic Programming](#4-dynamic-programming)
5. [Heap/Priority Queue](#5-heappriority-queue)
6. [ML-Specific Algorithms](#6-ml-specific-algorithms)

---

## 1. Array Manipulation

### Problem: Top-K Frequent Elements

**Often used in:** Feature selection, token frequency

```python
from collections import Counter
import heapq

def top_k_frequent(nums: list, k: int) -> list:
    """
    Find k most frequent elements.
    Time: O(n log k)
    Space: O(n)
    """
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)

# Test
nums = [1, 1, 1, 2, 2, 3]
print(top_k_frequent(nums, 2))  # [1, 2]
```

### Problem: Merge Sorted Arrays

**Used in:** Merging ranked lists, ensemble outputs

```python
def merge_k_sorted(arrays: list) -> list:
    """
    Merge k sorted arrays into one sorted array.
    Time: O(n log k) where n = total elements
    """
    import heapq
    
    result = []
    heap = []
    
    # Initialize with first element from each array
    for i, arr in enumerate(arrays):
        if arr:
            heapq.heappush(heap, (arr[0], i, 0))
    
    while heap:
        val, arr_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        
        if elem_idx + 1 < len(arrays[arr_idx]):
            next_val = arrays[arr_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, arr_idx, elem_idx + 1))
    
    return result

# Test
arrays = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
print(merge_k_sorted(arrays))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

---

## 2. Matrix Operations

### Problem: Matrix Spiral Order

**Used in:** Image processing, convolutional ops

```python
def spiral_order(matrix: list) -> list:
    """
    Traverse matrix in spiral order.
    Time: O(m*n), Space: O(1) extra
    """
    if not matrix:
        return []
    
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1
    
    while top <= bottom and left <= right:
        # Right
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1
        
        # Down
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1
        
        # Left
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1
        
        # Up
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1
    
    return result
```

### Problem: Matrix Multiplication

```python
import numpy as np

def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication from scratch.
    A: (m, n), B: (n, p) -> C: (m, p)
    """
    m, n = A.shape
    n2, p = B.shape
    
    assert n == n2, "Dimension mismatch"
    
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    
    return C
```

---

## 3. Graph Algorithms

### Problem: Build Dependency Graph

**Used in:** Pipeline DAG, feature dependencies

```python
from collections import defaultdict, deque

def topological_sort(n: int, edges: list) -> list:
    """
    Topological sort for DAG.
    Returns order or empty if cycle exists.
    """
    graph = defaultdict(list)
    in_degree = [0] * n
    
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1
    
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == n else []

# Test: Task execution order
edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
print(topological_sort(4, edges))  # [0, 1, 2, 3] or [0, 2, 1, 3]
```

### Problem: Shortest Path (Dijkstra)

**Used in:** Graph neural networks, pathfinding

```python
import heapq

def dijkstra(graph: dict, start: int) -> dict:
    """
    Single-source shortest path.
    graph: {node: [(neighbor, weight), ...]}
    """
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    
    heap = [(0, start)]
    
    while heap:
        dist, node = heapq.heappop(heap)
        
        if dist > distances[node]:
            continue
        
        for neighbor, weight in graph[node]:
            new_dist = dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))
    
    return distances
```

---

## 4. Dynamic Programming

### Problem: Edit Distance

**Used in:** String matching, NLP evaluation (WER)

```python
def edit_distance(s1: str, s2: str) -> int:
    """
    Levenshtein distance.
    Time: O(m*n), Space: O(n)
    """
    m, n = len(s1), len(s2)
    
    # Space-optimized: only keep current and previous row
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
        prev, curr = curr, prev
    
    return prev[n]

# Test
print(edit_distance("kitten", "sitting"))  # 3
```

### Problem: Longest Common Subsequence

**Used in:** Sequence alignment, ROUGE-L

```python
def lcs(s1: str, s2: str) -> int:
    """
    Longest common subsequence length.
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]
```

---

## 5. Heap/Priority Queue

### Problem: Median in Data Stream

**Used in:** Online learning, streaming statistics

```python
import heapq

class MedianFinder:
    """
    Find median from data stream.
    Two heaps: max-heap for lower half, min-heap for upper half.
    """
    def __init__(self):
        self.small = []  # max-heap (negate values)
        self.large = []  # min-heap
    
    def add_num(self, num: int):
        heapq.heappush(self.small, -num)
        
        # Balance: move to large if needed
        if self.small and self.large and -self.small[0] > self.large[0]:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        
        # Keep sizes balanced
        if len(self.small) > len(self.large) + 1:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        if len(self.large) > len(self.small) + 1:
            heapq.heappush(self.small, -heapq.heappop(self.large))
    
    def find_median(self) -> float:
        if len(self.small) > len(self.large):
            return -self.small[0]
        if len(self.large) > len(self.small):
            return self.large[0]
        return (-self.small[0] + self.large[0]) / 2
```

---

## 6. ML-Specific Algorithms

### Problem: Implement Softmax

```python
import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax.
    """
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

### Problem: Implement Batch Normalization

```python
def batch_norm(x: np.ndarray, gamma: float = 1.0, beta: float = 0.0, 
               eps: float = 1e-5) -> np.ndarray:
    """
    Batch normalization forward pass.
    """
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
```

### Problem: Implement Cosine Similarity

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot / (norm_a * norm_b)
```

### Problem: Implement AUC-ROC

```python
def auc_roc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Area Under ROC Curve.
    """
    # Sort by scores descending
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[order]
    
    # Calculate TPR and FPR at each threshold
    tpr_list = []
    fpr_list = []
    
    P = np.sum(y_true)
    N = len(y_true) - P
    
    tp = 0
    fp = 0
    
    for label in y_true_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / P if P > 0 else 0)
        fpr_list.append(fp / N if N > 0 else 0)
    
    # Trapezoidal integration
    auc = 0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i-1]) * (tpr_list[i] + tpr_list[i-1]) / 2
    
    return auc
```

---

## Tips for ML Coding Interviews

1. **Clarify problem** - Input format, edge cases, constraints
2. **State assumptions** - Data types, sizes, sorted/unsorted
3. **Analyze complexity** - Time and space
4. **Test with examples** - Walk through your solution
5. **Optimize if asked** - Know when to use heaps, DP, etc.
