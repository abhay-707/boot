# HPC Practicals — In-Depth Theory

---

## Practical 1: Parallel BFS and DFS using OpenMP

### 1.1 Introduction to Graph Traversal

Graph traversal is the process of visiting every vertex in a graph in a systematic manner. The two most fundamental traversal strategies are Breadth First Search (BFS) and Depth First Search (DFS). In High Performance Computing, parallelizing these algorithms allows us to exploit multi-core processors to speed up traversal on large graphs and trees.

---

### 1.2 Breadth First Search (BFS)

#### What is BFS?
BFS explores a graph level by level. Starting from a source node, it visits all immediate neighbors first, then their neighbors, and so on. It uses a **Queue** (FIFO) as its primary data structure.

#### Sequential BFS Algorithm
```
1. Mark source as visited, enqueue it
2. While queue is not empty:
   a. Dequeue a node
   b. Print / process it
   c. Enqueue all unvisited neighbors, mark them visited
```

#### BFS on a Tree (Example)
```
        0          ← Level 0
       / \
      1   2        ← Level 1
     / \ / \
    3  4 5  6      ← Level 2

BFS Order: 0 → 1 → 2 → 3 → 4 → 5 → 6
```

#### Parallel BFS — Concept
The key insight for parallelizing BFS is that **all nodes at the same level are independent of each other** — they do not depend on each other's results, only on the previous level. This is called **level-synchronous BFS**.

**Strategy:**
- Process all nodes at the current level in parallel using `#pragma omp parallel for`
- Use `#pragma omp critical` to protect shared resources (visited array, queue)
- After all threads finish the current level, move to the next level

#### OpenMP Directives Used
| Directive | Purpose |
|---|---|
| `#pragma omp parallel for` | Distribute loop iterations across threads |
| `#pragma omp critical` | Ensure only one thread at a time modifies shared data |

#### Time Complexity
| | Sequential | Parallel |
|---|---|---|
| Time | O(V + E) | O((V + E) / p) where p = number of threads |
| Space | O(V) | O(V) |

---

### 1.3 Depth First Search (DFS)

#### What is DFS?
DFS explores as far as possible along each branch before backtracking. It uses a **Stack** (LIFO) or **recursion** as its primary data structure.

#### Sequential DFS Algorithm
```
1. Mark source as visited, print it
2. For each unvisited neighbor:
   a. Recursively call DFS on that neighbor
```

#### DFS on a Tree (Example)
```
        0
       / \
      1   2
     / \ / \
    3  4 5  6

DFS Order: 0 → 1 → 3 → 4 → 2 → 5 → 6
```

#### Parallel DFS — Concept
DFS is harder to parallelize than BFS because it has inherent sequential dependencies (each step depends on the last). However, we can parallelize the **exploration of sibling branches** — when a node has multiple neighbors, each neighbor's subtree can be explored by a different thread simultaneously.

**Strategy:**
- Use `#pragma omp parallel for` over the neighbor list of each node
- Each thread recursively calls DFS on a different neighbor
- Use a `skip` flag with `#pragma omp critical` to safely check and mark the visited array (a `return` inside a critical block is invalid in OpenMP)

#### The `skip` Flag Fix
```cpp
// WRONG — return inside critical is illegal in OpenMP
#pragma omp critical
{
    if (visited[node]) return;   // ❌ compiler error
}

// CORRECT — use a flag, return outside
bool skip = false;
#pragma omp critical
{
    if (visited[node]) skip = true;
    else { visited[node] = true; }
}
if (skip) return;   // ✅ legal
```

#### Time Complexity
| | Sequential | Parallel |
|---|---|---|
| Time | O(V + E) | O((V + E) / p) approximately |
| Space | O(V) for recursion stack | O(V) |

---

### 1.4 OpenMP Overview

OpenMP (Open Multi-Processing) is an API for shared-memory parallel programming in C, C++, and Fortran. It uses compiler directives (`#pragma omp`) to parallelize code sections.

**Key Concepts:**
- **Fork-Join Model:** The master thread forks into multiple threads for parallel sections, then joins back
- **Shared Memory:** All threads share the same address space
- **Race Condition:** When two threads access/modify the same memory simultaneously → solved with `critical`

---

## Practical 2: Parallel Bubble Sort and Merge Sort using OpenMP

### 2.1 Introduction to Sorting

Sorting is one of the most fundamental problems in computer science. Parallelizing sorting algorithms can dramatically reduce execution time for large datasets by dividing the work across multiple CPU cores.

---

### 2.2 Bubble Sort

#### What is Bubble Sort?
Bubble Sort repeatedly steps through the list, compares adjacent elements, and swaps them if they are in the wrong order. The pass through the list is repeated until the list is sorted.

#### Sequential Bubble Sort
```
For i = 0 to n-1:
    For j = 0 to n-i-2:
        If arr[j] > arr[j+1]:
            Swap arr[j] and arr[j+1]
```

#### Why Sequential Bubble Sort Cannot Be Trivially Parallelized
In standard bubble sort, each comparison depends on the result of the previous swap, creating data dependencies. Two adjacent threads could try to swap overlapping pairs, causing **race conditions**.

For example, if Thread 1 swaps `arr[2]` and `arr[3]`, and Thread 2 simultaneously reads `arr[3]` and `arr[4]`, Thread 2 gets an inconsistent value.

#### Odd-Even Transposition Sort (Parallel Bubble Sort)
The solution is **Odd-Even Transposition Sort**. In each pass, we alternate between:
- **Even phase:** Compare and swap pairs at indices (0,1), (2,3), (4,5), ...
- **Odd phase:** Compare and swap pairs at indices (1,2), (3,4), (5,6), ...

This ensures no two threads ever touch the same element simultaneously, eliminating race conditions.

```
Pass 1 (even): [0-1] [2-3] [4-5] [6-7]  ← all independent, parallelizable
Pass 2 (odd):  [1-2] [3-4] [5-6]        ← all independent, parallelizable
Pass 3 (even): [0-1] [2-3] [4-5] [6-7]
...repeat n times
```

#### OpenMP in Bubble Sort
```cpp
for (int i = 0; i < n; i++) {
    int start = i % 2;  // alternates between 0 (even) and 1 (odd)
    #pragma omp parallel for
    for (int j = start; j < n - 1; j += 2)
        if (arr[j] > arr[j+1]) swap(arr[j], arr[j+1]);
}
```

#### Time Complexity
| | Sequential | Parallel |
|---|---|---|
| Time | O(n²) | O(n) with n processors |
| Space | O(1) | O(1) |

---

### 2.3 Merge Sort

#### What is Merge Sort?
Merge Sort is a divide-and-conquer algorithm that splits the array in half, recursively sorts each half, and then merges the two sorted halves.

#### Sequential Merge Sort
```
MergeSort(arr, l, r):
    if l >= r: return
    mid = (l + r) / 2
    MergeSort(arr, l, mid)       // Sort left half
    MergeSort(arr, mid+1, r)     // Sort right half
    Merge(arr, l, mid, r)        // Merge both halves
```

#### Why Merge Sort is Ideal for Parallelism
The left and right halves are **completely independent** of each other. There is no data dependency between sorting `arr[l..mid]` and `arr[mid+1..r]`. This makes Merge Sort a natural fit for parallelism — each half can be sorted by a different thread simultaneously.

#### Parallel Merge Sort using OpenMP Sections
```cpp
#pragma omp parallel sections
{
    #pragma omp section
    parMergeSort(arr, l, m, depth+1);    // Thread 1 sorts left half

    #pragma omp section
    parMergeSort(arr, m+1, r, depth+1);  // Thread 2 sorts right half
}
merge(arr, l, m, r);  // Sequential merge after both halves are sorted
```

#### Depth Limiting — Why It's Critical
Without depth limiting, the recursion would spawn threads exponentially:
- Depth 0: 2 threads
- Depth 1: 4 threads
- Depth 2: 8 threads
- Depth 10: 1024 threads!

This causes **thread overhead** to exceed the benefit of parallelism. We limit recursion depth to 4 (giving max 16 parallel sections), then fall back to sequential sort.

```cpp
if (depth < 4)
    // use parallel sections
else
    seqMergeSort(arr, l, r);  // fall back
```

#### Time Complexity
| | Sequential | Parallel |
|---|---|---|
| Time | O(n log n) | O(n) with n processors (theoretically) |
| Practical Parallel | — | O(n log n / p) where p = processors |
| Space | O(n) | O(n) |

---

### 2.4 Performance Measurement with OpenMP

`omp_get_wtime()` returns wall-clock time in seconds — ideal for measuring parallel performance.

```cpp
double t1 = omp_get_wtime();
// ... code to measure ...
double t2 = omp_get_wtime();
cout << "Time: " << (t2 - t1) << " sec\n";
```

**Speedup** = Sequential Time / Parallel Time  
**Efficiency** = Speedup / Number of Threads

---

## Practical 3: Parallel Reduction using OpenMP

### 3.1 What is Reduction?

Reduction is the process of combining all elements of an array into a single value using an associative operation such as sum, min, max, or product. It is one of the most common patterns in parallel computing.

**Examples:**
- Sum: `1 + 2 + 3 + 4 = 10`
- Min: `min(3, 1, 4, 1, 5) = 1`
- Max: `max(3, 1, 4, 1, 5) = 5`
- Average: `Sum / Count`

---

### 3.2 The Race Condition Problem

If we naively parallelize a sum loop:
```cpp
int sum = 0;
#pragma omp parallel for
for (int i = 0; i < N; i++)
    sum += arr[i];   // ❌ RACE CONDITION — multiple threads write to sum
```

Two threads might read `sum = 5`, both add their element, and both write back — one update is lost. The result is incorrect and non-deterministic.

---

### 3.3 Parallel Reduction — How It Works

OpenMP's `reduction` clause solves this cleanly:

```cpp
#pragma omp parallel for reduction(+:sum) reduction(min:mn) reduction(max:mx)
for (int i = 0; i < N; i++) {
    sum += arr[i];
    if (arr[i] < mn) mn = arr[i];
    if (arr[i] > mx) mx = arr[i];
}
```

#### Internal Mechanism (3 Phases)

**Phase 1 — Split:**  
Each thread gets its own **private copy** of the reduction variables, initialized to the identity value for that operation:
| Operation | Identity Value |
|---|---|
| `+` (sum) | 0 |
| `min` | INT_MAX |
| `max` | INT_MIN |
| `*` (product) | 1 |

**Phase 2 — Compute:**  
Each thread independently processes its portion of the array, updating only its private copy. No synchronization needed — no race conditions.

**Phase 3 — Merge:**  
After all threads finish, OpenMP combines all private copies using the specified operator into the final shared variable.

```
Array: [3, 1, 4, 1, 5, 9, 2, 6]

Thread 0: [3,1,4,1]  → local_min=1, local_max=4, local_sum=9
Thread 1: [5,9,2,6]  → local_min=2, local_max=9, local_sum=22

Final reduction:
  min = min(1, 2) = 1
  max = max(4, 9) = 9
  sum = 9 + 22    = 31
```

---

### 3.4 Average

Average cannot be reduced directly with a `reduction` clause since it's not a simple associative operation. It is derived after the parallel reduction completes:

```cpp
double avg = (double)sum / N;
```

---

### 3.5 Supported Reduction Operators in OpenMP

| Operator | Symbol | Identity |
|---|---|---|
| Addition | `+` | 0 |
| Multiplication | `*` | 1 |
| Minimum | `min` | INT_MAX |
| Maximum | `max` | INT_MIN |
| Bitwise AND | `&` | ~0 |
| Bitwise OR | `\|` | 0 |
| Logical AND | `&&` | 1 |
| Logical OR | `\|\|` | 0 |

---

### 3.6 Time Complexity

| | Sequential | Parallel |
|---|---|---|
| Time | O(n) | O(n/p + log p) |
| Space | O(1) | O(p) for private copies |

The `log p` term comes from the tree-based merge phase at the end.

---

## Practical 4: CUDA — Vector Addition and Matrix Multiplication

### 4.1 Introduction to CUDA

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model. It allows developers to write programs that run on the GPU (Graphics Processing Unit), which can have thousands of cores — far more than a CPU.

**CPU vs GPU:**
| | CPU | GPU |
|---|---|---|
| Cores | 4–64 | Thousands (3000–10000+) |
| Clock Speed | High (3–5 GHz) | Lower (1–2 GHz) |
| Best For | Complex sequential logic | Massive parallel data operations |
| Memory | RAM (DDR5) | VRAM (GDDR6) |

---

### 4.2 CUDA Programming Model

#### Hierarchy of Parallelism

```
Grid
└── Block (many blocks per grid)
    └── Thread (many threads per block)
```

- **Thread:** The smallest unit of execution. Each thread runs the kernel function.
- **Block:** A group of threads that can cooperate via shared memory and synchronize.
- **Grid:** A collection of blocks launched for one kernel call.

#### Thread Indexing (1D)
```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```
- `blockIdx.x` — which block is this thread in?
- `blockDim.x` — how many threads per block?
- `threadIdx.x` — which thread within the block?

#### Thread Indexing (2D) — for matrices
```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

---

### 4.3 CUDA Memory Model

```
CPU (Host)  ←──PCIe Bus──→  GPU (Device)
   RAM                          VRAM
```

Data must be explicitly transferred between CPU and GPU memory:

| Function | Direction | Description |
|---|---|---|
| `cudaMalloc(&d_ptr, size)` | — | Allocate memory on GPU |
| `cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)` | CPU → GPU | Send data to GPU |
| `cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost)` | GPU → CPU | Get results from GPU |
| `cudaFree(d_ptr)` | — | Free GPU memory |

---

### 4.4 Vector Addition

#### Concept
Given two vectors A and B of size N, compute C[i] = A[i] + B[i] for all i.

Sequentially this takes O(N) time. With CUDA, we assign one thread per element — all N additions happen simultaneously.

#### CUDA Kernel
```cpp
__global__ void vectorAdd(int* a, int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

- `__global__` — marks this function as a CUDA kernel (runs on GPU, called from CPU)
- The `if (i < n)` guard prevents out-of-bounds access when N is not a multiple of blockDim

#### Launch Configuration
```cpp
int threads = 256;                          // 256 threads per block
int blocks = (N + threads - 1) / threads;  // enough blocks to cover all N elements
vectorAdd<<<blocks, threads>>>(da, db, dc, N);
```

For N = 1,000,000 and 256 threads/block → ~3907 blocks, each with 256 threads → 1,000,192 threads total (slightly more than N, hence the guard).

---

### 4.5 Matrix Multiplication

#### Concept
Given two N×N matrices A and B, compute C = A × B where:

```
C[row][col] = Σ A[row][k] * B[k][col]  for k = 0 to N-1
```

Sequentially this takes O(N³). With CUDA, each element of C is computed by a separate thread in parallel.

#### CUDA Kernel
```cpp
__global__ void matMul(int* a, int* b, int* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++)
            sum += a[row * n + k] * b[k * n + col];
        c[row * n + col] = sum;
    }
}
```

Matrices are stored as 1D arrays in row-major order: `A[row][col]` → `a[row * n + col]`

#### Launch Configuration (2D)
```cpp
dim3 threads(16, 16);                         // 16×16 = 256 threads per block
dim3 blocks((N+15)/16, (N+15)/16);            // 2D grid of blocks
matMul<<<blocks, threads>>>(da, db, dc, N);
```

For N = 512: blocks = 32×32 = 1024 blocks, each 16×16 threads → 262,144 threads total, one per output element.

---

### 4.6 CUDA Workflow (Both Programs)

```
1. Allocate arrays on CPU (Host)
2. Initialize data on CPU
3. cudaMalloc — allocate GPU memory
4. cudaMemcpy — copy data CPU → GPU
5. Launch kernel <<<blocks, threads>>>
6. cudaMemcpy — copy results GPU → CPU
7. Use results on CPU
8. cudaFree — free GPU memory
9. delete[] — free CPU memory
```

---

### 4.7 Comparison Table

| Feature | Vector Addition | Matrix Multiplication |
|---|---|---|
| Operation | C[i] = A[i] + B[i] | C[r][c] = Σ A[r][k]·B[k][c] |
| Thread grid | 1D | 2D |
| Threads/block | 256 (linear) | 16×16 = 256 (2D tile) |
| Work per thread | 1 addition | N multiplications |
| Complexity (seq) | O(N) | O(N³) |
| Complexity (GPU) | O(1) effectively | O(N) per thread |

---

### 4.8 Why Use CUDA Over OpenMP for These Tasks?

| Aspect | OpenMP | CUDA |
|---|---|---|
| Hardware | CPU (4–64 cores) | GPU (thousands of cores) |
| Best For | General parallel tasks | Data-parallel numerical tasks |
| Memory | Shared RAM | Separate VRAM |
| Programming | Pragma-based, easy | Explicit memory management |
| Speedup (matrix mul) | ~4–32× | ~100–1000× |
| Requires | Any modern CPU | NVIDIA GPU + CUDA Toolkit |
```
