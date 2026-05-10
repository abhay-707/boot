#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

// ─────────────────────────────────────────
//  BUBBLE SORT
// ─────────────────────────────────────────

void seqBubbleSort(vector<int> arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
}

void parBubbleSort(vector<int> arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        int start = i % 2; // odd-even transposition
        #pragma omp parallel for
        for (int j = start; j < n - 1; j += 2)
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
    }
}

// ─────────────────────────────────────────
//  MERGE SORT
// ─────────────────────────────────────────

void merge(vector<int>& arr, int l, int m, int r) {
    vector<int> left(arr.begin() + l, arr.begin() + m + 1);
    vector<int> right(arr.begin() + m + 1, arr.begin() + r + 1);
    int i = 0, j = 0, k = l;
    while (i < (int)left.size() && j < (int)right.size())
        arr[k++] = (left[i] <= right[j]) ? left[i++] : right[j++];
    while (i < (int)left.size())  arr[k++] = left[i++];
    while (j < (int)right.size()) arr[k++] = right[j++];
}

void seqMergeSort(vector<int>& arr, int l, int r) {
    if (l >= r) return;
    int m = (l + r) / 2;
    seqMergeSort(arr, l, m);
    seqMergeSort(arr, m + 1, r);
    merge(arr, l, m, r);
}

void parMergeSort(vector<int>& arr, int l, int r, int depth = 0) {
    if (l >= r) return;
    int m = (l + r) / 2;
    if (depth < 4) { // limit thread spawning depth
        #pragma omp parallel sections
        {
            #pragma omp section
            parMergeSort(arr, l, m, depth + 1);
            #pragma omp section
            parMergeSort(arr, m + 1, r, depth + 1);
        }
    } else {
        seqMergeSort(arr, l, m);
        seqMergeSort(arr, m + 1, r);
    }
    merge(arr, l, m, r);
}

// ─────────────────────────────────────────
//  MAIN
// ─────────────────────────────────────────

int main() {
    const int N = 10000;
    vector<int> base(N);
    srand(42);
    for (int& x : base) x = rand() % 10000;

    vector<int> arr;
    double t1, t2;

    cout << "Array size: " << N << "\n";
    cout << "Threads: " << omp_get_max_threads() << "\n\n";

    cout << "========== BUBBLE SORT ==========\n";

    arr = base;
    t1 = omp_get_wtime();
    seqBubbleSort(arr);
    t2 = omp_get_wtime();
    cout << "Sequential: " << (t2 - t1) << " sec\n";

    arr = base;
    t1 = omp_get_wtime();
    parBubbleSort(arr);
    t2 = omp_get_wtime();
    cout << "Parallel:   " << (t2 - t1) << " sec\n";

    cout << "\n========== MERGE SORT ===========\n";

    arr = base;
    t1 = omp_get_wtime();
    seqMergeSort(arr, 0, N - 1);
    t2 = omp_get_wtime();
    cout << "Sequential: " << (t2 - t1) << " sec\n";

    arr = base;
    t1 = omp_get_wtime();
    parMergeSort(arr, 0, N - 1);
    t2 = omp_get_wtime();
    cout << "Parallel:   " << (t2 - t1) << " sec\n";

    return 0;
}
