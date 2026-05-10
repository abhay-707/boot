#include <iostream>
#include <vector>
#include <climits>
#include <omp.h>
using namespace std;

int main() {
    const int N = 10000000;
    vector<int> arr(N);
    srand(42);
    for (int& x : arr) x = rand() % 10000;

    int mn = INT_MAX, mx = INT_MIN, sm = 0;

    #pragma omp parallel for reduction(min:mn) reduction(max:mx) reduction(+:sm)
    for (int i = 0; i < N; i++) {
        if (arr[i] < mn) mn = arr[i];
        if (arr[i] > mx) mx = arr[i];
        sm += arr[i];
    }

    double avg = (double)sm / N;

    cout << "Min: " << mn  << "\n";
    cout << "Max: " << mx  << "\n";
    cout << "Sum: " << sm  << "\n";
    cout << "Avg: " << avg << "\n";

    return 0;
}
