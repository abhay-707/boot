#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
using namespace std;

const int N = 7;
vector<int> adj[N];
bool visited[N];

void addEdge(int u, int v) {
    adj[u].push_back(v);
    adj[v].push_back(u);
}

void parallelBFS(int start) {
    fill(visited, visited + N, false);
    queue<int> q;
    visited[start] = true;
    q.push(start);

    cout << "BFS: ";
    while (!q.empty()) {
        int size = q.size();
        vector<int> level;

        for (int i = 0; i < size; i++) {
            level.push_back(q.front());
            q.pop();
        }

        #pragma omp parallel for
        for (int i = 0; i < (int)level.size(); i++) {
            int node = level[i];
            #pragma omp critical
            cout << node << " ";

            for (int neighbor : adj[node]) {
                #pragma omp critical
                {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        q.push(neighbor);
                    }
                }
            }
        }
    }
    cout << "\n";
}

void parallelDFS(int node) {
    bool skip = false;
    #pragma omp critical
    {
        if (visited[node]) skip = true;
        else { visited[node] = true; cout << node << " "; }
    }
    if (skip) return;

    #pragma omp parallel for
    for (int i = 0; i < (int)adj[node].size(); i++) {
        if (!visited[adj[node][i]])
            parallelDFS(adj[node][i]);
    }
}

int main() {
    //        0
    //       / \
    //      1   2
    //     / \ / \
    //    3  4 5  6
    addEdge(0, 1); addEdge(0, 2);
    addEdge(1, 3); addEdge(1, 4);
    addEdge(2, 5); addEdge(2, 6);

    parallelBFS(0);

    fill(visited, visited + N, false);
    cout << "DFS: ";
    parallelDFS(0);
    cout << "\n";

    return 0;
}
