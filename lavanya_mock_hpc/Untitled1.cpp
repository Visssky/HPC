
#include <bits/stdc++.h>

using namespace std;

// Function to perform Parallel BFS
void parallelBFS(vector<vector<int>>& adj_list, int source, vector<bool>& visited, vector<int>& bfs_order) {
    queue<int> q;
    q.push(source);

    // Parallel loop over the queue
    #pragma omp parallel
    {
        while (!q.empty()) {
            // Get the next vertex from the queue
            #pragma omp for
            for (int i = 0; i < q.size(); i++) {
                int curr = q.front();
                q.pop();

                // If the current vertex has not been visited, mark it as visited
                // and explore all its neighbors
                if (!visited[curr]) {
                    #pragma omp critical
                    {
                        visited[curr] = true;
                        bfs_order.push_back(curr); // add the visited node to the bfs_order vector
                    }
                    for (int j = 0; j < adj_list[curr].size(); j++) {
                        int neighbor = adj_list[curr][j];

                        // Add the neighbor to the queue if it has not been visited
                        if (!visited[neighbor]) {
                            q.push(neighbor);
                        }
                    }
                }
            }
        }
    }
}

// Function to perform Parallel DFS
void parallelDFS(vector<vector<int>>& adj_list, int source, vector<bool>& visited, vector<int>& dfs_order) {
    stack<int> s;
    s.push(source);

    // Parallel loop over the stack
    #pragma omp parallel
    {
        while (!s.empty()) {
            // Get the next vertex from the stack
            #pragma omp for
            for (int i = 0; i < s.size(); i++) {
                int curr = s.top();
                s.pop();

                // If the current vertex has not been visited, mark it as visited
                // and explore all its neighbors
                if (!visited[curr]) {
                    #pragma omp critical
                    {
                        visited[curr] = true;
                        dfs_order.push_back(curr); // add the visited node to the dfs_order vector
                    }
                    for (int j = 0; j < adj_list[curr].size(); j++) {
                        int neighbor = adj_list[curr][j];

                        // Add the neighbor to the stack if it has not been visited
                        if (!visited[neighbor]) {
                            s.push(neighbor);
                        }
                    }
                }
            }
        }
    }
}

int main() {
    // Construct the adjacency list
    vector<vector<int>> adj_list = {
        {1, 2},
        {0, 3, 4},
        {0, 5, 6},
        {1},
        {1},
        {2},
        {2}
    };

    // Perform Parallel BFS from node 0
    int source = 0;
    int n = adj_list.size();
    vector<bool> visited(n, false);
    vector<int> bfs_order;
    parallelBFS(adj_list, source, visited, bfs_order);

    // Print the visited nodes and the BFS order
    cout << "BFS order: ";
    for (int i = 0; i < bfs_order.size(); i++) {
        cout << bfs_order[i] << " ";
    }
    cout << endl;

    for (int i = 0; i < n; i++) {
        if (visited[i]) {
            cout << "Node " << i << " has been visited" << endl;
        }
    }

    // Perform Parallel DFS from
// reset the visited vector
fill(visited.begin(), visited.end(), false);
vector<int> dfs_order;
parallelDFS(adj_list, source, visited, dfs_order);

// Print the visited nodes and the DFS order
cout << "DFS order: ";
for (int i = 0; i < dfs_order.size(); i++) {
    cout << dfs_order[i] << " ";
}
cout << endl;

for (int i = 0; i < n; i++) {
    if (visited[i]) {
        cout << "Node " << i << " has been visited" << endl;
    }
}

return 0;
}
