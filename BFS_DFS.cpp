#include<bits/stdc++.h>
using namespace std;

void Normal_BFS( int start, map<int,vector<int>> &adj ){
	int n = adj.size();
	queue<int> Q;
	vector<bool> vis(n+1, false);
	Q.push(start);
	vis[start] = true;
	cout << "BFS : ";
	while ( !Q.empty() ){
		int k = Q.size();
		while ( k-- ){
			int p = Q.front();
			Q.pop();
			cout << p << " ";
			for ( auto i : adj[p] ){
				if ( !vis[i] ){
					vis[i] = true;
					Q.push(i);
				} 
			}		
		}
	} cout << "\n"; 	
}

void Parallel_BFS( int start, map<int,vector<int>> &adj, vector<bool> &vis ){
	queue<int> Q;
	Q.push(start);
	vis[start] = true;
	#pragma omp parallel
	{
        	#pragma omp single
        	{
		    	while ( !Q.empty() ) {
				int p = Q.front();
				cout << p << " ";
				Q.pop();
				#pragma omp task
				{
					for ( auto i : adj[p] ){
				    		if ( !vis[i] ) {
							vis[i] = true;
							Q.push(i);
							#pragma omp task
								Parallel_BFS(i, adj, vis);
						}
					}
				}
            		}
        	}
    	}
}

void Call_Parallel_BFS( int start, map<int,vector<int>> &adj ){
	int n = adj.size();
	vector<bool> vis(n+1, false);
	cout << "Parallel BFS : ";
	Parallel_BFS(start, adj, vis);
	cout << "\n";
}

void Normal_DFS( int start, map<int,vector<int>> &adj ){
	int n = adj.size();
	stack<int> S;
	vector<bool> vis(n+1, false);
	S.push(start);
	vis[start] = true;
	cout << "DFS : ";
	while ( !S.empty() ){
		int p = S.top();
		S.pop();
		cout << p << " ";
		for ( auto i : adj[p] ){
			if ( !vis[i] ){
				vis[i] = true;
				S.push(i);
			}
		}
	} cout << "\n";
}

void Parallel_DFS( int start, map<int,vector<int>> &adj, vector<bool> &vis ){
	stack<int> S;
	S.push(start);
	vis[start] = true;
	#pragma omp parallel
	{
		#pragma omp single
		{
			while (!S.empty()){
				int p = S.top();
				cout << p << " ";
                		S.pop();
				#pragma omp task
				{
                    			for ( auto i : adj[p] ) {
                        			if ( !vis[i] ) {
                            				vis[i] = true;
                            				S.push(i);
							#pragma omp task
                            					Parallel_DFS( i, adj, vis );
                        			}
                    			}
                		}
            		}
        	}
    	}
}

void Call_Parallel_DFS( int start, map<int,vector<int>> &adj ){
	int n = adj.size();
	vector<bool> vis(n+1, false);
	cout << "Parallel DFS : ";
	Parallel_DFS(start, adj, vis);
	cout << "\n";
}

int main(){

	map<int,vector<int>> adj;
	adj[0] = {1, 2};
    	adj[1] = {0, 2, 3, 4};
    	adj[2] = {0, 1, 5, 6};
    	adj[3] = {1, 4};
    	adj[4] = {1, 3};
    	adj[5] = {2};
    	adj[6] = {2};
    	
    	Normal_BFS(0, adj);
    	Call_Parallel_BFS(0, adj);
    	Normal_DFS(0, adj);
    	Call_Parallel_DFS(0, adj);
	
	return 0;
}
