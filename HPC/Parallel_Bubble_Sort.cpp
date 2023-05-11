#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

void Serial_Bubble_Sort(vector<int> &a, int n){

	for ( int i = 0; i<n; i++ ){
		for ( int j = i+1; j<n; j++ ){
			if ( a[i] > a[j] ) swap(a[i],a[j]);
		} 
	}

}

void Parallel_Bubble_Sort(vector<int> &arr, int n)
{
    int i, j;
    for (i = 0; i < n - 1; i++)
    {
        if (i % 2 == 0)
        {
	    #pragma omp parallel for shared(arr)
            for (j = 0; j < n - 1; j += 2)
            {
                if (arr[j] > arr[j + 1])
                {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
        else
        {
	    #pragma omp parallel for shared(arr)
            for (j = 1; j < n - 1; j += 2)
            {
                if (arr[j] > arr[j + 1])
                {
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }
}

int main()
{
    vector<int> a = {64, 34, 25, 12, 22, 11, 90};
    vector<int> b = {64, 34, 25, 12, 22, 11, 90};
    int n = a.size();
    
    double start_time, end_time;
    
    start_time = omp_get_wtime();
    Parallel_Bubble_Sort(a, n);
    end_time = omp_get_wtime();
    cout << "Serial Bubble Sort : ";
    for ( auto i : a ) cout << i << " ";
    cout << "\n";
    cout << "Time Taken : " << (end_time - start_time) << "\n\n";
    
    start_time = omp_get_wtime();
    Serial_Bubble_Sort(b, n);
    end_time = omp_get_wtime();
    cout << "Parallel Bubble Sort : ";
    for ( auto i : b ) cout << i << " ";
    cout << "\n";
    cout << "Time Taken : " << (end_time - start_time) << "\n\n";
    
    return 0;
}
