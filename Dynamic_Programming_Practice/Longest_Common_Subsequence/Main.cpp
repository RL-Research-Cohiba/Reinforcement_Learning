#include <iostream>
#include <string>
using namespace std;

// Function to find length of Longest Common Subsequence of
// sequences X[0..m-1] and Y[0..n-1]
int LCSLength(string X, string Y, int m, int n)
{
	// return if we have reached the end of either sequence
	if (m == 0 || n == 0)
		return 0;

	// if last character of X and Y matches
	if (X[m - 1] == Y[n - 1])
		return LCSLength(X, Y, m - 1, n - 1) + 1;

	// else if last character of X and Y don't match
	return max(LCSLength(X, Y, m, n - 1), LCSLength(X, Y, m - 1, n));
}

// Longest Common Subsequence
int main()
{
	string X = "ABCBDAB", Y = "BDCABA";

	cout << "The length of LCS is " <<
			LCSLength(X, Y, X.length(), Y.length());

	return 0;
}