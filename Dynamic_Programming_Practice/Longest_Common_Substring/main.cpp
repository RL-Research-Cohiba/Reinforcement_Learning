#include <iostream>
#include <string>
#include <cstring>
using namespace std;

// Function to find the Longest common substring of sequences
// X[0..m-1] and Y[0..n-1]
string LCS(string X, string Y, int m, int n)
{
  int maxlen = 0; // Stores the max length of LCS
  int endingIndex = m; // Stores the ending index of LCS in X

  // Lookup[i][j] stores the length of LCS substring
  // X[0..i-1], Y[0..j-1]
  int lookup[m + 1][n + 1];

  // Initialize all cells of lookup table to 0
  memset(lookup, 0, sizeof(lookup));
  

  



// Main function
int main()
{
  string X = "ABC", Y = "BABA";
  int m = X.length(), n = Y.length();

  // Find longest common substring
  cout << "The Longest Common Substring is " << LCS(X, Y, m, n);

  return 0;
}

 
