#include <iostream>
#include <string>
#include <cstring>
using namespace std;




// main function
int main()
{
  string X = "ABC", Y = "BABA";
  int m = X.length(), n = Y.length();

  // find longest common substring
  cout << "The Longest Common Substring is " << LCS(X, Y, m, n);

  return 0;
}
