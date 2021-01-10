int main()
{
    int arr[] = {1, 101, 2, 3, 100, 4, 5};
    int n = sizeof(arr) / sizeof(arr[0]);
    printf("Sum of maximum sum increasing "
            "subsequence is %d\n",
            maxSumIS(arr, n ) );
    return 0;
}