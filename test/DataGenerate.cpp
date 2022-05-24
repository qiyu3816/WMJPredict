#include <time.h>
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;

int main()
{
    srand((unsigned)time(NULL));

    double x, y, z;
    ofstream ofs("data.txt");

    int n = 10000;
    while(n--)
    {
        if(rand() % 100 > 95)
            x += std::pow(-1, n) * double(rand() % 20) / 10.0, y += std::pow(-1, n) * double(rand() % 20) / 10.0, z += std::pow(-1, n) * double(rand() % 20) / 10.0;
        else
            x += std::pow(-1, n) * double(rand() % 10) / 10.0, y += std::pow(-1, n) * double(rand() % 10) / 10.0, z += std::pow(-1, n) * double(rand() % 10) / 10.0;
        
        ofs << x << " " << y << " " << z << "\n";
    }

    return 0;    
}