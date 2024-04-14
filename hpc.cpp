#include<bits/stdc++.h> 
#include <omp.h>
using namespace std;   

bool custom_sort(double a, double b) /* this custom sort function is defined to 
                                     sort on basis of min absolute value or error*/
{
    double  a1=abs(a-0);
    double  b1=abs(b-0);
    return a1<b1;
}
void LinearRegression(double *x,double *y, double alpha,int i,float final[][3])
{
    vector<double>error;  
    double err;
    double b0 = 0;                   
    double b1 = 0;   
    for (int i = 0; i < 20; i ++) {   // since there are 5 values and we want 4 epochs so run for loop for 20 times
    int idx = i % 5;              //for accessing index after every epoch
    double p = b0 + b1 * x[idx];  //calculating prediction
    err = p - y[idx];              // calculating error
    b0 = b0 - alpha * err;         // updating b0
    b1 = b1 - alpha * err * x[idx];// updating b1
    //cout<<"B0="<<b0<<" "<<"B1="<<b1<<" "<<"error="<<err<<endl;
    error.push_back(err);
    }
    sort(error.begin(),error.end(),custom_sort);//sorting based on error values
    final[i][0]=b0;
    final[i][1]=b1;
    final[i][2]=error[0];
}
int main()
{
    double x[] = {1, 2, 4, 3, 5};    
    double y[] = {1, 3, 3, 2, 5};    
    //dummy values for x and y
    int test=10;
    int num=2;
    float final[num][3];
    double parameters[4]={0.01,0.02,0.03,0.04};
    omp_set_num_threads(num);
    double start = omp_get_wtime();
    #pragma omp parallel for
    for(int i = 0; i < 4; ++i) {
        (LinearRegression(x,y,parameters[i],i,final));
    }
    double duration = omp_get_wtime() - start;
    cout << "Time taken " << duration << "s" << endl;
}