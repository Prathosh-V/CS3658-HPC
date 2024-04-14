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
void LinearRegression(vector<double> x,vector<double> y, double alpha,int i,map<double,double>  &res, float (*params)[2])
{
    vector<double>error;  
    double err;
    double b0 = 0;                   
    double b1 = 0;   
    for (int i = 0; i < 8000; i ++) {   // since there are 1000 values and we want 5 epochs so run for loop for 20 times
    int idx = i % 5;              //for accessing index after every epoch
    double p = b0 + b1 * x[idx];  //calculating prediction
    err = abs(p - y[idx]);              // calculating error
    b0 = b0 - alpha * err;         // updating b0
    b1 = b1 - alpha * err * x[idx];// updating b1
    //cout<<"B0="<<b0<<" "<<"B1="<<b1<<" "<<"error="<<err<<endl;
    error.push_back(err);
    }
    sort(error.begin(),error.end(),custom_sort);//sorting based on error values
    res[error[0]]=i;
}
int main()
{
     std::ifstream file("C:\\Users\\Micro\\Downloads\\archive (3)\\placement.csv");
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file." << std::endl;
        return 1;
    }

    // Define a vector to store each line of the CSV file
    std::vector<std::vector<std::string>> data;

    // Read the CSV file line by line
    std::string line;
    while (std::getline(file, line)) {
        // Create a string stream from the line
        std::istringstream iss(line);
        std::string token;

        // Define a vector to store the tokens of the current line
        std::vector<std::string> tokens;

        // Extract tokens from the line
        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }

        // Add the tokens to the data vector
        data.push_back(tokens);
    }

    // Close the file
    file.close();
    int c=1;
    vector <double> x,y;

    for (const auto& row : data) {
        for (const auto& token : row) {
            c+=1;
            if(c%2) x.push_back(stod(token));
            else y.push_back(stod(token));
        }
    }
    
    
    int num=4;
    map<double,double> err;  
    float params[num][2];

    double parameters[7]={0.01,0.02,0.03,0.04,0.05,0.06,0.07};
    omp_set_num_threads(num);
    double start = omp_get_wtime();
    #pragma omp parallel for
    for(int i = 0; i < 7; ++i) {
        (LinearRegression(x,y,parameters[i],i,err,params));
    }
    double duration = omp_get_wtime() - start;
    cout << "Time taken " << duration << "s" << endl;
    cout<<"Optimal error and optimal co-efficients:"<<endl;
    for(auto i: err) 
    {
        cout<<i.first<<" "<<params[(int)i.second][0]<<" "<<params[(int)i.second][1]<<endl;
    }
}