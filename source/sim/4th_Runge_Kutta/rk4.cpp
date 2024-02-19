// Example program
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <cassert>
#include <math.h>

using namespace std;
// Utility functions and whatnot
vector<double> simple(vector<double> y0, double t0);
void print_vec(vector<double> vec);
void print_2d(vector<vector<double>> vec);
vector<double> scalar_mult(vector<double> vec, double mult);
vector<double> vector_add(vector<double> vec1, vector<double> vec2);
vector<double> vector_add4(vector<double> v1, vector<double> v2, vector<double>v3, vector<double> v4);

// todo: comment this mess!
class ODEint {
    public:
        int m_NUM_EQ{ 0 };
        vector<double> m_y0{ NULL };
        vector<vector<double>> m_history{ NULL };
        vector<vector<double>> m_work{ NULL };
        function<vector<double>(vector<double>, double)> m_fx;
    
    private:
        void rk4_single_step(vector<double> &y0, vector<vector<double>> work, function<vector<double>(vector<double>, double)> fx, double &t0, double dt)
        {
            assert(y0.size() == m_NUM_EQ);
            assert(work[0].size() == m_NUM_EQ);
            work[0] = fx(y0, t0);
            work[1] = fx(vector_add(y0, scalar_mult(work[0], (dt/2))), t0 + (dt/2));
            work[2] = fx(vector_add(y0, scalar_mult(work[1], (dt/2))), t0 + (dt/2));
            work[3] = fx(vector_add(y0, scalar_mult(work[2], dt)), t0 + dt);
            // behemoth line - maybe split?
            y0 = vector_add(y0, scalar_mult(vector_add4(work[0], scalar_mult(work[1], 2), scalar_mult(work[2], 2), work[3]), (dt/6)));
            t0 += dt;
        }
        
    public:
        // constructor
        ODEint(int num_eq, function<vector<double>(vector<double>, double)> func)
            : m_NUM_EQ(num_eq), m_fx(func)
        {
            vector<double> row (num_eq, 0);
            m_work = vector<vector<double>> (4, row);
        }
        
        // returns the populated history matrix
        vector<vector<double>> run(double t0, double T, double dt, vector<double> init_cond)
        {
            // initialize history matrix
            assert(init_cond.size() == m_NUM_EQ);
            int num_steps = int((T-t0) / dt) + 1;
            vector<double> row (m_NUM_EQ + 1);
            m_history = vector<vector<double>> (num_steps, row); // row[0] is time, unneccesary now but will be later
            m_history[0][0] = t0;
            for(int i = 0; i < m_NUM_EQ; i++)
            {
                m_history[0][i+1] = init_cond[i];
            }
            
            // initialize state to be init condition
            m_y0 = vector<double> (init_cond);
            
            // run the thingy
            // at the end of each step, y0 holds the new state of y (state being the values of )
            for(int i = 0; i < num_steps - 1; i++)
            {
                rk4_single_step(m_y0, m_work, m_fx, t0, dt);
                m_history[i+1][0] = t0;
                for (int j = 0; j < m_NUM_EQ; j++)
                {
                    m_history[i+1][j+1] = m_y0[j];
                }
            }
            return(m_history);
        }
};

// UTILS ---------------------------------------------------------------------------------

// scalar multiply a vector
vector<double> scalar_mult(vector<double> vec , double mult)
{
    for(int i=0; i < vec.size(); i++)
    {
        vec[i] *= mult;
    }
    return vec;
}

// adds vec2 to vec1
vector<double> vector_add(vector<double> vec1, vector<double> vec2)
{
    assert(vec1.size() == vec2.size());
    for(int i = 0; i < vec1.size(); i++)
    {  
        vec1[i] += vec2[i];
    }
    return vec1;
}

// adds 4 vectors. Useful for runge-kutta 4.
vector<double> vector_add4(vector<double> v1, vector<double> v2, vector<double>v3, vector<double> v4)
{
    assert(v1.size() == v2.size() && v1.size() == v3.size() && v1.size() == v4.size());
    for(int i = 0; i < v1.size(); i++)
    {
        v1[i] += (v2[i] + v3[i] + v4[i]);
    }
    return v1;
}

void print_vec(vector<double> vec)
{
    for(int i = 0; i < vec.size() ; i++)
    {
        cout << vec[i] << " ";
    }
    cout << endl;
}

void print_2d(vector<vector<double>> vec)
{
    for (int i = 0; i < vec.size(); i++)
    {
        for (int j = 0; j < (vec[i]).size(); j++) 
        {
            cout << vec[i][j] << " ";
        }
        cout << endl;
    }   
}

// END UTILS --------------------------------------------------------------

// simple test ODE
vector<double> simple(vector<double> y0, double t0)
{
    y0 = {y0[0] + y0[1],
          y0[0]};
    return y0;
}

vector<double> x_sqr(vector<double> y0, double t0)
{
    y0 = { 3*(pow(t0, 3.0)) };
    return y0;
}

int main()
{
    
    ODEint a(2.0, simple);
    
    vector<double> init = {1, 2};
    print_2d(a.run(0.0, 10.0, 0.1, init));
    
    cout << "-------------------" << endl;
    
    ODEint b(1, x_sqr);
    vector<double> b_init = {0.0};
    cout << "running" << endl;
    vector<vector<double>> out = b.run(0.0, 10.0, 0.1, b_init);
    cout << "out.run returned" << endl;
    print_2d(out);
    
    
}

