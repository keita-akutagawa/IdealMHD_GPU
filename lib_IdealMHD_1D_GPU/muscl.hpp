#include <vector>

class MUSCL
{
public:
    void getLeftComponent(
        const std::vector<double> q, 
        std::vector<double>& qLeft
    );
    void getRightComponent(
        const std::vector<double> q, 
        std::vector<double>& qRight
    );
};

