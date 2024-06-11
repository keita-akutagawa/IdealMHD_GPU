#include <vector>

class MUSCL
{
private:
    int nDirection;

public:
    MUSCL(int nSize) : nDirection(nSize) {}
    
    void getLeftComponent(
        const std::vector<double>& q, 
        std::vector<double>& qLeft
    );
    void getRightComponent(
        const std::vector<double>& q, 
        std::vector<double>& qRight
    );
};

