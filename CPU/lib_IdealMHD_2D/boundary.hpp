#include <vector>
#include <string>


class Boundary
{
private:

public:
    std::string boundaryType;

    void periodicBoundary(
        std::vector<std::vector<std::vector<double>>>& U
    );

    void symmetricBoundary2ndX(
        std::vector<std::vector<std::vector<double>>>& U
    );

    void symmetricBoundary2ndY(
        std::vector<std::vector<std::vector<double>>>& U
    );

private:

};


