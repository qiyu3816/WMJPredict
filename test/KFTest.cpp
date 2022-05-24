// KF 二维输入数据预测测试 支持命令行指定数据文件输入和无文件随机数展示两种运行方式
// 程序使用C++调用Python的matplotlib数据可视化模块 在运行过程中实时显示每一个输入数据及其滤波后数据的折线图
#include "../include/KF.hpp"
#include <matplotlibcpp.h>
#include <fstream>

namespace plt = matplotlibcpp;

int main(int argc, char* argv[])
{
    std::ifstream ifs;
    if(argc != 2)
        std::cout << "Usage: ./EKFTest [input file]\n";
    else
        ifs.open(argv[1]);
    
    srand((unsigned)time(NULL));

    std::shared_ptr<predict::KF> kf = std::make_shared<predict::KF>(predict::KF::POSE);

    int x_lim = 300, y_lim = 10;
    std::vector<int> t(x_lim);
    for(int i = 0; i < x_lim; i++) t[i] = i;
    std::vector<float> src_x(x_lim), proc_x(x_lim), src_y(x_lim), proc_y(x_lim);
    cv::Point2f src_point, proc_point;

    plt::ion();
    plt::figure_size(1000, 500);
    plt::xlim(0, x_lim);
    plt::ylim(-y_lim, y_lim);
    int n = 1000;
    while((ifs.is_open() ? 1 : (n--)))
    {
        if(ifs.is_open())
            if(ifs >> src_point.x >> src_point.y) ;
            else break;
        else
            src_point.x += std::pow(-1, n) * float(rand() % 100) / 100.0, src_point.y += std::pow(-1, n) * float(rand() % 100) / 100.0;
        
        std::cout << "src_point : " << src_point << std::endl;
        proc_point = kf->predict(src_point);
        std::cout << "proc_point: " << proc_point << std::endl;
        std::cout << "--------------------\n";

        src_x.erase(src_x.begin());
        src_y.erase(src_y.begin());
        proc_x.erase(proc_x.begin());
        proc_y.erase(proc_y.begin());
        src_x.push_back(src_point.x);
        src_y.push_back(src_point.y);
        proc_x.push_back(proc_point.x);
        proc_y.push_back(proc_point.y);

        plt::clf();
        plt::subplot(1, 2, 1);
        plt::plot(t, src_x, "b-");
        plt::plot(t, proc_x, "r--");
        plt::title("x(b:src,r:proc)");

        plt::subplot(1, 2, 2);
        plt::plot(t, src_y, "b-");
        plt::plot(t, proc_y, "r--");
        plt::title("y(b:src,r:proc)");

        plt::pause(0.001);
    }

    if(argc == 2) ifs.close();
    return 0;
}