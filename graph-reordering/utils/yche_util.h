//
// Created by yche on 8/26/18.
//

#ifndef SRC_YCHE_UTIL_H
#define SRC_YCHE_UTIL_H


#include <string>

#include <iomanip>
#include <locale>
#include <sstream>

using namespace std;

template<class T>
std::string FormatWithCommas(T value) {
    std::stringstream ss;
    ss.imbue(std::locale(""));
    ss << std::fixed << value;
    return ss.str();
}

#endif //SRC_YCHE_UTIL_H
