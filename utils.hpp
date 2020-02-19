
#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <string>
#include <ctime>

namespace bosched
{
    inline std::string
    format_current_time() noexcept
    {
        time_t rawtime;
        struct tm * timeinfo;
        char buffer[80];

        time (&rawtime);
        timeinfo = localtime(&rawtime);

        strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeinfo);
        return std::string(buffer);
    }
}

#endif
