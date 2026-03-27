#ifndef __DEBUG_UTILS_HPP__
#define __DEBUG_UTILS_HPP__
#include <iostream>
#include <iomanip>
#include <string>

#ifndef VERBOSE
#define VERBOSE 0
#endif

// General logging macro
#if VERBOSE >= 1
    #define LOG_VERBOSE(level, msg) if (level <= VERBOSE) { std::cout << "[log" << level <<"] " << msg << std::endl; }
#else
    #define LOG_VERBOSE(level, msg) ((void)0) // No-op
#endif

// Conditional verbose logging macro
#if VERBOSE >= 1
    #define LOG_VERBOSE_IF(level, condition, msg) \
        if ((level <= VERBOSE) && (condition)) { std::cout << "[log" << level << "] " << msg << std::endl; }
#else
    #define LOG_VERBOSE_IF(level, condition, msg) ((void)0) // No-op
#endif

// Conditional verbose logging macro with else
#if VERBOSE >= 1
    #define LOG_VERBOSE_IF_ELSE(level, condition, msg_true, msg_false) \
        if ((level <= VERBOSE) && (condition)) { std::cout << "[log" << level << "] " << msg_true << std::endl; } \
        else if ((level <= VERBOSE)) { std::cout << "[log" << level << "] " << msg_false << std::endl; }
#else
    #define LOG_VERBOSE_IF_ELSE(level, condition, msg_true, msg_false) ((void)0) // No-op
#endif

#define DO_VERBOSE(level, operations) if (level <= VERBOSE) { operations }
#define MSG_HLINE(box_width) std::cout << std::string(box_width, '-') << std::endl;
#define MSG_BONDLINE(box_width) std::cout << '+' << std::string(box_width - 2, '-') << '+' << std::endl;

#define MSG_BOX_LINE(box_width, msg)                          \
    do {                                                 \
        std::ostringstream oss;                          \
        oss << msg;                                      \
        std::cout << "| " << std::left                   \
                  << std::setw(box_width - 4) << oss.str() \
                  << " |" << std::endl;                  \
    } while (0)

#define MSG_BOX(box_width, msg)                          \
    do {                                                 \
        std::ostringstream oss;                          \
        oss << msg;                                      \
        MSG_BONDLINE(box_width);                        \
        std::cout << "| " << std::left                   \
                  << std::setw(box_width - 4) << oss.str() \
                  << " |" << std::endl;                  \
        MSG_BONDLINE(box_width);                        \
    } while (0)

#define HEADER_PRINT(header, msg) \
    do { \
        std::ostringstream oss; \
        oss << msg; \
        std::cout << '[' << header << "]  " << oss.str() << std::endl; \
    } while (0)

#define OSTREAM2STRING(os) \
    do { \
        std::ostringstream oss; \
        oss << os; \
        return oss.str(); \
    } while (0)

#define header_print(header, msg) \
    do { \
        std::ostringstream oss; \
        oss << msg; \
        std::cout << '[' << header << "]  " << oss.str() << std::endl; \
    } while (0)

inline void box_print(std::string msg, int width = 40){
    MSG_BOX(width, msg);
}

inline void box_print_bound(int width = 40){
    MSG_BONDLINE(width);
}

inline void box_print_line(std::string msg, int width = 40){
    MSG_BOX_LINE(width, msg);
}

inline std::string size_t_to_string(size_t size){
    if (size /  1024 == 0) {
        return std::to_string(size);
    } else if (size / (1024 * 1024) == 0) {
        return std::to_string(size / 1024) + "K";
    } else if (size / (1024 * 1024) == 0){
        return std::to_string(size / (1024 * 1024)) + "M";
    } else {
        return std::to_string(size / (1024 * 1024 * 1024)) + "G";
    }
}

#endif
