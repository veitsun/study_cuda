#ifndef TLPI_HDR_H
#define TLPI_HDR_H
// 系统调用所需要的头文件

// #include <errno.h>
// #include <stdio.h>
#include <stdlib.h>
// #include <string.h>
#include <sys/types.h>
#include <unistd.h>

// #include "get_num.h"
#include "error_functions.h"

typedef enum { FALSE, TRUE } Boolean;
#define min(m, n) ((m) < (n) ? (m) : (n))
#define max(m, n) ((m) > (n) ? (m) : (n))

#endif