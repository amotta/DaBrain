#ifndef LOG_H
#define LOG_H

#include <stdio.h>
#include "net.h"

void logFiring(const net_t * pNet, FILE * file);
void logCurrent(const net_t * pNet, FILE * logFile);

#endif
