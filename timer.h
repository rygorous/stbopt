#ifndef TIMER_H
#define TIMER_H

void timer_init();
long long timer_sample();
double timer_duration(long long sample_diff);

unsigned char *stbi_orig_load(char const *filename, int *x, int *y, int *comp, int req_comp);

#endif