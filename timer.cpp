#include <Windows.h>

static LARGE_INTEGER s_freq;

void timer_init()
{
    QueryPerformanceFrequency(&s_freq);
}

long long timer_sample()
{
    LARGE_INTEGER r;
    QueryPerformanceCounter(&r);
    return r.QuadPart;
}

double timer_duration(long long sample_diff)
{
    return (double)sample_diff / (double)s_freq.QuadPart;
}

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image_orig.h"

stbi_uc *stbi_orig_load(char const *filename, int *x, int *y, int *comp, int req_comp)
{
   return stbi_load(filename, x, y, comp, req_comp);
}