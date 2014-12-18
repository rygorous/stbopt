#define STBI_SIMD
#define STB_IMAGE_IMPLEMENTATION
#include "../stb/stb_image.h"
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include "timer.h"

#include <emmintrin.h>

static void panic(char const *fmt, ...)
{
   va_list arg;
   va_start(arg, fmt);
   vfprintf(stderr, fmt, arg);
   va_end(arg);
   exit(1);
}

static void test_correct(char const *filename)
{
   int x0, y0, n0;
   int x1, y1, n1;
   unsigned char *data0 = stbi_load(filename, &x0, &y0, &n0, 0);
   unsigned char *data1 = stbi_orig_load(filename, &x1, &y1, &n1, 0);

   printf("%dx%d n=%d\n", x0, y0, n0);
   if (x0 != x1 || y0 != y1 || n0 != n1)
      panic("image dimension mismatch!\n");

   if (memcmp(data0, data1, x0*y0*n0) != 0)
      panic("image data mismatch!\n");

   stbi_image_free(data0);
   stbi_image_free(data1);
   
   printf("%s decodes correctly.\n", filename);
}

static void bench(char const *filename, int reqcomp)
{
   static const int kRuns = 30;

   timer_init();
   long long tstart = timer_sample();

   for (int run = 0; run < kRuns; ++run) {
      int x, y, n;
      unsigned char *data = stbi_load(filename, &x, &y, &n, reqcomp);
      stbi_image_free(data);
   }

   double dur_ms = timer_duration(timer_sample() - tstart) * 1000.0;
   printf("%d runs in %.3f ms -> %.3f ms/run\n", kRuns, dur_ms, dur_ms / kRuns);
}

static void my_YCbCr_to_RGB(stbi_uc *out, stbi_uc const *y, stbi_uc const *pcb, stbi_uc const *pcr, int count, int step)
{
   int i=0;

   if (step == 4) {
      // this is a fairly straightforward implementation and not super-optimized.
      __m128i signflip = _mm_set1_epi8(-0x80);
      __m128i cr_const0 = _mm_set1_epi16((short) ( 1.40200f*4096.0f));
      __m128i cr_const1 = _mm_set1_epi16((short) (-0.71414f*4096.0f));
      __m128i cb_const0 = _mm_set1_epi16((short) (-0.34414f*4096.0f));
      __m128i cb_const1 = _mm_set1_epi16((short) ( 1.77200f*4096.0f));
      __m128i y_bias = _mm_set1_epi16(8);
      __m128i xw = _mm_set1_epi16(255);

      for (; i+7 < count; i += 8) {
         // load
         __m128i zero = _mm_setzero_si128();
         __m128i y_bytes = _mm_loadl_epi64((__m128i *) (y+i));
         __m128i cr_bytes = _mm_loadl_epi64((__m128i *) (pcr+i));
         __m128i cb_bytes = _mm_loadl_epi64((__m128i *) (pcb+i));
         __m128i cr_bias = _mm_xor_si128(cr_bytes, signflip); // -128
         __m128i cb_bias = _mm_xor_si128(cb_bytes, signflip); // -128

         // unpack to short (and left-shift cr, cb by 8)
         __m128i yw  = _mm_unpacklo_epi8(y_bytes, zero);
         __m128i crw = _mm_unpacklo_epi8(_mm_setzero_si128(), cr_bias);
         __m128i cbw = _mm_unpacklo_epi8(_mm_setzero_si128(), cb_bias);

         // color transform
         __m128i yws = _mm_slli_epi16(yw, 4);
         __m128i cr0 = _mm_mulhi_epi16(cr_const0, crw);
         __m128i cb0 = _mm_mulhi_epi16(cb_const0, cbw);
         __m128i ywb = _mm_add_epi16(yws, y_bias);
         __m128i cb1 = _mm_mulhi_epi16(cbw, cb_const1);
         __m128i cr1 = _mm_mulhi_epi16(crw, cr_const1);
         __m128i rws = _mm_add_epi16(cr0, ywb);
         __m128i gwt = _mm_add_epi16(cb0, ywb);
         __m128i bws = _mm_add_epi16(ywb, cb1);
         __m128i gws = _mm_add_epi16(gwt, cr1);

         // descale
         __m128i rw = _mm_srai_epi16(rws, 4);
         __m128i bw = _mm_srai_epi16(bws, 4);
         __m128i gw = _mm_srai_epi16(gws, 4);

         // back to byte, set up for transpose
         __m128i brb = _mm_packus_epi16(rw, bw);
         __m128i gxb = _mm_packus_epi16(gw, xw);

         // transpose to interleave channels
         __m128i t0 = _mm_unpacklo_epi8(brb, gxb);
         __m128i t1 = _mm_unpackhi_epi8(brb, gxb);
         __m128i o0 = _mm_unpacklo_epi16(t0, t1);
         __m128i o1 = _mm_unpackhi_epi16(t0, t1);

         // store
         _mm_storeu_si128((__m128i *) (out + 0), o0);
         _mm_storeu_si128((__m128i *) (out + 16), o1);
         out += 32;
      }
   }

   for (; i < count; ++i) {
      int y_fixed = (y[i] << 16) + 32768; // rounding
      int r,g,b;
      int cr = pcr[i] - 128;
      int cb = pcb[i] - 128;
      r = y_fixed + cr*float2fixed(1.40200f);
      g = y_fixed - cr*float2fixed(0.71414f) - cb*float2fixed(0.34414f);
      b = y_fixed                            + cb*float2fixed(1.77200f);
      r >>= 16;
      g >>= 16;
      b >>= 16;
      if ((unsigned) r > 255) { if (r < 0) r = 0; else r = 255; }
      if ((unsigned) g > 255) { if (g < 0) g = 0; else g = 255; }
      if ((unsigned) b > 255) { if (b < 0) b = 0; else b = 255; }
      out[0] = (stbi_uc)r;
      out[1] = (stbi_uc)g;
      out[2] = (stbi_uc)b;
      out[3] = 255;
      out += step;
   }
}

#undef stbi__f2f
#define stbi__f2f(x)  ((int) (((x) * 4096 + 0.5)))

// dot product constant: even elems=x, odd elems=y
#define dpconst(x,y)  _mm_setr_epi16((x),(y),(x),(y),(x),(y),(x),(y))

// out(0) = c0[even]*x + c0[odd]*y   (c0, x, y 16-bit, out 32-bit)
// out(1) = c1[even]*x + c1[odd]*y
#define dct_rot(out0,out1, x,y,c0,c1) \
   __m128i c0##lo = _mm_unpacklo_epi16((x),(y)); \
   __m128i c0##hi = _mm_unpackhi_epi16((x),(y)); \
   __m128i out0##_l = _mm_madd_epi16(c0##lo, c0); \
   __m128i out0##_h = _mm_madd_epi16(c0##hi, c0); \
   __m128i out1##_l = _mm_madd_epi16(c0##lo, c1); \
   __m128i out1##_h = _mm_madd_epi16(c0##hi, c1)

// out = in << 12  (in 16-bit, out 32-bit)
#define dct_widen(out, in) \
   __m128i out##_l = _mm_srai_epi32(_mm_unpacklo_epi16(_mm_setzero_si128(), (in)), 4); \
   __m128i out##_h = _mm_srai_epi32(_mm_unpackhi_epi16(_mm_setzero_si128(), (in)), 4)

// wide add
#define dct_wadd(out, a, b) \
   __m128i out##_l = _mm_add_epi32(a##_l, b##_l); \
   __m128i out##_h = _mm_add_epi32(a##_h, b##_h)

// wide sub
#define dct_wsub(out, a, b) \
   __m128i out##_l = _mm_sub_epi32(a##_l, b##_l); \
   __m128i out##_h = _mm_sub_epi32(a##_h, b##_h)

// butterfly a/b, add bias, then shift by "s" and pack
#define dct_bfly32o(out0, out1, a,b,bias,s) \
   { \
      __m128i abiased_l = _mm_add_epi32(a##_l, bias); \
      __m128i abiased_h = _mm_add_epi32(a##_h, bias); \
      dct_wadd(sum, abiased, b); \
      dct_wsub(dif, abiased, b); \
      out0 = _mm_packs_epi32(_mm_srai_epi32(sum_l, s), _mm_srai_epi32(sum_h, s)); \
      out1 = _mm_packs_epi32(_mm_srai_epi32(dif_l, s), _mm_srai_epi32(dif_h, s)); \
   }

// 8-bit interleave step (for transposes)
#define dct_interleave8(a, b) \
   tmp = a; \
   a = _mm_unpacklo_epi8(a, b); \
   b = _mm_unpackhi_epi8(tmp, b)

// 16-bit interleave step (for transposes)
#define dct_interleave16(a, b) \
   tmp = a; \
   a = _mm_unpacklo_epi16(a, b); \
   b = _mm_unpackhi_epi16(tmp, b)

#define dct_pass(bias,shift) \
   { \
      /* even part */ \
      dct_rot(t2e,t3e, row2,row6, rot0_0,rot0_1); \
      __m128i sum04 = _mm_add_epi16(row0, row4); \
      __m128i dif04 = _mm_sub_epi16(row0, row4); \
      dct_widen(t0e, sum04); \
      dct_widen(t1e, dif04); \
      dct_wadd(x0, t0e, t3e); \
      dct_wsub(x3, t0e, t3e); \
      dct_wadd(x1, t1e, t2e); \
      dct_wsub(x2, t1e, t2e); \
      /* odd part */ \
      dct_rot(y0o,y2o, row7,row3, rot2_0,rot2_1); \
      dct_rot(y1o,y3o, row5,row1, rot3_0,rot3_1); \
      __m128i sum17 = _mm_add_epi16(row1, row7); \
      __m128i sum35 = _mm_add_epi16(row3, row5); \
      dct_rot(y4o,y5o, sum17,sum35, rot1_0,rot1_1); \
      dct_wadd(x4, y0o, y4o); \
      dct_wadd(x5, y1o, y5o); \
      dct_wadd(x6, y2o, y5o); \
      dct_wadd(x7, y3o, y4o); \
      dct_bfly32o(row0,row7, x0,x7,bias,shift); \
      dct_bfly32o(row1,row6, x1,x6,bias,shift); \
      dct_bfly32o(row2,row5, x2,x5,bias,shift); \
      dct_bfly32o(row3,row4, x3,x4,bias,shift); \
   }

#if 0
// this is the version with dequant in IDCT
#define dct_load(data, dequantize, row) \
   _mm_mullo_epi16(_mm_load_si128((const __m128i *) (data + (row)*8)), \
                   _mm_loadu_si128((const __m128i *) (dequantize + (row)*8)))
#else
#define dct_load(data, dequantize, row) \
   _mm_load_si128((const __m128i *) (data + (row)*8))
#endif

static void my_IDCT(stbi_uc *out, int out_stride, short data[64], unsigned short *dequantize)
{
   // This is constructed to match the IJG slow IDCT exactly.
   // ("dequantize" ignored for now since caller always passes all-1s.)
   __m128i rot0_0 = dpconst(stbi__f2f(0.5411961f), stbi__f2f(0.5411961f) + stbi__f2f(-1.847759065f));
   __m128i rot0_1 = dpconst(stbi__f2f(0.5411961f) + stbi__f2f( 0.765366865f), stbi__f2f(0.5411961f));
   __m128i rot1_0 = dpconst(stbi__f2f(1.175875602f) + stbi__f2f(-0.899976223f), stbi__f2f(1.175875602f));
   __m128i rot1_1 = dpconst(stbi__f2f(1.175875602f), stbi__f2f(1.175875602f) + stbi__f2f(-2.562915447f));
   __m128i rot2_0 = dpconst(stbi__f2f(-1.961570560f) + stbi__f2f( 0.298631336f), stbi__f2f(-1.961570560f));
   __m128i rot2_1 = dpconst(stbi__f2f(-1.961570560f), stbi__f2f(-1.961570560f) + stbi__f2f( 3.072711026f));
   __m128i rot3_0 = dpconst(stbi__f2f(-0.390180644f) + stbi__f2f( 2.053119869f), stbi__f2f(-0.390180644f));
   __m128i rot3_1 = dpconst(stbi__f2f(-0.390180644f), stbi__f2f(-0.390180644f) + stbi__f2f( 1.501321110f));

   // Rounding biases in column/row passes.
   // See stbi__idct_block for explanation.
   __m128i bias_0 = _mm_set1_epi32(512);
   __m128i bias_1 = _mm_set1_epi32(65536 + (128<<17));

   __m128i row0, row1, row2, row3, row4, row5, row6, row7;
   __m128i tmp;

   // load
   row0 = dct_load(data, dequantize, 0);
   row1 = dct_load(data, dequantize, 1);
   row2 = dct_load(data, dequantize, 2);
   row3 = dct_load(data, dequantize, 3);
   row4 = dct_load(data, dequantize, 4);
   row5 = dct_load(data, dequantize, 5);
   row6 = dct_load(data, dequantize, 6);
   row7 = dct_load(data, dequantize, 7);

   // column pass
   dct_pass(bias_0, 10);

   {
      // 16bit 8x8 transpose pass 1
      dct_interleave16(row0, row4);
      dct_interleave16(row1, row5);
      dct_interleave16(row2, row6);
      dct_interleave16(row3, row7);

      // transpose pass 2
      dct_interleave16(row0, row2);
      dct_interleave16(row1, row3);
      dct_interleave16(row4, row6);
      dct_interleave16(row5, row7);

      // transpose pass 3
      dct_interleave16(row0, row1);
      dct_interleave16(row2, row3);
      dct_interleave16(row4, row5);
      dct_interleave16(row6, row7);
   }

   // row pass
   dct_pass(bias_1, 17);

   {
      // pack
      __m128i p0 = _mm_packus_epi16(row0, row1); // a0a1a2a3...a7b0b1b2b3...b7
      __m128i p1 = _mm_packus_epi16(row2, row3);
      __m128i p2 = _mm_packus_epi16(row4, row5);
      __m128i p3 = _mm_packus_epi16(row6, row7);

      // 8bit 8x8 transpose pass 1
      dct_interleave8(p0, p2); // a0e0a1e1...
      dct_interleave8(p1, p3); // c0g0c1g1...

      // transpose pass 2
      dct_interleave8(p0, p1); // a0c0e0g0...
      dct_interleave8(p2, p3); // b0d0f0h0...

      // transpose pass 3
      dct_interleave8(p0, p2); // a0b0c0d0...
      dct_interleave8(p1, p3); // a4b4c4d4...

      // store
      _mm_storel_epi64((__m128i *) out, p0); out += out_stride;
      _mm_storel_epi64((__m128i *) out, _mm_shuffle_epi32(p0, 0x4e)); out += out_stride;
      _mm_storel_epi64((__m128i *) out, p2); out += out_stride;
      _mm_storel_epi64((__m128i *) out, _mm_shuffle_epi32(p2, 0x4e)); out += out_stride;
      _mm_storel_epi64((__m128i *) out, p1); out += out_stride;
      _mm_storel_epi64((__m128i *) out, _mm_shuffle_epi32(p1, 0x4e)); out += out_stride;
      _mm_storel_epi64((__m128i *) out, p3); out += out_stride;
      _mm_storel_epi64((__m128i *) out, _mm_shuffle_epi32(p3, 0x4e));
   }
}

static void dct_print(stbi_uc const *a, stbi_uc const *b, int i)
{
   for (int y=0; y < 8; ++y) {
      for (int x=0; x < 8; ++x)
         printf(" %02x", a[y*8+x]);
      printf("   ");
      for (int x=0; x < 8; ++x)
         printf(" %02x", b[y*8+x]);
      printf("\n");
   }
}

static void test_dct()
{
   __declspec(align(16)) short coeffs[64];
   stbi_uc out_ref[64], out_sse[64];
   unsigned short dq1[64];

   for (int i=0; i < 64; ++i)
      dq1[i] = 1;

   for (int i=0; i < 64; ++i) {
      memset(coeffs, 0, sizeof(coeffs));
      coeffs[i] = 512;

      stbi__idct_block(out_ref, 8, coeffs);
      my_IDCT(out_sse, 8, coeffs, dq1);
      if (memcmp(out_ref, out_sse, 64) != 0) {
         dct_print(out_ref, out_sse, i);
         panic("mismatch i=0%o\n", i);
      }
   }
}

int main()
{
   test_dct();
   timer_init();
   //test_correct("test.png");
   //bench("test.png", 0);

   stbi_install_YCbCr_to_RGB(my_YCbCr_to_RGB);
   stbi_install_idct(my_IDCT);

   test_correct("anemones.jpg");
   bench("anemones.jpg", 4);

   return 0;
}