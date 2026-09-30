# 12-bit JPEG references

Synthetic 12-bit DCT images and their libjpeg-turbo decodes, used by the JPEG reader tests in
`src/codec/jpeg.rs`. No clinical data was used. Copied from
medcognetics/jpeg-decoder commit 982d7ac (`tests/reftest/images/extended12`).

Each `.jpg` was encoded by libjpeg-turbo 3.1.4 `cjpeg -precision 12`. Each `.png` is
`djpeg -dct int -pnm` output stored as 16-bit grayscale with values 0-4095; the reader must
match it exactly.

Source `source.pgm` (37 x 23, maxval 4095):
`v(x, y) = (x*97 + y*53 + ((x*y) % 37)*29) % 4096`, then `v //= 3` where
`(x//5 + y//4) % 2 == 0`.

| File | `cjpeg` options |
| --- | --- |
| `sequential_37x23_q90.jpg` | `-quality 90` |
| `sequential_37x23_restart1.jpg` | `-quality 90 -restart 1` |
| `sequential_37x23_q5_16bit_dqt.jpg` | `-quality 5` (16-bit quantization tables) |
| `sequential_37x23_sampling2x2.jpg` | `-quality 90 -sample 2x2` |
| `progressive_37x23_q90.jpg` | `-quality 90 -progressive` |
| `sequential_64x16_q100_checker.jpg` | `-quality 100` on 8 x 8 blocks alternating 0 and 4095 (DC categories up to 15) |
| `color/rgb_37x23.jpg` | `-quality 90` on an RGB version of the source (planes `v`, `v` reversed, `v // 2`) |
