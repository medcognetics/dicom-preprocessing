# JPEG 2000 fixtures

Synthetic lossless JPEG 2000 streams used by the reader tests in `src/codec/jpeg2000.rs`. No
clinical data was used. The tests compare decoded samples with the source pattern, which is
computed independently of any codec.

Source (37 x 23, 12-bit): `v(x, y) = (x*97 + y*53 + ((x*y) % 37)*29) % 4096`, then
`v //= 3` where `(x//5 + y//4) % 2 == 0`.

| File | Encoder |
| --- | --- |
| `j2k_lossless_37x23.j2c` | imagecodecs 2026.3.6 `jpeg2k_encode(level=0, reversible=True, bitspersample=12, codecformat="J2K")` (OpenJPEG) |
| `j2k_lossless_37x23.jp2` | Same, with `codecformat="JP2"`: a JP2-wrapped stream |
| `j2k_lossless_rgb_37x23.j2c` | Same as the first, on three components: `v`, `v` with rows reversed, and `v // 2` |
| `htj2k_lossless_37x23.j2c` | OpenJPH 0.27.4 `ojph_compress -reversible true -num_decomps 3` from a 12-bit PGM |
