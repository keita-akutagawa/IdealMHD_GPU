理想MHDシミュレーションのコードです。
C++で書かれています。

Thrustライブラリ(CUDA)を用いてGPU並列化を施しています。

## スキーム

- HLLD
- MUSCL(minmod) : 空間2次精度
- CT-Contact
- RK2 : 時間2次精度
