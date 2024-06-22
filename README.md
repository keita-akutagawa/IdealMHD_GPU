理想MHDシミュレーションのコードです。\
C++で書かれています。

Thrustライブラリを用いてGPU並列化を施します。

## スキーム
- HLLD
- MUSCL(minmod) : 空間2次精度
- CT(average)
- RK2 : 時間2次精度
