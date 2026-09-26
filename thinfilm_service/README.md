# thinfilm-service

多层介质膜光谱核算服务：特征矩阵法（characteristic / transfer matrix method）
固化成统一口径，设计与检测两边都经 HTTP + JSON 调用。无网页界面。

- 复数 2×2 特征矩阵逐层连乘，吸收膜（复折射率 `n + iκ`）自然支持
- s / p 偏振的等效光学导纳各自独立计算，另有 `avg`（非偏振，取两者平均）
- 单点 R / T / A 与波段光谱扫描，同源同算法
- 有限厚基底通道：正面膜系相干、基底两界面间按光强非相干累加（几何级数闭式），
  同时报「只算正面」与「连背面一起算」两种口径
- 所有非法输入在计算前被拦截，返回结构化错误

## 运行（容器，一条命令）

```bash
docker build -t thinfilm-service . && docker run --rm -p 8080:8080 thinfilm-service
```

或 `docker compose up --build`。基础镜像 `python:3.12-slim`，服务由
waitress 以多线程方式提供（默认 8 线程，`THREADS` 环境变量可调），
监听 `PORT`（默认 8080）。

## 本地开发与测试

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest          # 184 个测试，钉死物理基准与接口行为
python run.py             # 本地起服务（HOST/PORT/THREADS 可调）
```

## 物理模型与约定

- 时间因子 `exp(-iωt)`；复折射率 `N = n + iκ`，`κ ≥ 0` 为吸收（无源）介质
- 层内角度由 Snell 定律给出：`N_j sinθ_j = N_0 sinθ_0`，复折射率同样适用
- 相位厚度 `δ = (2π/λ) · N · d · cosθ` —— 含入射角余弦因子，
  斜入射光谱峰位由它决定（测试里有专门的峰位蓝移锁定）
- 等效光学导纳（自由空间单位）：
  - s 偏振（TE）：`η = N·cosθ`
  - p 偏振（TM）：`η = N/cosθ`
- 单层特征矩阵 `[[cosδ, -i·sinδ/η], [-i·η·sinδ, cosδ]]`，按入射顺序连乘
  `M = M_1·M_2·…·M_k`；`[B; C] = M·[1; η_基底]`，输入导纳 `Y = C/B`
- 振幅系数 `r = (η_0 − Y)/(η_0 + Y)`，`t = 2η_0 / (B·(η_0 + Y))`
- 能量系数 `R = |r|²`，`T = Re(η_基底)/Re(η_0)·|t|²`，`A = 1 − R − T`
  （入射介质要求无吸收，否则入射能流无定义——校验层会拒绝）
- 无膜时 `M` 为单位阵，结果精确退化为单界面菲涅耳公式（有测试锁定）

### 有限厚基底（非相干背面通道）

实测样品是一片有确定厚度的基底：正面镀膜、背面是裸的基底/外界界面。
正面膜系很薄，仍按振幅相干叠加（同一套特征矩阵求解，从入射侧与从
基底内侧各解一次，后者即层序反转后的同一个求解器）；基底本身远超
相干长度，两个界面之间的多次反射相位抹平，只能按光强累加。记

- `R_f, T_f`：正面膜系（入射侧，相干）
- `R'_f, T'_f`：正面膜系（从基底内侧看，相干）
- `R_b, T_b`：背面裸界面（基底 → 背面外侧介质）
- `α = exp(−2·Im δ_基底)`：基底单程光强透过率（含吸收衰减与斜入射路径）

每在基底内往返一趟，被囚光强乘上 `q = R'_f·R_b·α²`，多次反射构成
几何级数，按闭式求和（不逐次截断）：

```
R_总 = R_f + T_f·T'_f·R_b·α² / (1 − q)
T_总 = T_f·T_b·α / (1 − q)
A_总 = 1 − R_总 − T_总
```

退化自洽：基底无吸收且背面外侧折射率等于基底时，背面界面消失
（`R_b = 0, α = 1`），两式精确回到只算正面的相干结果（测试锁定到
1e-12）。能量守恒 `R_总 + T_总 + A_总 = 1` 依旧成立。

波长与膜厚同单位即可（通常 nm），角度用度。

## 接口

### `POST /api/v1/reflectance` — 单点计算

```json
{
  "incident": 1.0,
  "substrate": 1.52,
  "layers": [{"index": {"re": 1.7, "im": 0.4}, "thickness": 180.0}],
  "wavelength": 550.0,
  "angle_deg": 45.0,
  "polarization": "s"
}
```

- `index` 支持三种写法：实数 `1.38`、`{"re": 1.7, "im": 0.4}`、`[1.7, 0.4]`
- `layers` 可省略或为空（裸界面）；`angle_deg` 默认 0；`polarization`
  取 `s` / `p` / `avg`，默认 `s`
- 响应含所求偏振的 `reflectance` / `transmittance` / `absorptance` /
  振幅系数，以及 `components` 下 s、p 两路的完整结果

### `POST /api/v1/reflectance/slab` — 整片样品（有限厚基底 + 背面）

在单点请求的基础上多认两个字段：`substrate_thickness`（基底物理厚度，
与波长同单位，0.5 mm 即 `500000.0`，必须 > 0）与 `back_index`（背面
外侧介质折射率，写法同 `index`，通常空气 `1.0`）。

```json
{
  "incident": 1.0,
  "substrate": 1.52,
  "layers": [{"index": 1.38, "thickness": 99.6377}],
  "substrate_thickness": 500000.0,
  "back_index": 1.0,
  "wavelength": 550.0,
  "angle_deg": 0.0,
  "polarization": "s"
}
```

响应顶层是连背面一起算的整片总 `reflectance` / `transmittance` /
`absorptance`；`front_only` 给出只算正面相干叠加的同一组量，
`back_contribution` 给出两者之差（即背面到底贡献了多少），另附
`back_interface_reflectance`（背面裸界面反射率）与
`round_trip_factor`（基底往返一趟的光强衰减因子 q）。`substrate`
块回显厚度、背面折射率与单程透过率 α；`components` 下是 s、p 两路
各自的完整双口径结果。示范膜系下：正面 1.26 %、整片 5.41 %，差值
4.15 % 正是裸玻璃背面的量级。

### `POST /api/v1/spectrum` — 光谱扫描

```json
{
  "incident": 1.0,
  "substrate": 1.52,
  "layers": [{"index": 1.38, "thickness": 99.6377}],
  "wavelength_start": 400.0,
  "wavelength_stop": 700.0,
  "points": 301,
  "angle_deg": 0.0,
  "polarization": "s"
}
```

返回 `points` 点列（`wavelength` / `reflectance` / `transmittance` /
`absorptance`），每一点都是同一个矩阵求解器的真值，无预设形状。

### `GET /api/v1/example` — 内置示范膜系

玻璃基底（n=1.52）上的单层四分之一波长增透膜（n=1.38，设计波长
550 nm，物理厚度 ≈ 99.64 nm）。响应里给出镀膜 / 裸基底两组可直接回放的
请求体和实时算出的反射率（≈1.26 % vs ≈4.26 %），加载即可复现
「设计波长处镀膜反射率明显低于裸基底」。

### `GET /health` — 存活探针

## 输入校验（计算前拒绝，HTTP 400）

负层厚、折射率实部 ≤ 0、入射介质带虚部、波长 ≤ 0、入射角 ≥ 90°、
非法偏振、光谱区间倒置、基底厚度 ≤ 0、背面介质折射率实部 ≤ 0 等，
统一返回：

```json
{
  "error": {
    "type": "validation_error",
    "message": "request failed validation",
    "details": [{"field": "layers[0].thickness", "message": "layer thickness must be >= 0"}]
  }
}
```

## 代码结构

```
thinopt/
  optics.py       Snell 定律、cosθ 分支选取、s/p 等效导纳、相位厚度
  matrices.py     复数特征矩阵构造与连乘（数据类 Layer/Stack）
  solver.py       单点 R/T/A 求解（含 s、p、avg）
  spectrum.py     波段扫描（逐点调用 solver，无捷径）
  substrate.py    有限厚基底：非相干背面通道、单程衰减、几何级数闭式
  validation.py   请求解析与物理合法性校验（先于一切计算）
  presets.py      内置四分之一波长增透膜示范
  api.py          Flask HTTP 层（仅 JSON）
run.py            waitress 入口
tests/            物理基准 + 接口 + 并发测试
```

## 测试锁定的正确性基准

- 无吸收膜系任意波长/角度/偏振下 `R + T = 1`
- 有吸收时 `R + A + T = 1` 且 `A ≥ 0`；单层吸收膜与独立的 Airy 求和公式逐点一致
- 无膜退化为单界面菲涅耳反射（与解析式对照到 1e-12）
- 四分之一波长增透膜在设计波长处反射率明显低于裸基底（并与闭式解一致）
- 半波长层在设计波长处「隐身」，反射率回到裸基底
- 反射率随入射角趋向掠射而升高
- 斜入射下增透谷蓝移至 `λ₀·cosθ_膜内` —— 锁定相位厚度里的 cosθ 因子
- 光谱扫描点与单点求解逐点一致；并发请求互不串扰
- 有限厚基底退化自洽：无吸收基底 + 背面折射率等于基底时，整片结果精确回到
  只算正面的相干口径（两条路径同一曲线）
- 非相干求和后整片 `R + T + A = 1`、`A ≥ 0` 依旧成立
- 增透膜样品的整片反射率明显高于只算正面的值，差值与裸玻璃背面界面反射
  （≈4.26 %）量级一致；裸玻片与教科书非相干公式 `R = 2r/(1+r)` 精确一致
- 吸收 / 厚度增大时多次反射串单调衰减趋于零；级数闭式与逐次叠加结果一致
