当前正在进行一个算法比赛，请帮我实现代码以取得更高的分数，具体要求如下：
【优化要求】
1. 方案要确保能够满足所有的约束
2. 时间分配策略可以基于每个用例输入的数据规模和约束条件类型来制定，确保不同的用例场景都能够最优的利用好时间
3. 代码实现要极致高性能，从数据结构，空间换时间等策略上进行性能优化
【demo代码接口示例】
import random
from time import sleep

def generate_random_box_positions(data):
    """
    根据data数据中box_size的个数，生成相同个数的随机box_position
    每个位置的x和y坐标都在0~10000之间随机生成

    参数:
        data: 包含box_size等信息的字典

    返回:
        字典，包含box_position字段，格式如示例所示
    """
    random.seed(42)  # 设置随机种子保证结果可重现，可以去掉这行获得完全随机结果

    num_boxes = len(data["box_size"])
    box_position = []

    for i in range(num_boxes):
        # 在0~1000范围内随机生成x和y坐标
        x = random.randint(0, 10000)
        y = random.randint(0, 10000)
        box_position.append([x, y])

    # 按照要求的格式返回字典
    result = {
        "box_position": box_position
    }

    return result


class Solution:
    def __init__(self):
        pass

    def solve(self, data):
        return generate_random_box_positions(data)


【支持的库】
absl-py==2.4.0
altgraph==0.17.5
cbcbox==2.929
cffi==2.0.0
colorama==0.4.6
contourpy==1.3.2
cycler==0.12.1
Cython==3.2.4
et_xmlfile==2.0.0
fonttools==4.62.1
immutabledict==4.3.1
json5==0.14.0
kiwisolver==1.5.0
llvmlite==0.47.0
matplotlib==3.10.8
mip==1.17.6
networkx==3.6.1
numba==0.65.1
numpy==2.2.6
openpyxl==3.1.5
ortools==9.11.4210
packaging==26.1
pandas==2.3.2
pillow==12.2.0
pip==26.1.2
protobuf==5.26.1
pybullet==3.2.7
pycparser==3.0
pyinstaller==6.20.0
pyinstaller-hooks-contrib==2026.5
pyparsing==3.3.2
python-dateutil==2.9.0.post0
pytz==2026.1.post1
PyYAML==6.0.3
scipy==1.16.3
setuptools==82.0.1
six==1.17.0
tk==0.1.0
tzdata==2026.1
ujson==5.10.0
wcwidth==0.6.0

【题目描述】
# 基于多规则的布局算法任务说明

## 题目说明与术语

### 题目说明

给定若干矩形（数量约 20~100，矩形不可旋转）及其约束关系，要求输出一个满足约束的布局方案，使：

1. **容器面积（Area）尽可能小**；
2. **网络连接的半周长线长（HPWL）尽可能小**。

### 关键术语

| 术语       | 英文                             | 说明                                                         |
| ---------- | -------------------------------- | ------------------------------------------------------------ |
| 容器       | Bounding Box                     | 包围一个或多个矩形的最小外接矩形框。                         |
| 半周长线长 | HPWL (Half-Perimeter Wirelength) | 框住网络内所有矩形中心点的最小矩形框，其周长的一半。常用于估算连线代价。 |

---

## 赛题背景与任务目标

输入包含：

- 每个矩形的尺寸 `box_size`（宽、高）；
- 矩形间约束（对称、对齐、重复组等）；
- 网络连接关系 `nets`。

输出需给出每个矩形的左下角坐标 `box_position`，并满足所有约束。在此基础上优化 Area 与 HPWL。

---

## 约束定义（不重叠、对齐、对称、重复组）

### 1) 不重叠约束

- 任意两个矩形的重叠面积必须为 0；
- 边界可接触，但不能相交。

### 2) 对称约束

一个对称组可以是symmetry_x（关于x轴对称）或者symmetry_y（关于y轴对称）。
每一种对称组都包含两类：

- `symmetry_pair`：成对矩形关于同一条轴对称；
- `self_symmetry`：单个矩形自身关于该轴对称。

对于每一组symmetry_x，需要确定一个x_axis，放置矩形使得
1.对于所有的symmetry_pair，
令(x1,y1)为对称对的第一个矩形的中心点坐标，(x2,y2)为对称对的第二个矩形的中心点坐标。
则要满足x1+x2=2*x_axis,y1=y2.
2.对于所有的self_symmetry，
令(x_i,y_i)为第i个自对称的矩形，则对于所有自对称矩形x_i=x_axis

补充说明：

- 每一个对称组的symmetry_pair和self_symmetry共享同一条对称轴；
- 不同对称组可有不同对称轴；
- symmetry_x对称轴是x_axis，symmetry_y对称轴是y_axis；

举例：
symmetry_x竖直对称的举例：如下json中，5 6 3 4 1 9共享一条竖直对称轴，其中5和6相互对称，3和4相互对称，1自对称，9自对称

```json
{
  "symmetry_x": [
    {
      "symmetry_pair": [[5, 6], [3, 4]],
      "self_symmetry": [1, 9]
    }
  ]
}
```

symmetry_y水平对称的举例：如下json中，7 8 2共享一条水平对称轴，其中7和8相互对称，2自对称

```json
{
  "symmetry_y": [
    {
      "symmetry_pair": [[7, 8]],
      "self_symmetry": [2]
    }
  ]
}
```

### 3) 对齐约束

`align` 包含四类：

- `left`、`right`、`top`、`bottom`。

同一组中的矩形需在对应边缘上共线（左对齐/右对齐/上对齐/下对齐）。具体对齐边界由选手选择。
例如，下面的对齐约束，4，13需要关于左边缘对齐；6,16需要关于下边缘对齐

```json
{
    "align": {
    "left": [[4, 13]],
    "right": [],
    "top": [],
    "bottom": [[6, 16]]
    }
}
```

### 4) 重复组约束

- 一个重复组内包含多个“矩形组合”；
- 各组合矩形个数相同，且对应下标的矩形尺寸一致(输入的box_size已保证)；
- 布局后，同一重复组内各组合的**相对位置关系必须一致**；
- 重复组外轮廓矩形不能与其他矩形或其他重复组重叠。

重复组的举例：如下json中

- 第一个重复组有三套矩形： (9 10 11)、(12 13 14)、(3 5 7)，每套矩形内部的布局完全一致，第一套9、10、11的相对位置，与第二套12、13、14的相对位置，与第三套3、5、7的相对位置完全一致 

```json
{
  "repeat_groups": [
    {
      "groups": [[9, 10, 11], [12, 13, 14], [3, 5, 7]]
    },
    {
      "groups": [[15, 16, 17], [18, 19, 20]]
    }
  ]
}
```

---

## 优化目标（Area 与 HPWL）

### 1) Area

布局所用到的所有矩形所占据的一个更大的矩形空间表示容器的大小，容器面积越小越好，代表空间利用率越高。

### 2) HPWL

对于每个 `net`：

- 取该 net 内所有矩形中心点；
- 计算包围这些中心点的最小外接矩形；
- 其周长的一半即该 net 的 HPWL。

同一矩形可属于多个net。一般希望同网内矩形更靠近，以降低 HPWL 代价。

---

## 评分与耗时限制

### 约束

必须严格满足：

1. 对齐、对称、重复组约束；
2. 矩形与矩形不重叠；
3. 重复组外轮廓与其他矩形/重复组不重叠。

### 耗时限制

- 单用例：120 秒；
- 单线程。


### 评分规则（文字说明）

- 若约束不满足或超时：该用例 `Score = 0`；
- 满足约束与时限后，尽量使得花费Cost = Area + 10 * HPWL尽可能小；
- 选手最终分数为所有用例得分之和；

---
