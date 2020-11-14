# Numpy教程

能力差，就努力提升自己，提高自己的编码能力，对于科学计算来说，先学好numpy开始。Numpy(Numerical Python)是Python语言的一个扩展程序库，支持大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库。

* 一个强大的N维数组对象ndarray
* 广播功能函数
* 整合C/C++/Fortran代码的工具
* 线性代数、傅里叶变换、随机数生成等功能。



## 1 Numpy Tutorials

### 1.1 数据类型（Data Types)

#### 1.1.1Array types and conversions between types

NumPy supports a much greater variety of numerical types than Python does. This section shows which are available, and how to modify an array’s data-type.

The primitive types supported are tied closely to those in C:

| Numpy type               | C type                | Description                                                  |
| :----------------------- | :-------------------- | :----------------------------------------------------------- |
| *np.bool_*               | `bool`                | Boolean (True or False) stored as a byte                     |
| *np.byte*                | `signed char`         | Platform-defined                                             |
| *np.ubyte*               | `unsigned char`       | Platform-defined                                             |
| *np.short*               | `short`               | Platform-defined                                             |
| *np.ushort*              | `unsigned short`      | Platform-defined                                             |
| *np.intc*                | `int`                 | Platform-defined                                             |
| *np.uintc*               | `unsigned int`        | Platform-defined                                             |
| *np.int_*                | `long`                | Platform-defined                                             |
| *np.uint*                | `unsigned long`       | Platform-defined                                             |
| *np.longlong*            | `long long`           | Platform-defined                                             |
| *np.ulonglong*           | `unsigned long long`  | Platform-defined                                             |
| *np.half* / *np.float16* |                       | Half precision float: sign bit, 5 bits exponent, 10 bits mantissa |
| *np.single*              | `float`               | Platform-defined single precision float: typically sign bit, 8 bits exponent, 23 bits mantissa |
| *np.double*              | `double`              | Platform-defined double precision float: typically sign bit, 11 bits exponent, 52 bits mantissa. |
| *np.longdouble*          | `long double`         | Platform-defined extended-precision float                    |
| *np.csingle*             | `float complex`       | Complex number, represented by two single-precision floats (real and imaginary components) |
| *np.cdouble*             | `double complex`      | Complex number, represented by two double-precision floats (real and imaginary components). |
| *np.clongdouble*         | `long double complex` | Complex number, represented by two extended-precision floats (real and imaginary components). |

Since many of these have platform-dependent definitions, a set of fixed-size aliases are provided:

| Numpy type                      | C type           | Description                                                  |
| :------------------------------ | :--------------- | :----------------------------------------------------------- |
| *np.int8*                       | `int8_t`         | Byte (-128 to 127)                                           |
| *np.int16*                      | `int16_t`        | Integer (-32768 to 32767)                                    |
| *np.int32*                      | `int32_t`        | Integer (-2147483648 to 2147483647)                          |
| *np.int64*                      | `int64_t`        | Integer (-9223372036854775808 to 9223372036854775807)        |
| *np.uint8*                      | `uint8_t`        | Unsigned integer (0 to 255)                                  |
| *np.uint16*                     | `uint16_t`       | Unsigned integer (0 to 65535)                                |
| *np.uint32*                     | `uint32_t`       | Unsigned integer (0 to 4294967295)                           |
| *np.uint64*                     | `uint64_t`       | Unsigned integer (0 to 18446744073709551615)                 |
| *np.intp*                       | `intptr_t`       | Integer used for indexing, typically the same as `ssize_t`   |
| *np.uintp*                      | `uintptr_t`      | Integer large enough to hold a pointer                       |
| *np.float32*                    | `float`          |                                                              |
| *np.float64* / *np.float_*      | `double`         | Note that this matches the precision of the builtin python *float*. |
| *np.complex64*                  | `float complex`  | Complex number, represented by two 32-bit floats (real and imaginary components) |
| *np.complex128* / *np.complex_* | `double complex` | Note that this matches the precision of the builtin python *complex*. |

有5种基本的数字类型，分别代表布尔值（booleans）、整数（int）、无符号整数（uint）浮点（float）和复数。那些名字中带有数字的类型表示该类型的位数大小（即在内存中表示一个值需要多少位）。有些类型，如int和intp，根据平台的不同，有不同的位数大小（如32位与64位机器）。当与原始内存寻址的低级代码（如C或Fortran）接口时，应考虑到这一点。

数据类型可以作为函数使用，将python数字转换为数组标量(详见数组标量部分)，将python数字序列转换为该类型的数组，或者作为许多numpy函数或方法所接受的dtype关键字的参数。以下是一些例子。

```python
import numpy as np

x = np.float32(1.0)
# 1.0

y = np.int_([1,2,4])
# array([1, 2, 4])

z = np.arange(-3,3,dtype=np.uint8)
# array([253, 254, 255,   0,   1,   2], dtype=uint8)

```

阵列类型也可以用字符代码来表示，主要是为了保持与旧的包（如Numeric）的向后兼容性。一些文档可能仍然会提及这些类型，例如：

```python
a = np.array([1,2,3],dtype='f')
# array([1., 2., 3.], dtype=float32)
b = np.array([-5,-1,4,5],dtype=np.uint8)
# array([251, 255,   4,   5], dtype=uint8)
```

**推荐使用dtype对象进行替换**

为了更换数组的类型，可以使用`.astype()`方法（首选）或将类型本身作为一个函数，例子如下：

```python
z.astype(float)
# array([253., 254., 255.,   0.,   1.,   2.])

np.int8(z)
# array([-3, -2, -1,  0,  1,  2], dtype=int8)
```

请注意，上面我们使用Python float对象作为dtype。NumPy 知道 int 是指 np.int_，bool 是指 np.bool_，float 是 np.float_，complex 是 np.complex_。其他的数据类型没有Python对应的类型。

要确定一个数组的类型，请看 dtype 属性。

```python
z.dtype
# dtype('uint8')
```

dtype对象还包含有关类型的信息，如它的位宽和字节顺序。数据类型也可以间接用于查询该类型的属性，例如它是否是一个整数。

```python
d = np.dtype(int)
# dtype('int32')
np.issubdtype(d,np.integer)
# True

np.issubdtype(d,np.floating)
# Float
```



#### 1.1.2 矩阵标量

NumPy 通常以数组标量 (一个带有关联 dtype 的标量) 的形式返回数组元素。数组标量不同于Python标量，但大多数情况下它们可以互换使用 (主要的例外是Python的v2.x以上的版本，在这些版本中，整数组标量不能作为列表和元组的索引)。也有一些例外，比如当代码需要一个标量的非常特定的属性，或者当它特别检查一个值是否是 Python 标量时。一般来说，通过使用相应的Python类型函数(例如int、float、complex、str、unicode)显式地将数组标量转换为Python标量，问题就很容易解决。

使用数组标量的主要优点是它们保留了数组类型(Python可能没有匹配的标量类型可用，例如int16)。因此，使用数组标量可以确保数组和标量之间的行为是相同的，不管值是否在数组中。NumPy标量也有许多与数组相同的方法。

#### 1.1.2 溢出错误(Overflow Errors)

NumPy数值类型的固定大小，当一个值需要的内存大于数据类型的可用内存时，可能会导致溢出错误。例如，numpy.power对于64位整数正确地评估100 * 10 ** 8，但对于32位整数却给出1874919424（不正确）。

```python
np.power(100,8,dtype=np.int64)
# 10000000000000000
np.power(100,8,dtype=np.int32)
# 1874919424
```







### 1.2 有结构的矩阵(Structured arrays)

结构矩阵是其数据类型由一系列简单的数据类型组成的矩阵，例如：

```python
import numpy as np

x = np.array([('Rex',9,81.0),('Fido',3,27.0)],dtype=[('name','U10'),('age','i4'),('weight','f4')])
```

其中，`x`是长度为2的一维矩阵，它的数据类型是由三个字段(fields)构成的结构：1-长度小于等于10的字符串，命名为’name'，2.一个32位的整数'age'，3.一个32位的浮点数'weight'。

通过`x[1]`可以访问，也可以获取和修改相应的数字：

```python
x[1]
('Fido', 3, 27.0)
```







### 1.3 矩阵创建（Array creation)

有5种创建数组的通用机制。

1. 从其他Python结构(如列表、元组)转换而来。

2. 固有的numpy数组创建对象(如arrange, ones, zeros等)

3. 从磁盘上读取数组，无论是标准格式还是自定义格式。

4. 通过使用字符串或缓冲区从原始字节创建数组。

5. 使用特殊的库函数（如随机）。

本节将不涉及复制、加入或以其他方式扩展或突变现有数组的方法，也不涉及创建对象数组或结构化数组。也不涉及创建对象数组或结构化数组。这两方面的内容将在各自的章节中介绍。

**将Python对象的矩阵转换为Numpy矩阵**

一般来说，在Python中以类似数组的结构排列的数值数据可以通过使用array()函数转换为数组。最明显的例子是列表和元组。关于它的使用细节，请参见 array() 的文档。一些对象可能支持array-protocol，并允许以这种方式转换为数组。一个简单的方法可以找出是否可以使用 array() 将对象转换为 numpy 数组，就是简单的交互式尝试，看看它是否有效! (Python方式)。













### 1.4 Numpy的I/O（I/O with Numpy）



### 1.5 索引(Indexing)





### 1.6 广播(Broadcasting)



### 1.7 

































































































































