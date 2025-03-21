# CPP知识点总结
# 1.面向对象编程概述
## 1.1程序设计的方法
- 面向过程、面向对象
- 面向过程中，程序=数据结构+算法。对象是数据结构和算法的封装体s
- 面向对象中，程序=对象+消息（数据+方法）
- 面向对象=对象+类+继承+消息
## 关系
- 聚类关系：has a、contains a
- 继承关系：is a
- 实例化： belongs to
## 类与抽象数据类型
- 对象由一组属性和一组操作构成
- 类是一组相同数据结构和操作的集合
- 封装有两个含义，一是把全部属性和行为结合在一起，二是尽量屏蔽对象的内部细节。
- 抽象包括数据抽象和行为抽象。数据抽象抽象共有的属性和状态。行为抽象抽象共有的行为和功能。

## 客户服务器模式和消息传递
- 在面向对象程序中，类和对象表示为服务器，使用类和对象的模块表现为客户
# 程序设计语言简介
## 命名空间和域解析操作符
- 使用完整命名空间
```cpp
using namespace_name
```
- 使用命名空间的某个特定变量
```cpp
using namespace_name::member_name
```
- 命名空间可以嵌套
- 命名空间里面的内容可以追加添加

## cpp的输入和输出
- 不要混合使用c和cpp的输入输出，这两种输出不会同步




# 3.类

# 4.继承


| 继承方式      | 基类的 `public` 成员在派生类中的访问权限 | 基类的 `protected` 成员在派生类中的访问权限 | 基类的 `private` 成员在派生类中的访问权限 | 子类访问权限       | 外部访问权限             |
|---------------|-----------------------------------------|--------------------------------------------|--------------------------------------------|--------------------|--------------------------|
| `public` 继承 | `public`                                | `protected`                               | 不可访问                                   | 保持 `public`、`protected` | 子类可继承，外部可访问 `public` 成员 |
| `protected` 继承 | `protected`                         | `protected`                               | 不可访问                                   | 保持 `protected`             | 子类可继承，外部不可访问          |
| `private` 继承 | `private`                             | `private`                                 | 不可访问                                   | 保持 `private`               | 子类不可继承，外部不可访问        |
- 同一个类中，不同函数头的函数会构成重载。但是，在继承中，只需相同函数名就可以构成覆盖。

## 派生类的构造函数与析构函数

定义简单派生类构造函数的一般形式为：
```
<派生类构造函数名>(<总参数列表>): <基类构造函数名>(<参数表>)
{
    <派生类新增数据成员初始化>
};
```
## 虚基类
- 虚函数的构造函数由且只由最底层的派生类调用，如果没有显式调用，就会调用默认构造函数。继承链中间的虚基类构造函数均不执行。

# 5.多态
- 运行期绑定和编译期绑定

## 虚函数
- 虚函数在声明时前面加上virtual
- 虚函数是动态绑定的运行时多态
- 虚函数通过虚函数表实现，所有声明为虚函数的函数会被放进虚函数表中。在调用时，查看虚函数表中的函数决定调用哪一个，子类的虚函数会覆盖父类的（动态绑定）。
- 注意，如果实现运行时多态必须使用virtual，简单的覆盖不能实现父类指针调用子类的函数。这一点与java不同。
- 在父类中被声明为虚函数的函数，在所有子类中，相同函数都自动成为虚函数。

- 构造函数不能声明为虚函数，但是析构函数常常被声明为虚函数。



## 重载、覆盖与遮蔽
- 重载是指**不同函数头**的函数可以共存，在编译期自动选择相应的函数。重载函数必须卸载同一个类中，父类和子类不能构成重载（但是同一层的类可以构成重载）。
- 重载的判断标准（满足如下条件之一即可构成重载）：变量列表、变量类型、顺序、函数被const修饰（变量被const修饰不能构成重载）（仅返回值不同不能构成重载）
- 覆盖是指相同函数名的虚函数可以共存，在**运行时动态绑定（基类指针根据实际的覆盖情况选择）**。
- 遮蔽是指相同函数名的非虚函数，子类的会遮蔽父类的，在**编译期选择（直接调用和指针类型相同的函数）**。注意，遮蔽只要是父子类有相同函数名（不需要参数列表返回值相同）就可以构成遮蔽。
- 我们可以将继承关系抽象成一个树，或者一个图。编译器在查找方法时，会沿着最底层的派生类一层一层向上寻找，如果在某一层有两个‘函数头相同的函数’，那么就会产生二义性。但是在不同的层上有无法区别的函数，则不会产生二义性。

## 抽象类
```cpp
#include <iostream>

class Shape {
public:
    // 纯虚函数，表示计算面积
    virtual double area() const = 0;

    // 纯虚函数，表示绘制形状
    virtual void draw() const = 0;

    // 虚析构函数，确保派生类析构时正确调用析构函数
    virtual ~Shape() {
        std::cout << "Shape destructor called" << std::endl;
    }
};

```
- 抽象类不能实例化，但是可以声明抽象类的指针和引用。这可以用作接口。



# 6.操作符重载

- 不能重载的运算符
```cpp
:?
.
.*
::
```
- 除了赋值操作符（=）之外，基类中所有被重载的操作符都将被派生类继承。

## 重载运算符语法：
```cpp
<返回值类型> operator<运算符> (<形式参数表>)
// 其中，形式参数列表是这个操作符所需要的参数
// 在类内重载，操作数不包括类本身
// 在类外重载，操作数包括所有运算用到的操作数
```
- 重载操作符不改变操作符的优先级

- 操作符重载实例
```cpp
// 重载加号，由于加法表达式返回一个临时的右值，因此返回值不加&
Vector operator+(const Vector& other) {
    return Vector(x + other.x, y + other.y);
}

// 在语义上，等于号返回原来的对象，所以返回引用（返回非引用不会报错，但是会有性能和语意的问题）
Vector& operator=(const Vector& other) {
    if (this != &other) {
        x = other.x;
        y = other.y;
    }
    return *this;
}


friend std::ostream& operator<<(std::ostream& os, const Vector& v) {
    os << "(" << v.x << ", " << v.y << ")";
    return os;
}

//前置自增返回对戏那个本身
Vector& operator++() { // 前置自增
    ++x; ++y;
    return *this;
}

//后置自增返回一个临时变量
Vector operator++(int) { // 后置自增
    Vector temp = *this;
    ++*this;
    return temp;
}
```

## 友元friend

- friend函数像static函数一样，不是类的成员，是独立的。

```cpp
// friend函数就是将一个类外的函数加上friend在类内声明一下
class Box {
private:
    int width;

public:
    Box(int w) : width(w) {}

    // 声明 friend 函数
    friend void showWidth(const Box& b);
};

// friend 函数定义在类外部
void showWidth(const Box& b) {
    std::cout << "Width: " << b.width << std::endl;
}
```
## 类型转换操作符的重载
```cpp
class Complex {
private:
    double real, imag;

public:
    Complex(double r, double i) : real(r), imag(i) {}

    // 正确：不显式标注返回类型
    operator double() const {
        return std::sqrt(real * real + imag * imag);
    }
};
```



# 7.模版与标准模版库
- 同名的const和非const函数也可以重载，const变量会调用const修饰的函数，非const对象会调用非const修饰的函数

- cpp中，模版分为函数模版和类模版

- 模版在编译时由编译器翻译为具体的模版类，这个过程称为模版的实例化。

- 函数模版定义语法如下
```cpp
template <typename 模板参数1,typename 模版参数2...>
<返回值类型> <函数名>(<参数表>)
{
    <函数体>
}
```

- 注意模版的传入变量不支持隐式类型转换，对于相同的T，传入的变量类型必须相同。

## 模版类
- 模版类的定义语法如下：
```cpp
template <模板参数表>
<返回值类型> <函数名>(<参数表>)
{
    <函数体>
}
```
- 模版类成员函数的定义
```cpp
template <模板参数表>
<返回值类型> <类模板名><类型名表>::<函数名>(<参数表>)
{
    <函数体>
}
```

- 类模版的实例化是在生成对象时发生。

- 类模版还可以有非类型的参数，成为函数类型参数（就是有类型的参数）。
# 8.UML

# 9.STL
## 容器
- 序列式容器：
```cpp
std::vector
std::list
std::deque
```
- 关联式容器
```cpp
std::set
std::map
std::unordered_map
std::unordered_set
std::multiset
std::multimap<int,int>
```
- 容器适配器
```cpp
std::stack
std::queue
std::priority_queue
```
## 容器的遍历
容器的遍历需要是可迭代的容器，队列和栈不可迭代。
```cpp
int main() {
    std::vector<int> vec = {1, 2, 3};
    for (const auto& num : vec) {
        // num = 10; // 错误，不能修改常量引用所引用的对象
        std::cout << num << " ";
    }
    for (auto num : vec) {
        num = 10; 
        std::cout << num << " ";
    }
    std::cout << std::endl;
    return 0;
}
```
## 改变容器和算法的排列顺序
```cpp
std::set<int, std::greater<int>>
std::sort(numbers.begin(), numbers.end(), std::greater<int>());
```
## 迭代器
迭代器类似于指针，用来访问容器中的元素，访问对应的元素需要加上*。
```cpp
begin()//返回第一个元素的迭代器
end()//返回一个指向容器末尾的迭代器，但不是最后一个元素，是最后一个元素再向后一个。
rbegin()//返回指向最后一个元素的迭代器
```
```cpp
std::prev()
std::next()
```
## algorithm
```cpp
std::sort
std::stable_sort
std::find
std::binary_search
std::accumulate
```
重写sort的排序算法
```cpp
// 定义自定义类型
struct Person {
    std::string name;
    int age;
};

// 使用 Lambda 表达式按年龄降序排序
int main() {
    std::vector<Person> people = {
        {"Alice", 25},
        {"Bob", 20},
        {"Charlie", 30}
    };
    std::sort(people.begin(), people.end(), [](const Person& a, const Person& b) {
        return a.age > b.age;
    });
    for (const auto& person : people) {
        std::cout << person.name << " (" << person.age << ") ";
    }
    std::cout << std::endl;
    return 0;
}
```
```cpp
//lower_bound和upper_bound函数

template< class ForwardIt, class T >
ForwardIt lower_bound( ForwardIt first, ForwardIt last, const T& value );

template< class ForwardIt, class T, class Compare >
ForwardIt lower_bound( ForwardIt first, ForwardIt last, const T& value, Compare comp );
```
# 10.CPP新特性

## lambda函数
```cpp
[capture list] (parameter list) -> return type { function body }
/*
capture list：捕获列表，用于捕获外部变量，让 lambda 函数能够访问外部作用域的变量。
parameter list：参数列表，和普通函数的参数列表类似。
return type：返回类型，可省略，编译器能自动推导。
function body：函数体，是 lambda 函数的具体实现。
*/
```
lambda函数的递归调用
```cpp
#include <iostream>

int main() {
    auto factorial = [](auto&& self, int n) -> int {
        if (n == 0) {
            return 1;
        }
        return n * self(self, n - 1);
    };

    int result = factorial(factorial, 5);
    std::cout << "5 的阶乘是: " << result << std::endl;
    return 0;
}
```
