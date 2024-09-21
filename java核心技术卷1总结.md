---
title: java核心技术卷1总结
tags:
---
# 第一章 Java程序设计概述
1. 简单性
2. 面向对象
3. 分布式
4. 健壮性
5. 安全性
6. 体系结构中立
7. 可移植性
8. 解释性
9. 高性能
10. 多线程
11. 动态性
# 第二章 Java程序设计环境
1. JDK（Java Development Kit） 和 JRE（Java Runtime Environment）： jdk不仅包含java运行时环境，还包含调试工具等开发工具。jre仅包含运行时环境
2. javaSE和javaEE：javaSE仅包含java的核心库，可以胜任一般的简单程序开发。javaEE包含网络和服务器组件，一般用来开发服务器。javaSE和javaEE使用相同的JDK，只是javaEE需要服务器环境支持
3. Java和JDK的版本问题：最初的 Java 版本采用的是 1.x 的版本号格式，主版本号一直是 1，次版本号才表示具体的 Java 版本，例如 Java 1.4、Java 1.5 等。  
从 Java 6 开始，Oracle 采用了更简洁的版本命名，抛弃了 1.x 的版本号格式，直接使用主版本号，如 Java 6、Java 7 等。  
高版本的JDK会兼容低版本的Java，使用命令可以指定特定的java版本。
```
javac --release 8 MyProgram.java
```
长期支持（LTS）：如 Java 8、Java 11、Java 17，Oracle 会提供多年的长期支持，包括安全补丁和错误修复。  

# 第三章 Java的基本程序设计结构
### 1. javadoc注释
Javadoc 注释使用 /** ... */ 的格式，通常用于类、接口、方法和字段的说明。  
每个注释可以包含描述和多个 Javadoc 标签，例如 @param、@return 等。这些标签帮助生成更清晰和结构化的文档。
```java
/**
 * 这是一个简单的计算器类，提供了加法和减法运算。
 * 
 * @author John Doe
 * @version 1.0
 * @since 2024
 */
public class Calculator {

    /**
     * 计算两个整数的和。
     * 
     * @param a 第一个整数
     * @param b 第二个整数
     * @return 两个整数的和
     */
    public int add(int a, int b) {
        return a + b;
    }

    /**
     * 计算两个整数的差值。
     * 
     * @param a 第一个整数
     * @param b 第二个整数
     * @return 两个整数的差值
     */
    public int subtract(int a, int b) {
        return a - b;
    }
}
```
常用的注释
```
@author：用于标注类或接口的作者。
@version：标注版本信息。
@param：描述方法参数，适用于每个方法参数。
@return：描述方法的返回值，适用于非 void 方法。
@throws 或 @exception：描述方法抛出的异常。
@since：表示类或方法自哪个版本开始存在。
@see：引用相关类或方法的链接。
```
### 2. 字符串
1. java字符串重载了+运算符，用于拼接字符串
2. 字符串不可变
3. 字符串比较重载equals
### 3. 数组
java可以定义不规则数组。
```java
int[][] jaggedArray = new int[3][];
jaggedArray[0] = new int[2]; // 第 1 行有 2 个元素
jaggedArray[1] = new int[4]; // 第 2 行有 4 个元素
jaggedArray[2] = new int[3]; // 第 3 行有 3 个元素
```
# 第四章 对象与类
java中所有对象都声明在堆中，不能声明在栈中。也就是说对象声明必须调用new
### 包

# 第五章 继承
1. java类只能有一个父类。
2. java中，子类可以覆写父类的方法。与此不同，c++中，子类只能覆写父类的虚方法。
3. 在java中，通过将子类对象赋值给基类引用来实现多态。 
4. 在 Java 中，当你将子类对象赋值给基类引用时，如果子类覆写了基类的方法，基类引用调用这个方法时，将会调用子类的实现。这种行为称为运行时多态或动态绑定，即方法的调用在程序运行时根据对象的实际类型（即子类类型）进行绑定。
5. java中，有抽象方法和抽象类。抽象方法只有在抽象类或接口中才能声明。抽象类中可以有实现过的方法。子类必须实现抽象方法。
6. java类的访问控制
    public：任何地方都可以访问，类、方法、成员均可使用。
    protected：同包内和子类可以访问。
    default（包级访问）：仅限同一个包内访问。
    private：只能在类内部访问，最严格的访问控制。
7. object类中的三个重要方法：equals，hashCode，toString
# 第六章 接口、lambda表达式与内部类
### 接口
1. 接口中的方法默认属性是public、abstract
2. 接口中的变量默认是常量
3. Java8中，接口中也可以有实现的方法，成为默认方法，使用default修饰。默认方法发和基类中的方法冲突时，遵从类优先原则。
4. 接口注入，接口注入是一种依赖注入的方式，允许通过外部注入接口的实现类，使得类不依赖于具体的实现。
5. 接口注入和基类引用的区别：一个是面向接口编程，一个是面向类编程。
### 几个重要的接口
1. cloneable接口，java中，对象的拷贝默认是浅拷贝，需要调用clone方法才能实现深拷贝。
2. comparable接口，实现comparable的类可以调用sort方法进行排序。

### lambda表达式
1. lambda表达式的语法如下：
```
(parameters) -> expression
(parameters) -> { statements }
```
2. lambda表达式是一个函数，而且整个表达式可以返回一个函数，可以使用一个变量接收这个函数。
3. lambda表达式可以实现函数式接口（即只有一个函数的接口）
4. lambda表达式可以结合foreach进行集合的遍历
# 第七章 异常、断言和日志
异常是可能解决的程序运行时问题，断言会抛出error，断言不能用于程序的控制流。
# 第八章 泛型程序设计
1. 泛型一般有类的泛型和方法的泛型
# 第九章 集合

# 第十章 图形用户界面程序设计

# 第十一章 Swing用户界面组件

# 第十二章 并发
