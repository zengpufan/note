# 简介
- JavaScript 不仅可以在浏览器中执行，也可以在服务端执行，甚至可以在任意搭载了 JavaScript 引擎 的设备中执行。
浏览器中嵌入了 JavaScript 引擎，有时也称作“JavaScript 虚拟机”。

- 最近出现了许多新语言，这些语言在浏览器中执行之前，都会被 编译（转化）成 JavaScript。

- javaScript的规范与手册
ECMA-262 规范 包含了大部分深入的、详细的、规范化的关于 JavaScript 的信息。这份规范明确地定义了这门语言。
MDN（Mozilla）JavaScript 索引 是一个带有用例和其他信息的主要的手册。
- JavaScript控制台
JavaScript是一个解释语言可以直接在控制台中输入代码执行和调试。在浏览器中，控制台的快捷键如下。在windows系统中，F12可以打开控制台。在Mac中，option+command+I可以打开控制台。
# javaScript基础知识

## 将js嵌入到html中
- 使用script标签，直接嵌入到html代码中
```html
  <script>
    alert('Hello, world!');
  </script>
```
- 引用外部的脚本
```html
<script src="/path/to/script.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/lodash.js/4.17.11/lodash.js"></script>
```
## use strict
这个指令看上去像一个字符串 "use strict" 或者 'use strict'。当它处于脚本文件的顶部时，则整个脚本文件都将以“现代”模式进行工作。
请确保 "use strict" 出现在脚本的最顶部，否则严格模式可能无法启用。

## 变量与常量
JS是一种弱类型语言，变量可以被赋值为任意的数据类型。
我们可以使用 var、let 或 const 声明变量来存储数据。
let — 现代的变量声明方式。
var — 老旧的变量声明方式。一般情况下，我们不会再使用它。但是，我们会在 老旧的 "var" 章节介绍 var 和 let 的微妙差别，以防你需要它们。
const — 类似于 let，但是变量的值无法被修改。

## 

# 数据结构
## 数组
- 数组的声明
js中，数组同样是弱类型的，可以将多种不同的数据类型存储到同一个数组中。
```js
let arr = new Array();
let arr = [];
```
- 数组的api
```js
//下标索引
fruits[3] = 'Lemon'; 
// 现在变成 ["Apple", "Orange", "Pear", "Lemon"]

//at索引，at索引可以使用负值，但是下表索引不行
let fruits = ["Apple", "Orange", "Plum"];
// 与 fruits[fruits.length-1] 相同
alert( fruits.at(-1) ); // Plum

//push pop shift unshift
//末尾添加，末尾取出，头部添加，头部取出
fruits.push("Orange", "Peach");
fruits.unshift("Pineapple", "Lemon");

```
- 数组的length
length变量中存储的是数组的长度。
length是可写的，可以修改length来延长和截断数组。
延长的数组新元素都是NaN，截断的数组元素会被删除，不可恢复。