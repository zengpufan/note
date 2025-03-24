# 概述
1. 做题之前仔细阅读题目，多读几遍，注意数据范围。
## 有单调性的枚举问题
## 动态规划问题
动态规划是一种求解最优解的算法。题目中一般蕴含有求解最大值\最小值的意思。
- 动态规划的适用性证明：
最优子结构+重叠子问题：最优子结构的意思是最优解由其子问题的最优解构成。重叠子结构的意思是，当一个问题取到最优解，不会影响其他问题取到最优解，子问题可以相互重叠。
- 动态规划并不总是完全符合上面两条规范
以leetcode198“打家劫舍”为例，当选择了第i个数据点之后，就不能选择第i+1个数据点，这使得一个问题取到最优解，会影响到另外一个问题取到最优解。但是实际上，为了使用动态规划，我们需要对问题进行进一步拆解，分为选取第i个点和不选去第i个点进行进一步的讨论。
- 动态规划的问题分析方法：
1. 定义最小子问题
2. 状态转移方程

## 数列问题
这类问题，元素和元素之间没有跨元素依赖，只需使用数组的结构就能表示元素之间的关系，不需要使用图数据结构。
## 区间维护问题
区间维护问题可以使用分块算法，这种算法具有较高的泛用性而且性能一般足以通过考试，写起来较为容易。
```cpp
    class divide_block{
        private:
            int block_size;
            int block_cnt;
            int len;
        public:
            vector<int>block_info;
            vector<int>vis;
            vector<int>vv;
            divide_block(const vector<int>&v){
                len=(int)v.size();
                block_size=sqrt(len);
                block_cnt=len/block_size+1;
                block_info=vector<int>(block_cnt);
                vv=v;
                vis=vector<int>(len);
                for( int i=0;i<len;i++){
                    int cur_block=i/block_size;
                    block_info[cur_block]=max(block_info[cur_block],v[i]);
                }
            }
            bool query( int fruit){
                for( int i=0;i<block_cnt;i++){
                    if(block_info[i]>=fruit){
                        int Max=0;
                        int flag=0;
                        for( int j=i*block_size;j<len&&j<(i+1)*block_size;j++){
                            if(vis[j]==true){
                                continue;
                            }
                            if(flag==0&&vv[j]>=fruit){
                                flag=1;
                                vis[j]=true;
                            }
                            else{
                                Max=max(Max,vv[j]);
                            }
                        }
                        block_info[i]=Max;
                        return false;
                    }
                    
                }
                return true;
            }
    };
```

# 没有思路的应对策略
1. 使用数据进行模拟，寻找思路。
2. 观察数据的范围，根据数据范围确定算法的复杂度，进而确定思路。
3. 将题目抽象为数学式子，通过推导确定思路。
4. 尝试解决化简版题目，将一些条件删除，先解决简单问题，在简单问题的基础上解决复杂问题。

# 检查
1. 注释调试点，cout会占用时间复杂度.
2. 注意有的OJ会卡空格。
