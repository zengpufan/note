# 概述
1. 做题之前仔细阅读题目，多读几遍，注意数据范围。

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
