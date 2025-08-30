# AccCTR

使用方法：

1、bash build.sh release //编译release

2、bash build.sh debug //编译debug

3、编译和运行UT：
  （1）bash build.sh ut //编译ut，覆盖率在tests/build/cov/gen目录下
  （2）cd build && bash build_test.sh ut //进入到build目录下并运行ut