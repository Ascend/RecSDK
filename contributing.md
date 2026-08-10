# 为Ascend RecSDK贡献

感谢您考虑为 Ascend RecSDK 做出贡献！我们欢迎任何形式的贡献，包括错误修复、功能增强、文档改进等，甚至只是反馈。无论您是经验丰富的开发者还是第一次参与开源项目，您的帮助都非常宝贵。

您可以通过多种方式支持本项目：

- 通过[RecSDK新手任务池](https://gitcode.com/Ascend/RecSDK/issues/1096)参与贡献。
- 审查 Pull Request 并协助其他贡献者。
- 传播项目：在博客文章、社交媒体上分享 RecSDK，或给仓库点个 ⭐。

参与贡献前，请先签署[开放项目贡献者许可协议（CLA）](https://clasign.osinfra.cn/sign/690ca9ddf91c03dee6082ab1)，并提前了解[《社区行为规范》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/code-of-conduct.md)。

## 贡献方式

### Pull Request

提交 PR 前，请先了解[《Pull Request (PR) 提交流程指南》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/pr-guide.md)，掌握从 Fork 到提交、从代码审查到合并的完整 PR 流程。RecSDK 的具体 PR 流程、评审与合入规则参见 [Pull Request](#pullrequest)。

### Issue

提交 Issue 前，请先了解[《Issue 创建与处理指南》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/issue-guide.md)，学习如何有效创建、分类和管理 Issue。如遇 bug，请[提交 Issue](https://gitcode.com/Ascend/RecSDK/issues)。

### SIG会议

可提前通过[sig-RecSDK Etherpad链接](https://etherpad.ascend.osinfra.cn/p/sig-RecSDK)进入共享文档 ，编辑申报议题，在[RecSDK SIG会议日历](https://meeting.ascend.osinfra.cn/?sig=sig-RecSDK)中找到对应会议链接并按时参会。

<a id="pullrequest"></a>

## Pull Request

### PR 创建流程

1. **Fork仓库**：在GitCode平台代码仓库右上角点击"Fork"按钮，Fork一份源代码到个人仓

2. **克隆到本地**：

   将Fork到个人仓的代码克隆到本地进行代码开发

   ```bash
   git clone https://gitcode.com/<your-username>/RecSDK.git
   cd RecSDK
   ```

3. **创建特性分支**

   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/issue-number
   ```

4. **进行开发**
   - 编写代码
   - 添加测试
   - 更新文档
   - 确保代码通过本地测试
5. **提交代码**

   ```bash
   git add .
   git commit -m "[feat] add new feature"
   ```

   本地提交代码前请先执行 pre-commit 检查，检查指导参见[《pre-commit 本地运行指南》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/pre-commit-guide.md)。

6. **推送到 Fork 仓库**

   ```bash
   git push origin feature/your-feature-name
   ```

7. **创建 Pull Request**
   - 访问gitcode仓库页面
   - 点击"Pull Request"或"合并请求"
   - 填写PR描述（见PR创建页面模板）

### PR最佳实践

1. **保持PR小规模**
   - 一次PR只解决一个问题
   - 便于评审和理解
   - 提高合并效率
   - 建议：单个PR不超过1000行（含测试）代码变更
2. **及时更新**
   - 定期同步上游主分支
   - 及时响应评审意见
   - 保持 PR 活跃
3. **清晰描述**
   - 详细描述变更原因和方式
   - 提供测试方法
   - 添加截图或示例（如适用）

### PR提交规范

#### Commit 消息格式

所有提交必须遵循以下格式：

```text
<type>(<scope>): <subject>

<body>
```

#### Type（类型）

- feat: 新功能
- fix: Bug修复
- docs: 文档更新
- style: 代码格式（不影响功能，如空格、格式化等）
- refactor: 重构（既不是 Bug 修复也不是新功能）
- perf: 性能优化
- test: 测试相关
- chore: 构建/工具变更（如依赖更新、构建配置等）
- ci: CI/CD 相关变更

#### Scope（范围，可选）

指定提交影响的范围，例如：

- `op`: 算子
- `emb_table`: 稀疏表相关
- `docs`: 文档
- `build`: 构建

#### Subject（主题）

- 使用祈使句，首字母小写
- 不超过50个字符
- 不以句号结尾
- 描述"做了什么"而不是"做了什么改动"

#### Body（正文，可选）

- 详细描述变更的原因和方式
- 说明与之前行为的对比
- 可以多行，每行不超过 72 个字符

### PR评审与合入规则

#### 评审要求

1. **评审人员要求**

   - 评审人员必须熟悉相关代码领域
   - 评审人员不能是PR作者本人
2. **评审检查项**
   - ✅ 代码质量和风格
   - ✅ 功能正确性
   - ✅ 测试覆盖率（分支60%，行80%）
   - ✅ 文档完整性
   - ✅ 性能影响
   - ✅ 安全性
   - ✅ 向后兼容性
3. **CI 检查要求**
   - ✅ 所有 CI 检查必须通过
4. **无 Block 评论**
   - PR不能有任何未解决问题

#### 合入规则

1. **Squash and Merge**
   - 将 PR 的所有提交合并为一个提交
   - 保持主分支历史清晰
   - 提交消息使用PR标题
2. **必须满足的条件**
   - ✅ 至少2位Maintainer或Committer的/lgtm，和1个/approve
3. **禁止的操作**
   - ❌ 禁止 Force Push 到主分支
   - ❌ 禁止合并自己的 PR（必须有他人评审）

#### 合并权限

- **Maintainer**：可以合并任何PR
- **Committer**：可以合并任何PR
- **Contributor**：无合并权限，需要等待Maintainer或Committer合并

## CI说明

CI检查项目有：

* 编译构建：
  * PR-build-SAST-check
  * Build_TF
  * Build_torchrec
  * Build_add_ons
* 恶意代码检查：Antipoison
* 编码安全与规范检查：CodeCheck
* 开源片段检查：SCA
* 开发者测试：
  * presmoke_TF
  * UT_torchrec
  * PreSmoke_torchrec
  * UT_python_common
  * UT_python_tf1
  * UT_AccCTR
  * UT_cpp_tf1
* 流水线：PR-pipeline_RecSDK

任意一项失败可以通过详情链接查看具体问题。如果是CI自身故障，请[联系committer](https://gitcode.com/Ascend/community/blob/master/MindSDK/sigs/RecSDK/sig-info.yaml)，或通过评论"rebuild"尝试重新构建。

### PreSmoke_torchrec运行说明

创建PR时，`PreSmoke_torchrec`检查会根据PR修改的代码文件运行相关的测试用例。`PreSmoke_torchrec`检查的运行入口为：[run_presmoke.sh](build/run_presmoke.sh)

调用方式如下：

```bash
cd build
bash run_presmoke.sh
```

## 分支与 Tag 命名规则

RecSDK 是基于开源社区项目的二次开发仓库，分支与 Tag 命名遵循[《第三方开源仓库分支指南》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/third-party-repo-branch-guide.md)。RecSDK 主要分支说明如下：

| 分支 | 说明 | 维护状态 |
| --- | --- | --- |
| develop | 主开发分支 | 长期维护 |
| develop_examples_and_tools | 示例、模型样例及模型相关工具 | 活跃维护 |
| develop_torch_benchmark | 基准模型适配 | 活跃维护 |

## 参考

- 开发规范
  - [《Ascend C++ 编码风格指南》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-cpp-coding-style-guide.md)
  - [《Ascend Python 编码风格指南》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-python-coding-style-guide.md)
- 安全编程指导
  - [《Ascend C++ 安全编程指南》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-cpp-secure-coding-guide.md)
  - [《Ascend Python 安全编程指南》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-python-secure-coding-guide.md)
- [《Ascend 安全编译选项指南 (C&C++)》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-secure-compile-guide.md)
- 更多社区相关规范，请访问 [Ascend 社区 community](https://gitcode.com/Ascend/community)

## Special Interest Group

### 工作目标和范围

1. 技术聚焦
围绕基于NPU的推荐系统SDK和算子进行深入研究，推动技术发展，解决实际问题。
2. 促进协作
通过组织会议、技术分享等方式，促进成员之间的协作和知识共享，提升技术水平。
3. 最佳实践
在技术实现、接口设计、开发流程等方面推动最佳实践，降低协作成本，提升系统兼容性和可维护性。
4. 社区建设
通过代码贡献、技术分享等方式，培养技术人才，推动社区生态建设。

### 角色说明

[Ascend 社区角色定义及晋升机制](https://gitcode.com/Ascend/community/blob/master/docs/role-definition-and-promotion-mechanism.md)

### 例会

* 周期：每1个月举行一次例会，可通过[Ascend开源社区](https://meeting.ascend.osinfra.cn/)搜索、查看sig-RecSDK的会议链接。
* 申报议题：通过[sig-RecSDK Etherpad链接](https://etherpad.ascend.osinfra.cn/p/sig-RecSDK)进入共享文档，编辑申报议题。
* 参会人员：maintainer、committer、contributor等核心成员，其他对本SIG感兴趣的人员。
* 会议内容：讨论遗留问题和进展；当期申报的议题；需求评审、任务和优先级；需求规划和进展（roadmap）；新晋maintainer、committer准入评审。
* 会议归档：会议纪要位于[sig-RecSDK Etherpad链接](https://etherpad.ascend.osinfra.cn/p/sig-RecSDK)。

### 成员列表

[SIG成员列表](https://gitcode.com/Ascend/community/blob/master/MindSDK/sigs/RecSDK/sig-info.yaml)。
