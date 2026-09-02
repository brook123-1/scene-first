# GitHub 维护者入门指南

这份指南假设你主要通过 GitHub 网页维护 Scene First。私有研发仓库与公开仓库必须保持独立：公开仓库只能从 `scripts/public-export-files.txt` 的干净导出创建，不能复制私有 Git 历史，也不能直接修改私有仓库可见性。

## 先认识六个页面

- **Code**：代码、分支、提交历史和 README。默认分支应始终是可运行、已审查的版本。
- **Issues**：报告 bug、提需求和跟踪明确任务。Issue 不是客服私聊，公开后所有人都能看见附件。
- **Pull requests**：别人提出的代码变更。这里看 diff、CI、review 和合并记录。
- **Actions**：自动测试。绿色表示已配置的检查通过，不代表隐私、许可和产品效果自动合格。
- **Releases**：面向使用者发布带版本号的稳定快照。Release 与部署网站是两件事。
- **Settings**：可见性、Issues、Security、Actions、Collaborators、Rules/Branches 等高风险设置。

## 第一次公开之前

不要先点 Public 再补材料。GitHub 会公开完整代码、提交历史和 Actions 日志，任何人都可以 fork。建议顺序：

1. 在私有源仓库的干净提交上运行 `npm run repo:export-public -- -Destination <仓库外空目录>`。
2. 对导出目录重新检查 Secret、真人数据、本机路径、第三方许可和图片 metadata。
3. 审计通过后才在导出目录执行 `git init`，并确认 `git log` 只有新的 root commit。
4. 先在 GitHub 新建一个全新的 **Private** staging repository，push 干净公开历史并观察首次 Actions。
5. 请至少一位不熟悉项目的人只看 README 完成安装。
6. 设置 repository description、topics 和只含 synthetic/许可清晰素材的 social preview。
7. 开启 Issues、Private Vulnerability Reporting 和默认分支保护。
8. 最后才对这个新的 staging repository 执行 **Settings → General → Danger Zone → Change repository visibility → Public**；绝不能对私有研发仓库这样做。

GitHub 的官方可见性说明：<https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility>

## 怎么看 CI

打开 **Actions**，点最新一次 `CI`。每个 job 都应为绿色。点进红色 job，展开第一个失败步骤，先看真实错误，不要反复点 Re-run 让偶发错误“变绿”。

本项目的 CI 只验证无 Key 的 Python 测试、编译、TypeScript、synthetic avatar schema 和确定性占位资产。Chrome 真机交互、付费 Provider、手机 HEIC 和 staging 都是人工项。PR 不能用“CI 绿”替代隐私审查。

## 处理 Issue

1. 先确认描述是否包含 Key、私人照片、个人数据或未打码日志；有就立即隐藏/删除敏感内容，并按泄漏流程处理。
2. 能复现就加合适 label，例如 `bug`；资料不够就只问最小必要信息。
3. 要求使用 synthetic fixture，不要让报告者私发陌生真人照片。
4. 重复 Issue 留下原 Issue 链接后关闭；spam 直接标记 spam、删除或 block，不争论。
5. 解决后链接修复 PR/commit 并关闭。

**Issue 与 Discussion**：Issue 适合有明确完成条件的 bug/任务；Discussion 适合开放问答、想法和社区交流。项目初期可以只开 Issues，等维护精力稳定后再开 Discussions。

## 审查、合并或拒绝 PR

按顺序看：

1. PR 作者说明了什么行为变化？是否超出一件事？
2. **Files changed** 是否出现 `.env`、`.local`、二进制模型、真实照片、crop、私人 screenshot、绝对路径或新域名？
3. Provider payload 是否扩大了上传范围？人工确认、fallback、mask 外验证有没有变弱？
4. 新依赖、模型和图片是否提供第一方许可来源与再分发依据？
5. 测试和文档是否同步？CI 是否真实通过？
6. 有疑问就点 **Review changes → Request changes**，写清必须改什么。无需因为对方投入了时间而合并。

适合合并时优先 **Squash and merge**，让默认分支历史清晰。拒绝时说明与项目范围、证据或安全边界不符，然后关闭；保持礼貌但不承诺未来合并。

### 绝不能直接 merge

- 含任何 Key、Token、Cookie、Authorization header、证书或 `.env`；
- 含未授权真人照片、个人数据、真实 crop、私人截图或 benchmark 原图；
- 含权属不清模型/字体/图标/avatar；
- 静默把 crop 改成完整图片上传，或把 Key 放进前端；
- 关闭人工确认、扩大公开调试路由、弱化 outside-mask 验证；
- 付费 Provider 测试替代 mock，或 CI 需要真实 Secret；
- 大量自动生成代码但作者不能解释数据流和许可；
- CI 失败、关键问题未回复、diff 与 PR 描述不符。

## Branch protection

公开前在 **Settings → Rules → Rulesets**（或 Branches）保护默认分支：要求 PR、要求 `CI / safe-tests` 状态检查、要求会话解决、禁止 force push 和删除。单人项目可以先不强制第二人批准，但仍通过 PR 保留自审记录。

官方说明：<https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches>

## Private Vulnerability Reporting

公开后到 **Settings → Security and quality / Advanced Security → Private vulnerability reporting → Enable**。再用匿名/另一个账户确认 Security 页面出现 **Report a vulnerability**。同时打开 Security alerts 通知。

官方步骤：<https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/configure-vulnerability-reporting/configure-for-a-repository>

## API Key 泄漏应急

1. 不要先只删 GitHub 上那一行，也不要在公开 Issue 继续粘贴 Key。
2. 立即到对应 Provider **撤销/轮换**，必要时暂停计费和检查调用日志。
3. 保存最小化的私人事件记录：Key 类型、首次暴露时间、路径/commit、处置时间；不保存完整 Key。
4. 删除当前树中的值，检查 Actions 日志、Release、artifact、PR、fork 和完整 Git 历史。
5. 如果进入 Git 历史，先保持仓库 Private；制定历史清理或干净公开仓库方案，得到作者授权后再执行。
6. 重新运行 gitleaks。旧 Key 即使已撤销，也要在报告中说明历史曾暴露。

## Collaborator

只给真正需要的人权限。阅读者无需 collaborator；普通贡献者通过 fork + PR。需要合并的人给 Write，管理设置的人才给 Maintain/Admin。定期到 **Settings → Collaborators** 删除不再需要的访问；不要共享账户或 Token。

## Fork、Star、Watch

- **Fork**：别人复制仓库到自己的账户，可独立修改；公开后无法保证删除所有 fork。
- **Star**：收藏/表达关注，不会自动收到每条通知。
- **Watch**：订阅通知。维护者至少订阅 Issues、PR 和 Security alerts，避免漏掉漏洞报告。

## 发布和撤回 Release

按 `docs/RELEASING.md` 操作。先更新 changelog、验证干净安装、确认许可和截图，再由明确 commit 创建 tag 和 Release。首个建议版本是 `v0.1.0`，但本 Sprint 不创建。

Release 出错时可以编辑说明或删除 Release，但公开 tag、源码归档、fork 和下载副本可能仍存在。普通 bug 发补丁版本；Secret/真人数据先处置泄漏，再评估历史和缓存，不能把“删除 Release”当作撤回互联网。

GitHub Release 官方说明：<https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository>

## 建议的最低维护频率

- 每周一次：看 Issues、PR、Actions、Security/Dependabot alerts；处理 spam。
- 每月一次：更新安全依赖，检查 README Quick Start 和链接；合并或关闭长期无响应 PR。
- 每个 Release 前：完整 clean setup、测试、两类 secret scan、图片/许可审计和 checklist。
- 暂时没精力时：在 README/Issue 写清维护状态，不要让安全报告长期无人看。
