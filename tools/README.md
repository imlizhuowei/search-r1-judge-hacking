# 抢卡 + 自动开跑

`grab_gpu.py` 在你自己的电脑上运行：反复对你的实例点"开机"，开机成功后通过 SSH 在服务器上执行
`scripts/on_boot.sh`（拉代码 → 开检索 → 检查工具 → 冒烟训练），每一步推送到手机。

**风险**：用的是网页控制台的接口和登录 token，不是公开 API；token 等同密码；平台可能限制或封禁自动化使用。

## 一次性准备

1. 电脑上：`pip install requests`
2. **控制台 token**：登录 autodl.com → F12 → Network → 点任意一个 `www.autodl.com/api/v1/...` 请求 →
   Request Headers → 复制 `Authorization` 的值，存成 `~/.autodl_token`（Windows 是 `C:\Users\<你>\.autodl_token`），一行。
3. **SSH 免密码登录**（只需做一次，公钥存在实例系统盘里，克隆后也在）：
   - 电脑上：`ssh-keygen -t ed25519`（一路回车）
   - 把 `~/.ssh/id_ed25519.pub` 的内容追加到服务器的 `/root/.ssh/authorized_keys`
   - 测试：`ssh -p <端口> root@<地址> echo ok` 不再要密码即成功（端口和地址在控制台"登录指令"里）
4. **手机推送**（可选）：iPhone 装 Bark，复制它给你的 key。

## 运行

```
python tools/grab_gpu.py --instance <实例UUID> --ssh "ssh -p <端口> root@<地址>" --bark-key <key>
```

- `--interval 3`：每 3 秒试一次（最低 2 秒）
- `--job smoke | smoke_gca | none`：开机后跑什么
- `--after keep | shutdown | shutdown_always`：任务结束后 保持开机（默认）/ 成功才关机（失败保留现场方便排查）/ 无论成败都关机
- 电脑要保持开机、不休眠
