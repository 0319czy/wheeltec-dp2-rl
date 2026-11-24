# 预编译固件（VLA 串口指令补丁版）· 2026-11-24

- 来源：`wheeltec_rl/firmware/DP2_HAL_work` + `vla_command_halwork.patch`（vla_cmd.h / usart3.c / control.c 三处）
- 工具链：PlatformIO `pio run -e dp2`（本机 `/home/czy/.local/bin/pio`）
- 结果：`SUCCESS`，RAM 46.2%（22,732 / 49,152 B），Flash 60.1%（157,668 / 262,144 B）
- 产物：
  - `DP2_HAL_work_vla_patch_firmware.bin`（158,176 B，烧写用）
  - `DP2_HAL_work_vla_patch_firmware.elf`（315,196 B，调试/符号用）
- 校验（sha256 见下，烧写前请核对）：
- `51fcaaa3314c0e94dc7b28d0a7b37cef0c17bd65812c0754d9e170e264933842`  `DP2_HAL_work_vla_patch_firmware.bin`

> 注意：这是**补丁版**固件，会新增 `#Vxxx,yyy;` 串口指令解析（USART3）与安全停机逻辑；
> 烧写前务必先备份车上原固件（README 有步骤），并用 `test_parse_vw` 单测确认协议行为。
