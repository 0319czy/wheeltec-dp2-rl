/* DP2 纯 PPO 站立策略的定点推理，最终输出范围为 ±1。 */
#include <stdint.h>
#include "rl_mj_fixed_weights.h"

#define OBS 6
#define HIDDEN 64

static inline int16_t tanh_q14(int32_t value, unsigned bits)
{
    uint32_t magnitude = value < 0 ? (uint32_t)(-value) : (uint32_t)value;
    uint32_t index = magnitude >> (bits - 7U);
    int32_t output = 16384;
    if (index < 768U) {
        uint32_t fraction = (magnitude >> (bits - 15U)) & 255U;
        int32_t slope = q_tanh[index + 1U] - q_tanh[index];
        output = q_tanh[index] + ((slope * (int32_t)fraction + 128) >> 8);
    }
    return (int16_t)(value < 0 ? -output : output);
}

#if defined(__GNUC__)
__attribute__((optimize("O3")))
#endif
void MJ_Policy_Forward(const float obs[OBS], float act[2])
{
    int16_t x[OBS], h[HIDDEN], g[HIDDEN];
    for (unsigned i = 0; i < OBS; ++i) {
        float v = (obs[i] - q_mean[i]) * q_inv_std[i];
        if (v < -10.0f) v = -10.0f;
        if (v > 10.0f) v = 10.0f;
        x[i] = (int16_t)(v * 2048.0f + (v < 0.0f ? -0.5f : 0.5f));
    }
    /* 导出时用输入上限和逐行绝对权重之和验证了 int32_t 累加范围。 */
    for (unsigned j = 0; j < HIDDEN; ++j) {
        int32_t sum = q_b1[j];
        for (unsigned i = 0; i < OBS; ++i) sum += q_w1[j * OBS + i] * (int32_t)x[i];
        h[j] = tanh_q14(sum, 26U);
    }
    for (unsigned j = 0; j < HIDDEN; ++j) {
        int32_t sum = q_b2[j];
        for (unsigned i = 0; i < HIDDEN; ++i) sum += q_w2[j * HIDDEN + i] * (int32_t)h[i];
        g[j] = tanh_q14(sum, 27U);
    }
    for (unsigned j = 0; j < 2; ++j) {
        int32_t sum = q_b3[j];
        for (unsigned i = 0; i < HIDDEN; ++i) sum += q_w3[j * HIDDEN + i] * (int32_t)g[i];
        /* 新策略末层用 Q15 权重，配合 Q14 隐藏层使累加保持在 int32_t 范围内。 */
        act[j] = (float)sum * (1.0f / 536870912.0f);
    }
}
