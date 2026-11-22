/* vla_cmd.h - continuous (v, omega) ASCII command helpers for DP2 RTOS firmware.
 *
 * Added by deploy/firmware-patch/vla_command.patch; not part of the upstream
 * WHEELTEC source tree.  Pure functions only: no STM32 HAL dependency, so the
 * same header is compiled by test_parse_vw.c on a PC with gcc.
 *
 * Wire format (host -> car, 10 Hz):
 *     #V<v_mm_s>,<w_mrad_s>;
 *     v_mm_s   signed int, forward positive,  mm/s,   clamped to +/-350
 *     w_mrad_s signed int, left/CCW positive, mrad/s, clamped to +/-1000
 *
 * Calibration: WHEEL_B_M / WHEEL_R_M are the HALF track width and wheel radius.
 * Current values come from assets/lowlevel/dp2_nopole3d.xml (b=0.095, r=0.0335);
 * replace them with measured values before real runs (marked TODO calibrate).
 */
#ifndef DP2_VLA_CMD_H
#define DP2_VLA_CMD_H

#include <stdint.h>

#define VLA_WHEEL_B_M   0.095f   /* TODO calibrate: half track width, m */
#define VLA_WHEEL_R_M   0.0335f  /* TODO calibrate: wheel radius, m */
#define VLA_CTRL_DT_S   0.01f    /* Normal() is called every 10 ms */
#define VLA_TIMEOUT_MS  300u     /* same dead-man timeout as the char remote */
#define VLA_VMAX_MM_S   350      /* spec section 2: |v| <= 0.35 m/s */
#define VLA_WMAX_MRAD_S 1000     /* spec section 2: |w| <= 1.0 rad/s */

/* Parse the body of a #...; frame, i.e. "V<v>,<w>" (no '#', no ';').
 * Returns 1 on success, 0 on any malformed input.  Values are clamped. */
static inline int vla_parse_vw(const char *body, int16_t *v_mm_s, int16_t *w_mrad_s)
{
    int i = 1;
    int sign = 1;
    int digits = 0;
    long v = 0;
    long w = 0;

    if (body == 0 || v_mm_s == 0 || w_mrad_s == 0) return 0;
    if (body[0] != 'V') return 0;

    if (body[i] == '-') { sign = -1; i++; }
    while (body[i] >= '0' && body[i] <= '9') {
        v = v * 10 + (long)(body[i] - '0');
        i++;
        digits++;
        if (digits > 6) return 0;
    }
    if (digits == 0) return 0;
    if (body[i] != ',') return 0;
    v *= sign;
    i++;

    sign = 1;
    digits = 0;
    if (body[i] == '-') { sign = -1; i++; }
    while (body[i] >= '0' && body[i] <= '9') {
        w = w * 10 + (long)(body[i] - '0');
        i++;
        digits++;
        if (digits > 6) return 0;
    }
    if (digits == 0) return 0;
    if (body[i] != '\0') return 0;
    w *= sign;

    if (v > VLA_VMAX_MM_S) v = VLA_VMAX_MM_S;
    if (v < -VLA_VMAX_MM_S) v = -VLA_VMAX_MM_S;
    if (w > VLA_WMAX_MRAD_S) w = VLA_WMAX_MRAD_S;
    if (w < -VLA_WMAX_MRAD_S) w = -VLA_WMAX_MRAD_S;

    *v_mm_s = (int16_t)v;
    *w_mrad_s = (int16_t)w;
    return 1;
}

/* Differential wheel-angle target increment for one 10 ms Normal() tick:
 *     dtheta_l = (v - w * b) * dt / r
 *     dtheta_r = (v + w * b) * dt / r
 * with v in m/s, w in rad/s.  Added to Target_theta_L / Target_theta_R. */
static inline void vla_delta_theta(int16_t v_mm_s, int16_t w_mrad_s,
                            float *dtheta_l, float *dtheta_r)
{
    float v = (float)v_mm_s / 1000.0f;
    float w = (float)w_mrad_s / 1000.0f;
    if (dtheta_l != 0) {
        *dtheta_l = (v - w * VLA_WHEEL_B_M) * VLA_CTRL_DT_S / VLA_WHEEL_R_M;
    }
    if (dtheta_r != 0) {
        *dtheta_r = (v + w * VLA_WHEEL_B_M) * VLA_CTRL_DT_S / VLA_WHEEL_R_M;
    }
}

/* 300 ms dead-man: a frame is fresh when now - last <= 300 ms (wrap-safe). */
static inline int vla_fresh(uint32_t now_ms, uint32_t last_ms)
{
    return (uint32_t)(now_ms - last_ms) <= VLA_TIMEOUT_MS;
}

#endif /* DP2_VLA_CMD_H */
